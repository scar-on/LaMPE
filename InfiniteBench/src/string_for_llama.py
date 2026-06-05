# -*- coding:utf-8 -*-

from typing import List, Optional, Tuple, Union

from torch import nn
import math
from transformers.models.llama.modeling_llama import rotate_half, repeat_kv
import torch
import transformers
from transformers.cache_utils import Cache
import pdb
import math
from transformers.modeling_outputs import BaseModelOutputWithPast, MoeCausalLMOutputWithPast
from transformers.modeling_outputs import CausalLMOutputWithPast
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from transformers import LlamaConfig, PretrainedConfig
from flash_attn import flash_attn_with_kvcache, flash_attn_func
import flash_attn_2_cuda as flash_attn_cuda
import math


def new_flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=None,
        v=None,
        rotary_cos=None,
        rotary_sin=None,
        cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
        cache_batch_idx: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        rotary_interleaved=True,
        alibi_slopes=None,
        num_splits=0,
):
    """
    If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
    k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
    the previous step, and update them with the new keys/values from the current step, and do
    attention with the updated cache, all in 1 kernel.

    If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
    For example, the KV cache could be pre-allocated with the max sequence length, and you can use
    cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.

    Also apply rotary embedding if rotary_cos and rotary_sin are passed in. The key @k will be
    rotated by rotary_cos and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If causal or local (i.e., window_size != (-1, -1)), the query @q will be rotated by rotary_cos
    and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If not causal and not local, the query @q will be rotated by rotary_cos and rotary_sin at
    indices cache_seqlens only (i.e. we consider all tokens in @q to be at position cache_seqlens).

    See tests/test_flash_attn.py::test_flash_attn_kvcache for examples of how to use this function.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Note: Does not support backward pass.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
            page_block_size must be a multiple of 256.
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
            k with k_cache, starting at the indices specified by cache_seqlens.
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim). Similar to k.
        rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). If not None, we apply rotary embedding
            to k and q. Only applicable if k and v are passed in. rotary_dim must be divisible by 16.
        rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Similar to rotary_cos.
        cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
            KV cache.
        block_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the KV cache.
            If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
            If the indices are not distinct, and k and v are provided, the values updated in the cache
                 might come from any of the duplicate indices.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        rotary_interleaved: bool. Only applicable if rotary_cos and rotary_sin are passed in.
            If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
            rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1
            (i.e. GPT-NeoX style).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
           If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
           to automatically determine the number of splits.
           Don't change this unless you know what you are doing.

    Return:
        out: (batch_size, seqlen, nheads, headdim).
    """
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    maybe_contiguous = lambda x: x.contiguous() if x is not None and x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)
    cache_batch_idx = maybe_contiguous(cache_batch_idx)
    block_table = maybe_contiguous(block_table)
    out, softmax_lse = flash_attn_cuda.fwd_kvcache(
        q,
        k_cache,
        v_cache,
        k,
        v,
        cache_seqlens,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        block_table,
        alibi_slopes,
        None,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        rotary_interleaved,
        num_splits,
    )
    return out, softmax_lse


def do_flash_decoding(query_states, key_states, value_states, k_cache, v_cache, cache_seqlens, intra=False):
    if key_states is not None:
        key_states = key_states.transpose(1, 2)
    if value_states is not None:
        value_states = value_states.transpose(1, 2)
    output, softmax_lse = new_flash_attn_with_kvcache(query_states.transpose(1, 2), k_cache, v_cache,
                                                      key_states, value_states, cache_seqlens=cache_seqlens)
    # return output.transpose(1, 2), softmax_lse
    return output, softmax_lse


def apply_rotary_pos_emb(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_emb = (x * cos) + (rotate_half(x) * sin)
    return x_emb


def _compute_default_rope_parameters(
        config: Optional[PretrainedConfig] = None,
        device: Optional["torch.device"] = None,
        seq_len: Optional[int] = None,
        **rope_kwargs,
) -> Tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        dim = int((config.hidden_size // config.num_attention_heads) * partial_rotary_factor)

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq


def _compute_llama3_parameters(
        config: PretrainedConfig, device: "torch.device", seq_len: Optional[int] = None, **rope_kwargs
) -> Tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies for llama 3.1.

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
    # Gets the default RoPE parameters
    inv_freq = _compute_default_rope_parameters(config, device, seq_len, **rope_kwargs)

    factor = config.rope_scaling["factor"] # `8` in the original implementation
    low_freq_factor = config.rope_scaling["low_freq_factor"]  # `1` in the original implementation
    high_freq_factor = config.rope_scaling["high_freq_factor"]  # `4` in the original implementation
    old_context_len = config.rope_scaling["original_max_position_embeddings"]  # `8192` in the original implementation
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in inv_freq:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
    inv_freq = torch.tensor(new_freqs, dtype=inv_freq.dtype, device=inv_freq.device)
    return inv_freq


class RotaryEmbedding(nn.Module):
    def __init__(
            self,
            dim=None,
            max_position_embeddings=2048,
            base=10000,
            device=None,
            scaling_factor=1.0,
            rope_type="default",
            config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        if config is not None and hasattr(config, "rope_scaling") and config.rope_scaling:
            self.rope_init_fn = _compute_llama3_parameters
        else:
            self.rope_init_fn = _compute_default_rope_parameters

        inv_freq = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", attention_factor * emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", attention_factor * emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if not isinstance(seq_len, int):
            seq_len = seq_len.size(-1)
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def string_flash_forward(
        neighbor_query_states,
        shifted_query_states,
        key_states,
        value_states,
):
    bsz, kv_seq_len, _, head_dim = neighbor_query_states.size()
    diag_out, diag_lse, _ = flash_attn_func(
        neighbor_query_states,
        key_states,
        value_states,
        causal=True,
        window_size=[diag_size, 0],
        return_attn_probs=True,
    )  # [bsz, L, h, d]
    triangle_len = (
            kv_seq_len - diag_size
    )  # here we should use kv_seq_len rather than max_kv_len since we have paddings in qkv and attention_mask
    if triangle_len < 0:
        return diag_out
    shifted_out, shifted_lse, _ = flash_attn_func(
        shifted_query_states[:, -triangle_len:, :, :],
        key_states[:, :triangle_len, :, :],
        value_states[:, :triangle_len, :, :],
        causal=True,
        window_size=[-1, -1],
        return_attn_probs=True,
    )  # [bsz, N, h, d]
    # to float32
    L = diag_out.size(1)
    N = shifted_out.size(1)
    S = L - N
    diag_lse = diag_lse.to(torch.float32)
    shifted_lse = shifted_lse.to(torch.float32)
    diag_out_head = diag_out[:, :S]
    diag_lse_tail = diag_lse[:, :, S:]
    diag_out_tail = diag_out[:, S:]

    lse_gap = 1 / (1 + torch.exp(diag_lse_tail - shifted_lse))
    lse_gap_re = 1 / (1 + torch.exp(shifted_lse - diag_lse_tail))
    lse_gap = lse_gap.transpose(1, 2).unsqueeze(-1)
    lse_gap_re = lse_gap_re.transpose(1, 2).unsqueeze(-1)
    merge_out_tail = diag_out_tail * lse_gap_re.to(diag_out_tail) + shifted_out * lse_gap.to(shifted_out)
    output = torch.cat([diag_out_head, merge_out_tail], dim=1)
    return output


def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    kv_seq_len += past_key_value["cache_seqlens"].item()
    past_key_value["cache_seqlens"] += key_states.shape[-2]

    q_seq_len = query_states.shape[-2]
    has_kv_cache = q_seq_len != kv_seq_len
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    key_states = apply_rotary_pos_emb(key_states, cos, sin, position_ids)

    key_cache = past_key_value[0][:, :, 0, :, :]
    value_cache = past_key_value[0][:, :, 1, :, :]

    if not has_kv_cache:
        # pdb.set_trace()
        key_cache[:, kv_seq_len - key_states.shape[-2]:kv_seq_len, :, :] = key_states.transpose(1, 2)
        value_cache[:, kv_seq_len - key_states.shape[-2]:kv_seq_len, :, :] = value_states.transpose(1, 2)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
    else:
        key_cache[:, kv_seq_len - 1, :, :] = key_states.transpose(1, 2)
        value_cache[:, kv_seq_len - 1, :, :] = value_states.transpose(1, 2)

    if not has_kv_cache:
        query_states_1 = apply_rotary_pos_emb(query_states, cos, sin, position_ids)
        position_ids = position_ids - diag_size + local_window
        query_states_2 = apply_rotary_pos_emb(query_states, cos, sin, position_ids)
        query_states_1 = query_states_1.transpose(1, 2)
        query_states_2 = query_states_2.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        attn_output = string_flash_forward(query_states_1, query_states_2, key_states, value_states)
    else:
        query_states_1 = apply_rotary_pos_emb(query_states, cos, sin, position_ids)
        position_ids = position_ids - diag_size + local_window
        query_states_2 = apply_rotary_pos_emb(query_states, cos, sin, position_ids)
        # flash decoding
        shifted_size = kv_seq_len - diag_size
        k_cache_diag = key_cache[:, shifted_size:kv_seq_len, :, :]
        v_cache_diag = value_cache[:, shifted_size:kv_seq_len, :, :]
        out1, lse1 = do_flash_decoding(query_states_1, None, None, k_cache_diag, v_cache_diag,
                                       cache_seqlens=diag_size)
        k_cache_shifted = key_cache[:, :shifted_size, :, :]
        v_cache_shifted = value_cache[:, :shifted_size, :, :]
        out2, lse2 = do_flash_decoding(query_states_2, None, None, k_cache_shifted, v_cache_shifted,
                                       cache_seqlens=shifted_size)
        lse1 = lse1.to(torch.float32)
        lse2 = lse2.to(torch.float32)
        gap21 = 1 / (1 + torch.exp(lse2 - lse1))
        gap12 = 1 / (1 + torch.exp(lse1 - lse2))
        attn_output = out1 * gap21.to(out1) + out2 * gap12.to(out2)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def allocate_inference_cache(
        max_batch_size,
        max_seqlen,
        nheads,
        headdim,
        layers,
        dtype=torch.float16,
):

    assert dtype in [torch.float16, torch.bfloat16, torch.float32]
    kv_cache_shape = (max_batch_size, max_seqlen, 2, nheads, headdim)
    allc_kv_cache = {i: {0: torch.empty(kv_cache_shape, device=layer.self_attn.k_proj.weight.device, dtype=dtype),
                         "cache_seqlens": torch.tensor([0], device=layer.self_attn.k_proj.weight.device).long()} for
                     i, layer in enumerate(layers)}

    return allc_kv_cache


# add cache_position = None for llama31
def flashdecoding_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position=None,
        output_router_logits=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            use_cache = False

    past_key_values_length = 0

    if past_key_values:
        input_ids = input_ids[:, -1].unsqueeze(-1)
        position_ids = position_ids[:, -1].unsqueeze(-1) if position_ids is not None else None

    if use_cache and (past_key_values is None or len(past_key_values)==0):
        num_kv_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // num_attention_heads
        print("allocate kv cache")
        past_key_values = allocate_inference_cache(
            batch_size,
            MAX_CACHE_LEN,
            num_kv_heads,
            head_dim,
            self.layers,
            dtype=self.dtype,
        )

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None

    # embed positions
    hidden_states = inputs_embeds
    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None
    for i, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[i],
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def causal_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        # cache_position=cache_position,
    )

    hidden_states = outputs[0]
    full_logits_length = 32000

    if hidden_states.shape[-2] < full_logits_length:
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        loss = None

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)

            loss = loss_fct(shift_logits, shift_labels)
    else:
        res = 0
        div_len = full_logits_length // 2
        if labels is None:
            # only produce the last logits
            logits = self.lm_head(hidden_states[..., -1:, :])
            logits = logits.float()
            # logits = logits.expand(-1, hidden_states.shape[-2], -1)
            loss = None
        else:
            # calculate loss by chunk
            shift_hidden_states = hidden_states[..., :-1, :]
            shift_labels = labels[..., 1:].contiguous()

            for i in range(0, shift_hidden_states.shape[-2], div_len):
                st = i
                ed = min(i + div_len, shift_hidden_states.shape[-2])
                logits = self.lm_head(shift_hidden_states[..., st:ed, :])
                logits = logits.float()

                shift_logits = logits.contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)

                res = res + loss_fct(shift_logits, shift_labels[st:ed]) * (ed - st)
            loss = res / (hidden_states.shape[-2] - 1)
            logits = None

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

# adapt transformers 4.37.2
def _init_rope(self):
    self.rotary_emb = RotaryEmbedding(
        self.head_dim,
        max_position_embeddings=self.max_position_embeddings,
        base=self.rope_theta,
        config=self.config
    )


diag_size = None
local_window = None

MAX_CACHE_LEN = None
attention_factor = None
MAX_NEW_TOKENS = 1300


def replace_with_string(max_test_length, shifted_offset, small_local_value=128):
    # this is used to pre-allocate KV cache, saving GPU memory
    global MAX_CACHE_LEN
    MAX_CACHE_LEN = max_test_length + MAX_NEW_TOKENS
    # String parameters
    global diag_size
    global local_window
    global attention_factor
    diag_size = shifted_offset
    local_window = small_local_value
    # STRING will make the attention map smooth, we use the attention_factor to recover it, similar with Yarn
    attention_factor = 0.1*math.log(2) + 1

    print("============== [STRING Config for Llama] ===============")
    print(f"Position ids for sliding window attention: {0}-{diag_size}")
    print(f"Position ids for Shifted self attention: {local_window}-{max_test_length-local_window}")
    print(f"Attention factor {attention_factor}")
    print("==============================================")

    transformers.models.llama.modeling_llama.LlamaAttention._init_rope = _init_rope
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = causal_forward
    transformers.models.llama.modeling_llama.LlamaModel.forward = flashdecoding_forward
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = forward
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = RotaryEmbedding

    # for mistral
    transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding = RotaryEmbedding
    transformers.models.mistral.modeling_mistral.MistralModel.forward = flashdecoding_forward
    transformers.models.mistral.modeling_mistral.MistralAttention.forward = forward
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = forward