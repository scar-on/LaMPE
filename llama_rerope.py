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
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_outputs import CausalLMOutputWithPast
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from transformers import LlamaConfig, PretrainedConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS 
from flash_attn import flash_attn_with_kvcache, flash_attn_func
import flash_attn_2_cuda as flash_attn_cuda
import math


def map_ids(X: torch.Tensor, to_min: float, to_max: float) -> torch.Tensor:
    device = X.device  
    x_min = X.min()
    x_max = X.max() 
    mapped_tensor = torch.floor((to_min + ((to_max - to_min) / (x_max - x_min)) * (X - x_min))).to(device)
    return X,mapped_tensor

def find_chunk_last_index(a, re_size):
    while re_size + 1 < a.shape[1] and a[0, re_size] == a[0, re_size + 1]:
        re_size += 1
    return re_size


class LlamaRotaryEmbedding(nn.Module):
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
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings
        
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        # print(inv_freq_expanded.device,position_ids_expanded.device)
        with torch.autocast(device_type=device_type, enabled=False):
            inv_freq_expanded = inv_freq_expanded.to(position_ids_expanded.device)
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)




def apply_rotary_pos_emb_map(q, k, cos, sin,inference): # position_ids=None, unsqueeze_dim=1
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)
    if inference == 1:
        q_embed = (q * cos[:,:,-1,:]) + (rotate_half(q) * sin[:,:,-1,:])if not q is None else None
        k_embed = (k * cos) + (rotate_half(k) * sin) if not k is None else None
    else:
        q_embed = (q * cos) + (rotate_half(q) * sin)if not q is None else None
        k_embed = (k * cos) + (rotate_half(k) * sin)if not k is None else None
    return q_embed, k_embed


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




def flash_prefill_merge(head_out,head_lse,map_out,map_lse,bias_1):
    
    L = head_out.size(1)  #torch.Size([1, seq_len, 32, 128])
    N = map_out.size(1)    #torch.Size([1, sqe_len-resize, 32, 128])

    head_lse,map_lse = head_lse.to(torch.float32),map_lse.to(torch.float32)

    head_out_head = head_out[:, :bias_1]

    head_out_mid = head_out[:, bias_1:]
    
    head_lse_mid = head_lse[:, :, bias_1:]  # 
    
    # 合并 mid
    lse_1 = 1 / (1 + torch.exp(head_lse_mid - map_lse))
    lse_2 = 1 / (1 + torch.exp(map_lse - head_lse_mid))
    lse_1 = lse_1.transpose(1, 2).unsqueeze(-1)
    lse_2 = lse_2.transpose(1, 2).unsqueeze(-1)
    merge_out_mid = head_out_mid * lse_2.to(head_out_mid) + map_out * lse_1.to(map_out)
    
    output = torch.cat([head_out_head, merge_out_mid], dim=1)

    return output

def flash_decode_merge(head_out,head_lse,map_out,map_lse):
    
    head_lse,map_lse = head_lse.to(torch.float32),map_lse.to(torch.float32)
    lse_1 = 1 / (1 + torch.exp(map_lse - head_lse))
    lse_2 = 1 / (1 + torch.exp(head_lse - map_lse))
    lse_1 = lse_1.transpose(1, 2).unsqueeze(-1)
    lse_2 = lse_2.transpose(1, 2).unsqueeze(-1)
    output = head_out * lse_1.to(head_out) + map_out * lse_2.to(map_out) 

    return output


def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        use_length = 4000,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    
    
    # print(attention_factor)
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # [bsz,head,len,dim]
    kv_seq_len = key_states.shape[-2]
    kv_seq_len += past_key_value["cache_seqlens"].item()
    past_key_value["cache_seqlens"] += key_states.shape[-2]

    q_seq_len = query_states.shape[-2]
    has_kv_cache = q_seq_len != kv_seq_len   # 判断是prefill 还是 decode

    # 正常 cos,sin 位置编码
    cos, sin = self.rotary_emb(value_states, position_ids)  #[len,dim]
    key_cache = past_key_value[0][:, :, 0, :, :]    # k cache
    value_cache = past_key_value[0][:, :, 1, :, :]  # v cache
    
    Q_mapping_ids = torch.full((1, kv_seq_len), use_length+1, device=value_states.device)
    K_mapping_ids = torch.full((1, kv_seq_len), 0, device=value_states.device)
    
    map_key_states = key_states.clone()
    if not has_kv_cache:
        # cal mapping_id bias_1 bias_2 re_size_2
        _, key_states = apply_rotary_pos_emb_map(None, key_states, cos, sin, has_kv_cache)  # k position_ids [1,len]
        key_cache[:, kv_seq_len - key_states.shape[-2]:kv_seq_len, :, :] = key_states.transpose(1, 2)  # add k cache
        value_cache[:, kv_seq_len - key_states.shape[-2]:kv_seq_len, :, :] = value_states.transpose(1, 2) # add v cache
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups) # head:8 -> 32 
    else:   
        # pdb.set_trace()
        _, key_states = apply_rotary_pos_emb_map(None, key_states, cos, sin, has_kv_cache)
        key_cache[:, kv_seq_len - 1, :, :] = key_states.transpose(1, 2)  # add k cache
        value_cache[:, kv_seq_len - 1, :, :] = value_states.transpose(1, 2) # add v cache
        
    
    if not has_kv_cache:
        triangle = kv_seq_len - use_length
        # head prefill
        head_query_states, _ = apply_rotary_pos_emb_map(query_states, _ , cos, sin, has_kv_cache) # Q
        head_out, head_lse, _ = flash_attn_func(head_query_states.transpose(1, 2), 
                                                key_states.transpose(1, 2),
                                                value_states.transpose(1, 2), 
                                                causal=True, 
                                                window_size=[use_length, 0], #位置编码需要bias+1才能匹配
                                                return_attn_probs=True)    
        if triangle<0:
            attn_output = head_out
        # tail prefill
        else:
            Q_tail_cos, Q_tail_sin = self.rotary_emb(value_states, Q_mapping_ids)
            K_tail_cos, K_tail_sin = self.rotary_emb(value_states, K_mapping_ids)
            
            
            tail_query_states, _ = apply_rotary_pos_emb_map(query_states[:,:,-triangle:,:],None, Q_tail_cos[:,-triangle:,:], 
                                                    Q_tail_sin[:,-triangle:,:], has_kv_cache) # tail Q
            _, tail_key_states = apply_rotary_pos_emb_map(None,map_key_states[:,:,:triangle,:], K_tail_cos[:,:triangle,:], 
                                                    K_tail_sin[:,:triangle,:], has_kv_cache) # tail K
            tail_key_states = repeat_kv(tail_key_states, self.num_key_value_groups)
            # flash attn func 传入shape[bsz,len,head,dim]
            tail_out, tail_lse, _ = flash_attn_func(
                tail_query_states.transpose(1, 2),
                tail_key_states.transpose(1, 2),
                value_states.transpose(1, 2)[:, :triangle, :, :],
                causal=True,
                window_size=[-1, -1],
                return_attn_probs=True,
            )   # [bsz, N, h, d]
            attn_output = flash_prefill_merge(head_out,head_lse,tail_out,tail_lse,use_length)         
    else:
        triangle = kv_seq_len - use_length
        tail_query_states = query_states.clone()
        head_query_states,_ = apply_rotary_pos_emb_map(query_states,None, cos, sin, has_kv_cache) # head Q
        
        # head decoding
        k_cache_head = key_cache[:, kv_seq_len-use_length:kv_seq_len, :, :]
        v_cache_head = value_cache[:, kv_seq_len-use_length:kv_seq_len, :, :]     
        head_out, head_lse, _  = flash_attn_func(head_query_states.transpose(1, 2), 
                                    k_cache_head,
                                    v_cache_head, 
                                    causal=True, 
                                    window_size=[-1, -1],
                                    return_attn_probs=True)
        if triangle<0:
            attn_output = head_out
        else:
            # tail decoding
            Q_tail_cos, Q_tail_sin = self.rotary_emb(value_states, Q_mapping_ids)
            K_tail_cos, K_tail_sin = self.rotary_emb(value_states, K_mapping_ids)
            
            
            tail_query_states, _ = apply_rotary_pos_emb_map(query_states, None, Q_tail_cos, Q_tail_sin, has_kv_cache)
            tail_key_states = key_cache[:,:kv_seq_len-use_length , :, :]  
            tail_value_states = value_cache[:, :kv_seq_len-use_length, :, :]
            
            _, tail_key_states = apply_rotary_pos_emb_map(None, tail_key_states.transpose(1,2), K_tail_cos[:, :kv_seq_len-use_length, :]
                                                        , K_tail_sin[:, :kv_seq_len-use_length, :], has_kv_cache)
             # [1, 2084, 32, 128]) torch.Size([1, 32, 1, 128]
            tail_out, tail_lse, _ = flash_attn_func(
                    tail_query_states.transpose(1, 2),
                    tail_key_states.transpose(1, 2),
                    tail_value_states,
                    causal=True,
                    window_size=[-1, -1],
                    return_attn_probs=True
            )
            attn_output = flash_decode_merge(head_out,head_lse,tail_out,tail_lse)
        
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
        # print("allocate kv cache")
        
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




diag_size = None
local_window = None

MAX_CACHE_LEN = None
attention_factor = None
MAX_NEW_TOKENS = 1300


def replace_with_rerope(max_test_length):
    print("rerope exe!!!!!!!!!!!!!!")
    # this is used to pre-allocate KV cache, saving GPU memory
    global MAX_CACHE_LEN
    MAX_CACHE_LEN = max_test_length + MAX_NEW_TOKENS
    # String parameters
    
    

    #transformers.models.llama.modeling_llama.LlamaAttention._init_rope = _init_rope
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = causal_forward
    transformers.models.llama.modeling_llama.LlamaModel.forward = flashdecoding_forward
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = forward
    # transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = RotaryEmbedding

   