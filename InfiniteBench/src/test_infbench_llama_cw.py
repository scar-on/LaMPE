import json
from lib2to3 import refactor
from pathlib import Path
import time
from typing import List, Tuple, Any
import pdb
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from compute_scores import *
import os
import sys
import math
from functools import partial
from types import MethodType
from eval_utils import (
    dump_jsonl,
    create_prompt,
    load_data,
    get_answer,
    DATA_NAME_TO_MAX_NEW_TOKENS,
)

from args import parse_args

def modify_method_of_instance(instance, target_class_name, target_method_name, new_method, visited_instances=None):
    """
        This function modifies the method of an instance of a model class. 
        It's part from chat-GPT.
        It will replace the method  with the new method.
        Currently, we only use this function to modify the attention method of a model. Do not test it further. 

        instance: 
            instance of a model to modify.
        target_class_name: 
            name of the attention class to modify. E.g. 'LlamaAttention', 'GPTNeoXAttention', etc.
        new_method: new method to replace the original method. E.g. 'self_extend_forward'. 
            It should include a parameter 'self' to be binded to the instance.
    """
    target_found = False
    if visited_instances is None:
        visited_instances = set()
    # Unique identifier for the instance (using id() since object's id is unique)
    instance_id = id(instance)
    if instance_id in visited_instances:
        target_found = False
        return target_found
    # Add the instance to the already_visited set
    visited_instances.add(instance_id)

    # Check if this instance is of the target class
    if instance.__class__.__name__ == target_class_name:
        bond_method = MethodType(new_method, instance) 
        setattr(instance, target_method_name, bond_method)
        target_found = True
        return target_found
    elif hasattr(instance, '__dict__'):
        for attr_name, attr_value in instance.__dict__.items():
            if isinstance(attr_value, object) and not isinstance(attr_value, (list, tuple, dict, set)):
                _found = modify_method_of_instance(attr_value, target_class_name, target_method_name, new_method, visited_instances)
                if _found:
                    target_found = True
            elif isinstance(attr_value, (list, tuple)):
                for item in attr_value:
                    if isinstance(item, object):
                        _found = modify_method_of_instance(item, target_class_name, target_method_name, new_method, visited_instances)
                        if _found:
                            target_found = True
            # If attribute value is a dictionary, iterate over its values and recurse
            # E.g, for a ModuleList, its moudels are stored in a dictionary: ._modules
            elif isinstance(attr_value, dict):
                for key, value in attr_value.items():
                    if isinstance(value, object):
                        _found = modify_method_of_instance(value, target_class_name, target_method_name, new_method, visited_instances)
                        if _found:
                            target_found = True
            # If attribute value is a set, iterate and recurse
            elif isinstance(attr_value, set):
                for item in attr_value:
                    if isinstance(item, object):
                        _found = modify_method_of_instance(item, target_class_name, target_method_name, new_method, visited_instances)
                        if _found:
                            target_found = True

    return target_found


def truncate_input(input: list, max_length: int, manner="middle"):
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    len_before = len(tokens)
    # print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    # print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens
    return tok.decode(tokens, skip_special_tokens=True)


def get_pred(
    args,
    model,
    tok: AutoTokenizer,
    input_text: str,
    max_tokens: int,
    verbose: bool = False,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    input_text = truncate_by_tokens(input_text, tok, args.max_len)
    inputs = tok(input_text, return_tensors="pt", return_token_type_ids=False).to(model.device)
    prompt_length = inputs.input_ids.size()[-1]
    print('document len', prompt_length)
    if args.method=='map':
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

        from llama_cache1 import  forward # map
        from modifed_forward import modify_method_of_instance
        change_forward = partial(forward, head=args.head, tail=args.tail, use_length=args.map_len) 
        modify_method_of_instance(model, "LlamaFlashAttention2", "forward", change_forward)
    sample = model.generate(**inputs, repetition_penalty=1, do_sample=False, max_new_tokens=max_tokens)
    output = tok.decode(sample[0][prompt_length:])
    output = " ".join(output.split())
    return output


def load_model(
    model_name,
    args,
    data_name
):
    
    print("Loading tokenizer")
    
    tok = AutoTokenizer.from_pretrained(args.model_path)
    tok.pad_token = tok.eos_token
    max_len = int(args.max_len/1024)
    result_dir = Path(args.output_dir) / data_name
    result_dir.mkdir(exist_ok=True, parents=True)
    model_name = "llama3"
    if args.method == "longlm":
        import SelfExtend
        llm = AutoModelForCausalLM.from_pretrained(args.model_path, attn_implementation="flash_attention_2", device_map="auto",
                                            trust_remote_code=True, torch_dtype=torch.bfloat16)
        SelfExtend.apply(llm, args.group, args.window, enable_flash_attention=True, flash_attention_impl="flash_attn") #8-32k
        output_path = (
            result_dir / f"cw_{model_name}_{max_len}k_{args.method}_{args.group}_{args.window}.jsonl"
        )
    if args.method == "dca":
        print("DCA")  
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
        from flash_decoding_chunkllama import replace_with_chunkllama  # DCA
        pretraining_length = int(args.pretraining_length*2/3)
        replace_with_chunkllama(pretraining_length=pretraining_length)
        llm = AutoModelForCausalLM.from_pretrained(args.model_path, attn_implementation="flash_attention_2", device_map="auto",
                                            trust_remote_code=True, torch_dtype=torch.bfloat16)
        output_path = (
            result_dir / f"cw_{model_name}_{max_len}k_{args.method}.jsonl"
        )
    if args.method == "yarn":
        print("yarn")   
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
        from llama_flashattn import replace_with_flashattn
        replace_with_flashattn(args.max_len)
        config = AutoConfig.from_pretrained(args.model_path)
        yarn_factor = 1.5

        config.rope_scaling = {"rope_type":"yarn","factor":yarn_factor}
        config.max_position_embeddings = int(config.max_position_embeddings*2/3)
        llm = AutoModelForCausalLM.from_pretrained(args.model_path,config = config, attn_implementation="flash_attention_2", device_map="auto",
                                            trust_remote_code=True, torch_dtype=torch.bfloat16)
        output_path = (
            result_dir / f"cw_{model_name}_{max_len}k_{args.method}_{yarn_factor}.jsonl"
        )
    if args.method == "ntk":
        print("ntk")   
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
        from llama_flashattn import replace_with_flashattn
        replace_with_flashattn(args.max_len)
        config = AutoConfig.from_pretrained(args.model_path)
        yarn_factor = 1.5
        config.rope_scaling = {"rope_type":"dynamic","factor":yarn_factor}
        config.max_position_embeddings = int(config.max_position_embeddings*2/3)
        llm = AutoModelForCausalLM.from_pretrained(args.model_path,config = config, attn_implementation="flash_attention_2", device_map="auto",
                                            trust_remote_code=True, torch_dtype=torch.bfloat16)
        output_path = (
            result_dir / f"cw_{model_name}_{max_len}k_{args.method}_{yarn_factor}.jsonl"
        )
    if args.method == "map":
        print("map")
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

        from llama_cache1 import replace_with_hmt, forward # map
        from modifed_forward import modify_method_of_instance

        replace_with_hmt(max_test_length=args.max_len, scale_factor=0.1*math.log(2) + 1)
        llm = AutoModelForCausalLM.from_pretrained(args.model_path, attn_implementation="flash_attention_2", device_map="auto",
                                            trust_remote_code=True, torch_dtype=torch.bfloat16)
        change_forward = partial(forward,head=args.head,
                                             tail=args.tail,use_length=args.map_len) # 
        modify_method_of_instance(llm, "LlamaFlashAttention2", "forward", change_forward)

        head, tail, map_len, max_len = int(args.head/1024), int(args.tail/1024), int(args.map_len/1024), int(args.max_len/1024)
        output_path = (
            result_dir / f"cw_{model_name}_{max_len}k_{args.method}_{head}k_{tail}k_{map_len}k.jsonl"
        )
    if args.method == "ori":
        print("原始")
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
        from llama_flashattn import replace_with_flashattn
        replace_with_flashattn(args.max_len)
        llm = AutoModelForCausalLM.from_pretrained(args.model_path, attn_implementation="flash_attention_2", device_map="auto",
                                            trust_remote_code=True, torch_dtype=torch.bfloat16)
        output_path = (
            result_dir / f"cw_{model_name}_{max_len}k_{args.method}.jsonl"
        )
    return llm, tok, output_path  # type: ignore


if __name__ == "__main__":
    args = parse_args()
    
    model_name = args.model_name
    print(json.dumps(vars(args), indent=4))
    data_name = args.task
    
    # Model
    max_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]

    model, tok,output_path = load_model(args.model_path,args,data_name)
    examples = load_data(data_name, data_dir=args.data_dir)
    args.stop_idx = len(examples)

    
    preds = []
    scores = []
    print("==== Evaluation ====")
    print(f"# examples: {len(examples)}")
    print(f"Max tokens: {max_tokens}")
    fw = open(output_path, "a")
    print("Your predictions are saved to", output_path)
    for i in range(args.start_idx, args.stop_idx):
        eg = examples[i]
        input_text = create_prompt(eg, data_name, model_name, args.data_dir)
        print(f"====== Example {i} ======")
        pred = get_pred(
            args, model, tok, input_text, max_tokens=max_tokens, verbose=args.verbose
        )

        ref =  get_answer(eg, data_name)
        score = get_score_one(pred=pred, label=ref, task_name=args.task, model_name=args.model_name)
        scores.append(score)
        print("prediction:", pred)
        print("Reference:", ref)
        print("score:", score)
        print(f"avg of {len(scores)} samples:", sum(scores) / len(scores))
        print(f"step {i}, avg score {sum(scores) / len(scores)}", file=fw)
        print("-" * 20)
        fw.flush()
        preds.append(
            {
                "id": i,
                "prediction": pred,
                "ground_truth": get_answer(eg, data_name),
            }
        )
    dump_jsonl(preds, output_path)
