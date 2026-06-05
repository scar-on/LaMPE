# -*- coding:utf-8 -*-
import importlib
from re import I
from types import MethodType
from functools import partial
import yaml
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import sys
import json
from utils import read_manifest
from functools import partial
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llama_cache1 import remap_uselen


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




def main():
    fw = open(file_name, "a")
    scores = []
    save_ds = []
    for i, data in enumerate(data_list):
        text_inputs = data["input"]
        inputs = tokenizer(text_inputs, return_tensors="pt", return_token_type_ids=False).to(model.device)
        prompt_length = inputs.input_ids.size()[-1]

        if args.method == "map":
            use_len = remap_uselen(prompt_length,"mistral")
            change_forward = partial(forward,head=args.head,
                                             tail=args.tail,use_length=use_len) # 
            print(prompt_length,use_len)
            modify_method_of_instance(model, "MistralFlashAttention2", "forward", change_forward)

        sample = model.generate(**inputs, repetition_penalty=1, do_sample=False, max_new_tokens=max_new_tokens)
        output = tokenizer.decode(sample[0][prompt_length:])
        output = " ".join(output.split())
        save_d = {}
        ref = data["outputs"]
        print(f"----------------- sample {i} -----------------")
        print('[Model Prediction]',output)
        print('[Ground Truth]', ref)
        if "qa" in args.task:
            score = max([r.lower() in output.lower() for r in ref])
        else:
            score_curr = [1.0 if r.lower() in output.lower() else 0.0 for r in ref]
            score = sum(score_curr) / len(score_curr)
        print("[score]:", score)
        scores.append(score)
        print(f"===== step {i}, ctx len {prompt_length}, avg score {sum(scores) / len(scores)} =====")
        print(f"step {i}, ctx len {prompt_length}, avg score {sum(scores) / len(scores)}", file=fw)
        fw.flush()
        save_d["ctx_len"] = prompt_length
        save_d["pred"] = output
        save_d["needle"] = ref
        save_d["score"] = score
        save_ds.append(save_d)

    for save_d in save_ds:
        fw.write(json.dumps(save_d) + '\n')
    fw.write(f"avg:{sum(scores) / len(scores)}\n")
    fw.close()
    print(f"avg:{sum(scores) / len(scores)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=os.environ.get("MISTRAL_MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.3"), type=str)
    parser.add_argument("--data_dir", type=str, help='path to load the dataset jsonl files')
    parser.add_argument("--benchmark", type=str, default='synthetic', help='Options: [synthetic]')
    parser.add_argument("--task", type=str, help='Options: tasks in benchmark')
    parser.add_argument('--head', default=512, type=int)
    parser.add_argument('--tail', default=200, type=int)
    parser.add_argument('--group', default=512, type=int)
    parser.add_argument('--window', default=200, type=int)
    parser.add_argument('--use_length', default=1536, type=int)
    parser.add_argument('--max_length', default=128*1024, type=int)
    parser.add_argument('--pretraining_length', default=8192, type=int)
    parser.add_argument('--method', type=str)
    args = parser.parse_args()

    

    # copied from https://github.com/hsiehjackson/RULER/blob/main/scripts/data/synthetic/constants.py#L24
    if "vt" in args.task :
        max_new_tokens = 30
    elif "cwe" in args.task:
        max_new_tokens = 120
    elif "fwe" in args.task:
        max_new_tokens = 50
    elif "qa" in args.task:
        max_new_tokens = 32
    elif "niah"  in args.task:
        max_new_tokens = 128
    else:
        raise NotImplementedError("Unsupported task")

    model_path = args.model_path
    open_source_model = model_path.split("/")[-1]
    if len(open_source_model) == 0:
        open_source_model = model_path.split("/")[-2]

    print("*" * 10, "Data loading", "*"*10)
    
    curr_folder = os.path.dirname(os.path.abspath(__file__))


    with open(os.path.join(curr_folder, f"{args.benchmark}.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)
        if args.task not in tasks_customized:
            raise ValueError(f'{args.task} is not found in config_tasks.yaml')

    task_file = args.data_dir
    data_list = read_manifest(task_file)
    print("*" * 10, "loading ends..", "*"*10)

    pred_save_path = os.path.join(os.environ.get("RULER_OUTPUT_DIR", "Predictions"), args.task, open_source_model)
    print(f"Your prediction file will be saved to: {pred_save_path}  , press enter to confirm...")
    os.makedirs(pred_save_path, exist_ok=True)
    use_len = int(args.use_length/1024)
    max_len = int(args.max_length/1024)
    head = int(args.head/1024)
    tail = int(args.tail) #/1024
    

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path)

    if args.method == "longlm":
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from llama_longlm import replace_with_longlm, forward
        from modifed_forward import modify_method_of_instance
        replace_with_longlm(args.max_length)
        model = AutoModelForCausalLM.from_pretrained(args.model_path,config=config, attn_implementation="flash_attention_2", device_map="auto",
                                            trust_remote_code=True, torch_dtype=torch.bfloat16)
       
        change_forward = partial(forward, group_size_1=args.group, group_size_2=args.window) 
        modify_method_of_instance(model, "LlamaFlashAttention2", "forward", change_forward)
        file_name = os.path.join(pred_save_path, f"{args.method}_{max_len}k_{args.group}_{args.window}.jsonl")
    if args.method == "dca":
        print("DCA")  
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from flash_decoding_chunkllama import replace_with_chunkllama  # DCA
        
        replace_with_chunkllama(pretraining_length=args.pretraining_length)
        model = AutoModelForCausalLM.from_pretrained(args.model_path,config=config, attn_implementation="flash_attention_2", device_map="auto",
                                            trust_remote_code=True, torch_dtype=torch.bfloat16)
        file_name = os.path.join(pred_save_path, f"{args.method}_{max_len}k.jsonl")  
    if args.method == "yarn":
        print("yarn")   
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from llama_flashattn import replace_with_flashattn
        replace_with_flashattn(args.max_length)
        config = AutoConfig.from_pretrained(args.model_path)
        yarn_factor = 16.0

        config.rope_scaling = {"rope_type":"yarn","factor":yarn_factor}
        
        model = AutoModelForCausalLM.from_pretrained(args.model_path,config = config, attn_implementation="flash_attention_2", device_map="auto",
                                            trust_remote_code=True, torch_dtype=torch.bfloat16)
        file_name = os.path.join(pred_save_path, f"{args.method}_{max_len}k_{yarn_factor}.jsonl")  
    if args.method == "ntk":
        print("ntk")   
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from llama_flashattn import replace_with_flashattn
        replace_with_flashattn(args.max_length)
        config = AutoConfig.from_pretrained(args.model_path)
        yarn_factor = 2.0
        config.rope_scaling = {"rope_type":"linear","factor":yarn_factor}
        
        model = AutoModelForCausalLM.from_pretrained(args.model_path,config = config, attn_implementation="flash_attention_2", device_map="auto",
                                            trust_remote_code=True, torch_dtype=torch.bfloat16)
        file_name = os.path.join(pred_save_path, f"{args.method}_{max_len}k_{yarn_factor}.jsonl")  
    if args.method == "map":
        print("map")
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from llama_cache1 import replace_with_hmt, forward # map
        from modifed_forward import modify_method_of_instance

        replace_with_hmt(max_test_length=args.max_length, scale_factor=0.1*math.log(2) + 1)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, attn_implementation="flash_attention_2", device_map="auto",
                                            trust_remote_code=True, torch_dtype=torch.bfloat16)
        change_forward = partial(forward,head=args.head,
                                             tail=args.tail,use_length=args.use_length) # 
        modify_method_of_instance(model, "MistralFlashAttention2", "forward", change_forward)

        file_name = os.path.join(pred_save_path, f"{args.method}_{max_len}k_{use_len}k_{head}k_{tail}.jsonl")
    if args.method == "ori":
        print("ori flash attn 实现")
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from llama_flashattn import replace_with_flashattn
        replace_with_flashattn(args.max_length)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, attn_implementation="flash_attention_2", device_map="auto",
                                            trust_remote_code=True, torch_dtype=torch.bfloat16)
        file_name = os.path.join(pred_save_path, f"{args.method}_{max_len}k.jsonl")
        
    if args.method == "rerope":
        print("rerope")
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from llama_rerope import replace_with_rerope,forward # map
        from modifed_forward import modify_method_of_instance
        replace_with_rerope(max_test_length=args.max_length)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, attn_implementation="flash_attention_2", device_map="auto",
                                            trust_remote_code=True, torch_dtype=torch.bfloat16)
        change_forward = partial(forward,use_length= 4096) # 
        modify_method_of_instance(model, "LlamaFlashAttention2", "forward", change_forward)
        file_name = os.path.join(pred_save_path, f"{args.method}_{max_len}k.jsonl")

    model = model.eval()
    sys.exit(main())


