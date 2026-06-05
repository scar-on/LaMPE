from types import MethodType
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaForCausalLM
from transformers.models.mistral.modeling_mistral import MistralFlashAttention2
from datasets import load_dataset
import torch
import os
import json
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers import AutoConfig


# import deepspeed
import torch.distributed as dist
import warnings
import math
warnings.filterwarnings("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import os



def load(model_path):# ,attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,device_map="auto",
                                                 torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2"
                                                  )
    tokenizer = AutoTokenizer.from_pretrained (model_path, legacy =False) 
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0
    return model, tokenizer

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

if __name__ == "__main__":
    file_name = os.environ.get("LAMPE_EXAMPLE_FILE", "dirty_files/passkey_examples.jsonl")
    model_dir = os.environ.get("LLAMA_MODEL_PATH", "meta-llama/Meta-Llama-3-8B-Instruct")
    replace_with_hmt(max_test_length=10*1024,scale_factor=1.0)
   
    
    model, tokenizer = load(model_dir)
    

    prompt_format = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" \
    "You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n" \
    "{context}{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\nAnswer:"

    for line in open(file_name, "r"):
        example = json.loads(line)
        prompt_postfix = "What is the pass key? The pass key is "
        prompt = example["input"] + prompt_postfix   
        # prompt = prompt_format.format(context=example["input"]*1, question=prompt_postfix)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        
        print( "-----------------------------------" )
        print( f"#Tokens of Prompt:", input_ids.shape[1]/1024, end=" " )
        print( "Passkey target:", example["target"] )
        decode_len = tokenizer(example["target"], return_tensors="pt").input_ids.shape[1]
           
    prompt_length = input_ids.shape[1]
    
    
    model.eval()
    with torch.no_grad():
        # model.to(model.device)
        use_len = remap_uselen(prompt_length, "llama3")
        print(prompt_length,use_len)

        change_forward = partial(forward,head=512,
                                             tail=200,use_length=use_len) # 
        modify_method_of_instance(model, "LlamaFlashAttention2", "forward", change_forward)

        tokens = model.generate(input_ids,max_new_tokens=128)# ,pad_token_id=tokenizer.eos_token_id
        answer = prompt_postfix + tokenizer.decode(tokens[0][prompt_length:])
        answer = answer.replace("\n", "\\n")
        answer= f"\n [ {answer} ]"
        print( answer )
        print( "-----------------------------------\n" )
        