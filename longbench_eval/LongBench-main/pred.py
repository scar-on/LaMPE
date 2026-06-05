import os
from datasets import load_dataset, load_from_disk
import torch
import json
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM, AutoModelForCausalLM,MistralForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse

import torch.distributed as dist
import torch.multiprocessing as mp
import os
import sys
from functools import partial
from types import MethodType
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))



from llama_cache1 import  forward
from modifed_forward import modify_method_of_instance



def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--exp_name', default='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--use_map', action='store_true')
    parser.add_argument('--use_length', default = 1536,type=int)
    parser.add_argument('--max_length', default = 1536,type=int)
    parser.add_argument('--head', default = 512,type=int)
    parser.add_argument('--tail', default = 200,type=int)
    parser.add_argument('--method', type=str, default=None)
    return parser.parse_args(args)


def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "Mistral-7B-Instruct-v0.3" in model_name:
       prompt = f"<s>[INST] {prompt} [/INST]</s>"
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "llama3.1" in model_name:
        prompt = prompt
    elif "llama" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt


def get_pred(args, rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path):
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(args, model2path[model_name], model_name, device)
    with torch.no_grad():
        for json_obj in tqdm(data):
            torch.cuda.empty_cache()
            prompt = prompt_format.format(**json_obj)
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if dataset == "narrativeqa":
                # 对narrativeqa进行长度截断
                if len(tokenized_prompt) > max_length:
                    half = int(max_length/2)
                    prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=False)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=False)
            if args.method !="map": # 是否使用映射方法
                print("截断")
                if len(tokenized_prompt) > args.max_length:
                    half = int(args.max_length/2)
                    prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
                prompt = build_chat(tokenizer, prompt, model_name)
             
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            context_length = input.input_ids.shape[-1]
            
            
            # use_len = remap_uselen(context_length, args.model_name)
            # print(use_len, context_length)
            if args.method=="map":
                if context_length<1536:
                    change_forward = partial(forward, head = 512,tail = 200,use_length = context_length,)
                else:
                    change_forward = partial(forward,head = 512,tail = 200,use_length = 1536)
                modify_method_of_instance(model, "LlamaFlashAttention2", "forward", change_forward)
        
            print(context_length)
            if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )[0]
            else:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,#pad_token_id=tokenizer.eos_token_id,
                    
                    temperature=1.0,
                )[0]
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
                f.write('\n')
    dist.destroy_process_group()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(args, path, model_name, device):
    if "llama" or "Llama-3" in model_name:
        if args.method == "yarn":
            print("yarn method")
            from llama_flashattn import replace_with_flashattn
            replace_with_flashattn(args.max_length)
            config = AutoConfig.from_pretrained(path)
            pre_len = config.max_position_embeddings
            yarn_factor = args.max_length/pre_len
            config.rope_scaling = {"rope_type":"yarn","factor":yarn_factor}
        if args.method == "ntk":
            print("ntk method")
            from llama_flashattn import replace_with_flashattn
            replace_with_flashattn(args.max_length)
            config = AutoConfig.from_pretrained(path)
            pre_len = config.max_position_embeddings
            yarn_factor = args.max_length/pre_len
            print(yarn_factor)
            config.static_ntk = True
            config.rope_scaling = {"rope_type":"dynamic","factor":yarn_factor}
        if args.method == "map": 
            print("map method")
            from llama_cache1 import replace_with_hmt
            replace_with_hmt(max_test_length=65536, scale_factor=0.1*math.log(2) + 1)
            config = AutoConfig.from_pretrained(path) 
        tokenizer = AutoTokenizer.from_pretrained(path) 
        model = LlamaForCausalLM.from_pretrained(path, config=config, trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16,device_map="auto").to(device) #, device_map="auto"
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    world_size = 1
    mp.set_start_method('spawn', force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    exp_name = args.exp_name
    # define your model
    max_length = args.max_length
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["hotpotqa", "musique"]
        # datasets = ["hotpotqa", "qasper", "multifieldqa_en",  "2wikimqa",\
        #              "multi_news", "trec", "triviaqa", 'musique', \
        #             "lcc", "repobench-p","gov_report","samsum", "passage_count","passage_retrieval_en","qmsum","narrativeqa"] #
    if "Llama-3" in model_name:
        dataset2prompt = json.load(open("config/dataset2prompt_llama3.json", "r"))
        args.model_name = "llama3"
    else:
        dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
        args.model_name = "llama2"
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        if args.e:
            longbench_e_dir = os.environ.get("LONGBENCH_E_DATA_DIR", "data/longbench_e")
            data = load_dataset('json', data_files=os.path.join(longbench_e_dir, f"{dataset}_e.json"))
            data = data['train']
            if not os.path.exists(f"pred_e/{model_name}_{exp_name}"):
                os.makedirs(f"pred_e/{model_name}_{exp_name}")
            out_path = f"pred_e/{model_name}_{exp_name}/{dataset}.jsonl"
        else:
            longbench_data_dir = os.environ.get("LONGBENCH_DATA_DIR", "data/longbench")
            data = load_from_disk(os.path.join(longbench_data_dir, dataset), dataset)
            if not os.path.exists(f"pred/{model_name}_{exp_name}"):
                os.makedirs(f"pred/{model_name}_{exp_name}")
            out_path = f"pred/{model_name}_{exp_name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=get_pred, args=(args, rank, world_size, data_subsets[rank], max_length, \
                        max_gen, prompt_format, dataset, device, model_name, model2path, out_path))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()