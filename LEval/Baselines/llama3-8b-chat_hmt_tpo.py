import math
from functools import partial

import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import transformers
# -*- coding:utf-8 -*-
import argparse
# from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from LEval_config import *
from tqdm import tqdm

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from llama_cache1 import replace_with_hmt,forward
from modifed_forward import modify_method_of_instance




def main():
    # openai.api_base = "https://api.openai-sb.com/v1"
    start_idx = 0
    for file_name in key_data_pairs:
        fw = open(file_name, "w")
        data = key_data_pairs[file_name]
        begin = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        assistant = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        sys_prompt = get_sys_prompt(args, file_name)

        for d in tqdm(data):
            document = d['input']
            instructions = d['instructions']
            outputs = d['outputs']

            for inst, out in zip(instructions, outputs):
                save_d = {}
                save_d['query'] = inst
                save_d['gt'] = out
                
                if "topic" in file_name:
                    context = document + "\n\n" + inst
                    message = begin + sys_prompt + user + context + assistant
                elif args.metric == "exam_eval":
                    context = "Document is as follows. {document} \nQuestion: {inst}.  Please directly give the answer without any additional output or explanation "
                    message = begin + sys_prompt + user + context + assistant
                    message += "\nAnswer:"
                    print("run")
                else:
                    context = "Document is as follows. {document} Instruction: {inst} " + f"\nAnswer this question with {len(out.split())} words."
                    message =begin + sys_prompt + user + context + assistant
                try:
                    text_inputs = message.format(document=document, inst=inst)
                except:
                    text_inputs = message
                save_d['prompt'] = message.replace(document, "<long document>")

                inputs = tokenizer(text_inputs, return_tensors="pt").to(device)
                prompt_length = inputs.input_ids.size()[-1]
                print(prompt_length)
                if prompt_length>max_length:
                    change_forward = partial(forward,head=args.head,
                                              tail=args.tail,use_length=max_length)#v
                else:
                    change_forward = partial(forward,head=args.head,
                                              tail=args.tail,use_length=prompt_length)
                modify_method_of_instance(model, "LlamaFlashAttention2", "forward", change_forward)
                
                sample = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
                prompt_length = inputs.input_ids.size()[-1]
                output = tokenizer.decode(sample[0][prompt_length:])

                save_d[f'{open_source_model}_pred'] = output.replace('</s>', '')
                save_d['evaluation'] = d['evaluation']

                # test the factuality in scientific fiction
                if "sci_fi" in file_name:
                    text_inputs = inst.replace("based on the world described in the document.", "based on the real-world knowledge and facts up until your last training") + "Please directly answer without any additional output or explanation. \nAnswer:"
                    inputs = tokenizer(text_inputs, return_tensors="pt").to(device)
                    sample = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
                    prompt_length = inputs.input_ids.size()[-1]
                    output = tokenizer.decode(sample[0][prompt_length:])
                    save_d[f'{open_source_model}_pred'] += f" [fact: {output}]"

                if start_idx < 5:
                    print('document len', num_tokens_from_string(document, tokenizer))
                    print("[document]:",text_inputs[:100] + "...")
                    print("----------------- [output] vs [ground truth] -----------------")
                    print('[output]:', save_d[f'{open_source_model}_pred'], "\n\n", '[ground truth]:', save_d['gt'])
                    start_idx += 1
                fw.write(json.dumps(save_d) + '\n')
                # break
        fw.close()
        # break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', choices=["llm_turbo_eval", "llm_gpt4_eval", "exam_eval", "ngram_eval", "human_eval"],
                        help='metric name from choices', required=True)
    parser.add_argument('--max_length', default="4k", help='max length of the input, e.g., 2k, 16k')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--head', type=int, default=512)
    parser.add_argument('--tail', type=int, default=100)
    # set this if you do not want to use data from huggingface
    parser.add_argument('--task_path', type=str, default=None,
                        help='set this if you want test a specific task , example: LEval-data/Closed-ended-tasks/coursera.jsonl or LEval-data/Closed-ended-tasks/ ')
    # set this if you do not want to test a specific task
    parser.add_argument('--task_name', type=str, default=None,
                        help='set this if you want test a specific task from huggingface, example: coursera')

    parser.add_argument('--mc_tasks', action='store_true', help='set this if you want to test all multiple choice tasks')
    parser.add_argument('--flash', action='store_true', help='set this if you want to use flash attention')
    parser.add_argument('--exp_name', default='store_true')
    args = parser.parse_args()


    model_path = os.environ.get("LLAMA_MODEL_PATH", "meta-llama/Meta-Llama-3-8B-Instruct")
    max_length = k_to_number(args.max_length) #- max_new_tokens
    open_source_model = args.exp_name
    data_save_path = os.path.join(LEVAL_DIR, "Predictions", args.metric, open_source_model)
    print(f"Your prediction file will be saved to: {data_save_path}  , press enter to confirm...")

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    replace_with_hmt(max_test_length=50*1024,scale_factor=0.1*math.log(2) + 1)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = LlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2").to(device)
    
    model = model.eval()

    key_data_pairs = {}
    build_key_data_pairs(args, key_data_pairs, data_save_path)
    sys.exit(main())
