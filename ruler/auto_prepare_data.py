import os
import argparse
import shlex
from concurrent.futures import ProcessPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

tasks = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2"
]

tasks = [
 "cwe", "fwe"
]


template = {
    "llama3.1": "llama3-chat",
    "llama2": "meta-chat",
    "qwen2": "qwen2",
    "llama3": "llama3-chat",
    "mistral": "mistral"
}

def execute_command(task):
    prepare_script = os.environ.get("RULER_PREPARE_SCRIPT", os.path.join(SCRIPT_DIR, "data", "prepare.py"))
    if not os.path.exists(prepare_script):
        raise FileNotFoundError(
            f"RULER data prepare script not found: {prepare_script}. "
            "Set RULER_PREPARE_SCRIPT to the prepare.py from your local RULER checkout."
        )
    save_dir = os.environ.get("RULER_DATA_JSONL_DIR", os.path.join(SCRIPT_DIR, "data-jsonl"))
    cmd = f"""
    python {shlex.quote(prepare_script)} \
    --save_dir {shlex.quote(save_dir)} \
    --benchmark synthetic \
    --task {task} \
    --tokenizer_path {shlex.quote(args.model_path)} \
    --tokenizer_type hf \
    --max_seq_length {args.max_length} \
    --model_template_type {temp} \
    --num_samples 500
    """
    print(f"Executing: {cmd}")
    result = os.system(cmd)
    if result != 0:
        print(f"Command for task {task} failed with exit code {result}")


# pip install pyyaml, tqdm, wonderwords, tenacity, nltk, filelock
if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--model_path", default=os.environ.get("LLAMA_MODEL_PATH", "meta-llama/Meta-Llama-3-8B-Instruct"), type=str)
    parse.add_argument("--max_length",default=4096, type=int)
    parse.add_argument("--temp", default="llama2") #choices=["llama3.1", "llama2", "qwen2"]
    args = parse.parse_args()
    temp = template[args.temp]

    with ProcessPoolExecutor() as executor:
        for task in tasks:
            executor.submit(execute_command, task)
