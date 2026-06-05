import json
import os

def process_text(text):
    # 1. 替换所有 `\n\n` 为 `\n`
    text = text.replace("\n\n", "\n")
    
    # 3. 在第一个 `\n` 前插入 user 标记
    first_n_index = text.find("\n")
    if first_n_index != -1:
        text = text[:first_n_index] + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n" + text[first_n_index+1:]

    # 4. 在最后一个 `\n` 之后插入 assistant 标记
    last_n_index = text.rfind("\n")
    if last_n_index != -1:
        text = text[:last_n_index] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n" + text[last_n_index+1:]
    
    # 2. 在最开始加上 system 标记
    text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" + text
    return text

def process_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 遍历所有键值对并修改文本
    for key in data:
        data[key] = process_text(data[key])

    # 保存修改后的 JSON
    output_path = file_path.replace(".json", "_processed.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Processed file saved to: {output_path}")

# Process the prompt config in this directory by default.
directory = os.environ.get(
    "LONGBENCH_PROMPT_CONFIG",
    os.path.join(os.path.dirname(__file__), "dataset2prompt.json"),
)

process_json_file(directory)
