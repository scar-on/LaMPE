import os
import subprocess

# Directory containing prediction .jsonl files.
pred_dir = os.environ.get(
    "LEVAL_PRED_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "Predictions", "exam_eval", "llama3_8b_cache1"),
)

# 遍历目录下的所有 .jsonl 文件
for filename in os.listdir(pred_dir):
    if filename.endswith(".jsonl"):
        pred_file = os.path.join(pred_dir, filename)
        # 构建命令
        cmd = ["python", "Evaluation/auto_eval.py", "--pred_file", pred_file]
        print(f"正在执行: {cmd}")
        # 执行命令
        subprocess.run(cmd)
