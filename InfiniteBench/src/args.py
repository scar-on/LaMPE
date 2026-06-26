import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from eval_utils import DATA_NAME_TO_MAX_NEW_TOKENS

INFINITEBENCH_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument(
        "--task",
        type=str,
        # choices=list(DATA_NAME_TO_MAX_NEW_TOKENS.keys()) + ["all"],
        required=True,
        help="Which task to use. Note that \"all\" can only be used in `compute_scores.py`.",  # noqa
    )
    p.add_argument(
        '--data_dir',
        type=str,
        default=os.environ.get("INFINITEBENCH_DATA_DIR", str(INFINITEBENCH_DIR / "data")),
        help="The directory of data."
    )
    p.add_argument("--output_dir", type=str, default=os.environ.get("INFINITEBENCH_OUTPUT_DIR", str(INFINITEBENCH_DIR / "results")), help="Where to dump the prediction results.")  # noqa
    p.add_argument(
        "--model_path",
        type=str,
        help="The path of the model (in HuggingFace (HF) style). If specified, it will try to load the model from the specified path, else, it wll default to the official HF path.",  # noqa
    )  # noqa
    p.add_argument(
        "--model_name",
        type=str,
        choices=["llama3.1","qwen2.5", "yarn-mistral", "llama3", "claude2", "rwkv", "yi-6b-200k", "yi-34b-200k", "chatglm3"],
        default="yarn-mistral",
        help="For `compute_scores.py` only, specify which model you want to compute the score for.",  # noqa
    )
    p.add_argument("--start_idx", type=int, default=0, help="The index of the first example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data.")  # noqa
    p.add_argument("--stop_idx", type=int, help="The index of the last example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data. Defaults to the length of dataset.")  # noqa
    p.add_argument("--verbose", action='store_true')
    p.add_argument("--device", type=str, default="cuda")
    
    p.add_argument('--head', type=int, default=8*1024)
    p.add_argument('--tail', type=int, default=1024)
    p.add_argument('--map_len', type=int, default=24*1024)
    p.add_argument('--max_len', type=int, default=128*1024)
    p.add_argument('--pretraining_length', type=int, default=8192)
    p.add_argument('--factor', type=int, default=4)
    p.add_argument('--group', type=int)
    p.add_argument('--window', type=int)
    p.add_argument('--method', type=str)
    return p.parse_args()