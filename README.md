# [ACL 2026] LaMPE: Length-aware Multi-grained Positional Encoding for Adaptive Long-context Scaling Without Training

LaMPE contains evaluation scripts for long-context position mapping experiments across RULER, InfiniteBench, LongBench, and LEval.

## Layout

- `ruler/`: RULER data preparation and evaluation scripts.
- `InfiniteBench/`: InfiniteBench evaluation scripts and task data.
- `longbench_eval/LongBench-main/`: LongBench prediction and evaluation scripts.
- `LEval/`: LEval baselines, data, and evaluation scripts.
- `run_all.sh`: top-level launcher for the benchmark scripts.

## Usage

Run one benchmark group:

```bash
bash run_all.sh ruler
bash run_all.sh infinitebench
bash run_all.sh longbench
bash run_all.sh leval
```

Run all benchmark groups in sequence:

```bash
bash run_all.sh all
```

Model paths, data paths, output directories, and Conda environments can be configured with environment variables such as `LLAMA_MODEL_PATH`, `LONGBENCH_DATA_DIR`, `RULER_OUTPUT_DIR`, `CONDA_PROFILE`, and `LAMPE_CONDA_ENV`.

## Requirements

Install the tested core environment:

```bash
pip install -r requirements.txt
```

Key package versions are pinned from the development environment:

- `torch==2.0.1`
- `transformers==4.43.3`
- `flash-attn==2.5.6`
- `accelerate==1.3.0`
- `datasets==3.6.0`
- `evaluate==0.4.3`

Benchmark-specific requirements are also kept in `InfiniteBench/requirements.txt` and `longbench_eval/LongBench-main/requirements.txt`.

## Data

Large benchmark inputs are not committed to this repository. Set `INFINITEBENCH_DATA_DIR`, `LONGBENCH_DATA_DIR`, and `LONGBENCH_E_DATA_DIR` to point to local copies of the corresponding benchmark data before running evaluations.
