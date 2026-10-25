# ttc-eval

Public code release for **"Strategic Scaling of Test-Time Compute: A Bandit
Learning Approach"** ([arXiv:2506.12721](https://arxiv.org/abs/2506.12721)),
accepted to ICLR 2026.

The step-splitting and PRM scoring logic lives in `prm_score.py`, called directly by `eliminate.py`.

## Install

```
conda env create -f environment.yml
conda activate ttc-eval
pip install -e .
```

Or without conda: `pip install -r requirements.txt && pip install -e .`

## Run

```
python src/ttc_eval/eliminate.py \
    model=meta-llama/Llama-3.1-8B-Instruct \
    save_dir=$SAVE_DIR/math500_run \
    dataset=math500 reward_mode=prm \
    total_budget=4000 per_round_budget=1 elimination_threshold=0.89 \
    exploration_rule=entropy
```

Key flags:

- `dataset`: `math500` | `livecodebench`
- `reward_mode`: `correctness` (ground-truth verifier) | `prm` (math500 only, `Qwen/Qwen2.5-Math-PRM-7B`)
- `total_budget`, `per_round_budget` (K), `elimination_threshold` (gamma), `max_samples` (per-query cap)
- `exploration_rule`: `uniform` | `ucb` | `gap` | `entropy` (`ucb_lambda`, `entropy_lambda`, `entropy_key`)

Output: one YAML per query in `save_dir` with `samples`, `scores`,
`is_corrects`, `best_response`, `best_score`, `eliminated`, `compute_used`;
overall coverage/accuracy/compute-used printed at the end.

## Notes
- Generation/evaluation logic reuses the major logic of
  [`llmonk`](https://github.com/ScalingIntelligence/large_language_monkeys),
  the official code release for
  [Large Language Monkeys](https://arxiv.org/abs/2407.21787), ported to
  vLLM's `/v1/chat/completions`.
