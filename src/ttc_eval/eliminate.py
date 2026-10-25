"""
Algorithm 1 ("Strategic Test-Time Compute Allocation") from
"Strategic Scaling of Test-Time Compute: A Bandit Learning Approach"
(arXiv:2506.12721), run end to end: generation (vLLM), scoring (a PRM or a
ground-truth verifier), and elimination all happen live, round by round, in
this one process -- there is no separate generate/score/pick pipeline.

    python src/ttc_eval/eliminate.py \
        model=meta-llama/Llama-3.1-8B-Instruct save_dir=$SAVE_DIR/math500_run \
        dataset=math500 reward_mode=prm \
        total_budget=4000 per_round_budget=1 elimination_threshold=0.89 \
        exploration_rule=entropy

See strategic_core.py for the algorithm/exploration-rule details.
"""

from pathlib import Path

import numpy as np
import pydra
import requests
from datasets import load_dataset

from ttc_eval.evaluate_math500 import get_unnormalized_answer, is_correct_minerva
from ttc_eval.lcb_prompts import SYSTEM_MESSAGE_GENERIC, build_question_prompt, extract_code
from ttc_eval.prompts import MATH_COT_PROMPT
from ttc_eval.strategic_core import run_algorithm1
from ttc_eval.utils import GenerateScriptConfig, save_yaml
from ttc_eval.vllm_utils import vllm_manager

from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics


def _math_boxed_answer_key(sample: str) -> str:
    try:
        from lm_eval.tasks.minerva_math.utils import normalize_final_answer

        return normalize_final_answer(get_unnormalized_answer(sample))
    except Exception:
        return get_unnormalized_answer(sample)


ENTROPY_KEY_FNS = {
    "raw": lambda s: s.strip(),
    "math_boxed_answer": _math_boxed_answer_key,
}


def build_math500_items(config) -> list:
    dataset = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    items = []
    for i, ex in enumerate(dataset):
        question_prompt = f"\n\nProblem:\n{ex['problem']}\n\nSolution:"
        items.append(
            {
                "id": i,
                "prompt": MATH_COT_PROMPT + question_prompt,
                "question_prompt": question_prompt,  # shown to the PRM, without the few-shot block
                "gt_answer": ex["solution"],
            }
        )
    return items


def build_livecodebench_items(config) -> list:
    problems = load_code_generation_dataset(release_version=config.release_version)
    return [
        {"id": p.question_id, "prompt": build_question_prompt(p), "eval_sample": p.get_evaluation_sample()}
        for p in problems
    ]


def make_math500_pull_fn(config, prm_scorer):
    url = f"http://localhost:{config.vllm_port}/v1/chat/completions"

    def pull(item, k):
        body = {
            "model": config.model,
            "messages": [{"role": "user", "content": item["prompt"]}],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "n": k,
        }
        response = requests.post(url, json=body)
        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.text}")
            return [], []

        samples = [c["message"]["content"] for c in response.json()["choices"]]
        if prm_scorer is not None:
            scores = [prm_scorer.score_response(item["question_prompt"], s)[0] for s in samples]
        else:
            scores = [1.0 if is_correct_minerva(s, item["gt_answer"]) else 0.0 for s in samples]
        return samples, scores

    return pull


def make_livecodebench_pull_fn(config):
    url = f"http://localhost:{config.vllm_port}/v1/chat/completions"

    def pull(item, k):
        body = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE_GENERIC},
                {"role": "user", "content": item["prompt"]},
            ],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "n": k,
        }
        response = requests.post(url, json=body)
        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.text}")
            return [], []

        samples = [c["message"]["content"] for c in response.json()["choices"]]
        codes = [extract_code(s) for s in samples]
        _, results, _ = codegen_metrics(
            [item["eval_sample"]],
            [codes],
            k_list=[1],
            num_process_evaluate=config.lcb_num_process_evaluate,
            timeout=config.lcb_timeout,
        )
        scores = [1.0 if all(g > 0 for g in sample_grades) else 0.0 for sample_grades in results[0]]
        return samples, scores

    return pull


class EliminateConfig(GenerateScriptConfig):
    dataset = "math500"  # math500 | livecodebench
    reward_mode = "correctness"  # correctness (ground-truth verifier) | prm (math500 only)
    release_version = "release_v1"  # livecodebench only

    total_budget = 4000
    per_round_budget = 1  # K: responses pulled per query per round
    elimination_threshold = 1.0  # gamma; use e.g. 0.89 with reward_mode=prm, 1.0 with correctness
    exploration_rule = "uniform"  # uniform | ucb | gap | entropy
    ucb_lambda = 1.0
    entropy_lambda = 3.0
    max_samples = None  # per-query pull cap; None = unbounded (limited only by total_budget)
    entropy_key = "raw"  # raw | math_boxed_answer (math500 only)

    lcb_num_process_evaluate = 4
    lcb_timeout = 6


@pydra.main(EliminateConfig)
def main(config: EliminateConfig):
    if config.dataset == "math500":
        items = build_math500_items(config)
    elif config.dataset == "livecodebench":
        items = build_livecodebench_items(config)
        assert config.reward_mode == "correctness", "reward_mode=prm is only supported for dataset=math500"
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    limit = config.limit if config.limit is not None else len(items)
    stride = config.stride if config.stride is not None else 1
    offset = config.offset if config.offset is not None else 0
    items = items[offset:limit:stride]
    print(f"Total number of items to process: {len(items)}")

    prm_scorer = None
    if config.reward_mode == "prm":
        from ttc_eval.prm_score import PRMScorer

        prm_scorer = PRMScorer()

    with vllm_manager(config) as vllm_port:
        config.vllm_port = vllm_port

        if config.dataset == "math500":
            pull_fn = make_math500_pull_fn(config, prm_scorer)
        else:
            pull_fn = make_livecodebench_pull_fn(config)

        def on_round(budget_remaining, active_count):
            print(f"budget remaining: {budget_remaining}, active queries: {active_count}")

        states = run_algorithm1(
            items,
            pull_fn,
            total_budget=config.total_budget,
            per_round_budget=config.per_round_budget,
            elimination_threshold=config.elimination_threshold,
            exploration_rule=config.exploration_rule,
            ucb_lambda=config.ucb_lambda,
            entropy_lambda=config.entropy_lambda,
            max_samples=config.max_samples,
            entropy_key_fn=ENTROPY_KEY_FNS[config.entropy_key],
            on_round=on_round,
        )

    items_by_id = {item["id"]: item for item in items}
    coverage, accuracy, total_pulled = 0, 0, 0

    for qid, state in states.items():
        item = items_by_id[qid]
        if config.dataset == "math500":
            is_corrects = [is_correct_minerva(s, item["gt_answer"]) for s in state.all_samples]
        else:
            is_corrects = [bool(s) for s in state.all_scores]  # ground-truth verifier reward == correctness

        if any(is_corrects):
            coverage += 1
        if state.all_scores:
            best_idx = int(np.argmax(state.all_scores))
            if is_corrects[best_idx]:
                accuracy += 1
        total_pulled += state.pulled

        save_yaml(
            config.save_dir / f"{qid}.yaml",
            {
                "samples": state.all_samples,
                "scores": state.all_scores,
                "is_corrects": is_corrects,
                "best_response": state.best_sample,
                "best_score": state.best_score,
                "eliminated": state.eliminated,
                "compute_used": state.pulled,
            },
        )

    n = len(states)
    print(f"Total compute used: {total_pulled} (budget: {config.total_budget})")
    print(f"Coverage: {coverage}/{n} = {coverage / n:.3f}")
    print(f"Accuracy: {accuracy}/{n} = {accuracy / n:.3f}")


if __name__ == "__main__":
    main()
