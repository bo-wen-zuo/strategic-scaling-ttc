"""
Grade generated MATH-500 samples against ground truth.

Answer extraction/equivalence follows the Minerva-style grading used by
lm-evaluation-harness (last \\boxed{...} on the "final answer" line,
compared via sympy-backed `is_equiv`).
"""

import re
import multiprocessing
from copy import deepcopy
from pathlib import Path

import pydra
from tqdm import tqdm
from lm_eval.tasks.minerva_math.utils import (
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
    is_equiv,
)

from ttc_eval.utils import EvaluateScriptConfig, load_yaml, save_yaml


def _strip_leading_zeros_int(s: str) -> str:
    t = s.strip()
    m = re.fullmatch(r"([+-]?)(0*)(\d+)$", t)
    if m:
        sign, _, digits = m.groups()
        return (sign if sign != "+" else "") + (digits.lstrip("0") or "0")
    return t


def get_unnormalized_answer(text: str) -> str:
    """
    Finds the line containing "the final answer is" + "\\boxed", and returns
    the contents of the \\boxed{...} on that line.
    """
    INVALID_ANSWER = "[invalidanswer]"
    for line in text.splitlines():
        if "the final answer is" in line.lower() and "\\boxed" in line:
            pos_boxed = line.find("\\boxed")
            start_brace = line.find("{", pos_boxed)
            end_brace = line.rfind("}")
            if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                extracted = line[start_brace : end_brace + 1].strip()
                if extracted.startswith("{") and extracted.endswith("}"):
                    return extracted[1:-1].strip()
                return extracted
    return INVALID_ANSWER


def is_correct_minerva(model_sample: str, gt_solution: str) -> bool:
    pred = normalize_final_answer(get_unnormalized_answer(model_sample))
    gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt_solution)))
    # string equality check needed because of
    # https://github.com/EleutherAI/lm-evaluation-harness/issues/2212
    return (
        pred == gt
        or is_equiv(pred, gt)
        or _strip_leading_zeros_int(pred) == _strip_leading_zeros_int(gt)
    )


def process_sample(task_config: "ScriptConfig"):
    if task_config.save_path.exists():
        return

    result = load_yaml(task_config.sample_path)
    is_corrects = [is_correct_minerva(str(sample), result["gt_answer"]) for sample in result["samples"]]
    result["is_corrects"] = is_corrects
    # Per-sample scalar used by bandit_allocate.py to rank queries during the
    # exploration phase. Defaults to correctness (0/1) as a self-consistency
    # proxy; swap in PRM/verifier scores here if you have them.
    result.setdefault("scores", [float(c) for c in is_corrects])
    save_yaml(task_config.save_path, result)


class ScriptConfig(EvaluateScriptConfig):
    pass


def get_tasks(config: ScriptConfig):
    tasks = []
    for sample_path in tqdm(sorted(config.samples_dir.glob("*.yaml")), desc="Loading generations"):
        task_config = deepcopy(config)
        task_config.sample_path = sample_path
        task_config.save_path = config.save_dir / sample_path.name
        tasks.append(task_config)
    return tasks


@pydra.main(base=ScriptConfig)
def main(config: ScriptConfig):
    tasks = get_tasks(config)
    tasks = sorted(tasks, key=lambda t: t.save_path)
    tasks = tasks[config.offset : config.limit : config.stride]

    print(f"Evaluating {len(tasks)} problems.")
    if config.num_workers not in (0, None):
        with multiprocessing.Pool(processes=config.num_workers) as pool:
            list(tqdm(pool.map(process_sample, tasks), total=len(tasks)))
    else:
        for task in tqdm(tasks):
            process_sample(task)


if __name__ == "__main__":
    main()
