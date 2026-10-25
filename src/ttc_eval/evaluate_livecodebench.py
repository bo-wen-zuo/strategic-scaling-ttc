"""
Grade generated LiveCodeBench samples by executing them against the
problem's public + private test cases (via lcb_runner's sandboxed grader).

Produces the same {samples, is_corrects, scores} schema as
evaluate_math500.py, so bandit_allocate.py works unmodified on LiveCodeBench
output too.
"""

from pathlib import Path

import pydra
from pydra import Config, REQUIRED

from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
from ttc_eval.lcb_prompts import extract_code
from ttc_eval.utils import load_yaml, save_yaml


class LCBEvaluateConfig(Config):
    samples_dir: Path = REQUIRED
    save_dir: Path = REQUIRED
    release_version = "release_v1"
    num_process_evaluate = 16
    timeout = 6

    def finalize(self):
        self.samples_dir = Path(self.samples_dir)
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)


@pydra.main(LCBEvaluateConfig)
def main(config: LCBEvaluateConfig):
    problems_by_id = {
        p.question_id: p for p in load_code_generation_dataset(release_version=config.release_version)
    }

    sample_paths = sorted(config.samples_dir.glob("*.yaml"))
    to_grade_paths = [p for p in sample_paths if not (config.save_dir / p.name).exists()]
    print(f"Grading {len(to_grade_paths)} / {len(sample_paths)} problems (rest already done).")

    results = [load_yaml(p) for p in to_grade_paths]
    eval_samples = [problems_by_id[r["question_id"]].get_evaluation_sample() for r in results]
    code_lists = [[extract_code(s) for s in r["samples"]] for r in results]

    metrics, graded_results, _ = codegen_metrics(
        eval_samples,
        code_lists,
        k_list=[1],
        num_process_evaluate=config.num_process_evaluate,
        timeout=config.timeout,
    )
    print(f"pass@1: {metrics.get('pass@1')}")

    for i, (path, result) in enumerate(zip(to_grade_paths, results)):
        # graded_results[i] is, per sample, a per-test-case list of results;
        # a sample passes only if every test case returned a truthy (>0) result.
        is_corrects = [all(g > 0 for g in sample_grades) for sample_grades in graded_results[i]]
        result["is_corrects"] = is_corrects
        result["scores"] = [float(c) for c in is_corrects]
        save_yaml(config.save_dir / path.name, result)


if __name__ == "__main__":
    main()
