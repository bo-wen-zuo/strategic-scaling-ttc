Files in this directory (`benchmarks/code_generation.py`,
`evaluation/testing_util.py`, `evaluation/pass_k_utils.py`,
`evaluation/compute_code_generation_metrics.py`) are vendored verbatim from
the official LiveCodeBench runner
(https://github.com/LiveCodeBench/LiveCodeBench, `lcb_runner` package).
They provide dataset loading and the sandboxed test-execution grader
(`run_test` / `codegen_metrics`) and are kept unmodified so grading stays
identical to upstream LiveCodeBench. Only a trimmed `__init__.py` was
written locally, to avoid pulling in the test-output-prediction and
code-execution scenario modules this project doesn't use.

Upstream LiveCodeBench is MIT-licensed; see `LICENSE` in this directory
(the rest of this repository is Apache-2.0, see the top-level `LICENSE`).
