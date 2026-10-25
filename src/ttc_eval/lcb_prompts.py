"""
Generic (model-agnostic) LiveCodeBench code-generation prompt + extraction,
trimmed from lcb_runner's prompts/code_generation.py and
utils/extraction_utils.py (which otherwise special-case ~15 proprietary
model families via LMStyle). We only need the "generic" path since vLLM's
`/v1/chat/completions` applies the chat template for us.
"""

from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

SYSTEM_MESSAGE_GENERIC = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program "
    "that matches the specification and passes all tests."
)

FORMATTING_MESSAGE_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)

FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to "
    "stdout (do not directly test on the sample inputs). Enclose your code "
    "within delimiters as follows. Ensure that when the python program "
    "runs, it reads the inputs, runs the algorithm and writes output to "
    "STDOUT."
)


def build_question_prompt(question: CodeGenerationProblem) -> str:
    prompt = f"### Question:\n{question.question_content}\n\n"
    if question.starter_code:
        prompt += f"### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{question.starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def extract_code(model_output: str) -> str:
    """Returns the contents of the last ```...``` fenced block, or "" if none."""
    lines = model_output.split("\n")
    fence_indices = [i for i, line in enumerate(lines) if "```" in line]
    if len(fence_indices) < 2:
        return ""
    return "\n".join(lines[fence_indices[-2] + 1 : fence_indices[-1]])
