"""
Process Reward Model (PRM) scoring: the reward oracle r(x, y) used live by
eliminate.py when reward_mode=prm (see arXiv:2506.12721).

Uses Qwen2.5-Math-PRM-7B: a response is split into reasoning steps, each
step boundary is marked with the model's <extra_0> separator token, and the
model outputs a per-step [P(bad), P(good)] distribution at each separator.
We take P(good) at the *last* step as the single scalar reward r(x, y) the
paper's reward oracle requires -- this is the standard aggregation for this
PRM family (a response is only as good as where it ends up).

`PRMScorer` is instantiated once by eliminate.py and called per response as
it's generated. This file's own CLI (`main`, below) is a separate utility:
it PRM-scores an already-generated pool of samples (e.g. the output of
evaluate_math500.py), for people who want PRM scores without running the
elimination algorithm:

    python src/ttc_eval/prm_score.py \
        samples_dir=$SAVE_DIR/math500_eval save_dir=$SAVE_DIR/math500_prm
"""

import re

import pydra
import torch
import torch.nn.functional as F
from pydra import Config, REQUIRED
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from ttc_eval.utils import load_yaml, save_yaml

PRM_MODEL_NAME = "Qwen/Qwen2.5-Math-PRM-7B"
PRM_SYSTEM_MESSAGE = "Please reason step by step, and put your final answer within \\boxed{}."
STEP_SEPARATOR = "<extra_0>"
CONTEXT_LIMIT = 4096  # hard cap on tokenized conversation length


def format_response_to_steps(response: str) -> list[str]:
    """
    Splits a response into steps at each line containing the word "step"
    (covers "Step 1:", "## Step 2", etc). Returns [] if the response has no
    step markers, or its last step doesn't contain a \\boxed{} answer --
    PRM scoring is skipped for such responses (reward defaults to 0).
    """
    pattern = r"(?i)(.*\bstep\b[^\n]*)"
    positions = [m.start() for m in re.finditer(pattern, response)]
    if not positions:
        return []

    positions.append(len(response))
    steps = [response[positions[i] : positions[i + 1]].strip() for i in range(len(positions) - 1)]

    if steps and "\\boxed{" in steps[-1]:
        return steps
    return []


class PRMScorer:
    def __init__(self, model_name: str = PRM_MODEL_NAME, device_map: str = "auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name, device_map=device_map, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).eval()
        self.step_sep_id = self.tokenizer.convert_tokens_to_ids(STEP_SEPARATOR)

    def _encode(self, messages):
        conversation_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return self.tokenizer.encode(
            conversation_str, return_tensors="pt", add_special_tokens=False
        )

    def score_response(self, question_prompt: str, response: str) -> tuple[float, list[float]]:
        """Returns (last_step_reward, all_step_rewards). Reward is 0.0 if the
        response has no parseable steps or exceeds CONTEXT_LIMIT tokens."""
        steps = format_response_to_steps(response)
        if not steps:
            return 0.0, []

        messages = [
            {"role": "system", "content": PRM_SYSTEM_MESSAGE},
            {"role": "user", "content": question_prompt},
            {"role": "assistant", "content": STEP_SEPARATOR.join(steps) + STEP_SEPARATOR},
        ]
        input_ids = self._encode(messages)
        if input_ids.size(-1) > CONTEXT_LIMIT:
            return 0.0, []

        input_ids = input_ids.to(self.model.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)

        logits = getattr(outputs, "logits", outputs[0])
        token_masks = input_ids == self.step_sep_id
        step_rewards = _make_step_rewards(logits, token_masks)[0]  # single-item batch
        if not step_rewards:
            return 0.0, []
        return float(step_rewards[-1]), [float(s) for s in step_rewards]

    def score_samples(self, question_prompt: str, samples: list[str]):
        scores, step_scores = [], []
        for sample in samples:
            score, steps = self.score_response(question_prompt, sample)
            scores.append(score)
            step_scores.append(steps)
        return scores, step_scores


def _make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1) * token_masks.unsqueeze(-1)
    all_scores = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
        all_scores.append(positive_probs.cpu().tolist())
    return all_scores


class PRMScoreConfig(Config):
    samples_dir = REQUIRED  # output of evaluate_math500.py / evaluate_livecodebench.py
    save_dir = REQUIRED
    question_field = "question"  # yaml field holding the prompt shown to the PRM

    def finalize(self):
        from pathlib import Path

        self.samples_dir = Path(self.samples_dir)
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)


@pydra.main(PRMScoreConfig)
def main(config: PRMScoreConfig):
    scorer = PRMScorer()

    sample_paths = sorted(config.samples_dir.glob("*.yaml"))
    for path in tqdm(sample_paths, desc="PRM scoring"):
        out_path = config.save_dir / path.name
        if out_path.exists():
            continue

        data = load_yaml(path)
        question_prompt = data.get(config.question_field, "")
        scores, step_scores = scorer.score_samples(question_prompt, data.get("samples", []))
        data["scores"] = scores
        data["step_scores"] = step_scores
        save_yaml(out_path, data)


if __name__ == "__main__":
    main()
