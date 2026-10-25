"""
Generate N samples per problem for HuggingFaceH4/MATH-500 using vLLM.

Talks to vLLM's OpenAI-compatible `/v1/chat/completions` endpoint with a
`messages` list, so vLLM applies the model's chat template itself -- we
never hand-assemble a raw chat-formatted string. For a plain few-shot CoT
prompt (as used here), that just means wrapping the whole thing as a
single user turn.
"""

import multiprocessing
import random
from functools import partial
from pathlib import Path

import pydra
import requests
from datasets import load_dataset
from tqdm import tqdm

from ttc_eval.prompts import MATH_COT_PROMPT
from ttc_eval.utils import GenerateScriptConfig, save_yaml
from ttc_eval.vllm_utils import vllm_manager


def run_inference(item: dict, config: GenerateScriptConfig):
    outpath = config.save_dir / f"{item['id']}.yaml"
    if outpath.exists():
        return

    prompt = MATH_COT_PROMPT + f"\n\nProblem:\n{item['problem']}\n\nSolution:"
    url = f"http://localhost:{config.vllm_port}/v1/chat/completions"

    num_samples = config.num_samples
    batch_size = config.batch_size
    assert num_samples % batch_size == 0

    samples = []
    for _ in tqdm(range(num_samples // batch_size), desc=f"Item {item['id']}"):
        body = {
            "model": config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "n": batch_size,
        }

        response = requests.post(url, json=body)
        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.text}")
            continue

        respj = response.json()
        samples.extend(choice["message"]["content"] for choice in respj["choices"])

    out = {
        "prompt": prompt,
        "question": item["problem"],
        "samples": samples,
        "gt_answer": item["solution"],
    }
    save_yaml(outpath, out)


@pydra.main(GenerateScriptConfig)
def main(config: GenerateScriptConfig):
    test_dataset = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    for i, item in enumerate(test_dataset):
        item["id"] = i

    random.seed(config.seed)
    random.shuffle(test_dataset)

    limit = config.limit if config.limit is not None else len(test_dataset)
    stride = config.stride if config.stride is not None else 1
    offset = config.offset if config.offset is not None else 0
    test_dataset = test_dataset[offset:limit:stride]

    print(f"Total number of items to process: {len(test_dataset)}")

    with vllm_manager(config) as vllm_port:
        config.vllm_port = vllm_port

        go_func = partial(run_inference, config=config)
        if config.num_workers not in (0, None):
            with multiprocessing.Pool(config.num_workers) as pool:
                list(tqdm(pool.imap_unordered(go_func, test_dataset), total=len(test_dataset)))
        else:
            for item in tqdm(test_dataset):
                go_func(item)


if __name__ == "__main__":
    main()
