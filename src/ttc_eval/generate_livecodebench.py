"""
Generate N samples per problem for LiveCodeBench code generation using vLLM.

Same style as generate_math500.py: talks to vLLM's OpenAI-compatible
`/v1/chat/completions` endpoint with a `messages` list, so vLLM applies the
model's chat template -- we never hand-assemble a raw chat-formatted string.
"""

import multiprocessing
import random
from functools import partial

import pydra
import requests
from tqdm import tqdm

from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from ttc_eval.lcb_prompts import SYSTEM_MESSAGE_GENERIC, build_question_prompt
from ttc_eval.utils import GenerateScriptConfig, save_yaml
from ttc_eval.vllm_utils import vllm_manager


class LCBGenerateConfig(GenerateScriptConfig):
    release_version = "release_v1"


def run_inference(item: dict, config: LCBGenerateConfig):
    outpath = config.save_dir / f"{item['id']}.yaml"
    if outpath.exists():
        return

    prompt = item["prompt"]
    url = f"http://localhost:{config.vllm_port}/v1/chat/completions"

    num_samples = config.num_samples
    batch_size = config.batch_size
    assert num_samples % batch_size == 0

    samples = []
    for _ in tqdm(range(num_samples // batch_size), desc=f"Item {item['id']}"):
        body = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE_GENERIC},
                {"role": "user", "content": prompt},
            ],
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
        "question_id": item["id"],
        "samples": samples,
    }
    save_yaml(outpath, out)


@pydra.main(LCBGenerateConfig)
def main(config: LCBGenerateConfig):
    problems = load_code_generation_dataset(release_version=config.release_version)
    items = [
        {"id": p.question_id, "prompt": build_question_prompt(p)} for p in problems
    ]

    random.seed(config.seed)
    random.shuffle(items)

    limit = config.limit if config.limit is not None else len(items)
    stride = config.stride if config.stride is not None else 1
    offset = config.offset if config.offset is not None else 0
    items = items[offset:limit:stride]

    print(f"Total number of items to process: {len(items)}")

    with vllm_manager(config) as vllm_port:
        config.vllm_port = vllm_port

        go_func = partial(run_inference, config=config)
        if config.num_workers not in (0, None):
            with multiprocessing.Pool(config.num_workers) as pool:
                list(tqdm(pool.imap_unordered(go_func, items), total=len(items)))
        else:
            for item in tqdm(items):
                go_func(item)


if __name__ == "__main__":
    main()
