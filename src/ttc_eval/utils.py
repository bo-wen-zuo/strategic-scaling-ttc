from pathlib import Path
import yaml
from pydra import Config, REQUIRED


def load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.CLoader)


def save_yaml(path: Path, data, sort_keys=True):
    with open(path, "w") as f:
        yaml.dump(data, f, sort_keys=sort_keys)


class GenerateScriptConfig(Config):
    model = REQUIRED
    save_dir = REQUIRED

    # vLLM OpenAI-compatible server settings. We talk to vLLM's own
    # `/v1/chat/completions` endpoint so the server applies the model's
    # chat template for us -- do not hand-format chat turns into a raw
    # text prompt.
    vllm_port = 8000
    gpus = None
    num_workers = None
    # Extra CLI flags forwarded to `vllm serve`. Only pass --chat-template
    # for models that don't ship their own (e.g. base Llama checkpoints
    # being used instruct-style); instruct models apply their tokenizer's
    # built-in template automatically.
    vllm_args = []

    seed = 0
    limit = None
    offset = None
    stride = None

    max_tokens = 1024
    num_samples = 4
    batch_size = 4
    top_p = 0.95
    temperature = 0.6

    def finalize(self):
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)


class EvaluateScriptConfig(Config):
    samples_dir: Path = REQUIRED
    save_dir: Path = REQUIRED

    num_workers: int = 1

    offset: int = 0
    stride: int = 1
    limit: int = 100_000

    def finalize(self):
        self.samples_dir = Path(self.samples_dir)
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
