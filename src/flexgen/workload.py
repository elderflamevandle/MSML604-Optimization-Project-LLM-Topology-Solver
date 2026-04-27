from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class WorkloadSpec:
    prompt_len: int
    decode_len: int

    def __post_init__(self):
        if self.prompt_len <= 0:
            raise ValueError("prompt_len must be positive")
        if self.decode_len <= 0:
            raise ValueError("decode_len must be positive")


_DEFAULTS = {"prompt_len": 512, "decode_len": 128}


def load_workload(path: str) -> WorkloadSpec:
    data = yaml.safe_load(Path(path).read_text()) or {}
    merged = {**_DEFAULTS, **{k: int(v) for k, v in data.items() if k in _DEFAULTS}}
    return WorkloadSpec(**merged)
