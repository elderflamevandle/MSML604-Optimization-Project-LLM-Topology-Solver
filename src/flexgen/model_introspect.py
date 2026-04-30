from dataclasses import dataclass
from pathlib import Path
import json
import logging
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    num_layers: int
    hidden_dim: int
    num_heads: int
    num_kv_heads: int
    intermediate_size: int
    vocab_size: int
    dtype_bytes: int

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads


_DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4, "int8": 1, "int4": 1}


def load_model_spec(hf_id: str) -> ModelSpec:
    model_path = Path(hf_id).expanduser()
    if model_path.exists():
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Local model path exists but has no config.json: {model_path}"
            )
        local_dir = model_path
        logger.info("Loaded model config from local path: %s", config_path)
    else:
        local_dir = Path(snapshot_download(repo_id=hf_id, allow_patterns=["config.json"]))
        logger.info("Loaded model config from HuggingFace repo: %s", hf_id)

    cfg = json.loads((local_dir / "config.json").read_text())

    num_heads = cfg["num_attention_heads"]
    num_kv_heads = cfg.get("num_key_value_heads", num_heads)
    dtype_str = cfg.get("torch_dtype", "float16")

    return ModelSpec(
        hf_id=hf_id,
        num_layers=cfg["num_hidden_layers"],
        hidden_dim=cfg["hidden_size"],
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=cfg.get("intermediate_size", 4 * cfg["hidden_size"]),
        vocab_size=cfg.get("vocab_size", 32000),
        dtype_bytes=_DTYPE_BYTES.get(dtype_str, 2),
    )


def params_per_layer(spec: ModelSpec) -> int:
    h = spec.hidden_dim
    h_kv_ratio = spec.num_kv_heads / spec.num_heads
    attn = h * h * (1 + 2 * h_kv_ratio + 1)
    ffn = 3 * spec.intermediate_size * h
    norms = 2 * h
    return int(attn + ffn + norms)


_QUANT_BYTES_PER_ELEM = {"fp16": 2.0, "int8": 1.0, "int4": 0.5}


def weights_per_layer_bytes(spec: ModelSpec, q: str) -> float:
    return params_per_layer(spec) * _QUANT_BYTES_PER_ELEM[q]


def kv_per_token_bytes(spec: ModelSpec, q: str) -> float:
    bytes_per_elem = _QUANT_BYTES_PER_ELEM[q] if q in ("fp16", "int4") else 2.0
    return 2 * spec.num_kv_heads * spec.head_dim * spec.num_layers * bytes_per_elem
