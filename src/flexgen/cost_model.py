from dataclasses import dataclass
from src.flexgen.model_introspect import ModelSpec, params_per_layer


@dataclass(frozen=True)
class EnumPoint:
    gbs: int
    num_gb: int
    q: str
    delegate: bool
    overlap: bool

    @property
    def block_size(self) -> int:
        return self.gbs * self.num_gb


@dataclass(frozen=True)
class PlacementFractions:
    w_g: float; w_c: float; w_d: float
    c_g: float; c_c: float; c_d: float
    h_g: float; h_c: float; h_d: float


def prefill_flops_per_layer(spec: ModelSpec, batch: int, seq_len: int) -> float:
    matmul = 2 * batch * seq_len * params_per_layer(spec)
    attn = 4 * batch * seq_len * seq_len * spec.num_kv_heads * spec.head_dim
    return float(matmul + attn)


def decode_flops_per_layer(spec: ModelSpec, batch: int, kv_len: int) -> float:
    matmul = 2 * batch * 1 * params_per_layer(spec)
    attn = 4 * batch * 1 * kv_len * spec.num_kv_heads * spec.head_dim
    return float(matmul + attn)
