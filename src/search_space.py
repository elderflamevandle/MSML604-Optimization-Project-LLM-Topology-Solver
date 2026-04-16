from dataclasses import dataclass

QUANTIZATIONS = ["fp16", "int8", "int4"]
BATCH_SIZES = [1, 4, 8, 16, 32]
PARALLELISMS = [1, 2]


@dataclass(frozen=True)
class Config:
    q: str
    b: int
    p: int


def get_search_space() -> list["Config"]:
    return [
        Config(q=q, b=b, p=p)
        for q in QUANTIZATIONS
        for b in BATCH_SIZES
        for p in PARALLELISMS
    ]
