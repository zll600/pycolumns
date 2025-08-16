# flake8: noqa
from typing import Dict, List, Any
import numpy as np

ALLOWED_EXTENSIONS: List[str] = [
    "array",
    "meta",
    "index",
    "index1",
    "sorted",
    "cols",
    "chunks",
    "json",
]


DEFAULT_CACHE_MEM: str = "1g"

DEFAULT_CNAME: str = "zstd"
DEFAULT_CLEVEL: int = 5
DEFAULT_SHUFFLE: str = "bitshuffle"

DEFAULT_COMPRESSION: Dict[str, Any] = {
    "cname": DEFAULT_CNAME,
    "clevel": DEFAULT_CLEVEL,
    "shuffle": DEFAULT_SHUFFLE,
}

# 1 megabyte
DEFAULT_CHUNKSIZE: str = "1m"

CHUNKS_DTYPE: np.dtype = np.dtype(
    [
        ("offset", "i8"),
        ("nbytes", "i8"),
        ("rowstart", "i8"),
        ("nrows", "i8"),
        ("is_external", bool),
    ]
)
