from typing import Dict, Any
import os
import hashlib
from functools import cache
import numpy as np

CACHE_ENABLED = True
CACHE_PATH = "~/Desktop/autocuepoints_cache/"


def cache_path(key: str) -> str:
    return os.path.expanduser(CACHE_PATH + key + ".npy")


def exists(key: str) -> bool:
    return os.path.exists(cache_path(key))


@cache
def convert_to_key(key: str) -> str:
    result = hashlib.md5(key.encode("utf-8"))
    result = result.hexdigest()
    return result


def get(key: str) -> Any:
    file_path = cache_path(key)
    if not os.path.exists(file_path):
        return None

    data = np.load(file_path)
    return data


def put(key: str, data: np.ndarray) -> None:
    file_path = cache_path(key)
    np.save(file_path, data)
