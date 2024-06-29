# This class allows a persistent storage of data in a json file
# We do not store anything in memory, instead we read and write
# to the file every time we need to access the data
# We use pathlib as much as possible

import json
from pathlib import Path
from typing import List


def get_storage_file():
    from src.classes import paths
    return Path(paths.root) / 'storage.json'


def get(key):
    storage_file = get_storage_file()
    if not storage_file.exists():
        return None
    with open(storage_file, 'r') as f:
        storage = json.load(f)
    return storage.get(key)


def set(key, value):
    storage_file = get_storage_file()
    if not storage_file.exists():
        with open(storage_file, 'w') as f:
            json.dump({}, f)
    with open(storage_file, 'r') as f:
        storage = json.load(f)
        storage[key] = value
    with open(storage_file, 'w') as f:
        json.dump(storage, f)

def get_paths(key):
    v = get(key)
    if isinstance(v, List):
        return [Path(p) for p in v]
    if isinstance(v, str):
        return [Path(v)]
    return []

def set_paths(key, paths):
    if isinstance(paths, List):
        set(key, [str(p) for p in paths])
    elif isinstance(paths, str):
        set(key, [str(paths)])


def has(key):
    storage_file = get_storage_file()
    if not storage_file.exists():
        return False
    with open(storage_file, 'r') as f:
        storage = json.load(f)
    return key in storage
