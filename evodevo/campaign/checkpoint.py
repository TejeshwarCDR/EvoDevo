"""Generation-boundary checkpoints, atomically persisted and never pruned."""
from __future__ import annotations

import json
import os


def write_checkpoint(run, directory: str) -> str:
    state = run.get_state()
    generation = state["generation"]
    os.makedirs(directory, exist_ok=True)
    destination = os.path.join(directory, f"gen_{generation}.json")
    temporary = destination + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(state, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return destination


def latest_checkpoint(directory: str):
    if not os.path.isdir(directory):
        return None
    paths = [os.path.join(directory, name) for name in os.listdir(directory)
             if name.startswith("gen_") and name.endswith(".json")]
    if not paths:
        return None
    path = max(paths, key=lambda p: int(os.path.basename(p)[4:-5]))
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
