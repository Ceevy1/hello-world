from __future__ import annotations

import numpy as np


def lomo_split(modules: np.ndarray) -> list[dict[str, np.ndarray | str]]:
    mods = np.asarray(modules)
    uniq = np.unique(mods)
    splits = []
    for test_module in uniq:
        test_mask = mods == test_module
        train_mask = ~test_mask
        splits.append({"train_mask": train_mask, "test_mask": test_mask, "test_module": str(test_module)})
    return splits


def lopo_split(presentations: np.ndarray) -> list[dict[str, np.ndarray | str]]:
    prs = np.asarray(presentations)
    uniq = np.unique(prs)
    out = []
    for test_p in uniq:
        test_mask = prs == test_p
        train_mask = ~test_mask
        out.append({"train_mask": train_mask, "test_mask": test_mask, "test_presentation": str(test_p)})
    return out
