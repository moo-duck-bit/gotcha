"""Loading the CBT-Bench splits.

The three Level II tasks share their input fields but each names its gold
field differently, so this module hides that behind one case shape:
``{"id", "ori_text", "thoughts", "situation", "gold"}``.

The dataset is not redistributed here; the records are first-person
mental-health narratives. Point ``--data-dir`` at your own copy.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

# level -> (file name, gold field)
SPLITS: Dict[str, tuple] = {
    "core_beliefs": ("core_major_test.json", "core_belief_major"),
    "fine_grained": ("core_fine_test.json", "core_belief_fine_grained"),
    "distortions": ("distortions_test.json", "distortions"),
}

DEFAULT_DIR = os.path.join("CBT-Bench", "test")


class DatasetNotFound(FileNotFoundError):
    """Raised with a message that says how to obtain the data."""


def split_path(level: str, data_dir: str = DEFAULT_DIR) -> str:
    if level not in SPLITS:
        raise ValueError(f"unknown level: {level!r}")
    return os.path.join(data_dir, SPLITS[level][0])


def load_split(
    level: str, data_dir: str = DEFAULT_DIR, limit: int = None
) -> List[Dict]:
    """Load one task's test split as a list of cases.

    Each case is ``{"id", "ori_text", "thoughts", "situation", "gold"}`` with
    ``gold`` always a list, whether the split stores one label or several.
    """
    path = split_path(level, data_dir)
    if not os.path.exists(path):
        raise DatasetNotFound(
            f"{path} not found. CBT-Bench is not redistributed with this "
            f"repository; obtain it from its original distributor and pass "
            f"--data-dir, or place it at {DEFAULT_DIR}."
        )

    with open(path, "r", encoding="utf-8") as fh:
        records = json.load(fh)

    field = SPLITS[level][1]
    cases = []
    for record in records:
        gold = record.get(field) or []
        cases.append(
            {
                "id": str(record.get("id", len(cases))),
                "ori_text": record.get("ori_text", ""),
                "thoughts": record.get("thoughts", ""),
                "situation": record.get("situation", ""),
                "gold": list(gold) if isinstance(gold, list) else [gold],
            }
        )
    return cases[:limit] if limit else cases


def case_inputs(case: Dict) -> Dict:
    """The fields the pipeline takes as its problem parameters."""
    return {
        "ori_text": case["ori_text"],
        "thoughts": case["thoughts"],
        "situation": case["situation"],
    }
