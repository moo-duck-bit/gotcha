#!/usr/bin/env python3
"""Join the CBT-Bench splits by ``id`` into one input file and one gold file.

Optional. ``python -m cbt.run`` reads the per-task test splits directly; this
is for inspecting all three labels of a case side by side.

    python merge.py --data-dir CBT-Bench/test --out-dir merged

Writes ``userinput.json`` (id, ori_text, thoughts, situation) and
``answer.json`` (id plus whichever of core_belief, fine_grained, distortions
the splits supply). Both contain source narratives and are git-ignored.
"""

import argparse
import json
import os
from collections import defaultdict
from glob import glob

from cbt.data import SPLITS

INPUT_FIELDS = ("ori_text", "thoughts", "situation")

# output key -> gold fields that may carry it, first match wins
ANSWER_FIELDS = {
    "core_belief": ("core_belief_major",),
    "fine_grained": ("core_belief_fine_grained", "fine_grained"),
    "distortions": ("distortions",),
}


def load_records(path):
    """Records from a JSON file, or from every JSON file in a directory."""
    paths = sorted(glob(os.path.join(path, "*.json"))) if os.path.isdir(path) else [path]
    records = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError as exc:
                print(f"skipping {p}: {exc}")
                continue
        records.extend(data if isinstance(data, list) else [data])
    return records


def merge(paths):
    """Fold records from every source into one entry per id."""
    by_id = defaultdict(dict)
    for path in paths:
        for record in load_records(path):
            item_id = str(record.get("id", "")).strip()
            if not item_id:
                continue
            entry = by_id[item_id]
            for field in INPUT_FIELDS:
                if record.get(field) and not entry.get(field):
                    entry[field] = record[field]
            for key, fields in ANSWER_FIELDS.items():
                for field in fields:
                    if field in record:
                        entry[key] = record[field]
                        break
    return by_id


def split(by_id):
    inputs, answers = [], []
    for item_id, fields in sorted(by_id.items()):
        if not any(key in fields for key in ANSWER_FIELDS):
            print(f"id {item_id}: no gold labels, skipped")
            continue
        inputs.append({"id": item_id, **{f: fields.get(f, "") for f in INPUT_FIELDS}})
        answers.append(
            {"id": item_id, **{k: fields[k] for k in ANSWER_FIELDS if k in fields}}
        )
    return inputs, answers


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        default=os.path.join("CBT-Bench", "test"),
        help="directory holding the CBT-Bench test splits",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        help="explicit split files or directories, overriding --data-dir",
    )
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args(argv)

    paths = args.inputs or [
        os.path.join(args.data_dir, name) for name, _ in SPLITS.values()
    ]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            "not found: " + ", ".join(missing) + "\nCBT-Bench is not "
            "redistributed with this repository; see the README."
        )

    inputs, answers = split(merge(paths))

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in (("userinput.json", inputs), ("answer.json", answers)):
        out = os.path.join(args.out_dir, name)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2, ensure_ascii=False)
        print(f"wrote {len(rows)} records to {out}")


if __name__ == "__main__":
    main()
