"""Run GoTCHA over a CBT-Bench split.

    python -m cbt.run --level distortions --limit 5
    python -m cbt.run --level distortions --limit 5 --dry-run

``--dry-run`` drives the whole graph from a stub, exercising every operation
without an API key. It checks that the pipeline is wired, not that a model is
any good.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from typing import Dict, List

from graph_of_thoughts import controller, language_models

from cbt import data, evaluate, similarity
from cbt.diagnostic import canonicalise, unresolved
from cbt.pipeline import (
    CBTParser,
    CBTPrompter,
    build_graph,
    level_operations,
)

LOGGER = logging.getLogger("cbt.run")


def _raw_labels(graph, level: str) -> List[str]:
    """What the graph's last selection said, before canonicalisation."""
    thoughts = graph.leaves[0].get_thoughts() if graph.leaves else []
    labels: List[str] = []
    for thought in thoughts:
        for label in thought.state.get(level, []) or []:
            if label not in labels:
                labels.append(label)
    return labels


def run_case(case: Dict, level: str, lm) -> Dict:
    """Run one case and return its prediction plus recovery counters."""
    graph = build_graph()
    ctrl = controller.Controller(
        lm, graph, CBTPrompter(), CBTParser(), data.case_inputs(case)
    )
    ctrl.run()

    counters = {
        name: {
            "refinements": op.n_refinements,
            "backtracks": op.n_backtracks,
            "recovered_first_try": op.n_recovered_first_try,
            "failed": op.n_failed,
        }
        for name, op in level_operations(graph).items()
    }
    raw = _raw_labels(graph, level)
    return {
        "id": case["id"],
        "predicted": canonicalise(raw, level),
        "unresolved": unresolved(raw, level),
        "gold": case["gold"],
        "recovery": counters,
    }


def build_lm(args):
    if args.dry_run:
        from cbt.stub_lm import StubLM

        return StubLM()
    if not os.path.exists(args.config):
        sys.exit(
            f"{args.config} not found. Copy config_template.json to "
            f"{args.config} and add your API key, or pass --dry-run."
        )
    return language_models.ChatGPT(args.config, model_name=args.model, cache=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run GoTCHA over a CBT-Bench split.")
    parser.add_argument(
        "--level",
        default="distortions",
        choices=list(data.SPLITS),
        help="which CBT-Bench task to run",
    )
    parser.add_argument("--limit", type=int, help="run only the first N cases")
    parser.add_argument("--data-dir", default=data.DEFAULT_DIR)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--model", default="chatgpt")
    parser.add_argument(
        "--similarity",
        default="lexical",
        choices=list(similarity.BACKENDS),
        help=(
            "backend for definitional alignment. 'sbert' is the paper's "
            "sentence-BERT measure and needs sentence-transformers; "
            "'lexical' is dependency-free"
        ),
    )
    parser.add_argument(
        "--sbert-model",
        default=similarity.DEFAULT_MODEL,
        help="sentence-transformers model to use with --similarity sbert",
    )
    parser.add_argument("--out", help="write per-case results to this JSON file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run the graph against a stub instead of a real model",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.similarity == "sbert":
        similarity.install("sbert", model_name=args.sbert_model)
    else:
        similarity.install(args.similarity)

    try:
        cases = data.load_split(args.level, args.data_dir, args.limit)
    except data.DatasetNotFound as exc:
        sys.exit(str(exc))

    lm = build_lm(args)

    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] case {case['id']}", file=sys.stderr)
        try:
            results.append(run_case(case, args.level, lm))
        except Exception as exc:  # one bad case should not end the run
            LOGGER.error("case %s failed: %s", case["id"], exc)
            results.append(
                {
                    "id": case["id"],
                    "predicted": [],
                    "gold": case["gold"],
                    "error": str(exc),
                }
            )

    report = evaluate.score(
        [r["predicted"] for r in results],
        [r["gold"] for r in results],
        args.level,
    )
    print()
    print(evaluate.format_report(report))

    drifted = Counter(x for r in results for x in r.get("unresolved", ()))
    if drifted:
        print("\nlabels the model produced that the taxonomy does not accept:")
        for label, count in drifted.most_common():
            print(f"  {count:4d}  {label}")

    cost = getattr(lm, "cost", 0.0)
    if cost:
        print(f"\nAPI cost: ${cost:.4f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(
                {"metrics": report, "results": results},
                fh,
                indent=2,
                ensure_ascii=False,
            )
        print(f"wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
