"""Multi-label metrics for the CBT-Bench tasks.

Micro and macro precision, recall and F1, exact match and Hamming accuracy,
with a per-label breakdown.

Predictions and gold labels both pass through canonicalisation first. Without
it a paraphrase counts as a miss and a false positive at once, which alone
pins F1 at 0 while accuracy still reads high.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from cbt.diagnostic import canonicalise


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def score(
    predictions: Sequence[Sequence[str]],
    gold: Sequence[Sequence[str]],
    level: str,
    labels: Sequence[str] = None,
) -> Dict:
    """Compare predicted against gold label sets, case by case.

    ``labels`` restricts the macro average to a fixed label set; by default
    it covers every label either side used.
    """
    if len(predictions) != len(gold):
        raise ValueError("predictions and gold differ in length")

    pred_sets = [set(canonicalise(p, level)) for p in predictions]
    gold_sets = [set(canonicalise(g, level)) for g in gold]

    if labels is None:
        labels = sorted({x for s in pred_sets + gold_sets for x in s})
    labels = list(labels)

    tp = fp = fn = 0
    per_label = {}
    for label in labels:
        l_tp = sum(1 for p, g in zip(pred_sets, gold_sets) if label in p and label in g)
        l_fp = sum(
            1 for p, g in zip(pred_sets, gold_sets) if label in p and label not in g
        )
        l_fn = sum(
            1 for p, g in zip(pred_sets, gold_sets) if label not in p and label in g
        )
        tp, fp, fn = tp + l_tp, fp + l_fp, fn + l_fn
        per_label[label] = {**_prf(l_tp, l_fp, l_fn), "support": l_tp + l_fn}

    micro = _prf(tp, fp, fn)
    macro = {
        key: (sum(v[key] for v in per_label.values()) / len(per_label))
        if per_label
        else 0.0
        for key in ("precision", "recall", "f1")
    }

    exact = (
        sum(1 for p, g in zip(pred_sets, gold_sets) if p == g) / len(gold_sets)
        if gold_sets
        else 0.0
    )
    hamming = (
        sum(
            1 - len(p ^ g) / len(labels)
            for p, g in zip(pred_sets, gold_sets)
        )
        / len(gold_sets)
        if gold_sets and labels
        else 0.0
    )

    return {
        "level": level,
        "num_cases": len(gold_sets),
        "micro": micro,
        "macro": macro,
        "exact_match": exact,
        "hamming": hamming,
        "per_label": per_label,
    }


def format_report(result: Dict) -> str:
    """A table in the shape the paper's results are reported in."""
    lines = [
        f"{result['level']}  ({result['num_cases']} cases)",
        "",
        f"{'':<32}{'P':>8}{'R':>8}{'F1':>8}",
        f"{'micro':<32}"
        f"{result['micro']['precision']:>8.3f}"
        f"{result['micro']['recall']:>8.3f}"
        f"{result['micro']['f1']:>8.3f}",
        f"{'macro':<32}"
        f"{result['macro']['precision']:>8.3f}"
        f"{result['macro']['recall']:>8.3f}"
        f"{result['macro']['f1']:>8.3f}",
        "",
        f"exact match {result['exact_match']:.3f}   hamming {result['hamming']:.3f}",
        "",
        f"{'label':<32}{'P':>8}{'R':>8}{'F1':>8}{'n':>6}",
    ]
    for label, scores in sorted(
        result["per_label"].items(), key=lambda kv: -kv[1]["support"]
    ):
        lines.append(
            f"{label[:32]:<32}"
            f"{scores['precision']:>8.3f}"
            f"{scores['recall']:>8.3f}"
            f"{scores['f1']:>8.3f}"
            f"{scores['support']:>6d}"
        )
    return "\n".join(lines)
