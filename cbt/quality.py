"""Level-specific base quality functions S_T, and the self-consistency term.

Each level weights a different mix of factors and every level's weights sum to
10, which puts S_T on the 1-10 rating scale the rest of the pipeline expects:

    core belief     accuracy 5, conciseness 3, clarity 2
    fine-grained    accuracy 4, consistency 3, conciseness 2, evidence 1
    distortion      accuracy 4, evidence 4, conciseness 2

Pure functions over parsed states, like the diagnostic layer.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Sequence

from cbt.diagnostic import VOCABULARY, _clamp, _norm, canonicalise, q_coh

WEIGHT_TOTAL = 10.0

WEIGHTS: Dict[str, Dict[str, float]] = {
    "core_beliefs": {"accuracy": 5.0, "conciseness": 3.0, "clarity": 2.0},
    "fine_grained": {
        "accuracy": 4.0,
        "consistency": 3.0,
        "conciseness": 2.0,
        "evidence": 1.0,
    },
    "distortions": {"accuracy": 4.0, "evidence": 4.0, "conciseness": 2.0},
}

# Label counts past which conciseness starts to decay.
EXPECTED_LABELS: Dict[str, int] = {
    "core_beliefs": 1,
    "fine_grained": 2,
    "distortions": 3,
}


def accuracy(labels: Sequence, level: str) -> float:
    """Share of labels drawn from the permissible set for the level."""
    labels = [_norm(x) for x in (labels or ())]
    if not labels:
        return 0.0
    allowed = {_norm(x) for x in VOCABULARY[level]}
    return _clamp(sum(1 for x in labels if x in allowed) / len(labels))


def conciseness(n_labels: int, level: str) -> float:
    """1 up to what the level expects, then expected / n.

    Zero labels score zero: an empty prediction is not concise, it is empty.
    """
    if n_labels <= 0:
        return 0.0
    expected = EXPECTED_LABELS.get(level, 3)
    return 1.0 if n_labels <= expected else _clamp(expected / n_labels)


def clarity(rationale: str, min_words: int = 20) -> float:
    """How articulated the reasoning is, saturating at ``min_words``."""
    return _clamp(len(str(rationale or "").split()) / min_words)


def evidence(state: Dict) -> float:
    """Whether the thought grounds itself in the patient's own statements.

    Direct quotes count for more than bare assertions; distortion detection in
    particular is expected to cite concrete utterances.
    """
    quotes = state.get("quotes") or []
    if quotes:
        return _clamp(len(quotes) / 2)
    supporting = state.get("evidence_for") or []
    if supporting:
        return _clamp(len(supporting) / 4)
    return 0.0


def base_quality(state: Dict, level: str) -> float:
    """S_T for one thought state, on a 0-10 scale."""
    if level not in WEIGHTS:
        raise ValueError(f"unknown level: {level!r}")

    labels = state.get(level, []) or []
    components = {
        "accuracy": accuracy(labels, level),
        "conciseness": conciseness(len(labels), level),
        "clarity": clarity(state.get("rationale", "")),
        "consistency": q_coh(labels, state.get("core_beliefs", [])),
        "evidence": evidence(state),
    }
    return sum(w * components[name] for name, w in WEIGHTS[level].items())


# --- Self-consistency ------------------------------------------------------
#
# The paper states the combination twice: as score = csc + beta * fsc with
# beta in {0, 0.15} in the method, and as a weight alpha balancing content
# against a frequency prior, robust over [0.80, 0.90], in the ablation. They
# are one parameterisation under beta = (1 - alpha) / alpha: alpha = 0.87
# gives beta = 0.149, the middle of the reported range.

BETA = 0.15
FREQUENCY_SCALE = 10.0


def beta_from_alpha(alpha: float) -> float:
    """Convert the ablation's alpha into the method's beta."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    return (1.0 - alpha) / alpha


def alpha_from_beta(beta: float) -> float:
    """Report a beta on the ablation's scale."""
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    return 1.0 / (1.0 + beta)


def frequency_scores(states: Sequence[Dict], level: str) -> list:
    """Ten times the share of candidates proposing the same label set.

    Agreement is counted over canonical labels: two candidates that paraphrase
    one answer agree, they are not two dissenters.
    """
    keys = [frozenset(canonicalise(s.get(level, []) or [], level)) for s in states]
    counts = Counter(keys)
    total = len(states) or 1
    return [FREQUENCY_SCALE * counts[key] / total for key in keys]


def base_score_max(beta: float = BETA) -> float:
    """The ceiling of csc + beta * fsc, for normalising into S_final."""
    return WEIGHT_TOTAL + beta * FREQUENCY_SCALE


def make_base_scorer(level: str, beta: float = BETA):
    """Build a ``scoring_function`` for the first Score of a level.

    Scores the candidate set together, because the frequency term is a
    property of the set. Each thought's combined rating is left on the state
    under ``base_score`` for the final Score to pick up.
    """

    def score(states):
        single = isinstance(states, dict)
        batch = [states] if single else list(states)

        content = [base_quality(s, level) for s in batch]
        frequency = frequency_scores(batch, level) if beta else [0.0] * len(batch)

        out = []
        for state, csc, fsc in zip(batch, content, frequency):
            state["content_score"] = csc
            state["frequency_score"] = fsc
            state["base_score"] = csc + beta * fsc
            out.append(state["base_score"])
        return out[0] if single else out

    return score
