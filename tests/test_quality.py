"""
Base quality S_T and the self-consistency term.
"""

import pytest

from cbt import diagnostic as D
from cbt import quality as Q


# --- level weights ---------------------------------------------------------


@pytest.mark.parametrize("level", list(Q.WEIGHTS))
def test_every_level_sums_to_the_rating_scale(level):
    assert sum(Q.WEIGHTS[level].values()) == pytest.approx(Q.WEIGHT_TOTAL)


def test_weights_match_the_paper():
    assert Q.WEIGHTS["core_beliefs"] == {
        "accuracy": 5.0,
        "conciseness": 3.0,
        "clarity": 2.0,
    }
    assert Q.WEIGHTS["fine_grained"] == {
        "accuracy": 4.0,
        "consistency": 3.0,
        "conciseness": 2.0,
        "evidence": 1.0,
    }
    assert Q.WEIGHTS["distortions"] == {
        "accuracy": 4.0,
        "evidence": 4.0,
        "conciseness": 2.0,
    }


def test_base_quality_rejects_an_unknown_level():
    with pytest.raises(ValueError):
        Q.base_quality({}, "not_a_level")


# --- components ------------------------------------------------------------


def test_accuracy_is_the_proportion_in_the_permissible_set():
    assert Q.accuracy(["labeling", "mind reading"], "distortions") == 1.0
    assert Q.accuracy(["labeling", "not-a-thing"], "distortions") == pytest.approx(0.5)
    assert Q.accuracy([], "distortions") == 0.0


def test_conciseness_decays_past_what_the_level_expects():
    assert Q.conciseness(1, "core_beliefs") == 1.0
    assert Q.conciseness(2, "core_beliefs") == pytest.approx(0.5)
    assert Q.conciseness(3, "distortions") == 1.0
    assert Q.conciseness(6, "distortions") == pytest.approx(0.5)


def test_an_empty_prediction_is_not_concise():
    assert Q.conciseness(0, "distortions") == 0.0


def test_clarity_saturates():
    assert Q.clarity("") == 0.0
    assert Q.clarity("word " * 10) == pytest.approx(0.5)
    assert Q.clarity("word " * 50) == 1.0


def test_evidence_prefers_direct_quotes():
    quoted = Q.evidence({"quotes": ["a", "b"]})
    asserted = Q.evidence({"evidence_for": ["a", "b"]})
    assert quoted > asserted
    assert Q.evidence({}) == 0.0


# --- self-consistency ------------------------------------------------------


def test_alpha_and_beta_are_the_same_parameterisation():
    """beta = 0.15 corresponds to alpha = 0.87, inside the reported range."""
    assert Q.beta_from_alpha(0.87) == pytest.approx(0.15, abs=1e-3)
    assert Q.alpha_from_beta(0.15) == pytest.approx(0.87, abs=1e-3)
    assert Q.alpha_from_beta(Q.beta_from_alpha(0.85)) == pytest.approx(0.85)


def test_alpha_one_means_content_only():
    assert Q.beta_from_alpha(1.0) == 0.0


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_alpha_out_of_range_is_rejected(bad):
    with pytest.raises(ValueError):
        Q.beta_from_alpha(bad)


def test_frequency_rewards_agreement():
    """Unanimous candidates score the full scale; a lone dissenter less."""
    agreed = [{"distortions": ["labeling"]} for _ in range(4)]
    assert Q.frequency_scores(agreed, "distortions") == [10.0] * 4

    split = agreed[:3] + [{"distortions": ["mind reading"]}]
    scores = Q.frequency_scores(split, "distortions")
    assert scores[:3] == [pytest.approx(7.5)] * 3
    assert scores[3] == pytest.approx(2.5)


def test_frequency_counts_paraphrases_as_agreement():
    """Two spellings of one answer are one answer, not two dissenters."""
    states = [
        {"fine_grained": ["I am undesirable"]},
        {"fine_grained": ["I am unwanted"]},
    ]
    assert Q.frequency_scores(states, "fine_grained") == [10.0, 10.0]


def test_frequency_distinguishes_different_label_sets():
    states = [
        {"distortions": ["labeling"]},
        {"distortions": ["labeling", "mind reading"]},
    ]
    assert Q.frequency_scores(states, "distortions") == [5.0, 5.0]


# --- the combined scorer ---------------------------------------------------


def _candidate(label, rationale="detail " * 25):
    return {
        "distortions": [label],
        "rationale": rationale,
        "quotes": ["q1", "q2"],
        "evidence_for": ["a"],
        "evidence_against": ["b"],
    }


def test_scorer_folds_frequency_into_the_base_score():
    scorer = Q.make_base_scorer("distortions")
    states = [_candidate("labeling"), _candidate("labeling")]
    scores = scorer(states)

    for state, total in zip(states, scores):
        assert state["frequency_score"] == pytest.approx(10.0)
        assert total == pytest.approx(
            state["content_score"] + Q.BETA * state["frequency_score"]
        )
        assert state["base_score"] == pytest.approx(total)


def test_agreement_outranks_an_equally_good_dissenter():
    scorer = Q.make_base_scorer("distortions")
    states = [
        _candidate("labeling"),
        _candidate("labeling"),
        _candidate("mind reading"),
    ]
    scores = scorer(states)
    assert scores[0] == scores[1] > scores[2]


def test_beta_zero_disables_self_consistency():
    scorer = Q.make_base_scorer("distortions", beta=0.0)
    states = [_candidate("labeling"), _candidate("mind reading")]
    scores = scorer(states)
    assert scores[0] == pytest.approx(scores[1])
    assert all(s["frequency_score"] == 0.0 for s in states)


def test_scorer_accepts_a_single_state():
    """Score(combined_scoring=False) hands over one state, not a list."""
    scorer = Q.make_base_scorer("distortions")
    assert isinstance(scorer(_candidate("labeling")), float)


def test_base_score_ceiling_tracks_beta():
    assert Q.base_score_max(0.0) == pytest.approx(10.0)
    assert Q.base_score_max(0.15) == pytest.approx(11.5)


def test_combined_score_normalises_into_the_unit_interval():
    """Whatever beta is, S_final's base term stays in [0, 1]."""
    assert D.normalise_base(Q.base_score_max(), Q.base_score_max()) == 1.0
