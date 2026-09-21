"""Tier 0: the diagnostic quality layer, verified without a language model."""

import pytest

from cbt import diagnostic as D


# --- constants -------------------------------------------------------------


def test_weights_sum_to_one():
    assert D.W_CON + D.W_COH + D.W_DEF + D.W_COUNTER == pytest.approx(1.0)


def test_paper_constants():
    assert (D.W_CON, D.W_COH, D.W_DEF, D.W_COUNTER) == (0.2, 0.3, 0.4, 0.1)
    assert D.TAU_DIAG == 0.6
    assert D.LAMBDA == 3.0


def test_taxonomy_sizes():
    assert len(D.CORE_BELIEFS) == 3
    assert sum(len(v) for v in D.FINE_GRAINED.values()) == 19
    assert len(D.DISTORTION_DEFINITIONS) == 10


# --- q_con -----------------------------------------------------------------


def test_q_con_zero_without_counter_evidence():
    assert D.q_con(["a", "b"], []) == 0.0


def test_q_con_balanced_evidence_is_one():
    assert D.q_con(["a", "b"], ["c", "d"]) == 1.0


def test_q_con_partial_ratio():
    assert D.q_con(["a", "b", "c", "d"], ["e"]) == pytest.approx(0.25)


def test_q_con_saturates_above_parity():
    assert D.q_con(["a"], ["b", "c", "d"]) == 1.0


def test_q_con_counter_evidence_only_saturates():
    assert D.q_con([], ["a"]) == 1.0


def test_q_con_empty_is_zero():
    assert D.q_con([], []) == 0.0


# --- q_coh -----------------------------------------------------------------


def test_q_coh_all_permitted():
    assert D.q_coh(["I am incompetent", "I am trapped"], ["helpless"]) == 1.0


def test_q_coh_half_permitted():
    # "I am unlovable" belongs to unlovable, not helpless.
    score = D.q_coh(["I am incompetent", "I am unlovable"], ["helpless"])
    assert score == pytest.approx(0.5)


def test_q_coh_union_over_multiple_parents():
    score = D.q_coh(
        ["I am incompetent", "I am unlovable"], ["helpless", "unlovable"]
    )
    assert score == 1.0


def test_q_coh_none_permitted():
    assert D.q_coh(["I am unlovable"], ["worthless"]) == 0.0


def test_q_coh_empty_children_is_zero():
    assert D.q_coh([], ["helpless"]) == 0.0


def test_q_coh_is_case_insensitive():
    assert D.q_coh(["  I AM TRAPPED "], ["Helpless"]) == 1.0


# --- q_def -----------------------------------------------------------------


def test_q_def_rewards_definition_overlap():
    aligned = D.q_def(
        "the speaker views the situation in only two categories instead of on "
        "a continuum",
        ["all-or-nothing thinking"],
    )
    unrelated = D.q_def("the weather today is pleasant", ["all-or-nothing thinking"])
    assert aligned > unrelated
    assert 0.0 <= unrelated < aligned <= 1.0


def test_q_def_unknown_label_contributes_zero():
    known = D.q_def(
        "predict the future negatively without considering other outcomes",
        ["fortune-telling"],
    )
    mixed = D.q_def(
        "predict the future negatively without considering other outcomes",
        ["fortune-telling", "not-a-real-distortion"],
    )
    assert mixed == pytest.approx(known / 2)


def test_q_def_empty_labels_is_zero():
    assert D.q_def("anything at all", []) == 0.0


def test_q_def_examples_can_raise_score():
    rationale = "he assumed his friend was annoyed at him"
    base = D.q_def(rationale, ["mind reading"])
    boosted = D.q_def(
        rationale,
        ["mind reading"],
        examples={"mind reading": ("he assumed his friend was annoyed at him",)},
    )
    assert boosted > base
    assert boosted == pytest.approx(1.0)


def test_q_def_respects_injected_similarity():
    D.set_similarity_fn(lambda a, b: 1.0)
    try:
        assert D.q_def("x", ["labeling"]) == 1.0
    finally:
        D.set_similarity_fn(D.lexical_similarity)


# --- q_counter -------------------------------------------------------------


def test_q_counter_zero_when_unaddressed():
    assert D.q_counter("nothing relevant here", ["mental filter"]) == 0.0


def test_q_counter_full_coverage():
    rationale = " ".join(D.COUNTER_QUESTIONS["mental filter"])
    assert D.q_counter(rationale, ["mental filter"]) == pytest.approx(1.0)


def test_q_counter_partial_coverage():
    rationale = D.COUNTER_QUESTIONS["fortune-telling"][0]
    assert D.q_counter(rationale, ["fortune-telling"]) == pytest.approx(0.5)


def test_q_counter_empty_labels_is_zero():
    assert D.q_counter("anything", []) == 0.0


# --- S_diag ----------------------------------------------------------------


def test_s_diag_is_the_weighted_sum():
    assert D.s_diag(con=1.0, coh=1.0, deff=1.0, counter=1.0) == pytest.approx(1.0)
    assert D.s_diag(con=0.0, coh=0.0, deff=0.0, counter=0.0) == 0.0
    assert D.s_diag(con=0.5, coh=0.5, deff=0.5, counter=0.5) == pytest.approx(0.5)


def test_s_diag_weighting_favours_definitional_alignment():
    """q_def carries 0.4, the largest single weight."""
    only_def = D.s_diag(deff=1.0)
    only_coh = D.s_diag(coh=1.0)
    only_con = D.s_diag(con=1.0)
    only_counter = D.s_diag(counter=1.0)
    assert only_def > only_coh > only_con > only_counter
    assert only_def == pytest.approx(0.4)
    assert only_counter == pytest.approx(0.1)


def test_s_diag_clamps_out_of_range_inputs():
    assert D.s_diag(con=5.0, coh=5.0, deff=5.0, counter=5.0) == 1.0
    assert D.s_diag(con=-5.0, coh=-5.0, deff=-5.0, counter=-5.0) == 0.0


# --- S_final ---------------------------------------------------------------


def test_s_final_applies_lambda():
    assert D.s_final(0.2, 0.5) == pytest.approx(0.2 + 3.0 * 0.5)


def test_normalise_base_maps_rating_scale_to_unit_interval():
    assert D.normalise_base(10.0) == 1.0
    assert D.normalise_base(5.0) == pytest.approx(0.5)
    assert D.normalise_base(0.0) == 0.0
    assert D.normalise_base(12.0) == 1.0  # clamped
    with pytest.raises(ValueError):
        D.normalise_base(5.0, base_max=0.0)


def test_s_final_lets_diagnostics_dominate_base_quality():
    """A fluent but clinically invalid thought must lose to a valid one.

    Only holds once S_T is normalised onto [0, 1]: on the raw 1-10 scale the
    diagnostic term caps at 3 while base quality swings 9, and the ordering
    reverses.
    """
    fluent_invalid = D.s_final(D.normalise_base(9.0), diag_score=0.1)
    plain_valid = D.s_final(D.normalise_base(5.0), diag_score=0.9)
    assert plain_valid > fluent_invalid


def test_s_final_on_raw_scale_would_not_dominate():
    """Guards the reason normalisation exists."""
    fluent_invalid = D.s_final(9.0, 0.1)
    plain_valid = D.s_final(5.0, 0.9)
    assert plain_valid < fluent_invalid


# --- category validity -----------------------------------------------------


def test_category_valid_accepts_known_labels():
    assert D.category_valid({"core_beliefs": ["helpless"]}, "core_beliefs")
    assert D.category_valid({"distortions": ["labeling"]}, "distortions")
    assert D.category_valid(
        {"fine_grained": ["I am a victim"]}, "fine_grained"
    )


def test_category_valid_rejects_unknown_and_empty():
    assert not D.category_valid({"core_beliefs": ["hopeless"]}, "core_beliefs")
    assert not D.category_valid({"distortions": []}, "distortions")


MISPLACED = {"fine_grained": ["I am incompetent"], "core_beliefs": ["unlovable"]}


def test_hierarchy_is_not_gated_by_default():
    """The default reads the formula literally: q_coh demotes, not rejects."""
    assert D.ENFORCE_HIERARCHY_GATE is False
    assert D.category_valid(MISPLACED, "fine_grained") is True


def test_hierarchy_gate_rejects_when_enabled():
    assert (
        D.category_valid(MISPLACED, "fine_grained", enforce_hierarchy=True) is False
    )
    placed = {**MISPLACED, "core_beliefs": ["helpless"]}
    assert D.category_valid(placed, "fine_grained", enforce_hierarchy=True) is True


def test_gate_falls_back_to_the_flat_set_without_a_parent():
    state = {"fine_grained": ["I am incompetent"]}
    assert D.category_valid(state, "fine_grained", enforce_hierarchy=True)


def test_validator_honours_the_hierarchy_flag():
    assert D.make_validator("fine_grained", enforce_hierarchy=True)(MISPLACED) is False


def test_scoring_alone_cannot_reject_a_hierarchy_violation():
    """The property that makes the gate a live question.

    With q_coh driven to 0 but the other three components perfect, S_diag
    still reaches 0.7, above tau. As a weight alone, q_coh cannot reject a
    thought that violates the hierarchy outright.
    """
    ceiling_without_coh = D.s_diag(con=1.0, coh=0.0, deff=1.0, counter=1.0)
    assert ceiling_without_coh == pytest.approx(0.7)
    assert ceiling_without_coh > D.TAU_DIAG


def test_q_coh_still_demotes_through_s_final():
    """Losing 0.3 of S_diag costs 0.9 of S_final, enough to reorder KeepBestN."""
    base = D.normalise_base(5.0)
    coherent = D.s_final(base, D.s_diag(con=1, coh=1, deff=1, counter=1))
    violating = D.s_final(base, D.s_diag(con=1, coh=0, deff=1, counter=1))
    assert coherent - violating == pytest.approx(0.3 * D.LAMBDA)
    assert coherent > violating


# --- validator / scorer factories -----------------------------------------


def _passing_distortion_state():
    label = "all-or-nothing thinking"
    return {
        "distortions": [label],
        "rationale": (
            D.DISTORTION_DEFINITIONS[label] + " " + " ".join(D.COUNTER_QUESTIONS[label])
        ),
        "evidence_for": ["a"],
        "evidence_against": ["b"],
    }


def test_validator_passes_a_well_formed_thought():
    validate = D.make_validator("distortions")
    assert validate(_passing_distortion_state()) is True


def test_validator_rejects_below_threshold():
    validate = D.make_validator("distortions")
    weak = {
        "distortions": ["mental filter"],
        "rationale": "unrelated text",
        "evidence_for": ["a"],
        "evidence_against": [],
    }
    assert validate(weak) is False


def test_validator_rejects_invalid_category_regardless_of_score():
    validate = D.make_validator("distortions")
    state = _passing_distortion_state()
    state["distortions"] = state["distortions"] + ["not-a-distortion"]
    assert validate(state) is False


def test_validator_threshold_is_tau_diag():
    """Exactly at tau the thought passes; just below it fails."""
    D.set_similarity_fn(lambda a, b: 1.0)
    try:
        state = {
            "distortions": ["labeling"],
            "rationale": "x",
            "evidence_for": ["a", "a", "a", "a"],
            "evidence_against": ["b"],
        }
        # con=0.25 -> 0.05, coh=1.0 -> 0.3, def=1.0 -> 0.4, counter=1.0 -> 0.1
        assert D.score_thought(state, "distortions")["s_diag"] == pytest.approx(0.85)
        assert D.make_validator("distortions", tau=0.85)(state) is True
        assert D.make_validator("distortions", tau=0.86)(state) is False
    finally:
        D.set_similarity_fn(D.lexical_similarity)


def test_scorer_returns_s_final_over_normalised_base():
    from cbt.quality import base_score_max

    score = D.make_scorer("distortions", base_max=base_score_max())
    state = _passing_distortion_state()
    state["base_score"] = 4.0
    diag = D.score_thought(state, "distortions")["s_diag"]
    assert score(state) == pytest.approx(4.0 / base_score_max() + 3.0 * diag)


def test_scorer_defaults_to_the_bare_rating_scale():
    """The caller passes the ceiling that matches the score it produced."""
    score = D.make_scorer("distortions")
    state = _passing_distortion_state()
    state["base_score"] = 4.0
    diag = D.score_thought(state, "distortions")["s_diag"]
    assert score(state) == pytest.approx(4.0 / D.BASE_SCORE_MAX + 3.0 * diag)


def test_scorer_ceiling_accounts_for_the_self_consistency_term():
    """base_score carries csc + beta * fsc, so the ceiling is above 10."""
    from cbt.quality import BETA, base_score_max

    assert base_score_max() == pytest.approx(10.0 + BETA * 10.0)
    score = D.make_scorer("distortions", base_max=base_score_max())
    state = _passing_distortion_state()
    state["base_score"] = base_score_max()
    diag = D.score_thought(state, "distortions")["s_diag"]
    assert score(state) == pytest.approx(1.0 + 3.0 * diag)


def test_scorer_ranks_diagnostics_above_base_quality():
    score = D.make_scorer("distortions")
    valid = _passing_distortion_state()
    valid["base_score"] = 5.0

    fluent = {
        "distortions": ["mental filter"],
        "rationale": "an eloquent but clinically unmoored paragraph",
        "evidence_for": ["a"],
        "evidence_against": [],
        "base_score": 10.0,
    }
    assert score(valid) > score(fluent)


# --- score_thought ---------------------------------------------------------


def test_score_thought_neutralises_only_undefined_components():
    """At the root there is no parent to check and no counter-question bank."""
    out = D.score_thought(
        {"core_beliefs": ["helpless"], "rationale": "r"}, "core_beliefs"
    )
    assert out["q_coh"] == 1.0
    assert out["q_counter"] == 1.0


def test_score_thought_evaluates_definitions_at_every_level():
    """q_def uses the level's own reference text, not a fixed 1.0."""
    aligned = D.score_thought(
        {
            "core_beliefs": ["helpless"],
            "rationale": D.CORE_BELIEF_DEFINITIONS["helpless"],
        },
        "core_beliefs",
    )
    unrelated = D.score_thought(
        {"core_beliefs": ["helpless"], "rationale": "the weather is pleasant"},
        "core_beliefs",
    )
    assert aligned["q_def"] > unrelated["q_def"]
    assert unrelated["q_def"] == 0.0


def test_upper_levels_can_fail_validation():
    """Holding q_def at 1.0 would floor S_diag at 0.8 and make tau unreachable."""
    weak = {
        "core_beliefs": ["helpless"],
        "rationale": "the weather is pleasant",
        "evidence_for": ["a"],
        "evidence_against": [],
    }
    assert D.score_thought(weak, "core_beliefs")["s_diag"] < D.TAU_DIAG
    assert D.make_validator("core_beliefs")(weak) is False


def test_score_thought_applies_coherence_at_fine_grained_level():
    out = D.score_thought(
        {
            "fine_grained": ["I am incompetent", "I am unlovable"],
            "core_beliefs": ["helpless"],
            "rationale": "r",
        },
        "fine_grained",
    )
    assert out["q_coh"] == pytest.approx(0.5)


def test_score_thought_rejects_unknown_level():
    with pytest.raises(ValueError):
        D.score_thought({}, "not_a_level")
