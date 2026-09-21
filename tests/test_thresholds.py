"""
The intermediate and final quality thresholds.

thr1 and thr2 sit on the 1-10 quality scale and decide which recovery a
failing thought receives. tau sits on S_diag in [0, 1] and decides whether a
thought is clinically valid at all. Both gates apply.
"""

import pytest

from cbt import diagnostic as D
from cbt.pipeline import (
    THR1,
    THR2,
    STAGES,
    CBTParser,
    CBTPrompter,
    build_graph,
    level_operations,
)
from graph_of_thoughts.controller import Controller
from tests.scripted_lm import ScriptedLM, payload, thought
from tests.test_pipeline import CASE, DISTORTION, GAP, aligned_rationale, good


def test_paper_threshold_values():
    assert (THR1, THR2) == (3.0, 5.0)
    assert THR1 < THR2, "the weaker candidate must get the more drastic action"


# --- the validator gate ----------------------------------------------------


def _valid_state(base_score):
    label = DISTORTION
    return {
        "distortions": [label],
        "rationale": aligned_rationale("distortions", label),
        "evidence_for": ["a"],
        "evidence_against": ["b"],
        "base_score": base_score,
    }


def test_thr2_rejects_a_clinically_valid_but_low_quality_thought():
    """Passing tau is not enough on its own once thr2 is set."""
    state = _valid_state(base_score=4.0)
    assert D.score_thought(state, "distortions")["s_diag"] >= D.TAU_DIAG
    assert D.make_validator("distortions")(state) is True
    assert D.make_validator("distortions", thr2=THR2)(state) is False


def test_thr2_accepts_at_the_threshold():
    validate = D.make_validator("distortions", thr2=THR2)
    assert validate(_valid_state(THR2)) is True
    assert validate(_valid_state(THR2 - 0.01)) is False


def test_thr2_does_not_rescue_a_clinically_invalid_thought():
    """A high quality score cannot buy past tau or category validity."""
    state = _valid_state(base_score=10.0)
    state["rationale"] = "entirely unrelated prose"
    state["evidence_against"] = []
    assert D.make_validator("distortions", thr2=THR2)(state) is False


def test_thresholds_are_off_unless_asked_for():
    assert D.make_validator("distortions")(_valid_state(0.0)) is True


# --- the recovery routing --------------------------------------------------


def _run(script, **op_kwargs):
    graph = build_graph()
    if op_kwargs:
        for op in level_operations(graph).values():
            for key, value in op_kwargs.items():
                setattr(op, key, value)
    lm = ScriptedLM(script)
    Controller(lm, graph, CBTPrompter(), CBTParser(), dict(CASE)).run()
    return graph, lm


def threadbare(level, label):
    """Valid category, but nothing else: no quotes, no evidence, no prose."""
    return thought(
        level,
        [label],
        "short",
        evidence_for=(),
        evidence_against=(),
        quotes=(),
    )


def base_script(overrides=None):
    script = {
        ("gap", "generate"): [GAP],
        ("core_beliefs", "generate"): [good("core_beliefs", "unlovable")],
        ("fine_grained", "generate"): [
            good("fine_grained", "i am unlovable", core_beliefs=["unlovable"])
        ],
        ("distortions", "generate"): [good("distortions", DISTORTION)],
    }
    script.update(overrides or {})
    return script


@pytest.mark.parametrize(
    "level,floor", [("distortions", 6.0), ("core_beliefs", 8.0), ("fine_grained", 9.0)]
)
def test_computed_quality_has_a_floor_above_both_thresholds(level, floor):
    """Why the paper's threshold values cannot fire against a computed S_T.

    Accuracy carries weight 4 or 5 and is 1.0 whenever the labels are in the
    permissible set; conciseness is 1.0 for any sensible count. A
    category-valid thought therefore banks ``floor`` before evidence or
    clarity contribute anything, which is above thr1 and thr2 both. Values
    of 2 to 6 are only reachable when S_T is an LLM's 1-10 rating.
    """
    from cbt.quality import base_quality

    barest = {
        "core_beliefs": ["helpless"],
        "fine_grained": ["i am helpless"],
        "distortions": ["labeling"],
        "rationale": "",
    }
    assert base_quality(barest, level) == pytest.approx(floor)
    assert floor > THR2 > THR1


def test_thresholds_fire_against_a_rater_scale_score():
    """Wired and working; it is the computed S_T that never gets that low."""
    graph, lm = _run(
        base_script(
            {
                ("distortions", "generate"): [threadbare("distortions", DISTORTION)],
                ("distortions", "backtrack"): [good("distortions", DISTORTION)],
            }
        ),
        thr1=9.0,  # stands in for a rater that scored the thought poorly
    )
    op = level_operations(graph)["distortions"]
    assert op.n_regenerated == STAGES["distortions"]["k1"]
    assert op.n_refinements == 0
    assert lm.count("distortions", "improve") == 0
    assert op.n_backtracks == STAGES["distortions"]["k1"]
    assert all(t.valid for t in op.get_thoughts())


def test_between_the_thresholds_is_refined_not_re_derived():
    """Weak on clinical alignment but not on quality: patch in place."""
    graph, lm = _run(
        base_script(
            {
                ("distortions", "generate"): [
                    thought(
                        "distortions",
                        [DISTORTION],
                        "a fluent paragraph about unrelated matters " * 3,
                        evidence_against=(),
                        quotes=(),
                    )
                ],
                ("distortions", "improve"): [good("distortions", DISTORTION)],
            }
        )
    )
    op = level_operations(graph)["distortions"]
    assert op.n_refinements > 0
    assert op.n_regenerated == 0
    assert lm.count("distortions", "backtrack") == 0


def test_disabling_thr1_restores_plain_refinement():
    graph, lm = _run(
        base_script(
            {
                ("distortions", "generate"): [threadbare("distortions", DISTORTION)],
                ("distortions", "improve"): [good("distortions", DISTORTION)],
            }
        ),
        thr1=None,
    )
    op = level_operations(graph)["distortions"]
    assert op.n_regenerated == 0
    assert op.n_refinements > 0


def test_a_clean_run_trips_neither_threshold():
    graph, lm = _run(base_script())
    for level, op in level_operations(graph).items():
        assert op.n_regenerated == 0, level
        assert op.n_refinements == 0, level
        assert all(t.valid for t in op.get_thoughts()), level
