"""
Tier 1: the reasoning graph, executed end to end against a scripted model.

Each test scripts the model's answers so one path through the graph is
forced, then asserts on what the graph did: how many thoughts survived each
selection, how many refinements fired, whether backtracking engaged, and
whether a broken thought was carried forward.

No API key, no network, no dataset.
"""

import pytest

from graph_of_thoughts.controller import Controller

from cbt import diagnostic as D
from cbt.pipeline import (
    STAGES,
    CBTParser,
    CBTPrompter,
    ValidateImproveBacktrack,
    build_graph,
    level_operations,
    stage_of,
)
from tests.scripted_lm import ScriptedLM, payload, thought

CASE = {"ori_text": "text", "thoughts": "thoughts", "situation": "situation"}

DISTORTION = "all-or-nothing thinking"


def aligned_rationale(level: str, label: str) -> str:
    """Rationale that clears the diagnostic threshold for ``label``."""
    text = D.DEFINITIONS_BY_LEVEL[level][label]
    if level == "distortions":
        text += " " + " ".join(D.COUNTER_QUESTIONS[label])
    return text


GAP = payload(objective_fact="he was not invited", subjective_thought="nobody wants me")


def good(level: str, label: str, **extra) -> str:
    return thought(
        level, [label], aligned_rationale(level, label) + " " + "detail " * 20, **extra
    )


def weak(level: str, label: str, **extra) -> str:
    """Valid category, rationale unmoored from the clinical definition."""
    return thought(
        level,
        [label],
        "a fluent paragraph about entirely unrelated matters " * 3,
        evidence_against=(),
        quotes=(),
        **extra,
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


def run(script):
    graph = build_graph()
    lm = ScriptedLM(script)
    Controller(lm, graph, CBTPrompter(), CBTParser(), dict(CASE)).run()
    return graph, lm


# --- prompt tagging --------------------------------------------------------


def test_every_prompt_carries_a_routable_stage_tag():
    _, lm = run(base_script())
    assert lm.prompts
    for prompt in lm.prompts:
        stage, op = stage_of(prompt)
        assert stage is not None and op is not None, prompt[:80]


# --- topology --------------------------------------------------------------


def test_graph_is_acyclic():
    """A literal back edge would make topological execution ill-defined."""
    graph = build_graph()
    seen, stack = set(), []

    def visit(op):
        if op in stack:
            pytest.fail("cycle through the operation graph")
        if op in seen:
            return
        stack.append(op)
        for succ in op.successors:
            visit(succ)
        stack.pop()
        seen.add(op)

    for root in graph.roots:
        visit(root)
    assert len(seen) == len(graph.operations)


def test_selection_widths_match_the_paper():
    graph, _ = run(base_script())
    ops = level_operations(graph)

    # k1: how many thoughts reached validation at each level
    assert len(ops["core_beliefs"].get_thoughts()) == STAGES["core_beliefs"]["k1"] == 2
    assert len(ops["fine_grained"].get_thoughts()) == STAGES["fine_grained"]["k1"] == 3
    assert len(ops["distortions"].get_thoughts()) == STAGES["distortions"]["k1"] == 5


def test_distortion_stage_keeps_all_candidates():
    """k1 == N at the distortion level: nothing is pruned before validation."""
    assert STAGES["distortions"]["k1"] == STAGES["distortions"]["n"] == 5


def test_final_output_width():
    graph, _ = run(base_script())
    assert len(graph.leaves[0].get_thoughts()) == STAGES["distortions"]["k2"] == 3


def test_hierarchy_conditions_each_level_on_the_one_above():
    """Core collapses to one consensus, which then fans out downstream."""
    _, lm = run(base_script())
    # k2_core == 1, so fine-grained is prompted once.
    assert lm.count("fine_grained", "generate") == 1
    # k2_fine == 3, so distortions are prompted once per surviving parent.
    assert lm.count("distortions", "generate") == STAGES["fine_grained"]["k2"] == 3


def test_final_thoughts_carry_the_full_hierarchy():
    graph, _ = run(base_script())
    for t in graph.leaves[0].get_thoughts():
        assert t.state["core_beliefs"] == ["unlovable"]
        assert t.state["fine_grained"] == ["i am unlovable"]
        assert t.state["distortions"] == [DISTORTION]
        assert t.state["objective_fact"]


# --- happy path ------------------------------------------------------------


def test_clean_run_triggers_no_recovery():
    graph, lm = run(base_script())
    for level, op in level_operations(graph).items():
        assert op.n_refinements == 0, level
        assert op.n_backtracks == 0, level
        assert op.n_failed == 0, level
        assert all(t.valid for t in op.get_thoughts()), level
    assert lm.count(op="improve") == 0
    assert lm.count(op="backtrack") == 0


def test_clean_run_call_count():
    """One gap call, one core call, one fine call, three distortion calls."""
    _, lm = run(base_script())
    assert len(lm.calls) == 6


# --- refinement ------------------------------------------------------------


def test_refinement_recovers_a_weak_thought():
    graph, lm = run(
        base_script(
            {
                ("distortions", "generate"): [weak("distortions", DISTORTION)],
                ("distortions", "improve"): [good("distortions", DISTORTION)],
            }
        )
    )
    op = level_operations(graph)["distortions"]
    assert op.n_refinements == STAGES["distortions"]["k1"] == 5
    assert op.n_recovered_first_try == 5
    assert op.n_backtracks == 0
    assert op.n_failed == 0
    assert all(t.valid for t in op.get_thoughts())


def test_refinement_feedback_names_the_failing_dimension():
    _, lm = run(
        base_script(
            {
                ("distortions", "generate"): [weak("distortions", DISTORTION)],
                ("distortions", "improve"): [good("distortions", DISTORTION)],
            }
        )
    )
    improve_prompts = [
        p for p in lm.prompts if stage_of(p) == ("distortions", "improve")
    ]
    assert improve_prompts
    joined = " ".join(improve_prompts).lower()
    assert "definitional alignment" in joined
    assert "contrastive balance" in joined


def test_refinement_is_capped_at_num_tries():
    """Two attempts, then the thought stops being refined."""
    graph, lm = run(
        base_script(
            {
                ("distortions", "generate"): [weak("distortions", DISTORTION)],
                ("distortions", "improve"): [weak("distortions", DISTORTION)],
                ("distortions", "backtrack"): [good("distortions", DISTORTION)],
            }
        )
    )
    op = level_operations(graph)["distortions"]
    n = STAGES["distortions"]["k1"]
    assert op.num_tries == 2
    assert op.n_refinements == 2 * n
    assert lm.count("distortions", "improve") == 2 * n


# --- backtracking ----------------------------------------------------------


def test_backtracking_engages_after_refinement_is_exhausted():
    graph, lm = run(
        base_script(
            {
                ("distortions", "generate"): [weak("distortions", DISTORTION)],
                ("distortions", "improve"): [weak("distortions", DISTORTION)],
                ("distortions", "backtrack"): [good("distortions", DISTORTION)],
            }
        )
    )
    op = level_operations(graph)["distortions"]
    n = STAGES["distortions"]["k1"]
    assert op.n_backtracks == n
    assert op.n_failed == 0
    assert all(t.valid for t in op.get_thoughts())
    assert all(t.state.get("backtracked") for t in op.get_thoughts())


def test_backtracking_is_not_reached_when_refinement_succeeds():
    graph, lm = run(
        base_script(
            {
                ("distortions", "generate"): [weak("distortions", DISTORTION)],
                ("distortions", "improve"): [good("distortions", DISTORTION)],
            }
        )
    )
    assert lm.count("distortions", "backtrack") == 0
    assert level_operations(graph)["distortions"].n_backtracks == 0


def test_exhausted_recovery_is_recorded_not_hidden():
    graph, _ = run(
        base_script(
            {
                ("distortions", "generate"): [weak("distortions", DISTORTION)],
                ("distortions", "improve"): [weak("distortions", DISTORTION)],
                ("distortions", "backtrack"): [weak("distortions", DISTORTION)],
            }
        )
    )
    op = level_operations(graph)["distortions"]
    assert op.n_failed == STAGES["distortions"]["k1"]
    assert not any(t.valid for t in op.get_thoughts())


# --- hierarchical consistency ---------------------------------------------


MISPLACED_SCRIPT = {
    ("core_beliefs", "generate"): [good("core_beliefs", "unlovable")],
    # "i am incompetent" sits under helpless, not unlovable.
    ("fine_grained", "generate"): [good("fine_grained", "i am incompetent")],
    ("fine_grained", "improve"): [good("fine_grained", "i am unlovable")],
}


def test_violation_is_demoted_but_not_rejected_by_default():
    """Default reading: q_coh lowers the score without failing validation."""
    graph, lm = run(base_script(MISPLACED_SCRIPT))
    op = level_operations(graph)["fine_grained"]
    assert op.n_refinements == 0
    assert all(t.valid for t in op.get_thoughts())
    # The penalty still lands: S_diag loses the full 0.3 coherence weight.
    for t in op.get_thoughts():
        assert D.score_thought(t.state, "fine_grained")["q_coh"] == 0.0


def test_violation_is_rejected_and_refined_when_gated(monkeypatch):
    monkeypatch.setattr(D, "ENFORCE_HIERARCHY_GATE", True)
    graph, lm = run(base_script(MISPLACED_SCRIPT))
    op = level_operations(graph)["fine_grained"]
    assert op.n_refinements > 0
    assert all(t.valid for t in op.get_thoughts())
    assert all(
        t.state["fine_grained"] == ["i am unlovable"] for t in op.get_thoughts()
    )


def test_consistency_violation_is_named_in_the_feedback(monkeypatch):
    monkeypatch.setattr(D, "ENFORCE_HIERARCHY_GATE", True)
    _, lm = run(base_script(MISPLACED_SCRIPT))
    prompts = [p for p in lm.prompts if stage_of(p) == ("fine_grained", "improve")]
    assert prompts
    assert "hierarchical consistency" in " ".join(prompts).lower()


# --- ranking ---------------------------------------------------------------


def test_final_selection_ranks_by_s_final():
    graph, _ = run(base_script())
    keep2 = graph.leaves[0]
    scores = [t.score for t in keep2.get_thoughts()]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 + D.LAMBDA for s in scores)


def test_invalid_categories_never_reach_the_output():
    graph, _ = run(
        base_script(
            {
                ("distortions", "generate"): [
                    thought("distortions", ["not-a-distortion"], "x" * 40)
                ],
                ("distortions", "improve"): [good("distortions", DISTORTION)],
            }
        )
    )
    for t in graph.leaves[0].get_thoughts():
        assert D.category_valid(t.state, "distortions")


# --- scripted model contract ----------------------------------------------


def test_scripted_lm_needs_no_config_or_key():
    lm = ScriptedLM({})
    assert lm.config == {}
    assert lm.cost == 0.0


def test_scripted_lm_rejects_an_unscripted_stage():
    with pytest.raises(KeyError):
        run({("gap", "generate"): [GAP]})
