"""The GoTCHA reasoning graph.

The three cognitive levels wired onto Graph of Thoughts operations with the
paper's branching parameters, plus its two-stage recovery: refinement with
dimension-specific feedback, then backtracking to the parent level.

Only ``Generate`` and the improve/backtrack steps of
``ValidateImproveBacktrack`` query a model. Scoring, validation and selection
are pure functions, which lets the whole graph run against a scripted stub.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List

from graph_of_thoughts.operations import (
    Generate,
    GraphOfOperations,
    KeepBestN,
    Score,
    ValidateAndImprove,
)
from graph_of_thoughts.operations.thought import Thought
from graph_of_thoughts.parser.parser import Parser
from graph_of_thoughts.prompter.prompter import Prompter

from cbt.diagnostic import (
    FINE_GRAINED,
    TAU_DIAG,
    VOCABULARY,
    make_scorer,
    make_validator,
    score_thought,
)
from cbt.quality import base_score_max, make_base_scorer

# --- Paper branching parameters -------------------------------------------
#                       N_T   k1_T  k2_T
STAGES: Dict[str, Dict[str, int]] = {
    "core_beliefs": {"n": 3, "k1": 2, "k2": 1},
    "fine_grained": {"n": 5, "k1": 3, "k2": 3},
    "distortions": {"n": 5, "k1": 5, "k2": 3},  # k1 = N: distortion keep-all
}

MAX_BACKTRACKS = 1
NUM_TRIES = 2

# Quality thresholds on the 1-10 scale, routing a failing thought to the right
# recovery. Since thr1 < thr2, the weaker the candidate the more drastic the
# action:
#
#     base_score >= thr2          accept
#     thr1 <= base_score < thr2   refine in place
#     base_score <  thr1          re-derive from the case
#
# The paper sweeps five combinations and reports 3.0 / 5.0 as most stable;
# 2.0 / 4.0 admits false positives, 4.0 / 6.0 costs recall. Both sit on a
# different axis from tau, which gates S_diag in [0, 1].
#
# Neither fires against the computed S_T in cbt.quality. Accuracy weighs 4 or
# 5 and is 1.0 for any thought whose labels are permissible, so a
# category-valid thought already banks 6.0 at distortions, 8.0 at core beliefs
# and 9.0 at fine-grained. Values of 2 to 6 need S_T to be an LLM's rating.
# The original implementation used one, and these were calibrated against it,
# so they stay wired for that case.
THR1 = 3.0
THR2 = 5.0

STAGE_TAG = re.compile(r"#\s*stage:\s*(\w+)\s*\|\s*op:\s*(\w+)")


def stage_of(prompt: str):
    """Recover ``(stage, op)`` from a prompt's machine-readable tag."""
    match = STAGE_TAG.search(prompt or "")
    return (match.group(1), match.group(2)) if match else (None, None)


# --- Prompter --------------------------------------------------------------


class CBTPrompter(Prompter):
    """Stage-aware prompting for the three cognitive levels.

    Every prompt opens with a ``# stage: <level> | op: <operation>`` tag, so
    tests can route responses deterministically and logs say which part of the
    graph issued a call.
    """

    def __init__(self, level: str = "core_beliefs") -> None:
        self.level = level

    def _tag(self, op: str) -> str:
        return f"# stage: {self.level} | op: {op}"

    def _vocabulary(self) -> str:
        separator = "; " if self.level == "fine_grained" else ", "
        return separator.join(VOCABULARY[self.level])

    def _context(self, kwargs: Dict) -> str:
        lines = []
        for key in ("ori_text", "thoughts", "situation"):
            if kwargs.get(key):
                lines.append(f"{key}: {kwargs[key]}")
        for key in ("objective_fact", "subjective_thought"):
            if kwargs.get(key):
                lines.append(f"{key}: {kwargs[key]}")
        for key in ("core_beliefs", "fine_grained"):
            if kwargs.get(key) and key != self.level:
                lines.append(f"{key}: {kwargs[key]}")
        return "\n".join(lines)

    def generate_prompt(self, num_branches: int, **kwargs) -> str:
        if self.level == "gap":
            return (
                f"{self._tag('generate')}\n"
                "You are a CBT expert. Separate what is verifiable in the case "
                "from how the speaker interprets it.\n\n"
                f"{self._context(kwargs)}\n\n"
                'Respond with JSON containing the keys "objective_fact" and '
                '"subjective_thought".'
            )
        return (
            f"{self._tag('generate')}\n"
            f"You are a CBT expert. Identify the {self.level} in the case below.\n"
            f"Permissible categories: {self._vocabulary()}\n\n"
            f"{self._context(kwargs)}\n\n"
            "Respond with JSON containing the keys "
            f'"{self.level}", "rationale", "evidence_for", "evidence_against", '
            '"quotes".'
        )

    def improve_prompt(self, **kwargs) -> str:
        feedback = kwargs.get("feedback", "")
        return (
            f"{self._tag('improve')}\n"
            f"The previous {self.level} analysis failed diagnostic validation.\n"
            f"{feedback}\n\n"
            f"Previous answer: {kwargs.get(self.level)}\n"
            f"Previous rationale: {kwargs.get('rationale', '')}\n\n"
            "Revise the analysis to address the feedback. Respond with the "
            "same JSON schema."
        )

    def backtrack_prompt(self, **kwargs) -> str:
        return (
            f"{self._tag('backtrack')}\n"
            f"Refinement did not recover a valid {self.level} analysis. "
            "Re-derive it from the case, correcting the parent-level "
            "assumptions.\n"
            f"{kwargs.get('feedback', '')}\n\n"
            f"{self._context(kwargs)}\n\n"
            "Respond with the same JSON schema."
        )

    def validation_prompt(self, **kwargs) -> str:
        return f"{self._tag('validate')}\nValidate the {self.level} analysis."

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        return f"{self._tag('score')}\nScore the {self.level} analyses."

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        return f"{self._tag('aggregate')}\nAggregate the {self.level} analyses."


# --- Parser ----------------------------------------------------------------

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

_LIST_KEYS = (
    "core_beliefs",
    "fine_grained",
    "distortions",
    "evidence_for",
    "evidence_against",
    "quotes",
)


class CBTParser(Parser):
    """Parses the JSON payload each stage is asked to emit."""

    def __init__(self, level: str = "core_beliefs") -> None:
        self.level = level

    @staticmethod
    def _extract(text: str) -> Dict:
        match = _JSON_BLOCK.search(str(text or ""))
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        for key in _LIST_KEYS:
            if key in payload and not isinstance(payload[key], list):
                payload[key] = [payload[key]]
        return payload

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        return [self._extract(text) for text in texts]

    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        return self._extract(texts[0]) if texts else {}

    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        return bool(texts) and "invalid" not in str(texts[0]).lower()

    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        scores = []
        for text in texts:
            found = re.search(r"\d+(?:\.\d+)?", str(text))
            scores.append(float(found.group(0)) if found else 0.0)
        return scores

    def parse_aggregation_answer(self, states: List[Dict], texts: List[str]) -> Dict:
        return self._extract(texts[0]) if texts else {}


# --- Level routing ---------------------------------------------------------
#
# A Controller carries one prompter and one parser for the whole graph, but
# each level needs its own vocabulary and output key, so operations announce
# their level before running. Three controllers over three graphs would break
# the hierarchy: each level reads the surviving thoughts of the one above.


class _LevelBound:
    level = "core_beliefs"

    def _bind(self, prompter, parser) -> None:
        if hasattr(prompter, "level"):
            prompter.level = self.level
        if hasattr(parser, "level"):
            parser.level = self.level


class CBTGenerate(Generate, _LevelBound):
    """``Generate`` bound to one cognitive level."""

    def __init__(
        self,
        level: str,
        num_branches_prompt: int = 1,
        num_branches_response: int = 1,
    ) -> None:
        super().__init__(num_branches_prompt, num_branches_response)
        self.level = level

    def _execute(self, lm, prompter, parser, **kwargs) -> None:
        self._bind(prompter, parser)
        super()._execute(lm, prompter, parser, **kwargs)


# --- Two-stage recovery ----------------------------------------------------


def diagnostic_feedback(state: Dict, level: str, tau: float = TAU_DIAG) -> str:
    """Name the dimensions that fell short, so refinement is targeted.

    The "dimension-specific feedback" the paper's first recovery stage relies
    on. A generic retry prompt gives the model nothing to act on.
    """
    scores = score_thought(state, level)
    complaints = []
    if scores["q_con"] < 0.5:
        complaints.append(
            "Contrastive balance is weak: gather evidence that contradicts the "
            "reading, not only evidence that supports it."
        )
    if scores["q_coh"] < 1.0:
        complaints.append(
            "Hierarchical consistency is violated: at least one belief does not "
            "belong to a subcategory of the inferred core beliefs."
        )
    if scores["q_def"] < 0.5:
        complaints.append(
            "Definitional alignment is weak: tie the rationale to the clinical "
            "definition of each category you assign."
        )
    if scores["q_counter"] < 0.5:
        complaints.append(
            "Counter-questioning is incomplete: address the diagnostic "
            "questions associated with each category."
        )
    if not complaints:
        complaints.append(
            f"Diagnostic score {scores['s_diag']:.2f} is below the {tau} threshold."
        )
    return " ".join(complaints)


class ValidateImproveBacktrack(ValidateAndImprove, _LevelBound):
    """Validation with refinement, then backtracking to the parent level.

    Adds the paper's second recovery stage to the upstream operation. A
    thought still invalid after ``num_tries`` refinements is re-derived from
    the case with corrective guidance instead of being carried forward broken.

    That re-derivation is one more prompt on this node, which keeps the graph
    acyclic. A literal back edge to the parent operation would leave the
    controller's topological execution undefined.

    The counters ``n_refinements``, ``n_backtracks``, ``n_regenerated``,
    ``n_recovered_first_try`` and ``n_failed`` record what recovery did.
    """

    def __init__(
        self,
        level: str = "core_beliefs",
        num_tries: int = NUM_TRIES,
        max_backtracks: int = MAX_BACKTRACKS,
        tau: float = TAU_DIAG,
        enforce_hierarchy: bool | None = None,
        thr1: float = THR1,
        thr2: float = THR2,
    ) -> None:
        super().__init__(
            num_samples=1,
            improve=True,
            num_tries=num_tries,
            validate_function=make_validator(level, tau, enforce_hierarchy, thr2),
        )
        self.level = level
        self.max_backtracks = max_backtracks
        self.tau = tau
        self.thr1 = thr1
        self.thr2 = thr2
        self.n_regenerated = 0
        self.n_refinements = 0
        self.n_backtracks = 0
        self.n_recovered_first_try = 0
        self.n_failed = 0

    def _query(self, lm, prompt: str) -> List[str]:
        return lm.get_response_texts(lm.query(prompt, num_responses=1))

    def _below_intermediate(self, state: Dict) -> bool:
        if self.thr1 is None:
            return False
        return float(state.get("base_score", 0.0) or 0.0) < self.thr1

    def _execute(self, lm, prompter, parser, **kwargs) -> None:
        self._bind(prompter, parser)
        previous = self.get_previous_thoughts()
        assert self.predecessors, "ValidateImproveBacktrack needs a predecessor"

        for thought in previous:
            chain = []
            current = Thought.from_thought(thought)
            attempts = 0

            while True:
                current.valid = self.validate_function(current.state)
                chain.append(current)
                if current.valid or attempts >= self.num_tries:
                    break
                if self._below_intermediate(current.state):
                    # Too weak to patch: re-derive rather than refine.
                    self.n_regenerated += 1
                    break

                feedback = diagnostic_feedback(current.state, self.level, self.tau)
                prompt = prompter.improve_prompt(
                    **{**current.state, "feedback": feedback}
                )
                responses = self._query(lm, prompt)
                update = parser.parse_improve_answer(current.state, responses)
                current = Thought({**current.state, **update, "feedback": feedback})
                attempts += 1
                self.n_refinements += 1

            if current.valid:
                if attempts == 1:
                    self.n_recovered_first_try += 1
            else:
                for _ in range(self.max_backtracks):
                    feedback = diagnostic_feedback(current.state, self.level, self.tau)
                    responses = self._query(
                        lm,
                        prompter.backtrack_prompt(
                            **{**current.state, "feedback": feedback}
                        ),
                    )
                    update = parser.parse_improve_answer(current.state, responses)
                    current = Thought(
                        {**current.state, **update, "backtracked": True}
                    )
                    self.n_backtracks += 1
                    current.valid = self.validate_function(current.state)
                    chain.append(current)
                    if current.valid:
                        break
                if not current.valid:
                    self.n_failed += 1

            self.thoughts.append(chain)

        self.logger.info(
            "%s: %d/%d valid, %d refinements, %d backtracks",
            self.level,
            sum(1 for c in self.thoughts if c[-1].valid),
            len(previous),
            self.n_refinements,
            self.n_backtracks,
        )


# --- Graph -----------------------------------------------------------------


def build_level(graph: GraphOfOperations, level: str) -> Dict:
    """Append one cognitive level's six-stage sequence to ``graph``.

    Generate -> Score -> KeepBestN(k1) -> Validate & Improve
             -> Score_final -> KeepBestN(k2)
    """
    params = STAGES[level]

    ops = {
        "generate": CBTGenerate(level, 1, params["n"]),
        # The self-consistency term is a property of the candidate set, so
        # this Score sees all of them at once.
        "score": Score(
            combined_scoring=True, scoring_function=make_base_scorer(level)
        ),
        "keep1": KeepBestN(params["k1"], True),
        "validate": ValidateImproveBacktrack(level=level),
        "score_final": Score(
            combined_scoring=False,
            scoring_function=make_scorer(level, base_max=base_score_max()),
        ),
        "keep2": KeepBestN(params["k2"], True),
    }
    for op in ops.values():
        graph.append_operation(op)
    return ops


def build_graph() -> GraphOfOperations:
    """The full GoTCHA operation graph.

    Gap extraction, then the three cognitive levels in sequence. Each level
    conditions on the surviving thoughts of the one above it; without that the
    hierarchy would be three independent classifiers.
    """
    graph = GraphOfOperations()
    graph.append_operation(CBTGenerate("gap", 1, 1))

    for level in ("core_beliefs", "fine_grained", "distortions"):
        build_level(graph, level)

    return graph


def level_operations(graph: GraphOfOperations) -> Dict[str, ValidateImproveBacktrack]:
    """Index the validation node of each level, for inspecting counters."""
    return {
        op.level: op
        for op in graph.operations
        if isinstance(op, ValidateImproveBacktrack)
    }
