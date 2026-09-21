"""A language model stub, for exercising the graph without an API key.

Answers every stage with a well-formed, diagnostically valid response, so
``--dry-run`` walks the whole pipeline and reports real selection widths and
recovery counters. It says nothing about how a real model performs.
"""

from __future__ import annotations

import json
import logging
from typing import List

from graph_of_thoughts.language_models import AbstractLanguageModel

from cbt.diagnostic import COUNTER_QUESTIONS, DEFINITIONS_BY_LEVEL, FINE_GRAINED
from cbt.pipeline import stage_of


def _answer(level: str, label: str, **extra) -> str:
    rationale = DEFINITIONS_BY_LEVEL[level].get(label, label)
    if level == "distortions":
        rationale += " " + " ".join(COUNTER_QUESTIONS.get(label, ()))
    return json.dumps(
        {
            level: [label],
            "rationale": rationale + " " + "supporting detail " * 8,
            "evidence_for": ["a supporting observation"],
            "evidence_against": ["a contradicting observation"],
            "quotes": ["a direct quote", "another direct quote"],
            **extra,
        }
    )


class StubLM(AbstractLanguageModel):
    """Answers each stage with a valid response for a fixed label."""

    def __init__(
        self,
        core: str = "helpless",
        distortion: str = "all-or-nothing thinking",
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_name = "stub"
        self.cache = False
        self.config = {}
        self.prompt_tokens = self.completion_tokens = 0
        self.cost = 0.0
        self.calls: List[tuple] = []
        self.core = core
        self.fine = FINE_GRAINED[core][0]
        self.distortion = distortion

    def load_config(self, path: str = "") -> None:
        """No config file, and so no API key, is required."""
        self.config = {}

    def query(self, query: str, num_responses: int = 1) -> List[str]:
        level, op = stage_of(query)
        self.calls.append((level, op))

        if level == "gap":
            body = json.dumps(
                {
                    "objective_fact": "an event that can be checked",
                    "subjective_thought": "how the speaker reads that event",
                }
            )
        elif level == "core_beliefs":
            body = _answer("core_beliefs", self.core)
        elif level == "fine_grained":
            body = _answer("fine_grained", self.fine, core_beliefs=[self.core])
        elif level == "distortions":
            body = _answer("distortions", self.distortion)
        else:
            body = "{}"

        return [body for _ in range(num_responses)]

    def get_response_texts(self, query_responses) -> List[str]:
        if isinstance(query_responses, str):
            return [query_responses]
        return list(query_responses)
