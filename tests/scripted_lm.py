"""
A language model stub that replays scripted responses.

The pipeline only queries a model in two places -- ``Generate`` and the
improve/backtrack steps of ``ValidateImproveBacktrack`` -- so replacing those
answers with a script exercises the entire graph, including the recovery
paths, at no API cost and with no randomness.

Responses are keyed by ``(stage, op)``, which the prompter writes into every
prompt as a ``# stage: <level> | op: <operation>`` tag.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

from graph_of_thoughts.language_models import AbstractLanguageModel

from cbt.pipeline import stage_of


def payload(**fields) -> str:
    """Render a response body the CBTParser will accept."""
    return json.dumps(fields)


def thought(
    level: str,
    labels,
    rationale: str = "",
    evidence_for=("a supporting observation",),
    evidence_against=("a contradicting observation",),
    quotes=("a direct quote", "another direct quote"),
    **extra,
) -> str:
    """A well-formed response for ``level`` carrying ``labels``."""
    return payload(
        **{
            level: list(labels),
            "rationale": rationale,
            "evidence_for": list(evidence_for),
            "evidence_against": list(evidence_against),
            "quotes": list(quotes),
            **extra,
        }
    )


class ScriptedLM(AbstractLanguageModel):
    """Replays canned responses and records every call.

    ``script`` maps ``(stage, op)`` to a list of response strings, consumed in
    order. When a key's list runs out the last entry repeats, so a script only
    has to spell out the responses a test actually cares about.
    """

    def __init__(self, script: Dict[Tuple[str, str], List[str]]) -> None:
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: List[Tuple[str, str]] = []
        self.prompts: List[str] = []
        self.model_name = "scripted"
        self.cache = False
        self.config = {}
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost = 0.0
        import logging

        self.logger = logging.getLogger(self.__class__.__name__)

    def load_config(self, path: str = "") -> None:
        """No config file, and so no API key, is required."""
        self.config = {}

    def query(self, query: str, num_responses: int = 1) -> List[str]:
        key = stage_of(query)
        self.calls.append(key)
        self.prompts.append(query)

        queue = self.script.get(key)
        if queue is None:
            raise KeyError(
                f"no scripted response for {key}; scripted keys: "
                f"{sorted(self.script)}"
            )

        out = []
        for _ in range(num_responses):
            out.append(queue[0] if len(queue) == 1 else queue.pop(0))
        return out

    def get_response_texts(self, query_responses) -> List[str]:
        if isinstance(query_responses, str):
            return [query_responses]
        return list(query_responses)

    # --- assertions helpers ------------------------------------------------

    def count(self, stage: str = None, op: str = None) -> int:
        """How many calls matched ``stage`` and/or ``op``."""
        return sum(
            1
            for s, o in self.calls
            if (stage is None or s == stage) and (op is None or o == op)
        )

    def ops_for(self, stage: str) -> List[str]:
        return [o for s, o in self.calls if s == stage]
