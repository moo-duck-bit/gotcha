"""Similarity backends for definitional alignment.

The paper computes q_def with "sentence-BERT cosine similarity normalized to
the range [0,1]". This module supplies that backend and installs it into the
diagnostic layer.

Sentence-BERT stays optional. The diagnostic layer defaults to a
dependency-free lexical measure, which keeps the graph and its tests runnable
without torch:

    pip install sentence-transformers
    python -m cbt.run --level distortions --similarity sbert
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Sequence

from cbt import diagnostic

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceBertSimilarity:
    """Cosine between sentence-BERT embeddings, mapped onto [0, 1].

    Cosine runs over [-1, 1] and the paper does not say which mapping it used,
    so both are available:

    ``clamp``   negatives become 0, leaving unrelated text near 0. This is
                what keeps tau able to reject anything: q_def weighs 0.4, so
                a floor of 0.5 would put S_diag at 0.8 for any well-formed
                thought whatever its rationale said.
    ``affine``  (cos + 1) / 2, which keeps the ordering of negative
                similarities at the cost of that floor.

    Embeddings are cached; each definition meets many rationales in a run.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        normalise: str = "clamp",
        model=None,
    ) -> None:
        if normalise not in ("clamp", "affine"):
            raise ValueError("normalise must be 'clamp' or 'affine'")
        self.normalise = normalise
        self.model_name = model_name
        self._cache: Dict[str, Sequence[float]] = {}

        if model is not None:
            self._model = model
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "sentence-transformers is not installed. Either "
                "`pip install sentence-transformers` or keep the default "
                "lexical similarity."
            ) from exc
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> Sequence[float]:
        key = str(text)
        if key not in self._cache:
            self._cache[key] = list(self._model.encode(key))
        return self._cache[key]

    def __call__(self, a: str, b: str) -> float:
        if not str(a).strip() or not str(b).strip():
            return 0.0
        cosine = _cosine(self.embed(a), self.embed(b))
        if self.normalise == "affine":
            return max(0.0, min(1.0, (cosine + 1.0) / 2.0))
        return max(0.0, min(1.0, cosine))


def _cosine(u: Sequence[float], v: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(u, v))
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(y * y for y in v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (nu * nv)


BACKENDS = ("lexical", "sbert")


def build(name: str = "lexical", **kwargs) -> Callable[[str, str], float]:
    """Construct a similarity function by name."""
    if name == "lexical":
        return diagnostic.lexical_similarity
    if name == "sbert":
        return SentenceBertSimilarity(**kwargs)
    raise ValueError(f"unknown similarity backend: {name!r}")


def install(name: str = "lexical", **kwargs) -> Callable[[str, str], float]:
    """Build a backend and make the diagnostic layer use it."""
    fn = build(name, **kwargs)
    diagnostic.set_similarity_fn(fn)
    return fn
