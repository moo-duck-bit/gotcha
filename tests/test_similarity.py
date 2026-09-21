"""
The sentence-BERT backend, tested without installing sentence-BERT.

A fake encoder stands in for the model, so the adapter's arithmetic, its
caching and the two [0, 1] mappings are all covered on a machine with no
torch. Whether the real embeddings are any good is the model's business.
"""

import math

import pytest

from cbt import diagnostic, similarity


class FakeEncoder:
    """Maps a handful of texts to fixed vectors and counts encode calls."""

    VECTORS = {
        "a": [1.0, 0.0],
        "same": [1.0, 0.0],
        "orthogonal": [0.0, 1.0],
        "opposite": [-1.0, 0.0],
        "half": [1.0, 1.0],
    }

    def __init__(self):
        self.calls = 0

    def encode(self, text):
        self.calls += 1
        return self.VECTORS.get(text, [0.0, 0.0])


@pytest.fixture
def sbert():
    return similarity.SentenceBertSimilarity(model=FakeEncoder())


# --- cosine mapping --------------------------------------------------------


def test_identical_text_scores_one(sbert):
    assert sbert("a", "same") == pytest.approx(1.0)


def test_orthogonal_text_scores_zero(sbert):
    assert sbert("a", "orthogonal") == pytest.approx(0.0)


def test_partial_overlap(sbert):
    assert sbert("a", "half") == pytest.approx(1 / math.sqrt(2))


def test_clamp_floors_negative_similarity_at_zero(sbert):
    assert sbert("a", "opposite") == 0.0


def test_affine_maps_opposites_to_zero_and_unrelated_to_a_half():
    affine = similarity.SentenceBertSimilarity(model=FakeEncoder(), normalise="affine")
    assert affine("a", "opposite") == pytest.approx(0.0)
    assert affine("a", "orthogonal") == pytest.approx(0.5)


def test_affine_floor_would_disable_the_threshold():
    """Why clamp is the default.

    q_def carries weight 0.4. Under the affine mapping an unrelated
    rationale still scores 0.5, so a well-formed thought reaches S_diag 0.8
    whatever it says, and tau = 0.6 can no longer reject anything.
    """
    unrelated_under_affine = 0.5
    assert diagnostic.s_diag(
        con=1.0, coh=1.0, deff=unrelated_under_affine, counter=1.0
    ) == pytest.approx(0.8)
    assert 0.8 > diagnostic.TAU_DIAG


def test_rejects_an_unknown_normalisation():
    with pytest.raises(ValueError):
        similarity.SentenceBertSimilarity(model=FakeEncoder(), normalise="nonsense")


# --- the edges -------------------------------------------------------------


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_blank_text_scores_zero(sbert, blank):
    assert sbert("a", blank) == 0.0
    assert sbert(blank, "a") == 0.0


def test_zero_vectors_do_not_divide_by_zero(sbert):
    assert sbert("unknown text", "other unknown text") == 0.0


# --- caching ---------------------------------------------------------------


def test_embeddings_are_cached():
    encoder = FakeEncoder()
    sim = similarity.SentenceBertSimilarity(model=encoder)
    for _ in range(5):
        sim("a", "half")
    assert encoder.calls == 2


# --- installation ----------------------------------------------------------


def test_build_returns_the_lexical_backend_by_default():
    assert similarity.build("lexical") is diagnostic.lexical_similarity


def test_build_rejects_an_unknown_backend():
    with pytest.raises(ValueError):
        similarity.build("word2vec")


def test_install_switches_the_diagnostic_layer():
    original = diagnostic.get_similarity_fn()
    try:
        installed = similarity.install("sbert", model=FakeEncoder())
        assert diagnostic.get_similarity_fn() is installed
        # q_def now runs through the installed backend.
        scored = diagnostic.q_def(
            "a", ["labeling"], definitions={"labeling": "same"}
        )
        assert scored == pytest.approx(1.0)
    finally:
        diagnostic.set_similarity_fn(original)


def test_the_default_backend_needs_no_optional_dependency():
    """The graph and its tests must run without torch installed."""
    assert diagnostic.get_similarity_fn() is diagnostic.lexical_similarity
