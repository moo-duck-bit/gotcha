"""
Label canonicalisation.

A model paraphrases its labels. Unless those paraphrases are mapped back onto
the benchmark's own strings, a correct answer is scored as both a miss and a
false positive. The observed effect is F1 pinned at 0.0 while accuracy still
reads around 0.5-0.8, because the per-label positives never line up.

The fixtures below are the label strings an actual run emitted.
"""

import pytest

from cbt.diagnostic import (
    CORE_BELIEFS,
    DISTORTION_DEFINITIONS,
    FINE_GRAINED,
    canonical_label,
    canonicalise,
    unresolved,
)

ALL_FINE = [v for vals in FINE_GRAINED.values() for v in vals]

LEVELS = (
    ("core_beliefs", list(CORE_BELIEFS)),
    ("fine_grained", ALL_FINE),
    ("distortions", list(DISTORTION_DEFINITIONS)),
)


# --- the vocabulary itself is well formed ---------------------------------


@pytest.mark.parametrize("level,vocabulary", LEVELS)
def test_every_canonical_label_resolves_to_itself(level, vocabulary):
    for label in vocabulary:
        assert canonical_label(label, level) == label


@pytest.mark.parametrize("level,vocabulary", LEVELS)
def test_no_canonical_label_is_ambiguous(level, vocabulary):
    """None may collide with another, or resolution silently drops it."""
    for label in vocabulary:
        assert canonical_label(label, level) is not None, label


def test_taxonomy_matches_the_paper_appendix():
    assert len(CORE_BELIEFS) == 3
    assert len(ALL_FINE) == 19
    assert len(DISTORTION_DEFINITIONS) == 10


def test_mind_reading_is_in_the_vocabulary():
    """The second most frequent gold label in CBT-CD."""
    assert "mind reading" in DISTORTION_DEFINITIONS
    assert canonical_label("mind reading", "distortions") == "mind reading"


def test_labels_absent_from_the_benchmark_are_not_accepted():
    """Burns categories the benchmark does not use must not be predictable."""
    for absent in (
        "jumping to conclusions",
        "disqualifying the positive",
        "catastrophizing",
    ):
        assert canonical_label(absent, "distortions") is None


# --- paraphrases actually observed in a run -------------------------------

OBSERVED = {
    "I am a failure": "i am a failure, loser",
    "I am a victim": "i am a victim",
    "I am abandoned": "i am bound to be abandoned",
    "I am alone": "i am bound to be alone",
    "I am defective": "i am defective",
    "I am out of control": "i am out of control",
    "I am powerless, weak, vulnerable": "i am powerless, weak, vulnerable",
    "I am rejected": "i am bound to be rejected",
    "I am trapped": "i am trapped",
    "I am unattractive": "i am unattractive",
    "I am undesirable": "i am undesirable, unwanted",
    "I am unwanted": "i am undesirable, unwanted",
    "I am vulnerable": "i am powerless, weak, vulnerable",
    "I am weak": "i am powerless, weak, vulnerable",
}


@pytest.mark.parametrize("emitted,expected", sorted(OBSERVED.items()))
def test_observed_paraphrases_resolve(emitted, expected):
    assert canonical_label(emitted, "fine_grained") == expected


# Paraphrases with no counterpart in the taxonomy. Mapping these onto a
# neighbouring label would manufacture agreement, so they stay unresolved and
# the right fix is prompting, not remapping.
OUT_OF_VOCABULARY = ("I am flawed", "I am inadequate")


@pytest.mark.parametrize("emitted", OUT_OF_VOCABULARY)
def test_out_of_vocabulary_paraphrases_stay_unresolved(emitted):
    assert canonical_label(emitted, "fine_grained") is None


# --- surface variation ----------------------------------------------------


@pytest.mark.parametrize(
    "emitted,expected",
    [
        ("Mind Reading", "mind reading"),
        ("  mind reading  ", "mind reading"),
        ("fortune telling", "fortune-telling"),
        ("all or nothing thinking", "all-or-nothing thinking"),
        ("should statement", "should statements"),
    ],
)
def test_surface_variation_is_absorbed(emitted, expected):
    assert canonical_label(emitted, "distortions") == expected


@pytest.mark.parametrize(
    "emitted",
    [
        "I don\u2019t deserve to live",  # the benchmark's own curly apostrophe
        "I don't deserve to live",
        "I do not deserve to live",
        "i dont deserve to live",
    ],
)
def test_typographic_punctuation_is_folded(emitted):
    assert canonical_label(emitted, "fine_grained") == "i do not deserve to live"


@pytest.mark.parametrize(
    "emitted", ["I am incompetent", "i am incompetent", "  I Am Incompetent  "]
)
def test_labels_missing_from_the_prior_vocabulary_resolve(emitted):
    """Absent from the 17-item list the pipeline used; 16 gold instances."""
    assert canonical_label(emitted, "fine_grained") == "i am incompetent"


@pytest.mark.parametrize(
    "alias",
    ["black-and-white thinking", "black and white thinking", "dichotomous thinking"],
)
def test_names_the_benchmark_definitions_give(alias):
    """Aliases sourced from the definition text, not from resemblance."""
    assert canonical_label(alias, "distortions") == "all-or-nothing thinking"


# --- batch helpers --------------------------------------------------------


def test_canonicalise_maps_and_deduplicates():
    """Two paraphrases of one label collapse to a single prediction."""
    assert canonicalise(
        ["I am undesirable", "I am unwanted", "I am trapped"], "fine_grained"
    ) == ["i am undesirable, unwanted", "i am trapped"]


def test_canonicalise_drops_what_it_cannot_resolve():
    assert canonicalise(["I am flawed", "I am trapped"], "fine_grained") == [
        "i am trapped"
    ]


def test_unresolved_reports_prompt_drift():
    assert unresolved(["I am flawed", "I am trapped"], "fine_grained") == [
        "I am flawed"
    ]


def test_canonical_label_rejects_an_unknown_level():
    with pytest.raises(ValueError):
        canonical_label("anything", "not_a_level")
