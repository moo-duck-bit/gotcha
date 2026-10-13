"""Diagnostic quality functions for the Validate & Improve stage.

The four quality measures, their weighted combination S_diag, and the final
ranking score S_final = S_T + lambda * S_diag. Everything here is a pure
function over parsed thought states, so the layer is testable without a model.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, Iterable, Sequence

W_CON, W_COH, W_DEF, W_COUNTER = 0.2, 0.3, 0.4, 0.1
TAU_DIAG = 0.6
LAMBDA = 3.0

# S_T is a 1-10 rating while S_diag is bounded to [0, 1], so S_T is normalised
# before the two are combined. Without that, lambda = 3 caps the diagnostic
# contribution at 3 points against a 9-point swing in base quality, and
# diagnostic validity does not dominate surface fluency as intended.
BASE_SCORE_MAX = 10.0

# A fine-grained belief outside its parent's subcategories: category violation
# or q_coh penalty?
#
# The paper gates on "a minimum diagnostic score and category validity" and
# says q_coh "ensures vertical coherence", which reads like a gate. But with
# these weights, q_coh = 0 still leaves S_diag at 0.7, above tau, so as a score
# it cannot reject anything. The reported ablation agrees: dropping q_coh
# entirely costs CBT-FC only 0.803 -> 0.734, far less than admitting every
# hierarchy violation would.
#
# The default therefore follows the formula literally and lets q_coh demote.
# Weighted by lambda it still moves S_final by 0.9 on a [0, 4] scale, enough to
# reorder KeepBestN. Set True to gate instead.
ENFORCE_HIERARCHY_GATE = False

LEVELS = ("core_beliefs", "fine_grained", "distortions")

CORE_BELIEFS = ("helpless", "unlovable", "worthless")

FINE_GRAINED: Dict[str, tuple] = {
    "helpless": (
        "i am incompetent",
        "i am helpless",
        "i am powerless, weak, vulnerable",
        "i am a victim",
        "i am needy",
        "i am trapped",
        "i am out of control",
        "i am a failure, loser",
    ),
    "unlovable": (
        "i am defective",
        "i am unlovable",
        "i am unattractive",
        "i am undesirable, unwanted",
        "i am bound to be rejected",
        "i am bound to be abandoned",
        "i am bound to be alone",
    ),
    "worthless": (
        "i am worthless, waste",
        "i am immoral",
        "i am bad - dangerous, toxic, evil",
        "i do not deserve to live",
    ),
}

CORE_BELIEF_DEFINITIONS: Dict[str, str] = {
    "helpless": (
        "being ineffective in getting things done, self-protection, and or "
        "measuring up to others"
    ),
    "unlovable": (
        "having personal qualities resulting in an inability to get or "
        "maintain love and intimacy from others"
    ),
    "worthless": "being an immoral sinner or dangerous to others",
}

DISTORTION_DEFINITIONS: Dict[str, str] = {
    "all-or-nothing thinking": (
        "black-and-white, polarized, or dichotomous thinking. View situation in "
        "only two categories instead of on a continuum"
    ),
    "mind reading": (
        "believe you know what others are thinking, failing to consider other, "
        "more likely possibilities"
    ),
    "fortune-telling": (
        "predict the future negatively without considering other, more likely "
        "outcomes"
    ),
    "personalization": (
        "believe others are behaving negatively because of you, without "
        "considering more plausible explanations"
    ),
    "emotional reasoning": (
        "think something must be true because you feel it so strongly, ignoring "
        "evidence to the contrary"
    ),
    "overgeneralization": (
        "make a sweeping negative conclusion that goes far beyond the current "
        "situation"
    ),
    "labeling": (
        "put a fixed, global label on yourself or others without considering "
        "evidence might lead to less extreme conclusion"
    ),
    "should statements": (
        "have a precise, fixed idea of how you or others should behave, "
        "overestimate how bad it is when expectations not met"
    ),
    "magnification": (
        "unreasonably magnify the negative and/or minimize the positive when "
        "evaluating yourself, another person, or situation"
    ),
    "mental filter": (
        "pay undue attention to one negative detail instead of seeing the whole "
        "picture"
    ),
}

ALL_FINE_GRAINED: tuple = tuple(
    belief for beliefs in FINE_GRAINED.values() for belief in beliefs
)

VOCABULARY: Dict[str, tuple] = {
    "core_beliefs": CORE_BELIEFS,
    "fine_grained": ALL_FINE_GRAINED,
    "distortions": tuple(DISTORTION_DEFINITIONS),
}

# Populate from the CD-Quest item bank when available.
DISTORTION_EXAMPLES: Dict[str, tuple] = {k: () for k in DISTORTION_DEFINITIONS}

# q_def applies at every level, so each needs its own reference text. A
# fine-grained belief is its own definition: the label is the clinical
# statement.
DEFINITIONS_BY_LEVEL: Dict[str, Dict[str, str]] = {
    "core_beliefs": CORE_BELIEF_DEFINITIONS,
    "fine_grained": {belief: belief for belief in ALL_FINE_GRAINED},
    "distortions": DISTORTION_DEFINITIONS,
}

# Standard CBT thought-record form. Replace with the official worksheet set
# when available.
COUNTER_QUESTIONS: Dict[str, tuple] = {
    "all-or-nothing thinking": (
        "is there a middle ground between the two extremes",
        "what would a continuum rather than two categories look like",
    ),
    "mind reading": (
        "what evidence do i have about what they are actually thinking",
        "what other explanations for their behaviour are possible",
    ),
    "fortune-telling": (
        "what other outcomes are possible",
        "how often has this prediction come true before",
    ),
    "personalization": (
        "what else could have caused this besides me",
        "how much of this was actually within my control",
    ),
    "emotional reasoning": (
        "what evidence contradicts this feeling",
        "would this still be true if i felt differently",
    ),
    "overgeneralization": (
        "is this one situation or a consistent pattern",
        "what counterexamples can i recall",
    ),
    "labeling": (
        "what specific behaviour am i describing instead of a global label",
        "does this label account for all the evidence",
    ),
    "should statements": (
        "where does this rule come from",
        "what happens if i replace should with prefer",
    ),
    "magnification": (
        "how important will this be in a year",
        "what positives am i discounting",
    ),
    "mental filter": (
        "what parts of the whole picture am i leaving out",
        "what went well that i am not counting",
    ),
}

# Alternative names the benchmark's definitions themselves supply, e.g.
# all-or-nothing thinking is "Also called black-and-white, polarized, or
# dichotomous thinking". Only names from those definitions belong here; this is
# not a place to map one clinical category onto a similar-sounding one.
ALIASES: Dict[str, str] = {
    "black-and-white thinking": "all-or-nothing thinking",
    "black and white thinking": "all-or-nothing thinking",
    "polarized thinking": "all-or-nothing thinking",
    "dichotomous thinking": "all-or-nothing thinking",
}


def _clamp(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


# Curly quotes and non-breaking space, spelled as escapes because they are
# invisible or confusable in source.
_SMART = str.maketrans(
    {
        "\u2019": "'",  # right single quote
        "\u2018": "'",  # left single quote
        "\u201c": '"',  # left double quote
        "\u201d": '"',  # right double quote
        "\u00a0": " ",  # non-breaking space
    }
)


def _norm(label: str) -> str:
    """Lowercase, trim, and fold typographic punctuation to ASCII.

    The benchmark writes "I don't deserve to live" with a curly apostrophe;
    a model answering with the ASCII one would otherwise never match.
    """
    return str(label).translate(_SMART).strip().lower()


# --- Similarity ------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_TOKEN.findall(str(text).lower()))


def lexical_similarity(a: str, b: str) -> float:
    """Cosine over token sets, in [0, 1]. No dependencies.

    The default backend. See cbt.similarity for the paper's sentence-BERT one.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / ((len(ta) ** 0.5) * (len(tb) ** 0.5))


_sim: Callable[[str, str], float] = lexical_similarity


def set_similarity_fn(fn: Callable[[str, str], float]) -> None:
    """Install the similarity function used by q_def and q_counter."""
    global _sim
    _sim = fn


def get_similarity_fn() -> Callable[[str, str], float]:
    """The currently installed similarity function."""
    return _sim


# --- Label canonicalisation ------------------------------------------------
#
# Models paraphrase. Left alone, "I am a failure" never matches the canonical
# "I am a failure, loser", and one correct answer counts as both a miss and a
# false positive. That alone drives F1 to zero while accuracy still reads 0.5
# to 0.8.
#
# The canonical labels supply their own paraphrases: several are written as
# comma-separated synonym groups ("powerless, weak, vulnerable"), and the rest
# carry filler a model drops ("bound to be abandoned" -> "abandoned").

_LEAD = re.compile(r"^(i\s*am|i'm|im|i\s+do\s+not|i\s+don'?t|i)\s+")
_FILLER = re.compile(r"\b(bound\s+to\s+be|going\s+to\s+be|always)\b")
_ARTICLE = re.compile(r"^(a|an|the)\s+")


def _terms(label: str) -> set:
    """The interchangeable surface terms a label stands for.

    "I am powerless, weak, vulnerable" -> {powerless, weak, vulnerable}
    "I am bound to be abandoned"       -> {abandoned}
    "I am a failure, loser"            -> {failure, loser}
    """
    text = _norm(label)
    text = _LEAD.sub("", text)
    text = _FILLER.sub(" ", text)
    # A spaced dash separates alternatives ("bad - dangerous, toxic"); a dash
    # inside a word is spelling ("fortune-telling" == "fortune telling").
    text = re.sub(r"\s+[-\u2013\u2014/]\s+", ",", text)
    text = re.sub(r"[-\u2013\u2014/]", " ", text)

    out = set()
    for part in text.split(","):
        part = _ARTICLE.sub("", part.strip(" .!?")).strip()
        part = re.sub(r"\s+", " ", part)
        if not part:
            continue
        out.add(part)
        # Models switch freely between "should statement" and the canonical
        # plural, so carry both forms.
        singular = part.endswith("s") and not part.endswith("ss")
        out.add(part[:-1] if singular else part + "s")
    return out


def _match(label: str, vocabulary: Iterable[str]) -> str | None:
    """The one canonical entry a surface form resolves to, if exactly one does.

    Matching two entries yields None: an ambiguous paraphrase is reported as
    unknown instead of being assigned to whichever candidate came first.
    """
    entries = list(vocabulary)
    target = ALIASES.get(_norm(label), _norm(label))
    for entry in entries:
        if target == _norm(entry):
            return entry

    terms = _terms(target)
    if not terms:
        return None
    hits = [e for e in entries if terms & _terms(e)]
    return hits[0] if len(hits) == 1 else None


def canonical_label(label: str, level: str) -> str | None:
    """Resolve a surface form to its canonical label, or None."""
    if level not in VOCABULARY:
        raise ValueError(f"unknown level: {level!r}")
    return _match(label, VOCABULARY[level])


def canonicalise(labels: Sequence, level: str) -> list:
    """Canonical forms of ``labels``, dropping what cannot be resolved.

    Deduplicates: distinct paraphrases often collapse onto one label.
    """
    out = []
    for label in labels or ():
        resolved = canonical_label(label, level)
        if resolved is not None and resolved not in out:
            out.append(resolved)
    return out


def unresolved(labels: Sequence, level: str) -> list:
    """Labels no canonical form accepts, for diagnosing prompt drift."""
    return [x for x in (labels or ()) if canonical_label(x, level) is None]


# --- The four quality measures ---------------------------------------------


def q_con(evidence_for: Sequence, evidence_against: Sequence) -> float:
    """Contrastive balance: min(1, |E-| / |E+|), or 0 when |E-| is 0.

    Zero without counter-evidence, which is the pressure to look for
    alternative explanations. Counter-evidence with no supporting evidence
    gives an unbounded ratio, saturated at 1.
    """
    n_pos, n_neg = len(evidence_for or ()), len(evidence_against or ())
    if n_neg == 0:
        return 0.0
    if n_pos == 0:
        return 1.0
    return min(1.0, n_neg / n_pos)


def q_coh(fine_grained: Sequence, core_beliefs: Sequence) -> float:
    """Hierarchical consistency: the share of children their parents permit.

    q_coh = |c intersect (union of F_y for y in Y_c)| / |c|
    """
    raw = list(fine_grained or ())
    if not raw:
        return 0.0

    permitted = set()
    for parent in core_beliefs or ():
        resolved = canonical_label(parent, "core_beliefs")
        if resolved:
            permitted.update(FINE_GRAINED[resolved])

    # Unresolvable children count against the score, not out of the total.
    children = [canonical_label(x, "fine_grained") for x in raw]
    return sum(1 for x in children if x in permitted) / len(raw)


def q_def(
    rationale: str,
    labels: Sequence,
    definitions: Dict[str, str] | None = None,
    examples: Dict[str, Iterable[str]] | None = None,
) -> float:
    """Definitional alignment of the rationale with clinical criteria.

    q_def = mean over l in Y of max(sim(R, D_l), max over e in E_l of sim(R, e))

    A label with no known definition contributes 0, so a hallucinated category
    is penalised, not ignored.
    """
    labels = list(labels or ())
    if not labels:
        return 0.0
    defs = DISTORTION_DEFINITIONS if definitions is None else definitions
    exs = DISTORTION_EXAMPLES if examples is None else examples

    total = 0.0
    for label in labels:
        key = _match(label, defs) or _norm(label)
        best = _sim(rationale, defs[key]) if key in defs else 0.0
        for example in exs.get(key, ()) or ():
            best = max(best, _sim(rationale, example))
        total += _clamp(best)
    return _clamp(total / len(labels))


def q_counter(
    rationale: str,
    labels: Sequence,
    questions: Dict[str, Iterable[str]] | None = None,
    threshold: float = 0.2,
) -> float:
    """Share of the relevant counter-questions the rationale addresses.

    A question counts as addressed once its similarity to the rationale clears
    ``threshold``. A label with no question set contributes 0.

    Labels resolve through the same canonicalisation as q_def, so a paraphrase
    such as "black-and-white thinking" finds the questions of its category.
    """
    labels = list(labels or ())
    if not labels:
        return 0.0
    bank = COUNTER_QUESTIONS if questions is None else questions

    total = 0.0
    for label in labels:
        key = _match(label, bank) or _norm(label)
        qs = tuple(bank.get(key, ()) or ())
        if not qs:
            continue
        total += sum(1 for q in qs if _sim(rationale, q) >= threshold) / len(qs)
    return _clamp(total / len(labels))


# --- S_diag and S_final ----------------------------------------------------


def s_diag(
    con: float = 0.0, coh: float = 0.0, deff: float = 0.0, counter: float = 0.0
) -> float:
    """S_diag = 0.2 q_con + 0.3 q_coh + 0.4 q_def + 0.1 q_counter."""
    return _clamp(
        W_CON * _clamp(con)
        + W_COH * _clamp(coh)
        + W_DEF * _clamp(deff)
        + W_COUNTER * _clamp(counter)
    )


def s_final(base_score: float, diag_score: float, lam: float = LAMBDA) -> float:
    """S_final = S_T + lambda * S_diag.

    ``base_score`` must already be normalised to [0, 1] by ``normalise_base``.
    Diagnostic validity then carries three times the weight of base quality and
    dominates surface fluency where the two conflict.
    """
    return base_score + lam * diag_score


def normalise_base(base_score: float, base_max: float = BASE_SCORE_MAX) -> float:
    """Map a raw rating onto [0, 1] for ``s_final``."""
    if base_max <= 0:
        raise ValueError("base_max must be positive")
    return _clamp(float(base_score) / base_max)


# --- Per-level scoring -----------------------------------------------------


def score_thought(state: Dict, level: str = "distortions") -> Dict[str, float]:
    """Every quality component for one thought state.

    Reads the keys a parsed thought carries: the label list for ``level``,
    ``rationale``, ``evidence_for`` / ``evidence_against``, and
    ``core_beliefs`` when scoring fine-grained beliefs.

    q_coh and q_counter are each undefined outside one level: there is no
    parent to check at the root, and the question bank is indexed by
    distortion type. Both are held at 1.0 there, neither rewarding nor
    penalising.
    """
    if level not in LEVELS:
        raise ValueError(f"unknown level: {level!r}")

    labels = state.get(level, []) or []
    rationale = state.get("rationale", "") or ""

    con = q_con(state.get("evidence_for", []), state.get("evidence_against", []))
    coh = (
        q_coh(labels, state.get("core_beliefs", []))
        if level == "fine_grained"
        else 1.0
    )
    deff = q_def(rationale, labels, definitions=DEFINITIONS_BY_LEVEL[level])
    counter = q_counter(rationale, labels) if level == "distortions" else 1.0

    return {
        "q_con": con,
        "q_coh": coh,
        "q_def": deff,
        "q_counter": counter,
        "s_diag": s_diag(con=con, coh=coh, deff=deff, counter=counter),
    }


def category_valid(
    state: Dict,
    level: str = "distortions",
    enforce_hierarchy: bool | None = None,
) -> bool:
    """Whether every label at ``level`` belongs to the permissible set.

    Under ``enforce_hierarchy`` the permissible set for fine-grained beliefs
    narrows from all 19 to the subcategories of the inferred core beliefs,
    which turns a hierarchy violation into an outright rejection. See
    ``ENFORCE_HIERARCHY_GATE`` for why that is off by default.
    """
    if enforce_hierarchy is None:
        enforce_hierarchy = ENFORCE_HIERARCHY_GATE

    raw = list(state.get(level, []) or [])
    if not raw:
        return False
    resolved = [canonical_label(x, level) for x in raw]
    if any(x is None for x in resolved):
        return False
    if level != "fine_grained":
        return True

    parents = canonicalise(state.get("core_beliefs", []) or [], "core_beliefs")
    if enforce_hierarchy and parents:
        allowed = set()
        for parent in parents:
            allowed.update(FINE_GRAINED[parent])
    else:
        allowed = set(ALL_FINE_GRAINED)
    return all(x in allowed for x in resolved)


def make_validator(
    level: str = "distortions",
    tau: float = TAU_DIAG,
    enforce_hierarchy: bool | None = None,
    thr2: float | None = None,
):
    """Build a ``validate_function`` for ValidateAndImprove.

    A thought passes only if it clears every gate the paper names: category
    validity, tau on S_diag, and the final quality threshold thr2 on the 1-10
    rating scale when one is given. The two thresholds sit on different axes:
    tau judges clinical validity, thr2 judges quality.
    """

    def validate(state: Dict) -> bool:
        if not category_valid(state, level, enforce_hierarchy):
            return False
        if score_thought(state, level)["s_diag"] < tau:
            return False
        if thr2 is None:
            return True
        return float(state.get("base_score", 0.0) or 0.0) >= thr2

    return validate


def make_scorer(
    level: str = "distortions",
    lam: float = LAMBDA,
    base_max: float = BASE_SCORE_MAX,
):
    """Build a ``scoring_function`` returning S_final for one thought state.

    ``base_max`` normalises the raw ``base_score`` the first Score left on the
    state. Pass the ceiling that matches it: with self-consistency folded in
    that is ``quality.base_score_max()``, not the bare rating scale.
    """

    def score(state: Dict) -> float:
        base = normalise_base(state.get("base_score", 0.0) or 0.0, base_max)
        return s_final(base, score_thought(state, level)["s_diag"], lam)

    return score
