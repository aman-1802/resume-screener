"""Local, free, per-resume scoring via sentence-transformer embeddings.

Runs entirely on-device (no API calls), so cost stays flat even for
hundreds of resumes. Semantic similarity catches reworded/synonym skills
(e.g. "Pythonic" ~ "Python") that plain keyword matching would miss.

A keyword also counts as matched if its core technical term appears
verbatim in the resume. This is needed because a single dense resume
sentence (e.g. "Deployed on AWS using Docker and Kubernetes") dilutes the
sentence-level embedding enough that some individually-listed JD
requirements fall under the similarity threshold even though the exact
term is right there in the text.
"""

import re

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.55
MUST_HAVE_WEIGHT = 2
NICE_TO_HAVE_WEIGHT = 1

# Generic recruiting-speak stripped out before literal-term matching, so a
# keyword like "Experience with Kubernetes" doesn't match on the word
# "experience" alone. Deliberately does NOT include words that are often
# the meaningful, distinguishing part of a keyword (e.g. "design" is the
# whole point of "Bachelor's degree in Design" or "design patterns") --
# stripping those made the keyword match on whatever was left over.
_STOPWORDS = {
    "experience", "experienced", "strong", "familiarity", "familiar",
    "understanding", "knowledge", "proficiency", "proficient", "deploying",
    "deployment", "years", "year", "with", "on", "of", "and",
    "or", "the", "a", "an", "in", "for", "to", "using", "skills", "plus",
}

# Common qualification abbreviations that resumes use in place of the
# words a JD spells out (e.g. "B.Tech" instead of "Bachelor's degree").
_DEGREE_LEVEL_PATTERNS = {
    "bachelor": [re.compile(p, re.IGNORECASE) for p in (r"\bb\.?\s?tech\b", r"\bb\.?\s?e\.?\b", r"\bb\.?\s?sc\.?\b", r"\bb\.?\s?a\.?\b", r"\bbachelor'?s?\b")],
    "master": [re.compile(p, re.IGNORECASE) for p in (r"\bm\.?\s?tech\b", r"\bm\.?\s?e\.?\b", r"\bm\.?\s?sc\.?\b", r"\bmba\b", r"\bmaster'?s?\b")],
    "doctorate": [re.compile(p, re.IGNORECASE) for p in (r"\bph\.?\s?d\.?\b", r"\bdoctorate\b")],
}
_DEGREE_KEYWORD_RE = re.compile(r"^(bachelor|master|doctorate)'?s?\s+degree(?:\s+in\s+(.+))?$", re.IGNORECASE)


def _degree_chunk_match(keyword: str, resume_text: str) -> bool | None:
    """For a "<Level> degree [in <field>]" keyword, require the degree
    level and the field words to co-occur on the *same resume line* --
    not just appear anywhere in the whole document.

    Without this, a resume mentioning "B.E." anywhere and the word
    "design" anywhere else (e.g. "design systems" in a work-experience
    bullet) would wrongly satisfy "Bachelor's degree in Design", since
    plain whole-document term matching doesn't care that those two
    mentions have nothing to do with each other.

    Splits on newlines only (not periods/bullets like the general
    chunker) because degree abbreviations like "B.E." or "B.Tech."
    contain periods -- splitting on those would sever the abbreviation
    from the field name that follows it on the same line.

    Returns None if `keyword` isn't a degree-requirement phrase at all,
    so the caller falls back to plain literal matching.
    """
    m = _DEGREE_KEYWORD_RE.match(keyword.strip())
    if not m:
        return None

    level_patterns = _DEGREE_LEVEL_PATTERNS.get(m.group(1).lower(), [])
    field_terms = [
        t for t in re.findall(r"[A-Za-z0-9]+", m.group(2) or "")
        if t.lower() not in _STOPWORDS and len(t) > 1
    ]

    for line in resume_text.lower().splitlines():
        if not any(p.search(line) for p in level_patterns):
            continue
        if not field_terms or all(re.search(rf"\b{re.escape(t.lower())}\b", line) for t in field_terms):
            return True
    return False


_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _chunk_text(text: str) -> list[str]:
    # Split on sentence-ish boundaries and newlines; drop tiny fragments.
    raw_chunks = re.split(r"[\n\.•;]+", text)
    return [c.strip() for c in raw_chunks if len(c.strip()) >= 3]


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


def _literal_match(keyword: str, resume_text_lower: str) -> bool:
    """True if the keyword's core (non-stopword) terms all appear verbatim.

    Requiring every term (not just one) matters for multi-word phrases: a
    single generic word like "system" or "tools" showing up anywhere in an
    unrelated sentence must not be enough to claim "operating system
    fundamentals" or "debugging tools" as matched.
    """
    terms = [t for t in re.findall(r"[A-Za-z0-9+#.]+", keyword) if t.lower() not in _STOPWORDS and len(t) > 1]
    if not terms:
        return False
    return all(re.search(rf"\b{re.escape(t.lower())}\b", resume_text_lower) for t in terms)


def score_resume(
    resume_text: str,
    must_have: list[str],
    nice_to_have: list[str],
    threshold: float = SIMILARITY_THRESHOLD,
) -> tuple[float, list[str], list[str]]:
    """Return (score 0-100, matched_keywords, missing_keywords)."""
    all_keywords = list(must_have) + list(nice_to_have)
    if not all_keywords:
        return 0.0, [], []

    chunks = _chunk_text(resume_text)
    if not chunks:
        return 0.0, [], list(all_keywords)

    model = get_model()
    keyword_embeddings = model.encode(all_keywords, convert_to_numpy=True)
    chunk_embeddings = model.encode(chunks, convert_to_numpy=True)

    sims = _cosine_sim_matrix(keyword_embeddings, chunk_embeddings)
    max_sims = sims.max(axis=1)
    resume_text_lower = resume_text.lower()

    matched, missing = [], []
    matched_weight = 0.0
    total_weight = 0.0

    for i, keyword in enumerate(all_keywords):
        weight = MUST_HAVE_WEIGHT if i < len(must_have) else NICE_TO_HAVE_WEIGHT
        total_weight += weight

        degree_result = _degree_chunk_match(keyword, resume_text)
        is_match = (
            degree_result if degree_result is not None
            else (max_sims[i] >= threshold or _literal_match(keyword, resume_text_lower))
        )

        if is_match:
            matched.append(keyword)
            matched_weight += weight
        else:
            missing.append(keyword)

    score = (matched_weight / total_weight) * 100 if total_weight else 0.0
    return round(score, 1), matched, missing
