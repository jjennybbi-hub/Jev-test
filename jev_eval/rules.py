"""Offline, rule-based checks: free and instant, but crude.

These match words and count things; they don't understand the text. They can't judge warmth,
lecturing, or talking down, so those dimensions are left blank. Each result is stored in the
same shape as a Jev answer (with probabilities of exactly 0 or 1) so the scorecard and report
work unchanged, plus a short note saying what each rule found.

All phrase lists and cut-offs are here, in one place, for editing.
"""

import re
from typing import Any

from jev_eval.inputs import Record

RULES_VERSION = "rules-1"

APOLOGY = r"\b(sorry|apologi[sz]e|apologies)\b"
DISCLAIMERS = [
    r"\bI(?:'m| am) not an? (?:\w+ )?(?:lawyer|doctor|advisor|professional|expert)\b",
    r"\bnot (?:legal|medical|financial|professional) advice\b",
    r"\bconsult (?:a|an|with a|your) (?:professional|doctor|lawyer|attorney|expert|specialist)\b",
    r"\bas an AI\b",
    r"\bI(?:'m| am) (?:just )?an AI\b",
    r"\bdouble-check with an? (?:authoritative|reliable|official) source\b",
]
HEDGES = r"\b(might|may|possibly|perhaps|arguably|it seems|it appears|could be|likely|probably|somewhat|generally)\b"
# A response is "hedging" when it has at least this many hedge words, and this many per 100 words.
HEDGE_MIN_COUNT = 2
HEDGE_PER_100_WORDS = 4.0

PREAMBLE = r"^\W*(great question|good question|certainly|sure|of course|absolutely|thanks for asking|I'd be happy|I would be happy|happy to help)\b"
FILLER = [
    r"\bI hope (?:this|that) helps\b",
    r"\blet me know if you have any (?:other|more|further) questions\b",
    r"\bfeel free to (?:ask|reach out)\b",
    r"\b(?:great|good|excellent) question\b",
    r"\bit(?:'s| is) (?:important|worth) (?:to note|noting)\b",
    r"\bin conclusion\b",
]
# A prompt "asks for an answer" if it has a question mark or opens like a request for one.
QUESTION_OPENERS = r"^\W*(what|how|which|when|where|who|why|is|are|can|could|does|do|did|should|will|convert|calculate)\b"

CASUAL = r"\b(gonna|wanna|gotta|kinda|yeah|yep|nope|hey|lol|cool|awesome|stuff)\b"
FORMAL = r"\b(furthermore|moreover|therefore|thus|hereby|herein|regarding|accordingly|consequently|shall|kindly)\b"

CODE_BLOCK = re.compile(r"```.*?```", re.S)
WORD = re.compile(r"[A-Za-z0-9']+")
SENTENCE_END = re.compile(r"[.!?]+(?:\s|$)")


def evaluate_one_rules(record: Record) -> dict[str, Any]:
    text = CODE_BLOCK.sub(" ", record.response)
    words = WORD.findall(text)
    answers: dict[str, Any] = {}
    notes: dict[str, str] = {}

    def flag(key: str, hit: bool, note: str) -> None:
        answers[key] = {"type": "noul", "noul": 1.0 if hit else 0.0}
        notes[key] = note

    def level(key: str, value: int, note: str) -> None:
        answers[key] = {"type": "score", "score": float(value), "probabilities": {str(i): float(i == value) for i in range(3)}}
        notes[key] = note

    apologies = _find(APOLOGY, text)
    flag("unneeded_apology", bool(apologies), _found(apologies) or "no apology")

    disclaimers = [m for pattern in DISCLAIMERS for m in _find(pattern, text)]
    flag("excess_disclaimers", bool(disclaimers), _found(disclaimers) or "no disclaimer phrases")

    hedges = _find(HEDGES, text)
    per_100 = 100 * len(hedges) / max(len(words), 1)
    flag("excess_hedging", len(hedges) >= HEDGE_MIN_COUNT and per_100 >= HEDGE_PER_100_WORDS, f"{len(hedges)} hedge words ({per_100:.1f} per 100 words)")

    asks = "?" in record.user_prompt or bool(re.search(QUESTION_OPENERS, record.user_prompt, re.I))
    flag("prompt_asks_for_answer", asks, "question mark or question opener" if asks else "no question found")
    preamble = re.search(PREAMBLE, text, re.I)
    flag("answer_up_front", not preamble, f"opens with {preamble[1]!r}" if preamble else "no preamble phrase at the start")

    filler = [m for pattern in FILLER for m in _find(pattern, text)] + _repeated_sentences(text)
    level("conciseness", 2 if not filler else 1 if len(filler) <= 2 else 0, _found(filler) or "no filler phrases")

    ease = _reading_ease(text, words)
    level("easy_to_follow", 2 if ease >= 60 else 1 if ease >= 30 else 0, f"reading ease {ease:.0f} (higher is easier)")

    casual, formal = _find(CASUAL, text), _find(FORMAL, text)
    contractions = sum("'" in w for w in words)
    lean = len(formal) - len(casual) - (contractions >= 3)
    level("formality", 2 if lean >= 2 else 0 if lean <= -2 else 1, f"{len(casual)} casual words, {len(formal)} formal words, {contractions} contractions")

    choice, note = _formatting(record.response, len(words))
    answers["formatting"] = {"type": "choice", "choice": choice, "probabilities": {c: float(c == choice) for c in ("over_formatted", "fits", "under_formatted")}}
    notes["formatting"] = note

    return {"id": record.id, "engine": "rules", "questions_version": RULES_VERSION, "answers": answers, "notes": notes}


def _find(pattern: str, text: str) -> list[str]:
    return [m[0] for m in re.finditer(pattern, text, re.I)]


def _found(matches: list[str]) -> str:
    return "found: " + ", ".join(repr(m) for m in matches[:3]) + (" ..." if len(matches) > 3 else "") if matches else ""


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(text) if WORD.search(s)]


def _repeated_sentences(text: str) -> list[str]:
    seen, repeats = set(), []
    for s in _sentences(text):
        norm = " ".join(WORD.findall(s.lower()))
        if norm in seen and len(norm.split()) >= 4:
            repeats.append(s)
        seen.add(norm)
    return repeats


def _syllables(word: str) -> int:
    groups = re.findall(r"[aeiouy]+", word.lower())
    count = len(groups) - (word.lower().endswith("e") and len(groups) > 1)
    return max(count, 1)


def _reading_ease(text: str, words: list[str]) -> float:
    """Flesch reading ease: about 60-70 is plain English, below 30 is hard going."""
    if not words:
        return 100.0
    sentences = max(len(_sentences(text)), 1)
    syllables = sum(_syllables(w) for w in words)
    return 206.835 - 1.015 * len(words) / sentences - 84.6 * syllables / len(words)


def _formatting(response: str, word_count: int) -> tuple[str, str]:
    lines = response.splitlines()
    headings = sum(bool(re.match(r"\s*#{1,6}\s", line)) for line in lines)
    bullets = sum(bool(re.match(r"\s*(?:[-*+]|\d+[.)])\s", line)) for line in lines)
    bold = len(re.findall(r"\*\*[^*]+\*\*", response))
    marks = f"{headings} headings, {bullets} list items, {bold} bold, {word_count} words"
    if word_count < 80 and (headings or bullets >= 3 or bold >= 3):
        return "over_formatted", f"short answer with heavy formatting ({marks})"
    step_words = len(re.findall(r"\b(first|then|next|after that|finally)\b", response, re.I))
    paragraphs = [p for p in re.split(r"\n\s*\n", response) if p.strip()]
    if not bullets and step_words >= 3:
        return "under_formatted", f"step-by-step text with no list ({marks})"
    if word_count > 200 and len(paragraphs) == 1 and not bullets:
        return "under_formatted", f"long single block of text ({marks})"
    return "fits", marks
