"""Turn raw Jev answers into a per-text scorecard. Pure code: no API calls.

Every dimension gets a `goodness` from 0 (worst) to 1 (best), derived from its probabilities:

- Noul flags: the probability of the good outcome (1 - p when a yes is the problem).
- Scores: closeness of the expected score to the target level in the weights config.
- The formatting Choice: the probability that the formatting fits.

The composite is the weighted mean of goodness over the dimensions that apply, scaled to 0-100.
Weights and targets live in a TOML file, so the scorecard can be re-ranked without new API calls.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from jev_eval.questions import DIMENSIONS, NO_THRESHOLD, QUESTIONS_VERSION, YES_THRESHOLD, Dimension

DIMENSION_BY_KEY = {d.key: d for d in DIMENSIONS}


@dataclass(frozen=True)
class DimensionWeight:
    weight: float = 1.0
    # Score dimensions only: the level (0-based) that counts as best.
    target: float | None = None


# Formality has no universally right level, so by default it is reported but not weighted.
DEFAULT_WEIGHTS: dict[str, DimensionWeight] = {
    "warmth": DimensionWeight(target=2),
    "formality": DimensionWeight(weight=0, target=1),
    "conciseness": DimensionWeight(target=2),
    "easy_to_follow": DimensionWeight(target=2),
}


def load_weights(path: Path | None) -> dict[str, DimensionWeight]:
    weights = {d.key: DEFAULT_WEIGHTS.get(d.key, DimensionWeight()) for d in DIMENSIONS}
    if path is None:
        return weights
    with path.open("rb") as f:
        config = tomllib.load(f)
    for key, values in config.get("dimensions", {}).items():
        if key not in weights:
            raise ValueError(f"{path}: unknown dimension {key!r}; known: {', '.join(weights)}")
        unknown = set(values) - {"weight", "target"}
        if unknown:
            raise ValueError(f"{path}: dimensions.{key}: unknown setting(s) {', '.join(sorted(unknown))}")
        weights[key] = DimensionWeight(weight=values.get("weight", weights[key].weight), target=values.get("target", weights[key].target))
    return weights


Verdict = Literal["yes", "no", "unsure"]


@dataclass
class Cell:
    """One dimension of one text."""

    key: str
    applies: bool | None = True  # None: the gating question was unsure
    goodness: float | None = None
    # Display values
    value: float | None = None  # Noul p, Score expected level, or P(fits)
    verdict: str | None = None  # Noul yes/no/unsure, Score top level, Choice top option
    split: bool = False  # Score probability is two-peaked; the expected value hides that
    levels: int | None = None


@dataclass
class Row:
    id: str
    error: str | None = None
    composite: float | None = None
    cells: dict[str, Cell] = field(default_factory=dict)
    stale: bool = False  # answered by an older question set


def noul_verdict(p: float) -> Verdict:
    if p >= YES_THRESHOLD:
        return "yes"
    if p <= NO_THRESHOLD:
        return "no"
    return "unsure"


def score_row(result: dict[str, Any], weights: dict[str, DimensionWeight]) -> Row:
    row = Row(id=result["id"], stale=result.get("questions_version") != QUESTIONS_VERSION)
    if "error" in result:
        row.error = result["error"]
        return row
    answers = result.get("answers", {})
    for dim in DIMENSIONS:
        answer = answers.get(dim.key)
        if answer is None:
            continue
        cell = _cell(dim, answer, weights[dim.key])
        if dim.applies_if:
            gate = answers.get(dim.applies_if)
            gate_verdict = noul_verdict(gate["noul"]) if gate else "unsure"
            cell.applies = {"yes": True, "no": False, "unsure": None}[gate_verdict]
        row.cells[dim.key] = cell
    row.composite = _composite(row.cells, weights)
    return row


def _cell(dim: Dimension, answer: dict[str, Any], weight: DimensionWeight) -> Cell:
    kind = answer["type"]
    if kind == "noul":
        p = answer["noul"]
        return Cell(key=dim.key, value=p, verdict=noul_verdict(p), goodness=1 - p if dim.yes_is_bad else p)
    if kind == "score":
        probabilities = {int(k): v for k, v in answer["probabilities"].items()}
        levels = len(probabilities)
        expected = answer["score"]
        top = max(probabilities, key=probabilities.__getitem__)
        span = max(levels - 1, 1)
        target = weight.target if weight.target is not None else levels - 1
        return Cell(
            key=dim.key,
            value=expected,
            verdict=str(top),
            levels=levels,
            split=levels >= 3 and probabilities[0] >= 0.3 and probabilities[levels - 1] >= 0.3,
            goodness=1 - abs(expected - target) / span,
        )
    if kind == "choice":
        p_fits = answer["probabilities"].get("fits", 0.0)
        return Cell(key=dim.key, value=p_fits, verdict=answer["choice"], goodness=p_fits)
    raise ValueError(f"unexpected answer type {kind!r} for {dim.key}")


def _composite(cells: dict[str, Cell], weights: dict[str, DimensionWeight]) -> float | None:
    total = weight_sum = 0.0
    for key, cell in cells.items():
        w = weights[key].weight
        if w <= 0 or cell.applies is not True or cell.goodness is None:
            continue
        total += w * cell.goodness
        weight_sum += w
    return round(100 * total / weight_sum, 1) if weight_sum else None
