"""Render scorecard rows as a text table, CSV, or JSON."""

import csv
import io
import json
from dataclasses import asdict

from jev_eval.questions import DIMENSIONS
from jev_eval.scorecard import DIMENSION_BY_KEY, Cell, Row

LEGEND = """\
Flags: YES >= 0.7, no <= 0.3, ? in between; number = P(yes). YES is a problem, except for Answer up front.
Scores: expected level / top level; * = probability split between the extremes.
Formatting: top option and P(fits). n/a = the prompt didn't ask for a direct answer.
Composite: weighted 0-100 over the dimensions that apply (see weights.example.toml)."""

RULES_LEGEND = """\
Rules engine: each check is a simple word/count rule, so numbers are 0 or 1 (it fired or not).
Blank = rules can't judge it (warmth, lecturing, talking down). Easy to follow = reading ease."""


def _cell_text(cell: Cell | None) -> str:
    if cell is None:
        return ""
    if cell.applies is False:
        return "n/a"
    prefix = "(?) " if cell.applies is None else ""
    if cell.levels is not None:
        return f"{prefix}{cell.value:.1f}/{cell.levels - 1}{'*' if cell.split else ''}"
    if cell.verdict in ("yes", "no", "unsure"):
        mark = {"yes": "YES", "no": "no", "unsure": "?"}[cell.verdict]
        return f"{prefix}{mark} {cell.value:.2f}"
    return f"{prefix}{cell.verdict} {cell.value:.2f}"


def table(rows: list[Row]) -> str:
    headers = ["id", "composite"] + [d.label for d in DIMENSIONS]
    lines = []
    errors = []
    for row in rows:
        stale = " (old questions)" if row.stale else ""
        if row.error:
            lines.append([row.id + stale, "ERROR"] + [""] * len(DIMENSIONS))
            errors.append(f"{row.id}: {row.error}")
            continue
        composite = "" if row.composite is None else f"{row.composite:.1f}"
        lines.append([row.id + stale, composite] + [_cell_text(row.cells.get(d.key)) for d in DIMENSIONS])
    widths = [max(len(h), *(len(line[i]) for line in lines)) if lines else len(h) for i, h in enumerate(headers)]
    out = [
        "  ".join(h.ljust(w) for h, w in zip(headers, widths)).rstrip(),
        "  ".join("-" * w for w in widths),
    ]
    out += ["  ".join(c.ljust(w) for c, w in zip(line, widths)).rstrip() for line in lines]
    if errors:
        out += ["", "Errors (no answers were recorded for these):"] + [f"  {e}" for e in errors]
    why = [f"  {row.id} / {DIMENSION_BY_KEY[c.key].label}: {c.note}" for row in rows for c in row.cells.values() if c.note and _flagged(c)]
    if why:
        out += ["", "Why (what the rules found):"] + why
    legend = LEGEND + ("\n" + RULES_LEGEND if any(row.engine == "rules" for row in rows) else "")
    return "\n".join(out) + "\n\n" + legend + "\n"


def _flagged(cell: Cell) -> bool:
    """A cell worth explaining: it counts against the text."""
    return cell.applies is True and cell.goodness is not None and cell.goodness < 1 and cell.key != "formality"


def to_csv(rows: list[Row]) -> str:
    fields = ["id", "engine", "composite", "error", "stale"]
    for d in DIMENSIONS:
        fields += [d.key, f"{d.key}_verdict", f"{d.key}_goodness", f"{d.key}_note"]
        if d.applies_if:
            fields.append(f"{d.key}_applies")
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        out: dict[str, object] = {"id": row.id, "engine": row.engine, "composite": row.composite, "error": row.error, "stale": row.stale}
        for key, cell in row.cells.items():
            out[key] = None if cell.value is None else round(cell.value, 4)
            out[f"{key}_verdict"] = cell.verdict
            out[f"{key}_goodness"] = None if cell.goodness is None else round(cell.goodness, 4)
            out[f"{key}_note"] = cell.note
            if DIMENSION_BY_KEY[key].applies_if:
                out[f"{key}_applies"] = {True: "yes", False: "no", None: "unsure"}[cell.applies]
        writer.writerow(out)
    return buf.getvalue()


def to_json(rows: list[Row]) -> str:
    return json.dumps([asdict(row) for row in rows], indent=2) + "\n"
