"""Load prompt/response records from CSV, JSONL, or a directory of text-file pairs."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Record:
    id: str
    user_prompt: str
    response: str


class InputError(ValueError):
    pass


def load_records(path: Path, *, id_field: str = "id", prompt_field: str = "prompt", response_field: str = "response") -> list[Record]:
    """Load records from a .csv, a .jsonl, or a directory of `<id>.prompt.txt` / `<id>.response.txt` pairs.

    CSV and JSONL rows need the prompt and response fields; the id field is optional and
    defaults to the row number.
    """
    if path.is_dir():
        return _load_dir(path)
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    elif path.suffix in (".jsonl", ".ndjson"):
        rows = []
        with path.open(encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise InputError(f"{path}:{n}: invalid JSON: {e}") from e
    else:
        raise InputError(f"{path}: expected a .csv, .jsonl, or a directory of .prompt.txt/.response.txt pairs")
    return _from_rows(path, rows, id_field, prompt_field, response_field)


def _from_rows(path: Path, rows: list[dict], id_field: str, prompt_field: str, response_field: str) -> list[Record]:
    records = []
    for n, row in enumerate(rows, 1):
        missing = [f for f in (prompt_field, response_field) if not isinstance(row.get(f), str) or not row[f].strip()]
        if missing:
            raise InputError(f"{path}: row {n} is missing {', '.join(repr(f) for f in missing)}")
        records.append(Record(id=str(row.get(id_field) or n), user_prompt=row[prompt_field], response=row[response_field]))
    _check_unique(path, records)
    return records


def _load_dir(path: Path) -> list[Record]:
    prompts = {p.name.removesuffix(".prompt.txt"): p for p in path.glob("*.prompt.txt")}
    responses = {p.name.removesuffix(".response.txt"): p for p in path.glob("*.response.txt")}
    unpaired = sorted(prompts.keys() ^ responses.keys())
    if unpaired:
        raise InputError(f"{path}: these ids lack a matching .prompt.txt or .response.txt: {', '.join(unpaired)}")
    if not prompts:
        raise InputError(f"{path}: no <id>.prompt.txt / <id>.response.txt pairs found")
    return [
        Record(id=key, user_prompt=prompts[key].read_text(encoding="utf-8"), response=responses[key].read_text(encoding="utf-8"))
        for key in sorted(prompts)
    ]


def _check_unique(path: Path, records: list[Record]) -> None:
    seen: set[str] = set()
    for r in records:
        if r.id in seen:
            raise InputError(f"{path}: duplicate id {r.id!r}")
        seen.add(r.id)
