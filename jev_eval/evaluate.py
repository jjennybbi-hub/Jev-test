"""Send each record to Jev and keep the raw answers.

Raw answers are stored as-is so the scorecard (weights, targets, thresholds) can be recomputed
without calling the API again. A failed call is stored as an error, never as an answer.
"""

import os
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

from typesafe_sdk import Question, SystemOneResponse, TypeSafeError
from typesafe_sdk.constants import DEFAULT_MODEL, DEFAULT_MODEL_ENV

from jev_eval.inputs import Record
from jev_eval.questions import QUESTIONS_VERSION, all_questions


class SystemOneClient(Protocol):
    def system_one(self, state: Any, questions: dict[str, Question], *, model: str | None = None) -> SystemOneResponse: ...


def build_state(record: Record) -> dict[str, str]:
    return {"user_prompt": record.user_prompt, "response": record.response}


def build_payload(record: Record, model: str | None) -> dict[str, Any]:
    """The request body as the SDK sends it, for logging and for rerunning a single case by hand."""
    return {
        "state": build_state(record),
        "model": model or os.environ.get(DEFAULT_MODEL_ENV, "").strip() or DEFAULT_MODEL,
        "questions": {name: q.model_dump(mode="json") for name, q in all_questions().items()},
    }


def evaluate_one(client: SystemOneClient, record: Record, model: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {"id": record.id, "questions_version": QUESTIONS_VERSION, "state": build_state(record)}
    try:
        response = client.system_one(state=build_state(record), questions=all_questions(), model=model)
    except TypeSafeError as e:
        result["error"] = f"{type(e).__name__}: {e}"
        return result
    result["model"] = response.model
    result["usage"] = response.usage.model_dump(mode="json")
    result["answers"] = {name: answer.model_dump(mode="json") for name, answer in response.answers.items()}
    return result


def evaluate_all(
    client: SystemOneClient,
    records: Iterable[Record],
    *,
    model: str | None = None,
    concurrency: int = 4,
) -> Iterator[dict[str, Any]]:
    """Evaluate records concurrently, yielding results in input order."""
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        yield from pool.map(lambda r: evaluate_one(client, r, model), records)
