import json
from pathlib import Path

import httpx2
import pytest
from typesafe_sdk import RetryPolicy, TypeSafeClient

from jev_eval import cli
from jev_eval.evaluate import build_payload, evaluate_all
from jev_eval.inputs import InputError, Record, load_records
from jev_eval.questions import DIMENSIONS, QUESTIONS_VERSION, all_questions
from jev_eval.rules import evaluate_one_rules
from jev_eval.scorecard import DimensionWeight, load_weights, score_row

EXAMPLES = Path(__file__).parent.parent / "examples"


def noul(p):
    return {"type": "noul", "noul": p}


def score(probabilities):
    expected = sum(level * p for level, p in enumerate(probabilities))
    return {
        "type": "score",
        "score": expected,
        "confidence": max(probabilities),
        "legend": {str(i): f"level {i}" for i in range(len(probabilities))},
        "probabilities": {str(i): p for i, p in enumerate(probabilities)},
    }


def choice(probabilities):
    return {"type": "choice", "choice": max(probabilities, key=probabilities.get), "confidence": max(probabilities.values()), "probabilities": probabilities}


GOOD_ANSWERS = {
    "prompt_asks_for_answer": noul(0.95),
    "warmth": score([0.0, 0.2, 0.8]),
    "formality": score([0.1, 0.8, 0.1]),
    "lectures": noul(0.02),
    "talks_down": noul(0.03),
    "unneeded_apology": noul(0.01),
    "excess_disclaimers": noul(0.05),
    "excess_hedging": noul(0.04),
    "answer_up_front": noul(0.97),
    "conciseness": score([0.0, 0.1, 0.9]),
    "easy_to_follow": score([0.0, 0.1, 0.9]),
    "formatting": choice({"over_formatted": 0.05, "fits": 0.9, "under_formatted": 0.05}),
}


def fake_client(answers=GOOD_ANSWERS, status=200, seen=None):
    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        if seen is not None:
            seen.append(body)
        if status != 200:
            return httpx2.Response(status, json={"detail": "boom"})
        return httpx2.Response(200, json={"model": body["model"], "usage": {"input_tokens": 100, "output_tokens": 12}, "answers": answers})

    return TypeSafeClient(api_key="test-key", transport=httpx2.MockTransport(handler), compression=False, retry=RetryPolicy(max_retries=0))


RECORD = Record(id="a", user_prompt="What port does Postgres use?", response="5432.")


def result(answers, version=QUESTIONS_VERSION):
    return {"id": "a", "questions_version": version, "answers": answers}


# --- inputs ---------------------------------------------------------------------------------


def test_load_jsonl_example():
    records = load_records(EXAMPLES / "sample.jsonl")
    assert [r.id for r in records] == ["direct", "preamble", "apologetic", "preachy", "poem"]


def test_load_csv_with_custom_fields(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text("q,a\nhello?,hi there\n")
    [record] = load_records(path, prompt_field="q", response_field="a")
    assert record == Record(id="1", user_prompt="hello?", response="hi there")


def test_load_dir_pairs(tmp_path):
    (tmp_path / "x.prompt.txt").write_text("p")
    (tmp_path / "x.response.txt").write_text("r")
    assert load_records(tmp_path) == [Record(id="x", user_prompt="p", response="r")]
    (tmp_path / "y.prompt.txt").write_text("p")
    with pytest.raises(InputError, match="y"):
        load_records(tmp_path)


def test_missing_response_is_an_error(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text('{"prompt": "hi"}\n')
    with pytest.raises(InputError, match="row 1"):
        load_records(path)


def test_duplicate_ids_rejected(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text('{"id": "a", "prompt": "p", "response": "r"}\n{"id": "a", "prompt": "p", "response": "r"}\n')
    with pytest.raises(InputError, match="duplicate"):
        load_records(path)


# --- request ----------------------------------------------------------------------------------


def test_one_request_per_record_with_all_questions_and_named_state():
    seen = []
    with fake_client(seen=seen) as client:
        [out] = list(evaluate_all(client, [RECORD], model="jev-test"))
    [body] = seen
    assert body["state"] == {"user_prompt": RECORD.user_prompt, "response": RECORD.response}
    assert set(body["questions"]) == set(all_questions())
    assert body["model"] == "jev-test"
    assert body == build_payload(RECORD, "jev-test")
    assert out["answers"]["warmth"]["probabilities"] == GOOD_ANSWERS["warmth"]["probabilities"]


def test_api_failure_is_stored_as_error_not_answer():
    with fake_client(status=500) as client:
        [out] = list(evaluate_all(client, [RECORD]))
    assert "error" in out and "answers" not in out
    row = score_row(out, load_weights(None))
    assert row.error and row.composite is None and not row.cells


# --- scorecard --------------------------------------------------------------------------------


def test_good_response_scores_high():
    row = score_row(result(GOOD_ANSWERS), load_weights(None))
    assert row.composite > 85
    assert row.cells["lectures"].verdict == "no"
    assert row.cells["answer_up_front"].applies is True


def test_bad_flag_lowers_composite():
    weights = load_weights(None)
    bad = {**GOOD_ANSWERS, "unneeded_apology": noul(0.95)}
    assert score_row(result(bad), weights).composite < score_row(result(GOOD_ANSWERS), weights).composite
    assert score_row(result(bad), weights).cells["unneeded_apology"].verdict == "yes"


@pytest.mark.parametrize(("gate", "applies"), [(0.9, True), (0.1, False), (0.5, None)])
def test_answer_up_front_gated_on_prompt_type(gate, applies):
    answers = {**GOOD_ANSWERS, "prompt_asks_for_answer": noul(gate), "answer_up_front": noul(0.05)}
    weights = load_weights(None)
    row = score_row(result(answers), weights)
    assert row.cells["answer_up_front"].applies is applies
    without = score_row(result({k: v for k, v in answers.items() if k != "answer_up_front"}), weights)
    # Only an applicable dimension counts toward the composite.
    assert (row.composite != without.composite) is (applies is True)


def test_noul_unsure_band():
    row = score_row(result({**GOOD_ANSWERS, "talks_down": noul(0.5)}), load_weights(None))
    assert row.cells["talks_down"].verdict == "unsure"


def test_score_target_and_split():
    weights = load_weights(None)
    weights["formality"] = DimensionWeight(weight=1, target=0)
    casual = score_row(result({**GOOD_ANSWERS, "formality": score([0.9, 0.1, 0.0])}), weights)
    formal = score_row(result({**GOOD_ANSWERS, "formality": score([0.0, 0.1, 0.9])}), weights)
    assert casual.cells["formality"].goodness > formal.cells["formality"].goodness
    split = score_row(result({**GOOD_ANSWERS, "warmth": score([0.45, 0.1, 0.45])}), weights)
    assert split.cells["warmth"].split and not casual.cells["warmth"].split


def test_formality_unweighted_by_default():
    weights = load_weights(None)
    a = score_row(result({**GOOD_ANSWERS, "formality": score([1.0, 0.0, 0.0])}), weights)
    b = score_row(result({**GOOD_ANSWERS, "formality": score([0.0, 0.0, 1.0])}), weights)
    assert a.composite == b.composite


def test_weights_file(tmp_path):
    path = tmp_path / "w.toml"
    path.write_text("[dimensions.formality]\nweight = 2\n")
    weights = load_weights(path)
    assert weights["formality"] == DimensionWeight(weight=2, target=1)
    path.write_text("[dimensions.nope]\nweight = 2\n")
    with pytest.raises(ValueError, match="unknown dimension"):
        load_weights(path)


def test_example_weights_file_loads():
    weights = load_weights(Path(__file__).parent.parent / "weights.example.toml")
    assert set(weights) == {d.key for d in DIMENSIONS}


def test_stale_question_version_flagged():
    assert score_row(result(GOOD_ANSWERS, version="old"), load_weights(None)).stale


# --- CLI --------------------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["table", "csv", "json"])
def test_cli_run_then_report(tmp_path, monkeypatch, capsys, fmt):
    monkeypatch.setattr(cli, "TypeSafeClient", lambda: fake_client())
    out = tmp_path / "results.jsonl"
    assert cli.main(["run", str(EXAMPLES / "sample.jsonl"), "--engine", "jev", "-o", str(out), "--format", fmt]) == 0
    assert len(out.read_text().splitlines()) == 5
    first = capsys.readouterr().out
    assert "preachy" in first
    assert cli.main(["report", str(out), "--format", fmt, "--sort", "composite"]) == 0
    assert "preachy" in capsys.readouterr().out


def test_cli_run_exits_nonzero_on_api_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "TypeSafeClient", lambda: fake_client(status=500))
    out = tmp_path / "results.jsonl"
    assert cli.main(["run", str(EXAMPLES / "sample.jsonl"), "--engine", "jev", "-o", str(out)]) == 1
    assert "ERROR" in capsys.readouterr().out


def test_cli_payload_sends_nothing(capsys):
    assert cli.main(["payload", str(EXAMPLES / "sample.jsonl"), "--model", "jev-x"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(lines) == 5 and lines[0]["request"]["model"] == "jev-x"


def test_cli_bad_sort_key(tmp_path, capsys):
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps(result(GOOD_ANSWERS)) + "\n")
    assert cli.main(["report", str(path), "--sort", "nope"]) == 2


# --- rules engine (offline) -------------------------------------------------------------------


def rules_row(prompt, response):
    return score_row(evaluate_one_rules(Record(id="r", user_prompt=prompt, response=response)), load_weights(None))


def test_rules_flags_apology_disclaimer_and_preamble():
    row = rules_row(
        "What is 3 miles in km?",
        "Great question! I'm sorry, I'm not a certified expert, but it's about 4.83 km. I hope this helps!",
    )
    assert row.cells["unneeded_apology"].verdict == "yes"
    assert row.cells["excess_disclaimers"].verdict == "yes"
    assert row.cells["answer_up_front"].verdict == "no" and row.cells["answer_up_front"].applies is True
    assert row.cells["conciseness"].value < 2
    assert "sorry" in row.cells["unneeded_apology"].note


def test_rules_clean_answer_scores_full_marks():
    row = rules_row("What port does Postgres use?", "5432. You can change it in postgresql.conf.")
    assert row.composite == 100.0
    assert not any(row.cells[k].verdict == "yes" for k in ("unneeded_apology", "excess_disclaimers", "excess_hedging"))


def test_rules_skip_what_they_cannot_judge_and_gate_non_questions():
    row = rules_row("Write a two-line poem about autumn.", "Leaves let go without a sound,\nand gold comes gently to the ground.")
    assert {"warmth", "lectures", "talks_down"}.isdisjoint(row.cells)
    assert row.cells["answer_up_front"].applies is False
    assert row.engine == "rules" and not row.stale


def test_rules_hedging_and_formatting():
    hedgy = rules_row("Is Paris the capital of France?", "It might possibly be Paris, perhaps, though it could be argued otherwise.")
    assert hedgy.cells["excess_hedging"].verdict == "yes"
    heavy = rules_row("Hi?", "# Answer\n\n- **Yes**\n- **Indeed**\n- **Sure**")
    assert heavy.cells["formatting"].verdict == "over_formatted"
    steps = rules_row("How do I make tea?", "First boil water. Then add the tea bag. Next wait three minutes. Finally remove the bag.")
    assert steps.cells["formatting"].verdict == "under_formatted"


def test_cli_defaults_to_rules_without_api_key(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(cli, "TypeSafeClient", lambda: pytest.fail("rules engine must not create an API client"))
    out = tmp_path / "results.jsonl"
    assert cli.main(["run", str(EXAMPLES / "sample.jsonl"), "-o", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "Why (what the rules found)" in printed and "Great question" in printed
    assert cli.main(["report", str(out), "--format", "csv"]) == 0
    assert "unneeded_apology_note" in capsys.readouterr().out
