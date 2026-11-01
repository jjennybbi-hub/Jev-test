"""jev-eval: score LLM responses on tone and clarity, offline with simple rules (free) or with TypeSafe's Jev."""

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

from typesafe_sdk import TypeSafeClient, TypeSafeError

from jev_eval import render
from jev_eval.evaluate import build_payload, evaluate_all
from jev_eval.inputs import InputError, load_records
from jev_eval.questions import DIMENSIONS
from jev_eval.rules import evaluate_one_rules
from jev_eval.scorecard import load_weights, score_row

FORMATS = {"table": render.table, "csv": render.to_csv, "json": render.to_json}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.handler(args)
    except (InputError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jev-eval", description=__doc__)
    sub = parser.add_subparsers(required=True)

    def add_input(p: argparse.ArgumentParser) -> None:
        p.add_argument("input", type=Path, help=".csv, .jsonl, or a directory of <id>.prompt.txt/<id>.response.txt pairs")
        p.add_argument("--id-field", default="id")
        p.add_argument("--prompt-field", default="prompt")
        p.add_argument("--response-field", default="response")

    def add_report_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--weights", type=Path, help="TOML file of dimension weights and targets")
        p.add_argument("--format", choices=FORMATS, default="table")
        p.add_argument("--sort", help="sort by 'composite' or a dimension key (best first)")

    run = sub.add_parser("run", help="evaluate texts, save raw answers, print the scorecard")
    add_input(run)
    run.add_argument(
        "--engine",
        choices=["rules", "jev"],
        default="rules",
        help="rules: free, offline word/count checks (default). jev: TypeSafe's API; needs TYPESAFE_API_KEY and may cost money",
    )
    run.add_argument("-o", "--output", type=Path, default=Path("results.jsonl"), help="raw answers (default: results.jsonl)")
    run.add_argument("--model", help="jev engine only: model name (default: the SDK's default, or TYPESAFE_DEFAULT_MODEL)")
    run.add_argument("--concurrency", type=int, default=4)
    add_report_options(run)
    run.set_defaults(handler=_run)

    report = sub.add_parser("report", help="rebuild the scorecard from saved answers, no API calls")
    report.add_argument("results", type=Path, help="results.jsonl written by 'run'")
    add_report_options(report)
    report.set_defaults(handler=_report)

    payload = sub.add_parser("payload", help="jev engine: print the exact API request for each text, without sending it")
    add_input(payload)
    payload.add_argument("--model")
    payload.set_defaults(handler=_payload)
    return parser


def _load(args: argparse.Namespace):
    return load_records(args.input, id_field=args.id_field, prompt_field=args.prompt_field, response_field=args.response_field)


def _run(args: argparse.Namespace) -> int:
    records = _load(args)
    weights = load_weights(args.weights)  # fail on a bad config before spending API calls
    if args.engine == "rules":
        client = nullcontext()
        results_iter = map(evaluate_one_rules, records)
    else:
        try:
            client = TypeSafeClient()
        except TypeSafeError as e:
            print(f"error: {e} (set TYPESAFE_API_KEY)", file=sys.stderr)
            return 2
        results_iter = evaluate_all(client, records, model=args.model, concurrency=args.concurrency)
    results = []
    with client, args.output.open("w", encoding="utf-8") as out:
        for n, result in enumerate(results_iter, 1):
            out.write(json.dumps(result) + "\n")
            out.flush()
            status = "ERROR " + result["error"] if "error" in result else "ok"
            print(f"[{n}/{len(records)}] {result['id']}: {status}", file=sys.stderr)
            results.append(result)
    print(f"raw answers saved to {args.output}", file=sys.stderr)
    _print_report(results, weights, args)
    return 1 if any("error" in r for r in results) else 0


def _report(args: argparse.Namespace) -> int:
    with args.results.open(encoding="utf-8") as f:
        results = [json.loads(line) for line in f if line.strip()]
    _print_report(results, load_weights(args.weights), args)
    return 0


def _payload(args: argparse.Namespace) -> int:
    for record in _load(args):
        print(json.dumps({"id": record.id, "request": build_payload(record, args.model)}))
    return 0


def _print_report(results: list[dict], weights, args: argparse.Namespace) -> None:
    rows = [score_row(r, weights) for r in results]
    if args.sort:
        rows = _sorted(rows, args.sort)
    if any(row.stale for row in rows):
        print("warning: some results were answered by an older question set; re-run them for comparable scores", file=sys.stderr)
    sys.stdout.write(FORMATS[args.format](rows))


def _sorted(rows, key: str):
    keys = {"composite"} | {d.key for d in DIMENSIONS}
    if key not in keys:
        raise ValueError(f"--sort must be one of: {', '.join(sorted(keys))}")

    def sort_value(row) -> float:
        if key == "composite":
            value = row.composite
        else:
            cell = row.cells.get(key)
            value = cell.goodness if cell and cell.applies is True else None
        return float("-inf") if value is None else value

    return sorted(rows, key=sort_value, reverse=True)


if __name__ == "__main__":
    sys.exit(main())
