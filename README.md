# jev-eval

A small CLI that scores LLM responses on **tone** and **clarity** using
[TypeSafe](https://typesafe.ai)'s Jev model, and prints a per-text scorecard.

Jev answers one simple question per dimension, for example "Does `response` apologize when
nothing went wrong?" It returns calibrated probabilities, not free text. Code turns those
answers into the scorecard and the weighted composite, so you can change the weights without
calling the API again.

## Setup

```sh
uv sync                       # installs typesafe-sdk (the official SDK) and this CLI
export TYPESAFE_API_KEY=...   # your key; never commit it
```

## Usage

```sh
# Score texts and save the raw answers to results.jsonl
uv run jev-eval run examples/sample.jsonl -o results.jsonl

# Rebuild the scorecard from saved answers with your own weights, sorted best-first (no API calls)
cp weights.example.toml weights.toml
uv run jev-eval report results.jsonl --weights weights.toml --sort composite
uv run jev-eval report results.jsonl --format csv > scorecard.csv

# Print the exact request body for each text, without sending anything
uv run jev-eval payload examples/sample.jsonl
```

### Input

Each record is a user prompt plus the LLM's response:

- **JSONL**: one object per line with `prompt` and `response`, and optionally `id`.
- **CSV**: columns `prompt` and `response`, and optionally `id`.
- **Directory**: `<id>.prompt.txt` and `<id>.response.txt` pairs.

To use different field names, pass `--prompt-field`, `--response-field` or `--id-field`.

## What gets evaluated

Each text is one API request. The state is `{"user_prompt": ..., "response": ...}`, and all of
the questions below go in that request. They are defined in [`jev_eval/questions.py`](jev_eval/questions.py),
together with their wording, criteria and thresholds. Edit that file to change them.

| Group | Dimension | Type | Good outcome |
|---|---|---|---|
| Tone | Warmth | Score: cold / matter-of-fact / warm | the `target` level (default: warm) |
| Tone | Formality | Score: casual / conversational / formal | the `target` level (unweighted by default) |
| Tone | Lectures/moralizes | Noul | no |
| Tone | Talks down | Noul | no |
| Tone | Unneeded apology | Noul | no |
| Tone | Excess disclaimers | Noul | no |
| Tone | Excess hedging | Noul | no |
| Clarity | Answer up front | Noul, applies only when the prompt asks for a direct answer | yes |
| Clarity | Conciseness | Score: padded / some filler / tight | tight |
| Clarity | Easy to follow | Score: confusing / needs re-reading / clear | clear |
| Clarity | Formatting fits | Choice: over-formatted / fits / under-formatted | fits |

Each question asks for a single judgment. For example, "hedging or over-apology" is three
separate questions (apology, disclaimers, hedging), so one doesn't blur into another.
"Answer up front" only makes sense when the prompt asks for an answer, so a separate question
in the same request checks that. When the prompt asks for something like a poem, the
scorecard shows `n/a` and leaves that dimension out of the composite.

### Reading the scorecard

- **Yes/no flags:** above 0.7 is YES, below 0.3 is no, and anything in between is `?`,
  meaning the text doesn't settle the question. An unsure "does this apply?" check shows as
  `(?)`, and that dimension is left out of the composite.
- **Scores:** the expected level, such as `1.6/2`. A `*` means the probability is split between
  the two extremes, which an average would hide. Look at the raw answers for those texts.
- **Composite:** a weighted average from 0 to 100 of each dimension's "goodness", counting only
  dimensions that apply. Set weights and target levels in a TOML file (see
  [`weights.example.toml`](weights.example.toml)).
- **Failed API calls** are recorded as errors, never as answers. `run` exits with status 1 if
  any call failed.
- Each saved result records the question-set version. If you edit the questions, `report` warns
  about results answered by the older set.

## Tests

```sh
uv run pytest
```

The tests run the real SDK against a fake HTTP transport, so they make no API calls.

## Status

The question wording hasn't been tested against the live API yet. Before relying on the
scores, run it on a few representative, hard and clearly negative examples. Read the
individual answers, and refine the wording or state in `questions.py`.
