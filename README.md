# jev-eval

A small command-line tool that scores LLM responses on **tone** and **clarity** and prints a
scorecard for each text.

It can judge the texts in two ways:

| Engine | Cost | Needs | How good |
|---|---|---|---|
| `rules` (default) | Free | Nothing, works offline | Crude: matches words and counts things |
| `jev` | Paid per use | A [TypeSafe](https://typesafe.ai) account and API key | Understands the text |

## Setup

```sh
uv sync
```

## Quick start (free, offline)

```sh
uv run jev-eval run examples/sample.jsonl
```

This scores the 5 example texts, prints a table, and saves the raw results to `results.jsonl`.
Under the table, a **"Why"** section lists what each rule found, for example
`opens with 'Great question'`.

```sh
# Rebuild the table from saved results, sorted best-first
uv run jev-eval report results.jsonl --sort composite

# Save it as a spreadsheet
uv run jev-eval report results.jsonl --format csv > scorecard.csv
```

### Your own texts

Each record is a user prompt plus the LLM's response:

- **JSONL**: one object per line with `prompt` and `response`, and optionally `id`.
  See `examples/sample.jsonl`.
- **CSV**: columns `prompt` and `response`, and optionally `id`.
- **Directory**: `<id>.prompt.txt` and `<id>.response.txt` pairs.

To use different field names, pass `--prompt-field`, `--response-field` or `--id-field`.

## What gets checked

| Group | Dimension | `rules` engine | `jev` engine |
|---|---|---|---|
| Tone | Warmth | — can't judge | Score: cold / matter-of-fact / warm |
| Tone | Formality | casual vs. formal words, contractions | Score: casual / conversational / formal |
| Tone | Lectures/moralizes | — can't judge | yes/no |
| Tone | Talks down | — can't judge | yes/no |
| Tone | Unneeded apology | any "sorry" or "apologize" | yes/no: apologizes when nothing went wrong |
| Tone | Excess disclaimers | phrases like "I'm not a lawyer" or "consult a professional" | yes/no |
| Tone | Excess hedging | too many words like "might", "perhaps" or "possibly" | yes/no |
| Clarity | Answer up front | opens with "Great question!", "Sure!" and the like | yes/no |
| Clarity | Conciseness | filler like "I hope this helps", repeated sentences | Score: padded / some filler / tight |
| Clarity | Easy to follow | reading-ease score (shorter words and sentences score higher) | Score: confusing / needs re-reading / clear |
| Clarity | Formatting fits | heavy formatting on a short answer, steps with no list | over-formatted / fits / under-formatted |

"Answer up front" only counts when the prompt asks a question. For a request like "write a
poem", it shows `n/a`.

**Know the limits of `rules`:** it only matches words. It will miss a preachy answer that
doesn't use the phrases it knows, and it will flag a "sorry" even when an apology was deserved.
Use its results as hints, and read the "Why" notes. To change a phrase list or a cut-off, edit
[`jev_eval/rules.py`](jev_eval/rules.py).

### Reading the scorecard

- **Yes/no checks:** YES means the problem was found, except for "Answer up front", where YES
  is good.
- **Scores:** shown as `level/2`, where 2 is best. Formality is the exception: there, 2 just
  means formal.
- **Composite:** a weighted average from 0 to 100, counting only the dimensions that were
  checked. To change the weights, copy [`weights.example.toml`](weights.example.toml) to
  `weights.toml`, edit it, and pass `--weights weights.toml`.

## Optional: the `jev` engine (paid)

This sends each prompt and response to TypeSafe's servers. Check their pricing and privacy
policy first, and don't send private data you aren't allowed to share.

```sh
export TYPESAFE_API_KEY=...   # from your TypeSafe account; never commit it
uv run jev-eval payload examples/sample.jsonl            # see exactly what would be sent, without sending it
uv run jev-eval run examples/sample.jsonl --engine jev
```

The questions sent to Jev are in [`jev_eval/questions.py`](jev_eval/questions.py). Jev
answers with probabilities. A yes/no answer above 0.7 counts as YES, below 0.3 as no, and
anything in between shows as `?`. A failed API call is recorded as an error, never as an
answer. The Jev question wording hasn't been tested against the live API yet.

## Tests

```sh
uv run pytest
```

No test calls the API or costs anything.
