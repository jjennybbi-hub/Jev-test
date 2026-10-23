"""Every question Jev is asked, in one place.

This module is the part of the tool people audit and edit most: question wording, criteria,
options, and the thresholds used to read the answers. Change a question here and bump
QUESTIONS_VERSION so stored results can be told apart.

Each record is sent as one request whose state is:

    {"user_prompt": "<what the user asked>", "response": "<the LLM output being evaluated>"}

All questions share that state, so they go in a single request and are answered in parallel.
Each question asks for one judgment a person could make at a glance; combining them into a
scorecard happens in code (see scorecard.py).
"""

from dataclasses import dataclass
from typing import Literal

from typesafe_sdk import Choice, Noul, Question, Score

QUESTIONS_VERSION = "2026-10-23.1"

# A Noul above YES_THRESHOLD reads as yes, below NO_THRESHOLD as no; between them the state
# doesn't settle the question and the scorecard says "unsure".
YES_THRESHOLD = 0.7
NO_THRESHOLD = 0.3


@dataclass(frozen=True)
class Dimension:
    """How one question appears on the scorecard."""

    key: str
    group: Literal["tone", "clarity"]
    label: str
    question: Question
    # For a Noul flag: True when a yes is the problem (e.g. "apologizes unnecessarily").
    yes_is_bad: bool = False
    # Answers only meaningful when another question's answer is yes (speculative fan-out):
    # the key of that gating Noul.
    applies_if: str | None = None


# --- Gating question (not shown as a dimension; it decides whether another one applies) -----

GATES: dict[str, Question] = {
    "prompt_asks_for_answer": Noul(
        instructions="Does `user_prompt` ask for a specific answer, fact, or recommendation?",
        criteria={
            "true": "The user asks something with a direct answer, such as 'What port does Postgres use?' "
            "or 'Should I use a list or a tuple here?'",
            "false": "The user asks for open-ended content, such as a poem, a story, a rewrite of their text, "
            "or a brainstorm.",
        },
    ),
}


# --- Tone & sentiment -------------------------------------------------------------------------

TONE = [
    Dimension(
        key="warmth",
        group="tone",
        label="Warmth",
        question=Score(
            instructions="How warm toward the user is `response`?",
            criteria=[
                "The response is curt or cold, as if the user were an interruption.",
                "The response is matter-of-fact: polite, but with no personal warmth.",
                "The response is friendly and encouraging toward the user.",
            ],
        ),
    ),
    Dimension(
        key="formality",
        group="tone",
        label="Formality",
        question=Score(
            instructions="How formal is the language of `response`?",
            criteria=[
                "The response is casual, with slang, contractions, or jokes, like a chat with a friend.",
                "The response is conversational but tidy, like a helpful colleague's message.",
                "The response is formal, like a business letter or official documentation.",
            ],
        ),
    ),
    Dimension(
        key="lectures",
        group="tone",
        label="Lectures/moralizes",
        yes_is_bad=True,
        question=Noul(
            instructions="Does `response` lecture or moralize about something `user_prompt` did not ask about?",
            criteria={
                "true": "The response adds unrequested ethical advice or tells the user how they ought to behave.",
                "false": "The response sticks to what the user asked; any warning it gives is one the user needs "
                "to act on the answer safely.",
            },
        ),
    ),
    Dimension(
        key="talks_down",
        group="tone",
        label="Talks down",
        yes_is_bad=True,
        question=Noul(
            instructions="Does `response` talk down to the user?",
            criteria={
                "true": "The response implies the user is ignorant or slow, for example 'As anyone knows...' or "
                "explaining basics the prompt shows they already understand.",
                "false": "The response treats the user as a capable adult at the level their prompt shows.",
            },
        ),
    ),
    Dimension(
        key="unneeded_apology",
        group="tone",
        label="Unneeded apology",
        yes_is_bad=True,
        question=Noul(
            instructions="Does `response` apologize when nothing went wrong?",
            criteria={
                "true": "The response says sorry or apologizes even though it made no mistake and is not "
                "declining anything.",
                "false": "The response contains no apology, or apologizes for a real mistake or a refusal.",
            },
        ),
    ),
    Dimension(
        key="excess_disclaimers",
        group="tone",
        label="Excess disclaimers",
        yes_is_bad=True,
        question=Noul(
            instructions="Does `response` include disclaimers the user does not need?",
            criteria={
                "true": "The response adds statements like 'I am not a lawyer' or 'consult a professional' to a "
                "question where they add nothing.",
                "false": "The response has no disclaimers, or only ones that matter for acting on its answer.",
            },
        ),
    ),
    Dimension(
        key="excess_hedging",
        group="tone",
        label="Excess hedging",
        yes_is_bad=True,
        question=Noul(
            instructions="Does `response` hedge claims it could state plainly?",
            criteria={
                "true": "The response wraps well-established facts in 'might', 'possibly', or 'it could be "
                "argued' so that its answer is hard to pin down.",
                "false": "The response states settled facts plainly and qualifies only what is uncertain.",
            },
        ),
    ),
]


# --- Clarity & structure ----------------------------------------------------------------------

CLARITY = [
    Dimension(
        key="answer_up_front",
        group="clarity",
        label="Answer up front",
        applies_if="prompt_asks_for_answer",
        question=Noul(
            instructions="Do the first two sentences of `response` give the answer to `user_prompt`?",
            criteria={
                "true": "The response opens with the answer, and any explanation follows it.",
                "false": "The response opens with a restatement of the question, background, or preamble, and the "
                "answer comes later or not at all.",
            },
        ),
    ),
    Dimension(
        key="conciseness",
        group="clarity",
        label="Conciseness",
        question=Score(
            instructions="How concise is `response` for what `user_prompt` asked?",
            criteria=[
                "The response is padded: repetition, filler, or restated points make it far longer than needed.",
                "The response has a few sentences of filler or repetition that could be cut.",
                "Every sentence of the response contributes something the user needs.",
            ],
        ),
    ),
    Dimension(
        key="easy_to_follow",
        group="clarity",
        label="Easy to follow",
        question=Score(
            instructions="How easy is `response` to follow for the person who wrote `user_prompt`?",
            criteria=[
                "The reader would get lost: ideas jump around, or key terms go unexplained.",
                "The reader can follow it with some re-reading.",
                "The reader can follow it on one pass; each point leads clearly to the next.",
            ],
        ),
    ),
    Dimension(
        key="formatting",
        group="clarity",
        label="Formatting fits",
        question=Choice(
            instructions="How well does the formatting of `response` fit its content?",
            criteria={
                "over_formatted": "Headings, bullet points, or bold text break up content that would read better "
                "as a few plain sentences.",
                "fits": "The formatting suits the content: prose for short answers, and lists, tables, or code "
                "blocks where they make the content easier to scan or copy.",
                "under_formatted": "Content such as steps, code, or a comparison is run together in paragraphs "
                "where a list, table, or code block would make it easier to use.",
            },
        ),
    ),
]

DIMENSIONS: list[Dimension] = TONE + CLARITY


def all_questions() -> dict[str, Question]:
    """The full question set for one request: every dimension plus the gates they depend on."""
    return {**GATES, **{d.key: d.question for d in DIMENSIONS}}
