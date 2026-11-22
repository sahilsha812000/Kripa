"""Fixed prompt set used for repeatable Model Arena comparisons."""

from typing import NamedTuple, Optional


class ArenaTask(NamedTuple):
    task_id: str
    prompt: str
    evaluation: str
    expected: Optional[object]


TASKS = (
    ArenaTask(
        "reasoning",
        "A library has 24 books on a cart. One third are borrowed, then 5 books are returned. "
        "How many books are on the cart now? Give the answer and briefly show the arithmetic.",
        "exact_answer",
        21,
    ),
    ArenaTask(
        "general_conversation",
        "I'm trying to build a habit of stretching each morning. What's one small step "
        "that could help me remember? Keep it to one sentence.",
        "practical_suggestion",
        {"required_any": ("reminder", "alarm", "note", "calendar", "schedule"), "sentences": 1},
    ),
    ArenaTask(
        "research",
        "Use only this source: [Northfield Library, June] The seed-sharing pilot gave "
        "packets to 34 households. The survey was voluntary and did not measure harvest "
        "size, so the results are not a yield estimate. In one sentence, state the number "
        "of households and the main limitation, and cite the source as [Northfield Library, June].",
        "source_grounded_answer",
        {
            "required_groups": (("34 households",), ("voluntary",), ("harvest size", "yield estimate")),
            "citation": "[Northfield Library, June]",
            "sentences": 1,
        },
    ),
    ArenaTask(
        "summarization",
        "Summarize this notice in exactly two bullet points, preserving the dates and "
        "service changes: Route 8 will detour from Monday through Thursday because Elm "
        "Bridge is closed for repairs. The eastbound stop at Pine Street will move to "
        "Oak Avenue during this period. The westbound route is unchanged, and normal "
        "service is expected Friday.",
        "key_fact_summary",
        {
            "required_groups": (
                ("monday through thursday", "monday to thursday"),
                ("elm bridge",),
                ("eastbound",),
                ("pine street",),
                ("oak avenue",),
                ("westbound",),
                ("unchanged",),
                ("friday",),
            ),
            "bullet_count": 2,
        },
    ),
    ArenaTask(
        "coding",
        "Write a Python function named is_palindrome(text) that returns a boolean, ignores "
        "letter case and spaces, and include one short example call. Keep the code easy to read. "
        "Use a simple function body with assignments and a return statement.",
        "python_tests",
        (("Race car", True), ("Kripa", False), ("", True), ("Never odd or even", True)),
    ),
    ArenaTask(
        "writing",
        "Write exactly three sentences for a warm, professional note encouraging a student "
        "who is learning robotics. Mention practice and progress.",
        "manual_review",
        None,
    ),
    ArenaTask(
        "instruction_following",
        "Return exactly these four lines, with no heading or extra text:\n"
        "KRIPA\n"
        "I can help with reasoning, coding, writing, and instructions.\n"
        "If unsure, I will say so.\n"
        "I keep API keys private.",
        "exact_text",
        "KRIPA\nI can help with reasoning, coding, writing, and instructions.\n"
        "If unsure, I will say so.\nI keep API keys private.",
    ),
)
