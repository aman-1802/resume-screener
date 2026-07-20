"""One OpenAI call per job description: extract structured screening keywords.

This is the only paid API call in the whole pipeline -- it runs once per JD,
not per resume, so cost stays negligible regardless of batch size.
"""

import json
import os

from openai import OpenAI

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are a technical recruiter's assistant. Given a job description, "
    "extract EVERY concrete, screenable requirement mentioned: specific "
    "languages, frameworks, libraries, tools, platforms, methodologies "
    "(e.g. SDLC, CI/CD, SOLID, design patterns), domain knowledge (e.g. "
    "data structures and algorithms, OS fundamentals, memory management, "
    "multi-threading), certifications, degree/education requirements, and "
    "years-of-experience thresholds. Be exhaustive and granular: if a "
    "sentence names several distinct things (e.g. 'Git and CI/CD pipelines' "
    "or 'Valgrind, gdb, or equivalent tools'), extract each as its own "
    "keyword rather than collapsing them into one bullet. Do not skip items "
    "just because they seem minor -- err on the side of including more, "
    "specific, atomic items over fewer, broad ones. "
    "Split them into 'must_have' (stated as required/strong proficiency/"
    "the 'Required' section) and 'nice_to_have' (stated as preferred/bonus/"
    "'good to have'). Keep each item short (a skill, tool, or requirement "
    "phrase, not a full sentence). "
    'Respond with strict JSON: {"must_have": [...], "nice_to_have": [...]}'
)


def extract_keywords(jd_text: str) -> dict[str, list[str]]:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": jd_text},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    payload = json.loads(response.choices[0].message.content)
    return {
        "must_have": payload.get("must_have", []),
        "nice_to_have": payload.get("nice_to_have", []),
    }
