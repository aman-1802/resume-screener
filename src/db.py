"""SQLite persistence for job descriptions and screening results."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "app.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS jds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role_title TEXT NOT NULL,
    source_filename TEXT NOT NULL,
    extracted_text TEXT NOT NULL,
    must_have TEXT NOT NULL,
    nice_to_have TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS screening_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    jd_id INTEGER NOT NULL REFERENCES jds(id),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS candidate_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES screening_runs(id),
    candidate_name TEXT NOT NULL,
    resume_filename TEXT NOT NULL,
    resume_path TEXT NOT NULL,
    score REAL NOT NULL,
    matched_keywords TEXT NOT NULL,
    missing_keywords TEXT NOT NULL
);
"""


@contextmanager
def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def _now():
    return datetime.now(timezone.utc).isoformat()


# ---- JDs ----

def add_jd(role_title: str, source_filename: str, extracted_text: str,
           must_have: list[str], nice_to_have: list[str]) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO jds (role_title, source_filename, extracted_text, "
            "must_have, nice_to_have, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (role_title, source_filename, extracted_text,
             json.dumps(must_have), json.dumps(nice_to_have), _now()),
        )
        return cur.lastrowid


def list_jds() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, role_title, source_filename, created_at FROM jds "
            "ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_jd(jd_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM jds WHERE id = ?", (jd_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["must_have"] = json.loads(d["must_have"])
        d["nice_to_have"] = json.loads(d["nice_to_have"])
        return d


def update_jd_keywords(jd_id: int, must_have: list[str], nice_to_have: list[str]):
    with get_conn() as conn:
        conn.execute(
            "UPDATE jds SET must_have = ?, nice_to_have = ? WHERE id = ?",
            (json.dumps(must_have), json.dumps(nice_to_have), jd_id),
        )


# ---- Screening runs ----

def create_run(jd_id: int) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO screening_runs (jd_id, created_at) VALUES (?, ?)",
            (jd_id, _now()),
        )
        return cur.lastrowid


def add_candidate_score(run_id: int, candidate_name: str, resume_filename: str,
                         resume_path: str, score: float,
                         matched_keywords: list[str], missing_keywords: list[str]):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO candidate_scores (run_id, candidate_name, resume_filename, "
            "resume_path, score, matched_keywords, missing_keywords) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, candidate_name, resume_filename, resume_path, score,
             json.dumps(matched_keywords), json.dumps(missing_keywords)),
        )


def get_candidates_for_jd(jd_id: int) -> list[dict]:
    """All candidates ever screened against this JD, across every upload
    batch, as one combined ranked list -- new uploads simply append."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT cs.* FROM candidate_scores cs "
            "JOIN screening_runs sr ON cs.run_id = sr.id "
            "WHERE sr.jd_id = ? ORDER BY cs.score DESC",
            (jd_id,),
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["matched_keywords"] = json.loads(d["matched_keywords"])
            d["missing_keywords"] = json.loads(d["missing_keywords"])
            results.append(d)
        return results
