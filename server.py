"""CV Screening Pipeline — Flask entry point."""

import csv
import io
import threading
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from flask import Flask, Response, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from src import db, extraction, keyword_extraction, matching, name_extraction
from src.ocr_client import UmiOcrUnavailable

BASE_DIR = Path(__file__).resolve().parent
JD_SOURCE_DIR = BASE_DIR / "data" / "jd_sources"
RESUME_DIR = BASE_DIR / "data" / "resumes"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB per request
db.init_db()

# In-memory progress tracking for background screening jobs. Fine for a
# single-process deployment; a job disappears if the server restarts
# mid-run, which just means the poller shows "unknown" and stops.
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


@app.route("/")
def index():
    return redirect(url_for("dashboard"))


# ---- Job Descriptions ----

@app.route("/job-descriptions")
def job_descriptions():
    jds = [db.get_jd(j["id"]) for j in db.list_jds()]
    return render_template(
        "job_descriptions.html",
        active_page="job_descriptions",
        jds=jds,
        error=request.args.get("error"),
    )


@app.route("/job-descriptions/add", methods=["POST"])
def add_job_description():
    role_title = request.form.get("role_title", "").strip()
    jd_file = request.files.get("jd_file")

    if not role_title or not jd_file or not jd_file.filename:
        return redirect(url_for("job_descriptions", error="Role title and a JD PDF are both required."))

    safe_name = secure_filename(jd_file.filename)
    if not safe_name.lower().endswith(".pdf"):
        return redirect(url_for("job_descriptions", error="Job description must be a PDF file."))

    JD_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = JD_SOURCE_DIR / safe_name
    jd_file.save(pdf_path)

    try:
        jd_text = extraction.extract_text(pdf_path)
    except UmiOcrUnavailable as exc:
        return redirect(url_for("job_descriptions", error=str(exc)))

    if not jd_text.strip():
        return redirect(url_for("job_descriptions", error="No text could be extracted from this PDF."))

    try:
        keywords = keyword_extraction.extract_keywords(jd_text)
    except Exception as exc:  # noqa: BLE001 - surface any API error to the user
        return redirect(url_for("job_descriptions", error=f"Keyword extraction failed: {exc}"))

    db.add_jd(
        role_title=role_title,
        source_filename=safe_name,
        extracted_text=jd_text,
        must_have=keywords["must_have"],
        nice_to_have=keywords["nice_to_have"],
    )
    return redirect(url_for("job_descriptions"))


@app.route("/job-descriptions/<int:jd_id>/keywords", methods=["POST"])
def update_job_description_keywords(jd_id):
    must_have = [line.strip() for line in request.form.get("must_have", "").splitlines() if line.strip()]
    nice_to_have = [line.strip() for line in request.form.get("nice_to_have", "").splitlines() if line.strip()]
    db.update_jd_keywords(jd_id, must_have, nice_to_have)
    return redirect(url_for("job_descriptions"))


# ---- Upload Resumes ----

@app.route("/upload-resumes")
def upload_resumes():
    jds = db.list_jds()
    jd_id = request.args.get("jd_id", type=int) or (jds[0]["id"] if jds else None)
    selected_jd = db.get_jd(jd_id) if jd_id else None

    return render_template(
        "upload_resumes.html",
        active_page="upload_resumes",
        jds=jds,
        selected_jd=selected_jd,
        uploaded_count=request.args.get("uploaded", type=int),
    )


def _run_screening_job(job_id: str, jd_id: int, saved_files: list[tuple[str, Path]]):
    jd = db.get_jd(jd_id)
    run_id = db.create_run(jd_id)

    for filename, resume_path in saved_files:
        try:
            text = extraction.extract_text(resume_path)
            candidate_name = name_extraction.guess_candidate_name(resume_path, text, fallback=resume_path.stem)
            score, matched, missing = matching.score_resume(text, jd["must_have"], jd["nice_to_have"])
            db.add_candidate_score(
                run_id=run_id,
                candidate_name=candidate_name,
                resume_filename=filename,
                resume_path=str(resume_path),
                score=score,
                matched_keywords=matched,
                missing_keywords=missing,
            )
        except UmiOcrUnavailable:
            pass
        with _jobs_lock:
            _jobs[job_id]["processed"] += 1

    with _jobs_lock:
        _jobs[job_id]["status"] = "done"


@app.route("/upload-resumes/screen", methods=["POST"])
def run_screening():
    jd_id = request.form.get("jd_id", type=int)
    jd = db.get_jd(jd_id)
    resume_files = request.files.getlist("resume_files")

    if not jd or not resume_files:
        return redirect(url_for("upload_resumes", jd_id=jd_id))

    RESUME_DIR.mkdir(parents=True, exist_ok=True)
    saved_files = []
    for uploaded in resume_files:
        if not uploaded.filename:
            continue
        safe_name = secure_filename(uploaded.filename)
        if not safe_name or not safe_name.lower().endswith(".pdf"):
            continue
        resume_path = RESUME_DIR / safe_name
        uploaded.save(resume_path)
        saved_files.append((safe_name, resume_path))

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "total": len(saved_files), "processed": 0}

    thread = threading.Thread(target=_run_screening_job, args=(job_id, jd_id, saved_files), daemon=True)
    thread.start()

    return redirect(url_for("screening_progress", job_id=job_id, jd_id=jd_id))


@app.route("/upload-resumes/processing/<job_id>")
def screening_progress(job_id):
    jd_id = request.args.get("jd_id", type=int)
    jd = db.get_jd(jd_id)
    return render_template("processing.html", active_page="upload_resumes", job_id=job_id, jd=jd)


@app.route("/upload-resumes/progress/<job_id>.json")
def screening_progress_json(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"status": "unknown"}), 404
    return jsonify(job)


# ---- Dashboard ----

@app.route("/dashboard")
def dashboard():
    jds = db.list_jds()
    selected_jd = None
    results = []
    selected_candidate = None
    selected_candidate_id = request.args.get("candidate_id", type=int)

    if jds:
        jd_id = request.args.get("jd_id", type=int) or jds[0]["id"]
        selected_jd = db.get_jd(jd_id)
        results = db.get_candidates_for_jd(jd_id)
        if selected_candidate_id:
            selected_candidate = next((r for r in results if r["id"] == selected_candidate_id), None)

    return render_template(
        "dashboard.html",
        active_page="dashboard",
        jds=jds,
        selected_jd=selected_jd,
        results=results,
        selected_candidate=selected_candidate,
        selected_candidate_id=selected_candidate_id,
    )


@app.route("/resume/<path:filename>")
def serve_resume(filename):
    return send_from_directory(RESUME_DIR, filename)


def _csv_safe(value: str) -> str:
    """Prefix values that would otherwise be read as a formula by Excel/
    Sheets when a candidate-supplied string (e.g. a name from a resume)
    starts with =, +, -, or @."""
    if value and value[0] in ("=", "+", "-", "@"):
        return "'" + value
    return value


@app.route("/export/<int:jd_id>.csv")
def export_jd_csv(jd_id):
    results = db.get_candidates_for_jd(jd_id)

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Candidate", "Score", "Matched Keywords", "Missing Keywords", "Resume File"])
    for r in results:
        writer.writerow([
            _csv_safe(r["candidate_name"]),
            r["score"],
            _csv_safe("; ".join(r["matched_keywords"])),
            _csv_safe("; ".join(r["missing_keywords"])),
            _csv_safe(r["resume_filename"]),
        ])

    return Response(
        buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=screening_results_jd{jd_id}.csv"},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8502, debug=False, threaded=True)
