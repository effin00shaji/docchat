import os
import shutil
import tempfile
import threading
import time
import uuid
from collections import defaultdict, deque

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI(title="DocChat API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Vercel URL later
    allow_methods=["*"],
    allow_headers=["*"],
)

# Limits that protect the free-tier server and your Gemini quota
MAX_FILES = 3
MAX_FILE_MB = 10
MAX_JOBS_PER_IP_PER_HOUR = 5
MAX_ACTIVE_JOBS = 3
JOB_TTL_SECONDS = 3600

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
run_lock = threading.Semaphore(1)  # one analysis at a time keeps memory use low
ip_hits: dict[str, deque] = defaultdict(deque)

_mongo_db = None


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def cleanup_jobs():
    cutoff = time.time() - JOB_TTL_SECONDS
    with jobs_lock:
        for job_id in [j for j, v in jobs.items() if v["created"] < cutoff]:
            del jobs[job_id]


def set_job(job_id: str, **fields):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(fields)


def run_job(job_id: str, workdir: str, saved: list):
    try:
        set_job(job_id, status="queued", message="Waiting for a free worker...")
        with run_lock:
            set_job(job_id, status="running", message="Starting analysis...")
            from extract import analyze_pdfs  # heavy imports load on first use

            result = analyze_pdfs(
                saved, progress=lambda msg: set_job(job_id, message=msg)
            )
        set_job(job_id, status="done", message="Done", result=result)
    except ValueError as e:
        set_job(job_id, status="error", error=str(e))
    except Exception as e:
        print(f"Job {job_id} failed: {e}")
        set_job(job_id, status="error",
                error="Something went wrong while analyzing. Please try again.")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@app.get("/")
def root():
    return {"message": "DocChat API is running"}


@app.post("/analyze-upload")
async def analyze_upload(
    request: Request,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
):
    cleanup_jobs()

    if not files or len(files) > MAX_FILES:
        raise HTTPException(400, f"Upload between 1 and {MAX_FILES} PDF files.")

    ip = client_ip(request)
    now = time.time()
    hits = ip_hits[ip]
    while hits and now - hits[0] > 3600:
        hits.popleft()
    if len(hits) >= MAX_JOBS_PER_IP_PER_HOUR:
        raise HTTPException(
            429, f"Limit reached: {MAX_JOBS_PER_IP_PER_HOUR} analyses per hour. Try again later."
        )

    with jobs_lock:
        active = sum(1 for j in jobs.values() if j["status"] in ("queued", "running"))
    if active >= MAX_ACTIVE_JOBS:
        raise HTTPException(429, "The server is busy with other analyses. Try again in a few minutes.")

    workdir = tempfile.mkdtemp(prefix="docchat_")
    saved = []
    try:
        for upload in files:
            name = os.path.basename(upload.filename or "file.pdf")
            if not name.lower().endswith(".pdf"):
                raise HTTPException(400, f"{name} is not a PDF.")
            data = await upload.read(MAX_FILE_MB * 1024 * 1024 + 1)
            if len(data) > MAX_FILE_MB * 1024 * 1024:
                raise HTTPException(413, f"{name} is larger than {MAX_FILE_MB} MB.")
            if not data.startswith(b"%PDF"):
                raise HTTPException(400, f"{name} is not a valid PDF.")
            path = os.path.join(workdir, f"{len(saved)}.pdf")
            with open(path, "wb") as out:
                out.write(data)
            saved.append((path, name))
    except HTTPException:
        shutil.rmtree(workdir, ignore_errors=True)
        raise

    hits.append(now)
    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "message": "Files received",
            "result": None,
            "error": None,
            "created": now,
        }
    background_tasks.add_task(run_job, job_id, workdir, saved)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(
                404, "This analysis expired or the server restarted. Please upload again."
            )
        return {k: job[k] for k in ("status", "message", "result", "error")}


@app.get("/results")
def get_saved_results():
    """Saved sample results (refresh them locally with `python extract.py`)."""
    global _mongo_db
    if _mongo_db is None:
        from pymongo import MongoClient

        _mongo_db = MongoClient(os.getenv("MONGO_URI"))["docchat"]
    docs = list(
        _mongo_db["ranked_clusters"].find({}, {"_id": 0}).sort("score", -1).limit(10)
    )
    return {"top_results": docs}