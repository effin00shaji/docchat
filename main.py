from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil

from extract import (
    extract_text_from_pdf,
    split_questions,
    get_embeddings,
    cluster_questions,
    rank_clusters,
    save_to_mongo,
    db
)

app = FastAPI()

# Allow requests from any frontend (needed later for React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"message": "DocChat API is running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"filename": file.filename, "status": "uploaded"}


@app.get("/analyze")
def analyze():
    all_questions = []

    for filename in os.listdir(UPLOAD_DIR):
        if filename.endswith(".pdf"):
            path = os.path.join(UPLOAD_DIR, filename)
            text = extract_text_from_pdf(path)
            questions = split_questions(text)
            for q in questions:
                all_questions.append({"text": q, "source_file": filename})

    if not all_questions:
        return {"error": "No PDFs found in uploads folder"}

    embeddings = get_embeddings(all_questions)
    clusters = cluster_questions(all_questions, embeddings, n_clusters=50)
    ranked = rank_clusters(clusters)

    save_to_mongo(all_questions, ranked)

    # Return a clean, JSON-friendly version (no raw embeddings)
    results = []
    for r in ranked[:20]:
        results.append({
            "cluster_id": r["cluster_id"],
            "size": r["size"],
            "distinct_sources": r["distinct_sources"],
            "score": r["score"],
            "sample_questions": [q["text"] for q in r["questions"][:3]],
            "sample_sources": [q["source_file"] for q in r["questions"][:3]]
        })

    return {
        "total_questions": len(all_questions),
        "total_clusters": len(clusters),
        "top_results": results
    }


@app.get("/results")
def get_saved_results():
    """Fetch previously saved results from MongoDB without recomputing."""
    clusters = list(db["ranked_clusters"].find({}, {"_id": 0}).sort("score", -1).limit(20))
    return {"top_results": clusters}