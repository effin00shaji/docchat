"""
DocChat analysis pipeline.

analyze_pdfs() takes a list of (path, display_name) PDFs and returns the most
important question groups. It does not touch a shared folder or the database,
so many visitors can use the site at the same time without seeing each other's data.

Run `python extract.py` locally to refresh the saved sample results in MongoDB
from the PDFs in the ./uploads folder.
"""
import json
import os
import re

import pdfplumber
from dotenv import load_dotenv

load_dotenv()

MAX_PAGES_PER_FILE = 80      # keeps extraction time bounded
MAX_QUESTIONS = 800          # keeps embedding time and memory bounded on the free tier
MAX_QUESTION_CHARS = 500

_model = None


def _get_model():
    """Load the embedding model once and reuse it for every request."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def extract_text_from_pdf(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages[:MAX_PAGES_PER_FILE]:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def split_questions(text: str) -> list[str]:
    # Main format: numbered questions ("1. What is ...")
    parts = re.split(r"\n?\d+\.\s*", text)
    questions = [q.strip() for q in parts[1:] if len(q.strip()) > 10]

    # Fallback for PDFs that are not numbered: lines that end with a question mark
    if len(questions) < 5:
        questions = [
            line.strip()
            for line in text.split("\n")
            if line.strip().endswith("?") and len(line.strip()) > 10
        ]

    return [q[:MAX_QUESTION_CHARS] for q in questions]


def choose_k(n_questions: int) -> int:
    """Number of groups to build: about one group per 5 questions, between 1 and 20."""
    return max(1, min(20, n_questions // 5))


def get_embeddings(questions: list[dict]):
    texts = [q["text"] for q in questions]
    return _get_model().encode(texts, batch_size=32, show_progress_bar=False)


def cluster_questions(all_questions: list[dict], embeddings, n_clusters: int):
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    clusters: dict[int, list[dict]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(all_questions[idx])
    return clusters


def rank_clusters(clusters: dict, total_questions: int) -> list[dict]:
    """Rank groups by size and by how many different files they appear in."""
    source_weight = max(2, round(total_questions * 0.01))
    ranked = []
    for cluster_id, items in clusters.items():
        size = len(items)
        distinct_sources = len({q["source_file"] for q in items})
        ranked.append({
            "cluster_id": cluster_id,
            "size": size,
            "distinct_sources": distinct_sources,
            "score": size + distinct_sources * source_weight,
            "questions": items,
        })
    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked


def summarize_top_clusters(ranked: list[dict], top_n: int = 5) -> dict[int, dict]:
    """One Gemini call for all top groups (cheaper and faster than one call per group).
    Returns {group_index: {"canonical_question", "key_points"}}. Empty dict on any failure."""
    top = ranked[:top_n]
    if not top or not os.getenv("GEMINI_API_KEY"):
        return {}
    try:
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))

        blocks = []
        for i, cluster in enumerate(top):
            lines = "\n".join(f"- {q['text'][:300]}" for q in cluster["questions"][:5])
            blocks.append(f"Group {i}:\n{lines}")

        prompt = (
            "Each group below holds similar exam or interview questions.\n"
            "For every group, write one clear version of the question and 2-3 short key points "
            "for answering it.\n\n"
            + "\n\n".join(blocks)
            + '\n\nRespond with ONLY a JSON array, no markdown, in this shape:\n'
              '[{"group": 0, "canonical_question": "...", "key_points": ["...", "..."]}]'
        )

        text = model.generate_content(prompt).text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text, strict=False)
        return {int(d["group"]): d for d in data}
    except Exception as e:  # the analysis still works without summaries
        print(f"Summaries unavailable: {e}")
        return {}


def analyze_pdfs(files, progress=None, top_n: int = 10, enrich_n: int = 5) -> dict:
    say = progress or (lambda msg: None)

    all_questions: list[dict] = []
    file_counts = []
    for path, name in files:
        say(f"Reading {name}...")
        questions = split_questions(extract_text_from_pdf(path))
        file_counts.append({"name": name, "questions": len(questions)})
        all_questions.extend({"text": q, "source_file": name} for q in questions)

    total_found = len(all_questions)
    if total_found == 0:
        raise ValueError(
            "No questions found. Upload text-based PDFs (not scans) with numbered "
            "questions or lines that end with a question mark."
        )

    truncated = total_found > MAX_QUESTIONS
    if truncated:  # sample evenly across all files instead of cutting off the last ones
        step = total_found / MAX_QUESTIONS
        all_questions = [all_questions[int(i * step)] for i in range(MAX_QUESTIONS)]

    n = len(all_questions)
    say(f"Reading {n} questions. This is the slow part, please keep this page open...")
    embeddings = get_embeddings(all_questions)

    say("Grouping similar questions...")
    clusters = cluster_questions(all_questions, embeddings, choose_k(n))
    ranked = rank_clusters(clusters, n)

    say("Writing summaries for the top questions...")
    summaries = summarize_top_clusters(ranked, enrich_n)

    top_results = []
    for i, cluster in enumerate(ranked[:top_n]):
        samples = [q["text"] for q in cluster["questions"][:5]]
        summary = summaries.get(i, {})
        top_results.append({
            "cluster_id": cluster["cluster_id"],
            "size": cluster["size"],
            "distinct_sources": cluster["distinct_sources"],
            "score": cluster["score"],
            "canonical_question": summary.get("canonical_question") or samples[0],
            "key_points": summary.get("key_points", []),
            "sample_questions": samples,
        })

    return {
        "total_questions": total_found,
        "analyzed_questions": n,
        "truncated": truncated,
        "files": file_counts,
        "top_results": top_results,
    }


if __name__ == "__main__":
    # Refresh the saved sample results shown by "See sample results".
    from pymongo import MongoClient

    folder = "uploads"
    pdfs = [
        (os.path.join(folder, f), f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(".pdf")
    ]
    if not pdfs:
        raise SystemExit("No PDFs found in the uploads folder.")

    result = analyze_pdfs(pdfs, progress=print, top_n=50)
    db = MongoClient(os.getenv("MONGO_URI"))["docchat"]
    db["ranked_clusters"].delete_many({})
    db["ranked_clusters"].insert_many(result["top_results"])
    print(f"Saved {len(result['top_results'])} sample groups from "
          f"{result['total_questions']} questions to MongoDB.")