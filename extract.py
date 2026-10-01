import pdfplumber
import re
import os
from dotenv import load_dotenv
load_dotenv()

def extract_text_from_pdf(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def split_questions(text: str) -> list[str]:
    pattern = r'\n?\d+\.\s*'
    parts = re.split(pattern, text)
    questions = [q.strip() for q in parts[1:] if q.strip()]
    return questions


from sentence_transformers import SentenceTransformer

def get_embeddings(questions: list[dict]):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [q["text"] for q in questions]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

from sklearn.cluster import KMeans
import numpy as np

def cluster_questions(all_questions: list[dict], embeddings, n_clusters: int = 50):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    clusters = {}
    for idx, label in enumerate(labels):
        label = int(label)
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(all_questions[idx])

    return clusters

def rank_clusters(clusters: dict) -> list:
    """Rank clusters by size AND number of distinct source files."""
    ranked = []
    for cluster_id, questions_in_cluster in clusters.items():
        size = len(questions_in_cluster)
        distinct_sources = len(set(q["source_file"] for q in questions_in_cluster))
        # Importance score: weight distinct sources more heavily than raw count
        score = size + (distinct_sources * 50)
        ranked.append({
            "cluster_id": cluster_id,
            "size": size,
            "distinct_sources": distinct_sources,
            "score": score,
            "questions": questions_in_cluster
        })
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked
import google.generativeai as genai
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm_model = genai.GenerativeModel("gemini-3.8-flash")

from pymongo import MongoClient
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["docchat"]
def summarize_cluster(cluster: dict) -> dict:
    sample_questions = [q["text"] for q in cluster["questions"][:5]]
    prompt = f"""Here are {len(sample_questions)} similar exam/interview questions that mean roughly the same thing:

{chr(10).join(f"- {q}" for q in sample_questions)}

Respond with ONLY valid JSON, no markdown formatting, in this exact structure:
{{"canonical_question": "one clear version of this question", "key_points": ["point 1", "point 2", "point 3"]}}"""

    response = llm_model.generate_content(prompt)
    text = response.text.strip()
    # Strip markdown code fences if present
    text = text.replace("```json", "").replace("```", "").strip()

    import json
    return json.loads(text, strict=False)


def save_to_mongo(all_questions: list[dict], ranked: list[dict]):
    # Save raw questions
    db["questions"].delete_many({})  # clear old data for a clean re-run
    if all_questions:
        db["questions"].insert_many(all_questions)

    # Save ranked clusters (without embeddings, just readable data)
    db["ranked_clusters"].delete_many({})
    cluster_docs = []
    for r in ranked:
        cluster_docs.append({
            "cluster_id": r["cluster_id"],
            "size": r["size"],
            "distinct_sources": r["distinct_sources"],
            "score": r["score"],
            "sample_questions": [q["text"] for q in r["questions"][:5]],
            "sample_sources": [q["source_file"] for q in r["questions"][:5]]
        })
    db["ranked_clusters"].insert_many(cluster_docs)

    print(f"\nSaved {len(all_questions)} questions and {len(cluster_docs)} ranked clusters to MongoDB.")
   
if __name__ == "__main__":
    all_questions = []
    try:
        mongo_client.admin.command('ping')
        print("MongoDB connected successfully!")
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
    for filename in os.listdir("uploads"):
        if filename.endswith(".pdf"):
            print(f"Processing {filename}...")

            path = os.path.join("uploads", filename)
            text = extract_text_from_pdf(path)
            questions = split_questions(text)
            print(f"  -> Found {len(questions)} questions")

            for q in questions:
                all_questions.append({
                    "text": q,
                    "source_file": filename
                })

    print(f"\nTotal questions across all files: {len(all_questions)}")

    print("\nGenerating embeddings (this downloads a model the first time, may take a minute)...")
    embeddings = get_embeddings(all_questions)
    print(f"Embeddings shape: {embeddings.shape}")

    print("\n--- Sample from deeper in the list ---")
    print(all_questions[100]["text"])

    print("\nClustering questions...")
    clusters = cluster_questions(all_questions, embeddings, n_clusters=50)

    print("\nRanking clusters by importance (size + cross-source frequency)...")
    ranked = rank_clusters(clusters)


    save_to_mongo(all_questions, ranked)

    print("\n--- Top 10 most important topics ---")
    ...

    print("\n--- Top 10 most important topics ---")
    import time  # add this near your other imports at the top

    print("\n--- Enriching top 5 clusters with LLM ---")
    for r in ranked[:5]:
        try:
            summary = summarize_cluster(r)
            print(f"\n[{r['distinct_sources']} sources, {r['size']} questions] {summary['canonical_question']}")
            for point in summary['key_points']:
                print(f"  • {point}")
        except Exception as e:
            print(f"  (failed to summarize cluster {r['cluster_id']}: {e})")
        time.sleep(13)  # wait between calls to respect free-tier rate limit
    for r in ranked[:10]:
        print(f"\nCluster {r['cluster_id']} | {r['size']} questions | appears in {r['distinct_sources']} different source(s) | score: {r['score']}")
        for q in r["questions"][:2]:
            print(f"  - [{q['source_file']}] {q['text'][:80]}")

    for filename in os.listdir("uploads"):
        if filename.endswith(".pdf"):
            print(f"Processing {filename}...")
            path = os.path.join("uploads", filename)
            text = extract_text_from_pdf(path)
            questions = split_questions(text)
            print(f"  -> Found {len(questions)} questions")

            for q in questions:
                all_questions.append({
                    "text": q,
                    "source_file": filename
                })