# DocChat — Exam & Interview Question Importance Analyzer

An AI-powered tool that ingests multiple exam papers and interview question sets (PDFs), then uses embeddings, clustering, and an LLM to identify and summarize the most frequently repeated — and therefore most important — questions across sources.

## Features
- Extracts and parses questions from multiple PDF sources
- Generates semantic embeddings (sentence-transformers) to detect similar questions phrased differently
- Clusters related questions using KMeans
- Ranks topics by cross-source frequency, not just raw count
- Uses Gemini to generate canonical questions + key study points per topic
- Persists results in MongoDB
- FastAPI backend with REST endpoints
- Simple web frontend to upload PDFs and view ranked results

## Tech Stack
Python, FastAPI, sentence-transformers, scikit-learn, MongoDB, Google Gemini API, HTML/JS

## How It Works
1. Upload PDFs (question papers, interview question sets)
2. Text is extracted and split into individual questions
3. Questions are embedded and clustered by semantic similarity
4. Clusters are ranked by size and number of distinct sources
5. Top clusters are summarized by an LLM into clean, study-ready points

## Setup
```bash
pip install -r requirements.txt
```
Create a `.env` file with: