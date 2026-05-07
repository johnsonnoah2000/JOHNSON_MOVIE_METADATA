print("APP STARTING...")
import os
import json
import requests
import pandas as pd
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
CSV_URL = "https://hiperc.buffalostate.edu/courses/movies_metadata.csv"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "minimax-m2.1:cloud"
TOP_K = 5

# ── Global state ─────────────────────────────────────────────────────────────
movies_df = None
tfidf_matrix = None
vectorizer = None


def load_movies():
    """Load and clean the movies dataset, build TF-IDF index."""
    global movies_df, tfidf_matrix, vectorizer

    print("Loading movies dataset...")
    try:
        df = pd.read_csv(CSV_URL, low_memory=False)
    except Exception:
        # fallback to local copy if URL fails
        local = os.path.join(os.path.dirname(__file__), "movies_metadata.csv")
        df = pd.read_csv(local, low_memory=False)

    # Keep useful columns
    keep = ["title", "overview", "genres", "release_date", "vote_average"]
    df = df[[c for c in keep if c in df.columns]].copy()

    # Clean
    df["title"] = df["title"].fillna("Unknown Title")
    df["overview"] = df["overview"].fillna("")
    df["genres"] = df["genres"].fillna("")
    df["release_date"] = df["release_date"].fillna("N/A")
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)

    # Drop rows with no useful text
    df = df[df["overview"].str.strip() != ""].reset_index(drop=True)

    # Build search text combining title + overview + genres
    df["search_text"] = (
        df["title"].str.lower() + " " +
        df["overview"].str.lower() + " " +
        df["genres"].str.lower()
    )

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    tfidf_matrix = vectorizer.fit_transform(df["search_text"])

    movies_df = df
    print(f"Loaded {len(movies_df)} movies.")


def retrieve_movies(question: str, top_k: int = TOP_K):
    """Return top_k most relevant movie rows with similarity scores."""
    query_vec = vectorizer.transform([question.lower()])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        row = movies_df.iloc[idx].to_dict()
        row["score"] = round(float(scores[idx]), 4)
        results.append(row)
    return results


def build_context(rows: list) -> str:
    """Create compact context text from retrieved rows."""
    parts = []
    for i, row in enumerate(rows, 1):
        parts.append(
            f"[Movie {i}]\n"
            f"Title: {row['title']}\n"
            f"Overview: {row['overview'][:300]}\n"
            f"Genres: {row['genres']}\n"
            f"Release Date: {row['release_date']}\n"
            f"Vote Average: {row['vote_average']}\n"
        )
    return "\n".join(parts)


def ask_ollama(question: str, context: str) -> str:
    """Send grounded prompt to Ollama and return the answer."""
    prompt = f"""You are a helpful movie assistant. Answer the user's question ONLY using the retrieved movie records below.
If the context does not contain enough information to answer, say so clearly instead of making up facts.

Retrieved Movies:
{context}

User Question: {question}

Answer:"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2}
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "No response received from model.").strip()
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Make sure `ollama serve` is running."
    except requests.exceptions.Timeout:
        return "Error: Ollama request timed out. The model may still be loading."
    except requests.exceptions.HTTPError as e:
        return f"Error: Ollama returned HTTP {e.response.status_code}. Check that model '{OLLAMA_MODEL}' is installed."
    except Exception as e:
        return f"Unexpected error: {str(e)}"


@app.route("/", methods=["GET", "POST"])
def index():
    question = ""
    retrieved_rows = []
    answer = ""
    error = ""

    if request.method == "POST":
        question = request.form.get("question", "").strip()

        if not question:
            error = "Please enter a question before submitting."
        else:
            retrieved_rows = retrieve_movies(question)
            if not retrieved_rows:
                error = "No relevant movies found for your question."
            else:
                context = build_context(retrieved_rows)
                answer = ask_ollama(question, context)

    return render_template(
        "index.html",
        question=question,
        retrieved_rows=retrieved_rows,
        answer=answer,
        error=error
    )


if __name__ == "__main__":
    load_movies()
    app.run(host="127.0.0.1", port=5005, debug=True)