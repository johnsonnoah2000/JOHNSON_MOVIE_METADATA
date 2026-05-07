# 🎬 JOHNSON Movie Metadata RAG Assistant

> A Flask web application that answers movie-related questions using Retrieval-Augmented Generation (RAG), TF-IDF retrieval, and a locally-hosted Ollama LLM.

---

## 📌 Project Purpose

This app allows users to ask natural language questions about movies and receive grounded answers based on retrieved movie records from a real dataset of 44,000+ films. It combines keyword-based retrieval (TF-IDF + cosine similarity) with a local LLM (Ollama) to generate accurate, context-aware responses.

**Course:** DSA502 — Data Science with AI  
**Student:** Johnson  
**Mini Project:** 2

---

## 🧠 RAG Flow

```
User Question
      ↓
TF-IDF Retrieval (top 5 matching movies from CSV)
      ↓
Build Context from retrieved rows
      ↓
Send to Ollama (minimax-m2.1:cloud)
      ↓
Display retrieved rows + grounded AI answer
```

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| Flask | Web framework |
| pandas | CSV loading and data cleaning |
| scikit-learn | TF-IDF vectorizer + cosine similarity |
| requests | Ollama API calls |
| Ollama | Local LLM runtime |
| minimax-m2.1:cloud | LLM model for answer generation |

---

## 📁 Project Structure

```
JOHNSON_MOVIE_METADATA/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── templates/
    ├── cover.html          # Landing/cover page
    └── index.html          # Main search UI
```

---

## ⚙️ Setup Steps

### 1. Clone the repo
```bash
git clone https://github.com/johnsonnoah2000/JOHNSON_MOVIE_METADATA.git
cd JOHNSON_MOVIE_METADATA
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Start Ollama in a separate terminal
```bash
ollama run minimax-m2.1:cloud
```

### 5. Run the Flask app
```bash
python app.py
```

### 6. Open in browser
```
http://127.0.0.1:5005
```

---

## 🚀 Run Instructions

- Make sure Ollama is running in a **separate terminal** before starting Flask
- The app loads ~44,000 movies on startup — wait for `Loaded 44494 movies.` before opening the browser
- App runs on port **5005**

---

## 💬 Sample Question & Output

**Question:** *"Find movies about space travel and exploration."*

**Retrieved Movies:**
| Title | Year | Rating | Score |
|-------|------|--------|-------|
| Operation Ganymed | 1977 | 3.3 | 0.322 |
| Space Tourists | 2010 | 5.5 | 0.311 |

**AI Answer:**
> Based on the retrieved movie records, here are the movies about space travel and exploration:
> 1. **Operation Ganymed** (1977) — A spaceship returns to Earth after several years of space exploration...
> 2. **Space Tourists** (2010) — A documentary following a journey into space...

---

## ✅ Deliverables Checklist

- [x] Flask app runs on port 5005
- [x] CSV loaded and cleaned with pandas
- [x] TF-IDF + cosine similarity retrieval (top 5)
- [x] Ollama call with grounded prompt
- [x] UI shows retrieved rows and generated answer
- [x] Cover page + main search page
- [x] GitHub repo submitted