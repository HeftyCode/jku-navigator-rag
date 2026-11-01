# JKU Linz & Austrian Student Life Navigator — Bilingual Production RAG

[![Live App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jku-navigator.streamlit.app)

An end-to-end, auditable **Bilingual (English / German) Retrieval-Augmented Generation (RAG)** system built for **Johannes Kepler University (JKU) Linz** students. It indexes **2,699 hierarchical chunks across 27 official JKU documents**—covering TNF Computing & Natural Sciences Bachelor curricula (AI, Computer Science, Bioinformatics, Data Science, Mathematics, Physics, Chemistry, Mechatronics, Electronics, Medical Engineering), credit transfer equivalence tables, examination law (*Satzungsteil Studienrecht*), and campus regulations (*Hausordnung*, *Parkordnung*, Library & ZID IT rules).

**Live Deployment:** [https://jku-navigator.streamlit.app](https://jku-navigator.streamlit.app)

---

## System Architecture & Why Standard RAG Fails Here

University regulations and curricula break naive "chat-with-PDF" tutorials because they mix **German legal paragraphs (`§ 14 Abs. 2`)**, **exact course abbreviations (`VL`, `KV`, `UE`)**, and **cross-lingual queries** (e.g., an international student asking in English about a German regulation).

1. **Page- & Section-Aware Hierarchical Ingestion (`src/ingestion/parser.py`):**
   - Extracts clean Markdown from 27 JKU PDFs using `PyMuPDF4LLM` while preserving credit transfer tables and injecting `[Source.pdf | Page X | Section: Y]` headers into all **2,699 chunks**.
2. **Bilingual Query Expansion (`src/generation/generator.py`):**
   - Automatically detects query language (`en` / `de`) and expands search queries with cross-lingual academic/legal synonyms and German compound nouns before retrieval.
3. **Hybrid Search + Reciprocal Rank Fusion (`src/retrieval/hybrid_search.py`):**
   - **Dense Retrieval:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` indexed in `Qdrant`.
   - **Sparse Keyword Retrieval:** `BM25Okapi` with a custom regex tokenizer (`[a-z0-9äöüß§]+(?:\.[0-9]+)*`), bilingual stopword filtering, and 4-character morphological root stemming so German plurals/compounds match (`Hund` $\leftrightarrow$ `Hunde` / `Haustierverbot`).
   - **Fusion:** Merges top-20 Dense and top-20 Sparse results via **Reciprocal Rank Fusion ($k=60$)**.
4. **Cross-Encoder Reranking:**
   - Re-scores fused candidates using `cross-encoder/ms-marco-MiniLM-L-6-v2` to pass only the highest-evidence top-$K$ chunks to the generator.
5. **Zero-Extrapolation Guardrails & Fault-Tolerant LLM Routing:**
   - Enforces strict `Pydantic` JSON output schemas with page-level inline citations (`[Filename.pdf, Page X]`), automatic abstention on unanswerable queries, and automatic rate-limit failover across Groq LLMs.

---

## Evaluation & 100-Query Ablation Benchmark (`src/evaluation/eval_runner.py`)

Evaluated across a 100-query benchmark and an 8-case adversarial LLM-as-a-Judge audit:

| Retrieval Architecture | Exact Chunk Hit Rate @ 5 | Document Hit Rate @ 5 | Mean Reciprocal Rank (MRR) |
| :--- | :---: | :---: | :---: |
| **1. Naive Dense Vector Only** | 72.0% | 90.0% | 0.569 |
| **2. Sparse BM25 Keyword Only** | 91.0% | 97.0% | 0.686 |
| **3. Hybrid RRF + Cross-Encoder (Ours)** | **95.0% (+23.0% vs. Dense)** | **99.0%** | **0.792 (+39.2% vs. Dense)** |

* **End-to-End Generation Audit (LLM-as-a-Judge):**
  * **Document Retrieval Accuracy:** `100.0%`
  * **Answer Relevance:** `100.0%`
  * **Context Faithfulness:** `100.0%`
  * **Hallucination Rate:** `0.0%`

---

## Engineering Post-Mortem: Failure Modes Identified & Fixed

1. **Cross-Lingual BM25 Dilution & German Morphology:**
   - *Failure:* Asking in English (*"Am I allowed to bring my dog onto the JKU campus?"*) initially failed to retrieve the German *Hausordnung* (`Haustierverbot`) because English conversational stopwords (`am, i, to, onto, the`) matched English study guides in BM25, and singular `Hund` missed plural `Hunde`.
   - *Fix:* Stripped bilingual stopwords in `_tokenize()`, added 4-character prefix stems for words $\ge 5$ chars, and constrained query expansion to high-signal bilingual terms and compound nouns.
2. **Pre-Trained Knowledge Extrapolation (12.5% $\rightarrow$ 0.0% Hallucination):**
   - *Failure:* Our LLM Judge flagged a German response about `KV` courses because the generator used outside knowledge to define "Combined Course" components not explicitly stated in the retrieved chunk.
   - *Fix:* Added strict zero-extrapolation constraints to the generator system prompt, dropping the audited hallucination rate from `12.5%` to `0.0%`.
3. **LLM API Rate Limits & Smaller-Model JSON Formatting:**
   - *Failure:* Running batch evaluations hit Groq's 200k daily token limit on the primary 120B model, and smaller fallback models occasionally tripped server-side `json_object` validation.
   - *Fix:* Implemented `chat_with_retry()` with automatic model-pool rotation on Daily Token Limits (`TPD`), backoff retries on Per-Minute Limits (`TPM`), and regex-based client-side JSON extraction fallback.

---

## Quickstart (Local & FastAPI)

```bash
pip install -r requirements.txt
python -m src.ingestion.parser
python -m src.ingestion.indexer
python -m src.evaluation.eval_runner

# Launch Streamlit UI
streamlit run app.py

# Or launch FastAPI REST Backend (Swagger UI at http://localhost:8000/docs)
uvicorn src.api.main:app --reload
```