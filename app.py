import json
from pathlib import Path
import streamlit as st
from src.generation.generator import JKUNavigatorGenerator

# --- Page Configuration ---
st.set_page_config(
    page_title="JKU & Austrian Student Navigator",
    page_icon="🎓",
    layout="wide",
)

# --- Cache the RAG Pipeline so Qdrant & Models Load Only ONCE ---
@st.cache_resource(show_spinner="Loading Multilingual Embedder, Cross-Encoder Reranker & Qdrant Index...")
def get_rag_engine() -> JKUNavigatorGenerator:
    return JKUNavigatorGenerator()


def load_json_file(filepath: str):
    path = Path(filepath)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


# --- Sidebar Controls & Architecture Info ---
with st.sidebar:
    st.title("JKU Navigator AI")
    st.caption("Production Bilingual RAG Engine (EN 🇬🇧 / DE 🇩🇪)")
    st.markdown("---")

    st.subheader("Retrieval & Generation Settings")
    lang_choice = st.selectbox(
        "Response Language",
        options=["auto", "en", "de"],
        format_func=lambda x: {
            "auto": "Auto-Detect (Match Question)",
            "en": "🇬🇧 Force English",
            "de": "🇩🇪 Force German (Deutsch)",
        }[x],
    )
    top_n = st.slider("Top-K Reranked Chunks Passed to LLM", min_value=2, max_value=8, value=5)

    st.markdown("---")
    st.subheader("Active Pipeline Stack")
    st.markdown(
        """
        - **Parser:** `PyMuPDF4LLM` (Page & Section aware)
        - **Dense Search:** `paraphrase-multilingual-MiniLM-L12-v2` + `Qdrant`
        - **Sparse Search:** Custom Regex `BM25Okapi` (preserves `§` & course codes)
        - **Fusion:** Reciprocal Rank Fusion ($k=60$)
        - **Reranker:** Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)
        - **Generator:** `openai/gpt-oss-120b` (Pydantic JSON schema + citations)
        """
    )

# --- Main Header ---
st.title("JKU Linz & Austrian Student Life Navigator")
st.markdown(
    "Ask questions in **English or German** across official JKU curriculum guides, credit transfer tables, "
    "house regulations (*Hausordnung*), and campus parking rules (*Parkordnung*)."
)

# --- Navigation Tabs ---
tab_chat, tab_eval, tab_corpus = st.tabs([
    "Bilingual RAG Assistant",
    "100-Query Evaluation & Ablation Dashboard",
    "Indexed JKU Document Corpus",
])

# =====================================================================
# TAB 1: BILINGUAL RAG ASSISTANT
# =====================================================================
with tab_chat:
    st.markdown("##### Quick-Test Prompts (Click to test Cross-Lingual Retrieval & Guardrails)")
    col1, col2, col3, col4 = st.columns(4)

    preset_query = None
    if col1.button("Dogs on Campus? (EN ➔ DE PDF)", use_container_width=True):
        preset_query = "Am I allowed to bring my dog onto the JKU campus and what are the rules?"
    if col2.button("VL vs. KV Courses (DE ➔ EN PDF)", use_container_width=True):
        preset_query = "Was ist der Unterschied zwischen einer VL und einer KV im AI Bachelorstudium, und gibt es Anwesenheitspflicht?"
    if col3.button("Informatik ➔ AI Credit Transfer", use_container_width=True):
        preset_query = "How does credit transfer work between Informatik and Artificial Intelligence at JKU?"
    if col4.button("Hallucination Trap Test", use_container_width=True):
        preset_query = "What is the minimum cryptocurrency deposit required to use the JKU submarine dock?"

    user_input = st.chat_input("Ask a question about JKU regulations, courses, parking, or credit transfers...")
    active_query = preset_query or user_input

    if active_query:
        with st.chat_message("user"):
            st.write(active_query)

        with st.chat_message("assistant"):
            try:
                bot = get_rag_engine()
                with st.spinner("1️⃣ Expanding query bilingually ➔ 2️⃣ Running Hybrid Search + RRF ➔ 3️⃣ Cross-Encoder Reranking ➔ 4️⃣ Generating cited response..."):
                    result = bot.answer_question(
                        user_query=active_query,
                        target_language=lang_choice,
                        top_n_chunks=top_n,
                    )

                # Show Guardrail Badge & Bilingual Expansion
                status_col1, status_col2 = st.columns([1, 2])
                if result["is_supported"]:
                    status_col1.success("Grounded in JKU Documents")
                else:
                    status_col1.warning("Abstention Triggered (Unsupported by Corpus)")

                status_col2.info(f"**Bilingual Search Expansion:** `{result['bilingual_search_query']}`")

                # Main Answer
                st.markdown("### Answer")
                st.markdown(result["answer"])

                # Structured Citations
                if result["citations"]:
                    st.markdown("#### Verified Citations")
                    for cit in result["citations"]:
                        st.markdown(
                            f"- **`[{cit['source']}, Page {cit['page']}]`** *(Section: {cit['section']})* — {cit['evidence_summary']}"
                        )

                # Under-the-Hood Retrieval Telemetry
                with st.expander("Inspect Under-the-Hood Hybrid Retrieval & Reranker Scores"):
                    for rank_idx, chunk in enumerate(result["retrieved_chunks"], 1):
                        st.markdown(
                            f"**Rank #{rank_idx}** | **`{chunk['source']}` (Page {chunk['page']})** | "
                            f"Section: `{chunk['section']}`  \n"
                            f"**Cross-Encoder Score:** `{chunk['rerank_score']}` | "
                            f"**RRF Score:** `{chunk['rrf_score']}` "
                            f"*(Dense Vector Rank: `#{chunk['dense_rank']}`, BM25 Keyword Rank: `#{chunk['bm25_rank']}`)*"
                        )
                        st.code(chunk.get("contextual_text", chunk["text"]), language="markdown")

            except Exception as e:
                st.error(f"Error executing RAG pipeline: {e}")

# =====================================================================
# TAB 2: 100-QUERY EVALUATION & ABLATION DASHBOARD
# =====================================================================
with tab_eval:
    eval_data = load_json_file("data/processed/eval_results.json")

    if not eval_data:
        st.warning("No evaluation report found yet. Run `python -m src.evaluation.eval_runner` first.")
    else:
        st.subheader("1. Retrieval Architecture Ablation Study (100 Queries, Top-5)")
        st.caption(
            f"Last benchmarked: {eval_data.get('timestamp')} | Compares Naive Dense Vector Search against "
            "Sparse BM25 and our Full Hybrid RRF + Cross-Encoder Reranker."
        )

        ab = eval_data["retrieval_100_ablation"]["ablation_results"]

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown("#### 1️⃣ Naive Dense Vector")
            st.metric("Exact Chunk Hit Rate @ 5", f"{ab['naive_dense']['exact_chunk_hit_rate_pct']}%")
            st.metric("Mean Reciprocal Rank (MRR)", f"{ab['naive_dense']['mrr']}")
            st.metric("Document Hit Rate @ 5", f"{ab['naive_dense']['document_hit_rate_pct']}%")

        with m2:
            st.markdown("#### 2️⃣ Sparse BM25 Keyword")
            st.metric(
                "Exact Chunk Hit Rate @ 5",
                f"{ab['sparse_bm25']['exact_chunk_hit_rate_pct']}%",
                delta=f"+{round(ab['sparse_bm25']['exact_chunk_hit_rate_pct'] - ab['naive_dense']['exact_chunk_hit_rate_pct'], 1)}% vs Dense",
            )
            st.metric(
                "Mean Reciprocal Rank (MRR)",
                f"{ab['sparse_bm25']['mrr']}",
                delta=f"+{round(ab['sparse_bm25']['mrr'] - ab['naive_dense']['mrr'], 3)} vs Dense",
            )
            st.metric("Document Hit Rate @ 5", f"{ab['sparse_bm25']['document_hit_rate_pct']}%")

        with m3:
            st.markdown("#### 3️⃣ Hybrid RRF + Reranker (Ours)")
            st.metric(
                "Exact Chunk Hit Rate @ 5",
                f"{ab['hybrid_reranked']['exact_chunk_hit_rate_pct']}%",
                delta=f"+{round(ab['hybrid_reranked']['exact_chunk_hit_rate_pct'] - ab['naive_dense']['exact_chunk_hit_rate_pct'], 1)}% vs Dense",
            )
            st.metric(
                "Mean Reciprocal Rank (MRR)",
                f"{ab['hybrid_reranked']['mrr']}",
                delta=f"+{round(ab['hybrid_reranked']['mrr'] - ab['naive_dense']['mrr'], 3)} vs Dense",
            )
            st.metric("Document Hit Rate @ 5", f"{ab['hybrid_reranked']['document_hit_rate_pct']}%")

        st.markdown("---")
        st.subheader("2. End-to-End Generation & Hallucination Audit (LLM-as-a-Judge)")
        gen = eval_data["generation_audit"]

        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Document Retrieval Accuracy", f"{gen['correct_document_retrieval_pct']}%")
        g2.metric("Answer Relevance", f"{gen['answer_relevance_pct']}%")
        g3.metric("Context Faithfulness", f"{gen['context_support_faithfulness_pct']}%")
        g4.metric("Hallucination Rate", f"{gen['hallucination_rate_pct']}%", delta_color="inverse")

        st.markdown("#### Detailed Audit Log (Every Test Case & Judge Reasoning)")
        for idx, case in enumerate(gen["cases"], 1):
            badge = "PASS" if (case["supported_by_context"] and not case["hallucinated"]) else "FLAGGED BY JUDGE"
            with st.expander(f"Case #{idx} [{badge}]: {case['question']}"):
                st.write(f"**Expected Source:** `{case['expected_source']}` | **Correct Doc Retrieved:** `{case['doc_retrieved_correctly']}`")
                st.write(f"**Answer Relevant:** `{case['answer_relevant']}` | **Supported by Context:** `{case['supported_by_context']}` | **Hallucinated:** `{case['hallucinated']}`")
                st.info(f"🧑‍⚖️ **LLM Judge Reasoning:** {case['judge_reasoning']}")
                st.caption(f"**Generated Answer Preview:** {case['answer_preview']}")

# =====================================================================
# TAB 3: INDEXED CORPUS EXPLORER
# =====================================================================
with tab_corpus:
    chunks_data = load_json_file("data/processed/chunks.json")
    if chunks_data:
        st.subheader(f"{len(chunks_data)} Contextual Chunks Indexed Across 5 JKU Documents")
        doc_counts = {}
        for c in chunks_data:
            doc_counts[c["source"]] = doc_counts.get(c["source"], 0) + 1

        for doc_name, count in doc_counts.items():
            st.markdown(f"- **`{doc_name}`** — `{count}` indexed chunks")

        st.markdown("---")
        selected_doc = st.selectbox("Filter chunks by document:", list(doc_counts.keys()))
        filtered = [c for c in chunks_data if c["source"] == selected_doc][:10]
        st.caption(f"Showing first 10 chunks of `{selected_doc}`:")
        for c in filtered:
            with st.expander(f"Chunk #{c['chunk_id']} | Page {c['page']} | Section: {c['section']}"):
                st.code(c.get("contextual_text", c["text"]), language="markdown")