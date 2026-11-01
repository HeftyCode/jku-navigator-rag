import json
from pathlib import Path
import streamlit as st
from src.generation.generator import JKUNavigatorGenerator

st.set_page_config(
    page_title="JKU & Austrian Student Navigator",
    page_icon="🎓",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading Multilingual Embedder, mMARCO Reranker & Qdrant Index...")
def get_rag_engine() -> JKUNavigatorGenerator:
    return JKUNavigatorGenerator()


@st.cache_data(ttl=60)
def load_json_file(filepath: str, _mtime: float = 0.0):
    path = Path(filepath)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def read_cached_json(filepath: str):
    path = Path(filepath)
    mtime = path.stat().st_mtime if path.exists() else 0.0
    return load_json_file(filepath, _mtime=mtime)


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
        - **Parser:** `PyMuPDF4LLM` (Table-header & 128-token aware)
        - **Dense Search:** `paraphrase-multilingual-MiniLM-L12-v2` + `Qdrant`
        - **Sparse Search:** `BM25Okapi` + Snowball DE/EN Stemmers
        - **Fusion:** Reciprocal Rank Fusion ($k=60$)
        - **Reranker:** Multilingual Cross-Encoder (`mmarco-mMiniLMv2-L12-H384-v1`)
        - **Guardrails:** Programmatic Citation Verifier + Rerank Confidence Gate
        """
    )

st.title("🎓 JKU Linz & Austrian Student Life Navigator")
st.markdown(
    "Ask questions in **English or German** across **27 official JKU documents (3,046 chunks)**: TNF Bachelor curricula, "
    "credit transfer tables, examination law (*Satzungsteil Studienrecht*), and campus regulations (*Hausordnung*, *Parkordnung*, Library & ZID IT)."
)

tab_chat, tab_eval, tab_corpus = st.tabs([
    "Bilingual RAG Assistant",
    "4-Arm Ablation & Evaluation Dashboard",
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

    user_input = st.chat_input(
        "Ask a question about JKU regulations, courses, parking, or credit transfers...",
        max_chars=500,
    )
    active_query = preset_query or user_input

    if active_query:
        try:
            bot = get_rag_engine()
            with st.spinner("1️⃣ Expanding query bilingually ➔ 2️⃣ Hybrid Search + RRF ➔ 3️⃣ mMARCO Reranking ➔ 4️⃣ Verifying citations..."):
                st.session_state["last_rag_result"] = bot.answer_question(
                    user_query=active_query,
                    target_language=lang_choice,
                    top_n_chunks=top_n,
                )
        except Exception:
            st.error("The RAG service encountered a temporary upstream rate limit or error. Please try again in a few seconds.")

    if "last_rag_result" in st.session_state:
        result = st.session_state["last_rag_result"]

        with st.chat_message("user"):
            st.write(result["user_query"])

        with st.chat_message("assistant"):
            status_col1, status_col2, status_col3 = st.columns([1.2, 1.2, 2.2])
            if result["is_supported"]:
                status_col1.success("Grounded in JKU Corpus")
            else:
                status_col1.warning("Abstention Triggered (Unsupported)")

            cit_audit = result.get("citation_audit", {})
            if cit_audit.get("all_citations_valid", True):
                status_col2.success(f"Citations Verified ({result.get('latency_ms', 0)} ms)")
            else:
                status_col2.warning(f"Filtered {cit_audit.get('hallucinated_citations_removed', 0)} Unverified Citation(s)")

            status_col3.info(f"**Bilingual Expansion:** `{result['bilingual_search_query']}`")

            st.markdown("### Answer")
            st.markdown(result["answer"])
            st.caption(f"Answered by `{result.get('model_used', 'Groq LLM')}` in `{result.get('latency_ms', 0)} ms`")

            if result["citations"]:
                st.markdown("#### Programmatically Verified Citations (Matched Against Retrieved Chunks)")
                for cit in result["citations"]:
                    st.markdown(
                        f"- **`[{cit['source']}, Page {cit['page']}]`** *(Section: {cit['section']})* — {cit['evidence_summary']}"
                    )

            with st.expander("Inspect Under-the-Hood Hybrid Retrieval & Multilingual Reranker Scores"):
                for rank_idx, chunk in enumerate(result["retrieved_chunks"], 1):
                    d_rank = f"#{chunk['dense_rank']}" if chunk.get("dense_rank") is not None else "—"
                    b_rank = f"#{chunk['bm25_rank']}" if chunk.get("bm25_rank") is not None else "—"
                    st.markdown(
                        f"**Rank #{rank_idx}** | **`{chunk['source']}` (Page {chunk['page']})** | "
                        f"Section: `{chunk['section']}`  \n"
                        f"**mMARCO Cross-Encoder Score:** `{chunk['rerank_score']}` | "
                        f"**RRF Score:** `{chunk['rrf_score']}` "
                        f"*(Dense Vector Rank: `{d_rank}`, Snowball BM25 Rank: `{b_rank}`)*"
                    )
                    st.code(chunk.get("contextual_text", chunk["text"]), language="markdown")

# =====================================================================
# TAB 2: 4-ARM ABLATION & EVALUATION DASHBOARD
# =====================================================================
with tab_eval:
    eval_data = read_cached_json("data/processed/eval_results.json")

    if not eval_data:
        st.warning("No evaluation report found yet. Run `python -m src.evaluation.eval_runner` first.")
    else:
        st.subheader("1. 4-Arm Retrieval Architecture Ablation Study (100 Non-Circular Queries, Top-5)")
        st.caption(
            f"Last benchmarked: {eval_data.get('timestamp')} across {eval_data.get('corpus_chunks', 3046)} chunks | "
            "Reports Page-Level Recall@5 (with 95% Wilson Score Confidence Intervals) and Mean Reciprocal Rank (MRR)."
        )

        ab = eval_data["retrieval_100_ablation"]["ablation_results"]
        base_hit = ab["naive_dense"]["exact_chunk_hit_rate_pct"]
        base_mrr = ab["naive_dense"]["mrr"]

        cols = st.columns(4)
        arm_configs = [
            ("1️⃣ Naive Dense Vector", "naive_dense", False),
            ("2️⃣ Sparse BM25 (Snowball)", "sparse_bm25", True),
            ("3️⃣ Hybrid RRF (No Rerank)", "hybrid_rrf_no_rerank", True),
            ("4️⃣ Hybrid RRF + mMARCO", "hybrid_reranked", True),
        ]

        for col, (title, key, show_delta) in zip(cols, arm_configs):
            if key not in ab:
                continue
            r = ab[key]
            ci = r.get("page_hit_ci_95", [0.0, 0.0])
            hit_diff = r["exact_chunk_hit_rate_pct"] - base_hit
            mrr_diff = r["mrr"] - base_mrr

            with col:
                st.markdown(f"#### {title}")
                st.metric(
                    "Page Hit Rate @ 5",
                    f"{r['exact_chunk_hit_rate_pct']}%",
                    delta=f"{hit_diff:+.1f} pp vs Dense" if show_delta else None,
                )
                st.caption(f"95% Wilson CI: `{ci[0]}% – {ci[1]}%`")
                st.metric(
                    "Mean Reciprocal Rank (MRR)",
                    f"{r['mrr']:.3f}",
                    delta=f"{mrr_diff:+.3f} vs Dense" if show_delta else None,
                )
                st.metric("Document Hit Rate @ 5", f"{r['document_hit_rate_pct']}%")

        st.markdown("---")
        st.subheader("2. End-to-End Generation, Citation & Abstention Audit")
        gen = eval_data["generation_audit"]

        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("Doc Retrieval Accuracy", f"{gen['correct_document_retrieval_pct']}%")
        g2.metric("Answer Relevance", f"{gen['answer_relevance_pct']}%")
        g3.metric("Context Faithfulness", f"{gen['context_support_faithfulness_pct']}%")
        g4.metric("Verified Citations", f"{gen.get('citation_verification_pct', 100.0)}%")
        g5.metric("Hallucination Rate", f"{gen['hallucination_rate_pct']}%", delta_color="inverse")

        st.markdown("#### Detailed Audit Log (Every Test Case, Model Used & Judge Reasoning)")
        for idx, case in enumerate(gen["cases"], 1):
            badge = "PASS" if (case["supported_by_context"] and not case["hallucinated"]) else "FLAGGED BY JUDGE"
            with st.expander(f"Case #{idx} [{badge}]: {case['question']}"):
                st.write(
                    f"**Expected Source:** `{case['expected_source']}` | "
                    f"**Correct Doc Retrieved:** `{case['doc_retrieved_correctly']}` | "
                    f"**Citations Verified:** `{case.get('citations_verified', True)}`"
                )
                st.write(
                    f"**Generator Model:** `{case.get('generator_model', 'N/A')}` | "
                    f"**Judge Model:** `{case.get('judge_model', 'N/A')}` | "
                    f"**Latency:** `{case.get('latency_ms', 0)} ms`"
                )
                st.info(f"🧑‍⚖️ **LLM Judge Reasoning:** {case['judge_reasoning']}")
                st.caption(f"**Generated Answer Preview:** {case['answer_preview']}")

# =====================================================================
# TAB 3: INDEXED CORPUS EXPLORER
# =====================================================================
with tab_corpus:
    chunks_data = read_cached_json("data/processed/chunks.json")
    if chunks_data:
        doc_counts = {}
        for c in chunks_data:
            doc_counts[c["source"]] = doc_counts.get(c["source"], 0) + 1

        st.subheader(
            f"{len(chunks_data)} Contextual Chunks Indexed Across {len(doc_counts)} JKU Documents"
        )

        for doc_name, count in sorted(doc_counts.items()):
            st.markdown(f"- **`{doc_name}`** — `{count}` indexed chunks")

        st.markdown("---")
        selected_doc = st.selectbox("Filter chunks by document:", sorted(doc_counts.keys()))
        filtered = [c for c in chunks_data if c["source"] == selected_doc][:10]
        st.caption(f"Showing first 10 chunks of `{selected_doc}`:")
        for c in filtered:
            with st.expander(f"Chunk #{c['chunk_id']} | Page {c['page']} | Section: {c['section']}"):
                st.code(c.get("contextual_text", c["text"]), language="markdown")