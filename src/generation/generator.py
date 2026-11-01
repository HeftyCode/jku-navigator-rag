import re
import time
import json
import threading
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from groq import Groq, RateLimitError, BadRequestError, APIConnectionError, InternalServerError
from src.config import settings
from src.retrieval.hybrid_search import HybridRetriever

load_dotenv()

CITATION_REGEX = re.compile(r"\[([^\[\],]+\.pdf),\s*Page\s*(\d+)\]")

# Strict allow-list of pure instruction/chat models (no web-search/audio/guard models)
ALLOWED_GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
]


class QueryExpansion(BaseModel):
    detected_language: str = Field(
        default="en",
        description="ISO 639-1 language code of the user's question (e.g., 'en', 'de', 'es', 'tr').",
    )
    bilingual_search_query: str = Field(
        description="Search string containing core keywords in BOTH English and German for hybrid retrieval."
    )


class CitationItem(BaseModel):
    source: str = Field(description="Exact PDF filename from the context header")
    page: int = Field(description="Exact page number from the context header")
    section: str = Field(description="Section title from the context header")
    evidence_summary: str = Field(description="Brief 1-sentence summary of what this source proves")


class RAGResponse(BaseModel):
    answer: str = Field(
        description="Comprehensive answer in the target language with inline citations like [Filename.pdf, Page 12]."
    )
    citations: List[CitationItem] = Field(
        default_factory=list,
        description="List of specific document sources used to construct the answer.",
    )
    is_supported_by_context: bool = Field(
        description="True if the retrieved documents contain the answer; False if the system must abstain."
    )


def verify_citations(
    answer: str, citations: List[CitationItem], retrieved_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Programmatic Citation Auditor:
    Verifies every inline [Filename.pdf, Page X] and structured CitationItem against
    the actual retrieved chunks passed to the LLM.
    """
    allowed_pairs = {(c["source"], int(c["page"])) for c in retrieved_chunks}
    chunk_lookup = {(c["source"], int(c["page"])): c for c in retrieved_chunks}

    inline_matches = [(src.strip(), int(pg)) for src, pg in CITATION_REGEX.findall(answer)]
    invalid_inline = [pair for pair in inline_matches if pair not in allowed_pairs]

    verified_list: List[Dict[str, Any]] = []
    seen_pairs = set()

    for cit in citations:
        pair = (cit.source.strip(), int(cit.page))
        if pair in allowed_pairs and pair not in seen_pairs:
            real_chunk = chunk_lookup[pair]
            verified_list.append({
                "source": real_chunk["source"],
                "page": int(real_chunk["page"]),
                "section": real_chunk["section"],
                "evidence_summary": cit.evidence_summary,
                "verified": True,
            })
            seen_pairs.add(pair)
        elif pair not in allowed_pairs:
            invalid_inline.append(pair)

    # Also ensure any valid inline citation in the text appears in the verified citation cards
    for pair in inline_matches:
        if pair in allowed_pairs and pair not in seen_pairs:
            real_chunk = chunk_lookup[pair]
            verified_list.append({
                "source": real_chunk["source"],
                "page": int(real_chunk["page"]),
                "section": real_chunk["section"],
                "evidence_summary": f"Referenced directly in response from {real_chunk['doc_title']}.",
                "verified": True,
            })
            seen_pairs.add(pair)

    return {
        "verified_citations": verified_list,
        "hallucinated_citations_removed": len(invalid_inline),
        "all_citations_valid": len(invalid_inline) == 0,
    }


class JKUNavigatorGenerator:
    def __init__(self):
        if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "your_copied_key_here":
            raise ValueError("GROQ_API_KEY is missing! Please set your free key inside the .env file.")

        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self._lock = threading.Lock()
        self._cooldown_until: Dict[str, float] = {}
        self.fallback_models = self._get_available_fallback_models()
        self.last_used_model: str = self.fallback_models[0]
        self.retriever = HybridRetriever()

    def _get_available_fallback_models(self) -> List[str]:
        """Pins active models strictly to ALLOWED_GROQ_MODELS."""
        preferred = [settings.LLM_MODEL] + [m for m in ALLOWED_GROQ_MODELS if m != settings.LLM_MODEL]
        try:
            available_ids = {m.id for m in self.client.models.list().data}
            active = [m for m in preferred if m in available_ids]
            if active:
                print(f"Pinned Groq Model Pool: {active}")
                return active
        except Exception:
            pass
        return preferred[:3]

    def _pick_model(self) -> str:
        """Thread-safely picks the highest-priority model whose cooldown has expired."""
        now = time.time()
        with self._lock:
            for m in self.fallback_models:
                if self._cooldown_until.get(m, 0.0) <= now:
                    return m
            # If all are in cooldown, pick the one whose cooldown expires soonest
            return min(self.fallback_models, key=lambda m: self._cooldown_until.get(m, 0.0))

    def _mark_cooldown(self, model_name: str, seconds: float):
        with self._lock:
            self._cooldown_until[model_name] = time.time() + seconds

    @staticmethod
    def _extract_json_object(raw_text: str) -> str:
        """Uses JSONDecoder().raw_decode to safely parse the first valid JSON object without greedy regex bugs."""
        if not raw_text:
            raise ValueError("Empty response from model")
        cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        start_idx = cleaned.find("{")
        if start_idx == -1:
            raise ValueError("No JSON object found in output")
        obj, _ = json.JSONDecoder().raw_decode(cleaned[start_idx:])
        return json.dumps(obj, ensure_ascii=False)

    def chat_with_retry(self, messages: List[dict], schema_cls=None, temperature: float = 0.0) -> str:
        """
        Thread-safe, schema-validated LLM caller with exponential backoff + model cooldowns.
        """
        max_attempts = 8
        use_strict_json_mode = True

        for attempt in range(max_attempts):
            current_model = self._pick_model()
            try:
                kwargs = {
                    "model": current_model,
                    "messages": messages,
                    "temperature": temperature,
                }
                if use_strict_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}

                resp = self.client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content or ""
                json_str = self._extract_json_object(content)

                if schema_cls is not None:
                    schema_cls(**json.loads(json_str))

                with self._lock:
                    self.last_used_model = current_model
                return json_str

            except BadRequestError as e:
                err_body = str(e).lower()
                if "json_validate_failed" in err_body or "failed_generation" in err_body:
                    use_strict_json_mode = False
                    continue
                raise

            except (json.JSONDecodeError, ValueError, ValidationError):
                use_strict_json_mode = False
                continue

            except RateLimitError as e:
                err_msg = str(e).lower()
                if "tokens per day" in err_msg or "tpd" in err_msg:
                    self._mark_cooldown(current_model, seconds=1800.0)
                    use_strict_json_mode = True
                    print(f"Daily quota reached on {current_model}. Cooling down for 30m & rotating...")
                    continue
                wait_sec = min(4.0 * (1.5 ** attempt), 15.0)
                print(f"TPM limit on {current_model}. Sleeping {wait_sec:.1f}s...")
                time.sleep(wait_sec)

            except (APIConnectionError, InternalServerError):
                time.sleep(2.0 * (attempt + 1))

        raise RuntimeError("All retry attempts exhausted across the Groq model pool.")

    def expand_query_bilingually(self, user_query: str) -> QueryExpansion:
        """
        Zero-Contamination Bilingual Query Expander:
        Contains NO test-set examples. Expands any student query into EN + DE academic/regulatory terms.
        """
        system_prompt = (
            "You are a bilingual search query expansion engine for an Austrian university (JKU Linz).\n"
            "Given a user question in any language:\n"
            "1. Identify its ISO 639-1 language code ('en', 'de', etc.) in 'detected_language'.\n"
            "2. Produce a 'bilingual_search_query' containing the core nouns and technical terms in BOTH "
            "English and German (including singular/plural forms and standard Austrian university compound nouns, "
            "such as 'tuition fee fees Studienbeitrag ÖH-Beitrag' or 'leave of absence Beurlaubung').\n"
            "Do NOT include conversational filler words.\n"
            "Respond ONLY with valid JSON matching this schema:\n"
            '{"detected_language": "en", "bilingual_search_query": "..."}'
        )

        raw_content = self.chat_with_retry(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            schema_cls=QueryExpansion,
            temperature=0.0,
        )
        return QueryExpansion(**json.loads(raw_content))

    def answer_question(
        self,
        user_query: str,
        target_language: Optional[str] = "auto",
        top_n_chunks: int = 5,
    ) -> dict:
        t0 = time.perf_counter()
        expansion = self.expand_query_bilingually(user_query)

        effective_target = target_language if target_language in ("auto", "en", "de") else "auto"
        reply_lang = expansion.detected_language if effective_target == "auto" else effective_target
        lang_map = {"en": "English", "de": "German (Deutsch)"}
        lang_name = lang_map.get(reply_lang, f"the user's language ({reply_lang})")

        # Decoupled Retrieval: BM25 gets bilingual expansion; Dense & Reranker get the natural user question!
        retrieved_chunks = self.retriever.retrieve_and_rerank(
            search_query=expansion.bilingual_search_query,
            rerank_query=user_query,
            dense_query=f"{user_query} {expansion.bilingual_search_query}",
            retrieve_top_k=20,
            final_top_n=top_n_chunks,
        )

        # Pre-LLM Reranker Confidence Gate
        top_score = retrieved_chunks[0]["rerank_score"] if retrieved_chunks else -99.0
        if not retrieved_chunks or top_score < settings.MIN_RERANK_SCORE:
            abstain_msg = (
                "Die vorliegenden JKU-Dokumente enthalten leider keine passenden Informationen zu dieser Anfrage."
                if reply_lang == "de"
                else "The indexed JKU documents do not contain information matching this query."
            )
            return {
                "user_query": user_query,
                "detected_language": expansion.detected_language,
                "reply_language": reply_lang,
                "bilingual_search_query": expansion.bilingual_search_query,
                "answer": abstain_msg,
                "is_supported": False,
                "citations": [],
                "citation_audit": {"all_citations_valid": True, "hallucinated_citations_removed": 0},
                "model_used": self.last_used_model,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "retrieved_chunks": retrieved_chunks,
            }

        context_blocks = []
        for idx, c in enumerate(retrieved_chunks, 1):
            context_blocks.append(
                f"--- SOURCE CHUNK #{idx} | Exact Citation Tag: [{c['source']}, Page {c['page']}] | Section: {c['section']} ---\n"
                f"{c['contextual_text']}"
            )
        formatted_context = "\n\n".join(context_blocks)

        system_prompt = f"""You are the official JKU & Austrian Student Navigator AI Assistant.
Answer the student's question using ONLY the provided JKU source chunks below.

CRITICAL RULES:
1. LANGUAGE: Write your entire 'answer' and 'evidence_summary' fields in **{lang_name}**.
2. INLINE CITATIONS: Every factual claim MUST include the exact inline citation tag from the chunk header, formatted strictly as [Filename.pdf, Page X].
3. STRICT ZERO-EXTRAPOLATION: State ONLY facts, definitions, and rules explicitly written in the provided chunks. Never use outside knowledge or extrapolate. If the chunks do not answer the question, set "is_supported_by_context" to false, set "citations" to [], and politely state in {lang_name} that the documents do not contain this information.
4. OUTPUT FORMAT: Return ONLY a valid JSON object matching this exact schema:
{{
  "answer": "Detailed response in {lang_name} with inline [Filename.pdf, Page X] citations...",
  "citations": [
    {{
      "source": "Filename.pdf",
      "page": 1,
      "section": "Section Name",
      "evidence_summary": "Brief summary of what this page states"
    }}
  ],
  "is_supported_by_context": true
}}"""

        user_message = (
            f"RETRIEVED JKU DOCUMENTS:\n{formatted_context}\n\n"
            f"STUDENT QUESTION: {user_query}\n"
            f"Return valid JSON in {lang_name}."
        )

        raw_output = self.chat_with_retry(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            schema_cls=RAGResponse,
            temperature=0.0,
        )

        validated = RAGResponse(**json.loads(raw_output))

        # Programmatically verify citations & enforce is_supported=False => citations=[]
        if not validated.is_supported_by_context:
            verified_citations = []
            citation_audit = {"all_citations_valid": True, "hallucinated_citations_removed": 0}
        else:
            audit_res = verify_citations(validated.answer, validated.citations, retrieved_chunks)
            verified_citations = audit_res["verified_citations"]
            citation_audit = {
                "all_citations_valid": audit_res["all_citations_valid"],
                "hallucinated_citations_removed": audit_res["hallucinated_citations_removed"],
            }

        return {
            "user_query": user_query,
            "detected_language": expansion.detected_language,
            "reply_language": reply_lang,
            "bilingual_search_query": expansion.bilingual_search_query,
            "answer": validated.answer,
            "is_supported": validated.is_supported_by_context,
            "citations": verified_citations,
            "citation_audit": citation_audit,
            "model_used": self.last_used_model,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "retrieved_chunks": retrieved_chunks,
        }

    def close(self):
        self.retriever.close()