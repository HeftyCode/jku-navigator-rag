import os
import re
import time
import json
from typing import List, Optional, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from groq import Groq, RateLimitError, BadRequestError
from src.config import settings
from src.retrieval.hybrid_search import HybridRetriever

load_dotenv()

# --- 1. Pydantic Schemas for Predictable Production Outputs ---

class QueryExpansion(BaseModel):
    detected_language: Literal["en", "de"] = Field(
        description="The language of the user's question ('en' for English, 'de' for German)."
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
        description="Comprehensive answer in the target language with inline citations like [Bachelor_AI_Studienleitfaden.pdf, Page 12]."
    )
    citations: List[CitationItem] = Field(
        description="List of specific document sources used to construct the answer."
    )
    is_supported_by_context: bool = Field(
        description="True if the retrieved documents contain the answer; False if the system must abstain."
    )


# --- 2. The Bilingual RAG Generator Pipeline ---

class JKUNavigatorGenerator:
    def __init__(self):
        if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "your_copied_key_here":
            raise ValueError("GROQ_API_KEY is missing! Please set your free key inside the .env file.")
        
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.fallback_models = self._get_available_fallback_models()
        self.model_idx = 0
        self.model = self.fallback_models[self.model_idx]
        self.retriever = HybridRetriever()

    def _get_available_fallback_models(self) -> List[str]:
        """Builds a prioritized list of active free Groq chat models."""
        preferred = [
            settings.LLM_MODEL,
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ]
        unique_preferred = list(dict.fromkeys(preferred))
        try:
            available_ids = {m.id for m in self.client.models.list().data}
            active = [m for m in unique_preferred if m in available_ids]
            # Also append any other active non-audio/non-guard models on the account as backup
            for mid in sorted(available_ids):
                if mid not in active and not any(skip in mid.lower() for skip in ["whisper", "guard", "tts", "embed", "vision", "orpheus", "allam", "playai"]):
                    active.append(mid)
            if active:
                print(f"Active Groq Model Pool: {active[:5]}")
                return active[:5]
        except Exception:
            pass
        return unique_preferred

    @staticmethod
    def _extract_json_object(raw_text: str) -> str:
        """Cleanly extracts a JSON object {...} even if the model wraps it in markdown or reasoning text."""
        if not raw_text:
            raise ValueError("Empty response from model")
        cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            return match.group(0)
        return cleaned

    def chat_with_retry(self, messages: List[dict], temperature: float = 0.0) -> str:
        """
        Production Fault-Tolerant LLM Caller:
        1. Auto-switches models on Daily Limit (TPD).
        2. Waits and retries on Per-Minute Limit (TPM).
        3. Falls back from strict server-side json_object mode to regex JSON extraction if a smaller model throws Error 400.
        """
        max_attempts = 8
        use_strict_json_mode = True

        for attempt in range(max_attempts):
            current_model = self.fallback_models[self.model_idx]
            try:
                kwargs = {
                    "model": current_model,
                    "messages": messages,
                    "temperature": temperature if use_strict_json_mode else 0.2,
                }
                if use_strict_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}

                resp = self.client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content or ""
                json_str = self._extract_json_object(content)
                # Verify it parses as valid JSON before returning
                json.loads(json_str)
                return json_str

            except BadRequestError as e:
                # Groq Error 400: json_validate_failed on smaller reasoning models
                print(f"Strict JSON validator tripped on {current_model}. Retrying with client-side JSON parser...")
                use_strict_json_mode = False
                continue

            except (json.JSONDecodeError, ValueError):
                print(f"Malformed JSON output from {current_model}. Retrying...")
                use_strict_json_mode = False
                continue

            except RateLimitError as e:
                err_msg = str(e).lower()
                if "tokens per day" in err_msg or "tpd" in err_msg:
                    if len(self.fallback_models) > 1:
                        old_model = current_model
                        self.model_idx = (self.model_idx + 1) % len(self.fallback_models)
                        self.model = self.fallback_models[self.model_idx]
                        use_strict_json_mode = True
                        print(f"Daily limit reached on {old_model}. Auto-switching to {self.model}...")
                        continue
                wait_sec = 8.0
                print(f"Rate limit (TPM) paused on {current_model}. Waiting {wait_sec}s ({attempt + 1}/{max_attempts})...")
                time.sleep(wait_sec)

        raise RuntimeError("All retry attempts exhausted across the Groq model pool.")

    def expand_query_bilingually(self, user_query: str) -> QueryExpansion:
        system_prompt = (
            "You are a bilingual search query optimizer for Johannes Kepler University (JKU) Linz.\n"
            "Given a user question in English or German:\n"
            "1. Detect if the user's question is in English ('en') or German ('de').\n"
            "2. Produce a 'bilingual_search_query' containing the core English terms AND their exact German "
            "translations, plurals, and compound regulation terms (for example, if asked about 'dog', include "
            "'dog dogs pet Hund Hunde Haustier Haustierverbot Leine Maulkorb Hausordnung'; if asked about 'VL and KV', "
            "include 'VL KV Vorlesung Combined Course Kombikurs lecture exercise').\n"
            "Do NOT include conversational filler words like 'Am I allowed to' or 'what are the'.\n"
            "Respond ONLY with valid JSON matching this exact structure:\n"
            '{"detected_language": "en", "bilingual_search_query": "..."}'
        )

        raw_content = self.chat_with_retry(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            temperature=0.0,
        )
        return QueryExpansion(**json.loads(raw_content))

    def answer_question(
        self,
        user_query: str,
        target_language: Optional[Literal["auto", "en", "de"]] = "auto",
        top_n_chunks: int = 5,
    ) -> dict:
        expansion = self.expand_query_bilingually(user_query)
        reply_lang = expansion.detected_language if target_language == "auto" else target_language
        lang_name = "English" if reply_lang == "en" else "German (Deutsch)"

        retrieved_chunks = self.retriever.retrieve_and_rerank(
            query=expansion.bilingual_search_query,
            retrieve_top_k=20,
            final_top_n=top_n_chunks,
        )

        context_blocks = []
        for idx, c in enumerate(retrieved_chunks, 1):
            context_blocks.append(
                f"--- SOURCE CHUNK #{idx} (Rerank Score: {c['rerank_score']}) ---\n"
                f"{c.get('contextual_text', c['text'])}"
            )
        formatted_context = "\n\n".join(context_blocks)

        system_prompt = f"""You are the official JKU & Austrian Student Navigator AI Assistant.
You must answer the student's question using ONLY the provided JKU source chunks below.

CRITICAL RULES:
1. LANGUAGE: Write your entire 'answer' and 'evidence_summary' fields in **{lang_name}** (even if the source PDF chunks are in a different language!).
2. INLINE CITATIONS: Every factual claim in your 'answer' MUST include an inline citation showing the exact PDF filename and Page number in brackets, like [Bachelor_AI_Studienleitfaden.pdf, Page 14] or [2116_Hausordnung_V17_2026_10_08.pdf, Page 4].
3. STRICT ZERO-EXTRAPOLATION (NO HALLUCINATION): State ONLY facts, definitions, and details that are EXPLICITLY written word-for-word in the provided chunks. NEVER use outside knowledge, never infer what an acronym means unless the chunk explicitly defines it, and never add extra examples. If a detail is missing from the chunks, omit it completely. If the chunks do not answer the question at all, set "is_supported_by_context" to false, leave "citations" empty, and politely state in {lang_name} that the current JKU documents do not contain this information.
4. OUTPUT FORMAT: Return ONLY a valid JSON object matching this exact schema:
{{
  "answer": "Detailed markdown response in {lang_name} with inline [Filename.pdf, Page X] citations...",
  "citations": [
    {{
      "source": "Filename.pdf",
      "page": 1,
      "section": "Section Name",
      "evidence_summary": "Brief summary of what this specific page states"
    }}
  ],
  "is_supported_by_context": true
}}"""

        user_message = (
            f"RETRIEVED JKU DOCUMENTS:\n{formatted_context}\n\n"
            f"STUDENT QUESTION: {user_query}\n"
            f"Remember to respond in {lang_name} as valid JSON."
        )

        raw_output = self.chat_with_retry(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
        )

        validated_response = RAGResponse(**json.loads(raw_output))

        return {
            "user_query": user_query,
            "detected_language": expansion.detected_language,
            "reply_language": reply_lang,
            "bilingual_search_query": expansion.bilingual_search_query,
            "answer": validated_response.answer,
            "is_supported": validated_response.is_supported_by_context,
            "citations": [c.model_dump() for c in validated_response.citations],
            "retrieved_chunks": retrieved_chunks,
        }

    def close(self):
        self.retriever.close()