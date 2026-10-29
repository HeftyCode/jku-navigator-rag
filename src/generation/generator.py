import os
import json
from typing import List, Optional, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from groq import Groq
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
        self.model = self._resolve_active_model()
        self.retriever = HybridRetriever()

    def _resolve_active_model(self) -> str:
        """Verifies the configured LLM is active on Groq, or auto-selects the best available free model."""
        preferred_Order = [
            settings.LLM_MODEL,
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
        ]
        try:
            available_ids = {m.id for m in self.client.models.list().data}
            for candidate in preferred_Order:
                if candidate in available_ids:
                    print(f"Using active Groq LLM: {candidate}")
                    return candidate
        except Exception:
            pass
        return settings.LLM_MODEL

    def expand_query_bilingually(self, user_query: str) -> QueryExpansion:
        """
        Expands an English or German question into high-signal EN+DE keywords and synonyms
        (including German plurals and compound nouns) so BM25 and Dense search hit both languages.
        """
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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        raw_json = json.loads(response.choices[0].message.content)
        return QueryExpansion(**raw_json)

    def answer_question(
        self,
        user_query: str,
        target_language: Optional[Literal["auto", "en", "de"]] = "auto",
        top_n_chunks: int = 5,
    ) -> dict:
        """
        Full End-to-End Pipeline:
        1. Bilingual Query Expansion (EN + DE)
        2. Hybrid Search (Dense + BM25) + RRF + Cross-Encoder Reranking
        3. Grounded LLM Generation with Pydantic Validation & Citations
        """
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
3. NO HALLUCINATION (ABSTENTION): If the provided chunks do not contain the answer, set "is_supported_by_context" to false, leave "citations" empty, and politely state in {lang_name} that the current JKU documents do not contain this information. Never guess or invent rules.
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

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        raw_output = json.loads(completion.choices[0].message.content)
        validated_response = RAGResponse(**raw_output)

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


if __name__ == "__main__":
    bot = JKUNavigatorGenerator()
    try:
        # Test 1: Asking in ENGLISH about a GERMAN document (Hausordnung / Parkordnung)
        q1 = "Am I allowed to bring my dog or ride an e-scooter inside the JKU campus buildings?"
        print(f"\n==========================================")
        print(f"🇬🇧 TEST 1 (English Question -> German PDF): {q1}")
        res1 = bot.answer_question(q1)
        print(f"Bilingual Query Generated: {res1['bilingual_search_query']}")
        print(f"\nAnswer ({res1['reply_language'].upper()}):\n{res1['answer']}")
        print(f"\nStructured Citations ({len(res1['citations'])}):")
        for cit in res1["citations"]:
            print(f"  - [{cit['source']}, Page {cit['page']}] ({cit['section']}): {cit['evidence_summary']}")

        # Test 2: Asking in GERMAN about an ENGLISH document (Bachelor AI Study Guide)
        q2 = "Was ist der Unterschied zwischen einer VL und einer KV im AI Bachelorstudium, und gibt es Anwesenheitspflicht?"
        print(f"\n==========================================")
        print(f"🇩🇪 TEST 2 (German Question -> English PDF): {q2}")
        res2 = bot.answer_question(q2)
        print(f"Bilingual Query Generated: {res2['bilingual_search_query']}")
        print(f"\nAnswer ({res2['reply_language'].upper()}):\n{res2['answer']}")
        print(f"\nStructured Citations ({len(res2['citations'])}):")
        for cit in res2["citations"]:
            print(f"  - [{cit['source']}, Page {cit['page']}] ({cit['section']}): {cit['evidence_summary']}")
    finally:
        bot.close()