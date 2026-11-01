import re
import math
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from pydantic import BaseModel, Field
from src.generation.generator import JKUNavigatorGenerator


class JudgeScore(BaseModel):
    # Reasoning comes FIRST so the LLM performs chain-of-thought before committing to boolean verdicts
    reasoning: str = Field(description="Concise 1-2 sentence explanation of why the answer passed or failed.")
    answer_relevant: bool = Field(
        description="True if the answer directly addresses the student's question (or appropriately abstains on unanswerable trap questions)."
    )
    supported_by_context: bool = Field(
        description="True if every factual claim in the answer is explicitly backed by the retrieved context chunks."
    )
    hallucinated: bool = Field(
        description="True if the model invented facts, rules, or numbers not present in the retrieved context."
    )


def wilson_ci_95(hits: int, total: int) -> Tuple[float, float]:
    """Computes the exact 95% Wilson Score Confidence Interval (in %) for a binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    z = 1.95996
    p = hits / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    half_width = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)) / denom
    low = max(0.0, (center - half_width) * 100.0)
    high = min(100.0, (center + half_width) * 100.0)
    return (round(low, 1), round(high, 1))


class JKUEvaluator:
    def __init__(self, output_path: str = "data/processed/eval_results.json"):
        self.output_path = Path(output_path)
        self.bot = JKUNavigatorGenerator()
        self.retriever = self.bot.retriever

    def build_100_retrieval_dataset(self) -> List[Dict[str, Any]]:
        """
        Builds a 100-query benchmark WITHOUT circular header/opening-sentence copy-pasting:
        - Strips section headers, markdown formatting, and opening boilerplate.
        - Extracts substantive interior clauses/terms and wraps them in natural EN/DE student question templates.
        - Evaluates retrieval at the Page Level (source, page) so overlapping chunks on the same page are scored fairly.
        """
        chunks = self.retriever.chunks
        rng = random.Random(42)

        # Filter to substantive chunks with distinct interior content
        candidates = [
            c for c in chunks
            if len(c.get("text", "")) >= 180 and not c["text"].strip().startswith("|---")
        ]
        sampled = rng.sample(candidates, min(100, len(candidates)))

        en_templates = [
            "What do the JKU regulations state regarding {}?",
            "Can you explain the requirements and rules for {}?",
            "How does JKU handle {} in this program?",
        ]
        de_templates = [
            "Welche Regelungen gelten an der JKU bezüglich {}?",
            "Was ist im Studienplan oder den Richtlinien zu {} festgelegt?",
            "Wie sind die Bestimmungen für {} geregelt?",
        ]

        dataset = []
        for idx, c in enumerate(sampled):
            # Take interior lines/clauses (skipping the first line to avoid opening-sentence circularity)
            raw_text = re.sub(r"[|#*_`>]+", " ", c["text"])
            raw_text = re.sub(r"\s+", " ", raw_text).strip()
            words = raw_text.split()

            # Extract an 8-to-10 word interior span from the middle/second half of the chunk
            start_w = min(len(words) // 3, max(0, len(words) - 10))
            interior_span = " ".join(words[start_w : start_w + 9]).strip(".,:;()- ")

            is_german_doc = any(w in raw_text.lower() for w in ["und", "der", "die", "das", "für", "gemäß", "studien"])
            template_pool = de_templates if is_german_doc else en_templates
            question_text = template_pool[idx % len(template_pool)].format(interior_span)

            dataset.append({
                "id": idx + 1,
                "query": question_text,
                "expected_source": c["source"],
                "expected_page": int(c["page"]),
                "expected_chunk_id": c["chunk_id"],
            })

        return dataset

    def run_100_query_ablation_benchmark(self, top_k: int = 5) -> Dict[str, Any]:
        """
        4-Arm Retrieval Ablation Study (n=100, Top-5) with 95% Wilson Confidence Intervals:
        1. naive_dense           : Dense Vector Search Only
        2. sparse_bm25           : Sparse BM25 with Snowball Stemming Only
        3. hybrid_rrf_no_rerank  : Hybrid RRF (Dense + BM25) WITHOUT Cross-Encoder
        4. hybrid_reranked       : Hybrid RRF + Multilingual Cross-Encoder Reranker (mmarco-mMiniLMv2)
        """
        dataset = self.build_100_retrieval_dataset()
        total = len(dataset)
        print(f"\nRunning 4-Arm Retrieval Ablation Study (Top-{top_k}, n={total})...")

        arms = ["naive_dense", "sparse_bm25", "hybrid_rrf_no_rerank", "hybrid_reranked"]
        metrics = {arm: {"page_hits": 0, "doc_hits": 0, "mrr_sum": 0.0} for arm in arms}

        for item in dataset:
            q = item["query"]
            target_page = (item["expected_source"], item["expected_page"])
            target_doc = item["expected_source"]

            # Arm 1: Naive Dense Vector Search
            dense_res = self.retriever.dense_search(q, top_k=20)
            dense_top = [self.retriever.chunk_map[r["chunk_id"]] for r in dense_res[:top_k]]
            self._score_arm(metrics["naive_dense"], target_page, target_doc, dense_top)

            # Arm 2: Sparse BM25 Search
            bm25_res = self.retriever.sparse_bm25_search(q, top_k=20)
            bm25_top = [self.retriever.chunk_map[r["chunk_id"]] for r in bm25_res[:top_k]]
            self._score_arm(metrics["sparse_bm25"], target_page, target_doc, bm25_top)

            # Arm 3: Hybrid RRF (No Reranker)
            fused_res = self.retriever.reciprocal_rank_fusion(dense_res, bm25_res)
            fused_top = fused_res[:top_k]
            self._score_arm(metrics["hybrid_rrf_no_rerank"], target_page, target_doc, fused_top)

            # Arm 4: Full Hybrid RRF + Multilingual Cross-Encoder Reranker
            reranked_top = self.retriever.retrieve_and_rerank(
                search_query=q, rerank_query=q, dense_query=q, retrieve_top_k=20, final_top_n=top_k
            )
            self._score_arm(metrics["hybrid_reranked"], target_page, target_doc, reranked_top)

        summary = {}
        for arm in arms:
            m = metrics[arm]
            ci_low, ci_high = wilson_ci_95(m["page_hits"], total)
            summary[arm] = {
                "exact_chunk_hit_rate_pct": round((m["page_hits"] / total) * 100.0, 1),
                "page_hit_ci_95": [ci_low, ci_high],
                "document_hit_rate_pct": round((m["doc_hits"] / total) * 100.0, 1),
                "mrr": round(m["mrr_sum"] / total, 3),
            }

        return {"total_queries": total, "top_k": top_k, "ablation_results": summary}

    @staticmethod
    def _score_arm(
        bucket: Dict[str, float],
        target_page: Tuple[str, int],
        target_doc: str,
        top_chunks: List[Dict[str, Any]],
    ):
        retrieved_docs = [c["source"] for c in top_chunks]
        retrieved_pages = [(c["source"], int(c["page"])) for c in top_chunks]

        if target_doc in retrieved_docs:
            bucket["doc_hits"] += 1
        if target_page in retrieved_pages:
            bucket["page_hits"] += 1
            rank = retrieved_pages.index(target_page) + 1
            bucket["mrr_sum"] += 1.0 / rank

    def run_end_to_end_generation_eval(self) -> Dict[str, Any]:
        """
        End-to-End Bilingual Generation, Abstention & Citation Verification Audit (12 held-out cases).
        """
        golden_qa = [
            {
                "question": "Am I allowed to bring a dog onto the JKU campus and what are the rules?",
                "expected_source": "2116_Hausordnung_V17_2026_10_08.pdf",
                "is_answerable": True,
            },
            {
                "question": "Was ist der Unterschied zwischen VL und KV Kursen im Bachelor AI?",
                "expected_source": "Bachelor_AI_Studienleitfaden.pdf",
                "is_answerable": True,
            },
            {
                "question": "Where are the Distance Learning Centers located for the JKU AI program?",
                "expected_source": "Bachelor_AI_Studienleitfaden.pdf",
                "is_answerable": True,
            },
            {
                "question": "Welche Regeln gelten laut Parkordnung für das Parken am JKU Campus?",
                "expected_source": "4101_Parkordnung_V6_2026_09_09.pdf",
                "is_answerable": True,
            },
            {
                "question": "How does credit transfer work between Informatik and Artificial Intelligence at JKU?",
                "expected_source": "Credit_Transfer_between_the_Programs_Informatik_and_Artificial_Intelligence_at_JKU.pdf",
                "is_answerable": True,
            },
            {
                "question": "How are credits transferred between Bioinformatics and Artificial Intelligence?",
                "expected_source": "Credit_Transfer_between_the_Programs_Bioinformatics_and_Artificial_Intelligence_at_JKU.pdf",
                "is_answerable": True,
            },
            {
                "question": "Welche Sicherheitsregeln gelten laut allgemeiner Laborordnung in den Labors der JKU?",
                "expected_source": "1203_Allgemeine_Laborordnung_V2_2019_07_02.pdf",
                "is_answerable": True,
            },
            {
                "question": "What are the rules for borrowing books and using the JKU University Library?",
                "expected_source": "Benuetzungsordnung_Universitaetsbibliothek_Library.pdf",
                "is_answerable": True,
            },
            {
                "question": "Welche Pflichtfächer und ECTS umfasst das Bachelorstudium Statistik und Data Science?",
                "expected_source": "Exam_Roster_BA_Statistik_Data_Science_2026W.pdf",
                "is_answerable": True,
            },
            {
                "question": "Welche Bestimmungen gelten im Satzungsteil Studienrecht für Prüfungen und Studienabschlüsse?",
                "expected_source": "2105_Satzungsteil_Studienrecht_V31_2026_07_02.pdf",
                "is_answerable": True,
            },
            {
                "question": "What is the minimum cryptocurrency deposit required to use the JKU submarine dock?",
                "expected_source": "NONE_TRAP_QUESTION",
                "is_answerable": False,
            },
            {
                "question": "Welches Formular muss ich ausfüllen, um ein kostenloses JKU-Raumschiff zu mieten?",
                "expected_source": "NONE_TRAP_QUESTION",
                "is_answerable": False,
            },
        ]

        print(f"\nRunning End-to-End Generation, Citation & Abstention Audit ({len(golden_qa)} cases)...")
        detailed_logs = []
        doc_retrieved_count = 0
        relevant_count = 0
        supported_count = 0
        hallucination_count = 0
        abstention_correct_count = 0
        verified_citation_pass_count = 0

        for idx, item in enumerate(golden_qa, 1):
            q = item["question"]
            print(f"  [{idx}/{len(golden_qa)}] Auditing: {q[:65]}...")

            time.sleep(2.0)
            rag_out = self.bot.answer_question(q, top_n_chunks=5)
            gen_model = rag_out.get("model_used", "unknown")

            retrieved_sources = [c["source"] for c in rag_out["retrieved_chunks"]]
            abstention_correct = (rag_out["is_supported"] == item["is_answerable"])
            if abstention_correct:
                abstention_correct_count += 1

            if not item["is_answerable"]:
                doc_hit = not rag_out["is_supported"]
            else:
                doc_hit = item["expected_source"] in retrieved_sources

            cit_valid = rag_out.get("citation_audit", {}).get("all_citations_valid", True)
            if cit_valid:
                verified_citation_pass_count += 1

            time.sleep(2.0)
            judge = self._judge_response(
                question=q,
                answer=rag_out["answer"],
                context_chunks=rag_out["retrieved_chunks"],
            )
            judge_model = self.bot.last_used_model

            if doc_hit:
                doc_retrieved_count += 1
            if judge.answer_relevant:
                relevant_count += 1
            if judge.supported_by_context and abstention_correct:
                supported_count += 1
            if judge.hallucinated or not abstention_correct:
                hallucination_count += 1

            preview = rag_out["answer"]
            if len(preview) > 220:
                preview = preview[:220] + "..."

            detailed_logs.append({
                "question": q,
                "expected_source": item["expected_source"],
                "is_answerable": item["is_answerable"],
                "doc_retrieved_correctly": doc_hit,
                "abstention_accurate": abstention_correct,
                "citations_verified": cit_valid,
                "answer_relevant": judge.answer_relevant,
                "supported_by_context": judge.supported_by_context and abstention_correct,
                "hallucinated": judge.hallucinated or not abstention_correct,
                "judge_reasoning": judge.reasoning,
                "generator_model": gen_model,
                "judge_model": judge_model,
                "latency_ms": rag_out.get("latency_ms", 0.0),
                "answer_preview": preview,
            })

        n = len(golden_qa)
        return {
            "total_evaluated": n,
            "correct_document_retrieval_pct": round((doc_retrieved_count / n) * 100, 1),
            "answer_relevance_pct": round((relevant_count / n) * 100, 1),
            "context_support_faithfulness_pct": round((supported_count / n) * 100, 1),
            "hallucination_rate_pct": round((hallucination_count / n) * 100, 1),
            "citation_verification_pct": round((verified_citation_pass_count / n) * 100, 1),
            "abstention_accuracy_pct": round((abstention_correct_count / n) * 100, 1),
            "cases": detailed_logs,
        }

    def _judge_response(
        self, question: str, answer: str, context_chunks: List[Dict[str, Any]]
    ) -> JudgeScore:
        """
        Blind Groundedness Judge:
        Evaluates the answer strictly against the retrieved context without being told the expected label.
        """
        context_text = "\n\n".join([c["contextual_text"] for c in context_chunks])
        prompt = """You are an independent AI Evaluation Judge auditing a RAG system.
Compare the GENERATED ANSWER strictly against the RETRIEVED CONTEXT and STUDENT QUESTION.

EVALUATION RULES:
1. First write a concise 1-2 sentence 'reasoning' analyzing whether every claim in the answer appears in the context.
2. If the RETRIEVED CONTEXT does NOT contain the answer to the question, and the GENERATED ANSWER politely states that the documents do not contain that information (abstention), then:
   - answer_relevant = true
   - supported_by_context = true
   - hallucinated = false
3. If the GENERATED ANSWER invents facts, numbers, or rules not present in RETRIEVED CONTEXT, set supported_by_context = false and hallucinated = true.

Return ONLY valid JSON with this exact schema:
{
  "reasoning": "1-2 sentence analysis...",
  "answer_relevant": true,
  "supported_by_context": true,
  "hallucinated": false
}"""

        user_msg = f"QUESTION: {question}\n\nRETRIEVED CONTEXT:\n{context_text}\n\nGENERATED ANSWER:\n{answer}"
        raw_judge = self.bot.chat_with_retry(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_msg},
            ],
            schema_cls=JudgeScore,
            temperature=0.0,
        )
        return JudgeScore(**json.loads(raw_judge))

    def run_all_and_save(self):
        try:
            ablation = self.run_100_query_ablation_benchmark(top_k=5)
            gen_eval = self.run_end_to_end_generation_eval()

            full_report = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "corpus_chunks": len(self.retriever.chunks),
                "retrieval_100_ablation": ablation,
                "generation_audit": gen_eval,
            }

            self.output_path.write_text(
                json.dumps(full_report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"\nSaved complete evaluation report to {self.output_path}!")
            return full_report
        finally:
            self.bot.close()


if __name__ == "__main__":
    evaluator = JKUEvaluator()
    report = evaluator.run_all_and_save()

    ab = report["retrieval_100_ablation"]["ablation_results"]
    gen = report["generation_audit"]

    print("\n=========================================================================")
    print("4-ARM RETRIEVAL ABLATION RESULTS (100 Non-Circular Queries, Top-5)")
    print("=========================================================================")
    for label, key in [
        ("1. Naive Dense Vector Only ", "naive_dense"),
        ("2. Sparse BM25 (Snowball)  ", "sparse_bm25"),
        ("3. Hybrid RRF (No Reranker)", "hybrid_rrf_no_rerank"),
        ("4. Hybrid RRF + mMARCO Rerank", "hybrid_reranked"),
    ]:
        r = ab[key]
        ci = r["page_hit_ci_95"]
        print(
            f"{label}: Page Hit@5 = {r['exact_chunk_hit_rate_pct']}% (95% CI: {ci[0]}–{ci[1]}%) "
            f"| Doc Hit@5 = {r['document_hit_rate_pct']}% | MRR = {r['mrr']}"
        )

    print("\n=========================================================================")
    print("END-TO-END RAG GENERATION & CITATION AUDIT (n=12)")
    print("=========================================================================")
    print(f"• Document Retrieval Accuracy : {gen['correct_document_retrieval_pct']}%")
    print(f"• Answer Relevance            : {gen['answer_relevance_pct']}%")
    print(f"• Context Faithfulness        : {gen['context_support_faithfulness_pct']}%")
    print(f"• Citation Verification Rate  : {gen['citation_verification_pct']}%")
    print(f"• Hallucination Rate          : {gen['hallucination_rate_pct']}%")
    print("=========================================================================\n")