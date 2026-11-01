import json
import time
import random
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from src.generation.generator import JKUNavigatorGenerator


class JudgeScore(BaseModel):
    answer_relevant: bool = Field(
        description="True if the answer directly addresses the student's question (or appropriately abstains on trap questions)."
    )
    supported_by_context: bool = Field(
        description="True if every claim in the answer is backed by the retrieved context chunks."
    )
    hallucinated: bool = Field(
        description="True if the model invented facts, rules, or numbers not present in the context."
    )
    reasoning: str = Field(description="1-sentence explanation of the evaluation score.")


class JKUEvaluator:
    def __init__(self, output_path: str = "data/processed/eval_results.json"):
        self.output_path = Path(output_path)
        # Instantiate a single generator (which holds one Qdrant connection to avoid Windows file locks)
        self.bot = JKUNavigatorGenerator()
        self.retriever = self.bot.retriever

    def build_100_retrieval_dataset(self) -> List[Dict[str, Any]]:
        """
        Builds a 100-question Golden Retrieval Dataset anchored to the exact chunks in data/processed/chunks.json.
        Combines specific JKU course/regulation queries with deterministic chunk-level queries.
        """
        chunks = self.retriever.chunks
        dataset = []

        # Seed for 100% reproducible benchmark numbers
        rng = random.Random(42)
        valid_chunks = [c for c in chunks if len(c.get("text", "")) > 120]
        sampled_chunks = rng.sample(valid_chunks, min(100, len(valid_chunks)))

        for idx, c in enumerate(sampled_chunks):
            raw_lines = [
                line.strip()
                for line in c["text"].split("\n")
                if len(line.strip()) > 25 and not line.strip().startswith("[")
            ]
            snippet = raw_lines[0] if raw_lines else c["section"]
            # Trim snippet to create a realistic search query targeting that exact regulation/course table
            words = snippet.split()
            query_phrase = " ".join(words[:12]).strip(".,:;*-# ")

            dataset.append({
                "id": idx + 1,
                "query": f"{c['section']}: {query_phrase}",
                "expected_chunk_id": c["chunk_id"],
                "expected_source": c["source"],
                "expected_page": c["page"],
            })

        return dataset

    def run_100_query_ablation_benchmark(self, top_k: int = 5) -> Dict[str, Any]:
        """
        Tests 100 queries across 3 architectures to prove why Hybrid + Reranking wins:
        1. Naive Dense Vector Search (Baseline)
        2. Sparse BM25 Keyword Search
        3. Full Hybrid RRF + Cross-Encoder Reranker (Our System)
        """
        dataset = self.build_100_retrieval_dataset()
        total = len(dataset)
        print(f"\nRunning 100-Query Retrieval Ablation Study (Top-{top_k}) across {total} test cases...")

        metrics = {
            "naive_dense": {"chunk_hits": 0, "doc_hits": 0, "mrr_sum": 0.0},
            "sparse_bm25": {"chunk_hits": 0, "doc_hits": 0, "mrr_sum": 0.0},
            "hybrid_reranked": {"chunk_hits": 0, "doc_hits": 0, "mrr_sum": 0.0},
        }

        for item in dataset:
            q = item["query"]
            target_cid = item["expected_chunk_id"]
            target_doc = item["expected_source"]

            # 1. Naive Dense Vector Search
            dense_res = self.retriever.dense_search(q, top_k=top_k)
            dense_cids = [r["chunk_id"] for r in dense_res]
            dense_docs = [self.retriever.chunk_map[cid]["source"] for cid in dense_cids]
            self._record_hit(metrics["naive_dense"], target_cid, target_doc, dense_cids, dense_docs)

            # 2. Sparse BM25 Search
            bm25_res = self.retriever.sparse_bm25_search(q, top_k=top_k)
            bm25_cids = [r["chunk_id"] for r in bm25_res]
            bm25_docs = [self.retriever.chunk_map[cid]["source"] for cid in bm25_cids]
            self._record_hit(metrics["sparse_bm25"], target_cid, target_doc, bm25_cids, bm25_docs)

            # 3. Full Hybrid (Dense + BM25 + RRF) + Cross-Encoder Reranker
            hybrid_res = self.retriever.retrieve_and_rerank(q, retrieve_top_k=20, final_top_n=top_k)
            hybrid_cids = [r["chunk_id"] for r in hybrid_res]
            hybrid_docs = [r["source"] for r in hybrid_res]
            self._record_hit(metrics["hybrid_reranked"], target_cid, target_doc, hybrid_cids, hybrid_docs)

        summary = {}
        for mode, m in metrics.items():
            summary[mode] = {
                "exact_chunk_hit_rate_pct": round((m["chunk_hits"] / total) * 100, 1),
                "document_hit_rate_pct": round((m["doc_hits"] / total) * 100, 1),
                "mrr": round(m["mrr_sum"] / total, 3),
            }

        return {"total_queries": total, "top_k": top_k, "ablation_results": summary}

    @staticmethod
    def _record_hit(
        bucket: Dict[str, float],
        target_cid: int,
        target_doc: str,
        retrieved_cids: List[int],
        retrieved_docs: List[str],
    ):
        if target_doc in retrieved_docs:
            bucket["doc_hits"] += 1
        if target_cid in retrieved_cids:
            bucket["chunk_hits"] += 1
            rank = retrieved_cids.index(target_cid) + 1
            bucket["mrr_sum"] += 1.0 / rank

    def run_end_to_end_generation_eval(self) -> Dict[str, Any]:
        """
        Evaluates the 4 core video metrics on bilingual & adversarial trap questions:
        1. Correct Document Retrieved
        2. Answer Relevance
        3. Supported by Context (Faithfulness)
        4. Hallucination Rate
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

        print(f"\nRunning End-to-End LLM Generation & Hallucination Audit ({len(golden_qa)} cases)...")
        detailed_logs = []
        doc_retrieved_count = 0
        relevant_count = 0
        supported_count = 0
        hallucination_count = 0

        for idx, item in enumerate(golden_qa, 1):
            q = item["question"]
            print(f"  [{idx}/{len(golden_qa)}] Auditing: {q[:65]}...")

            # Pace API calls to stay comfortably inside Groq's free tier rate limit
            time.sleep(2.2)
            rag_out = self.bot.answer_question(q, top_n_chunks=5)

            retrieved_sources = [c["source"] for c in rag_out["retrieved_chunks"]]
            if not item["is_answerable"]:
                # For trap questions, retrieval passes if the model properly abstains (is_supported == False)
                doc_hit = not rag_out["is_supported"]
            else:
                doc_hit = item["expected_source"] in retrieved_sources

            # Run LLM as a Judge to verify Faithfulness, Relevance, and Hallucination
            time.sleep(2.0)
            judge = self._judge_response(
                question=q,
                answer=rag_out["answer"],
                context_chunks=rag_out["retrieved_chunks"],
                is_answerable=item["is_answerable"],
            )

            if doc_hit:
                doc_retrieved_count += 1
            if judge.answer_relevant:
                relevant_count += 1
            if judge.supported_by_context:
                supported_count += 1
            if judge.hallucinated:
                hallucination_count += 1

            detailed_logs.append({
                "question": q,
                "expected_source": item["expected_source"],
                "is_answerable": item["is_answerable"],
                "doc_retrieved_correctly": doc_hit,
                "answer_relevant": judge.answer_relevant,
                "supported_by_context": judge.supported_by_context,
                "hallucinated": judge.hallucinated,
                "judge_reasoning": judge.reasoning,
                "answer_preview": rag_out["answer"][:220] + "...",
            })

        n = len(golden_qa)
        summary = {
            "total_evaluated": n,
            "correct_document_retrieval_pct": round((doc_retrieved_count / n) * 100, 1),
            "answer_relevance_pct": round((relevant_count / n) * 100, 1),
            "context_support_faithfulness_pct": round((supported_count / n) * 100, 1),
            "hallucination_rate_pct": round((hallucination_count / n) * 100, 1),
            "cases": detailed_logs,
        }
        return summary

    def _judge_response(
        self, question: str, answer: str, context_chunks: List[Dict[str, Any]], is_answerable: bool
    ) -> JudgeScore:
        context_text = "\n\n".join([c.get("contextual_text", c["text"]) for c in context_chunks])
        prompt = f"""You are an strict AI Evaluation Judge auditing a RAG system.
Evaluate the GENERATED ANSWER against the RETRIEVED CONTEXT and STUDENT QUESTION.

Note: Is this question answerable from JKU documents? {is_answerable}.
If is_answerable is False, and the model politely states the documents do not contain that information without inventing facts, then:
- answer_relevant = true
- supported_by_context = true
- hallucinated = false

Return ONLY valid JSON with this exact schema:
{{
  "answer_relevant": true,
  "supported_by_context": true,
  "hallucinated": false,
  "reasoning": "1 sentence explanation"
}}"""

        user_msg = f"QUESTION: {question}\n\nRETRIEVED CONTEXT:\n{context_text}\n\nGENERATED ANSWER:\n{answer}"
        raw_judge = self.bot.chat_with_retry(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
        )
        return JudgeScore(**json.loads(raw_judge))

    def run_all_and_save(self):
        try:
            ablation = self.run_100_query_ablation_benchmark(top_k=5)
            gen_eval = self.run_end_to_end_generation_eval()

            full_report = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
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

    print("\n============================================================")
    print("100-QUERY RETRIEVAL ABLATION RESULTS (Top-5)")
    print("============================================================")
    print(f"1. Naive Dense Vector Only : Chunk Hit = {ab['naive_dense']['exact_chunk_hit_rate_pct']}% | Doc Hit = {ab['naive_dense']['document_hit_rate_pct']}% | MRR = {ab['naive_dense']['mrr']}")
    print(f"2. Sparse BM25 Keyword Only: Chunk Hit = {ab['sparse_bm25']['exact_chunk_hit_rate_pct']}% | Doc Hit = {ab['sparse_bm25']['document_hit_rate_pct']}% | MRR = {ab['sparse_bm25']['mrr']}")
    print(f"3. Hybrid RRF + Reranker   : Chunk Hit = {ab['hybrid_reranked']['exact_chunk_hit_rate_pct']}% | Doc Hit = {ab['hybrid_reranked']['document_hit_rate_pct']}% | MRR = {ab['hybrid_reranked']['mrr']}")

    print("\n============================================================")
    print("END-TO-END RAG GENERATION & HALLUCINATION METRICS")
    print("============================================================")
    print(f"• Document Retrieval Accuracy : {gen['correct_document_retrieval_pct']}%")
    print(f"• Answer Relevance            : {gen['answer_relevance_pct']}%")
    print(f"• Supported by Context        : {gen['context_support_faithfulness_pct']}%")
    print(f"• Hallucination Rate          : {gen['hallucination_rate_pct']}%")
    print("============================================================\n")