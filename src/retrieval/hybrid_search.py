import re
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from src.config import settings

class HybridRetriever:
    def __init__(self, chunks_path: str = "data/processed/chunks.json"):
        self.chunks_path = Path(chunks_path)
        self.chunks = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        self.chunk_map = {c["chunk_id"]: c for c in self.chunks}

        print("Initializing Embedder, Cross-Encoder Reranker, and Qdrant...")
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.reranker = CrossEncoder(settings.RERANKER_MODEL)
        self.qdrant = QdrantClient(path=settings.QDRANT_PATH)

        # Build BM25 sparse keyword index over all chunks
        tokenized_corpus = [self._tokenize(c.get("contextual_text", c["text"])) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Custom bilingual tokenizer:
        1. Preserves German umlauts (äöüß), legal symbols (§), and dotted course codes (326.001).
        2. Filters out English/German stopwords so filler words don't pollute BM25.
        3. Emits 4-char root prefixes for words >= 5 chars so German plurals/compounds match (Hund <-> Hunde, Parken <-> Parkordnung).
        """
        stopwords = {
            "the", "and", "what", "are", "is", "to", "in", "on", "onto", "of", "for", "am", "i", "my",
            "how", "does", "do", "can", "a", "an", "with", "between", "at", "by", "from", "or", "be",
            "der", "die", "das", "und", "ist", "was", "wie", "im", "für", "ein", "eine", "einer",
            "gibt", "es", "laut", "am", "den", "dem", "des", "sich", "auf", "mit", "von", "zu", "oder"
        }
        raw_tokens = re.findall(r"[a-z0-9äöüß§]+(?:\.[0-9]+)*", text.lower())
        final_tokens = []
        for t in raw_tokens:
            if t in stopwords:
                continue
            final_tokens.append(t)
            # Add 4-character stem for morphological/compound matching (e.g. "hunde" -> "hund*")
            if len(t) >= 5 and t.isalpha():
                final_tokens.append(t[:4] + "*")
        return final_tokens

    def dense_search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Semantic vector search using Qdrant."""
        query_vector = self.embedder.encode(query, normalize_embeddings=True).tolist()
        
        # Compatible with all qdrant-client versions (1.8 through 1.13+)
        if hasattr(self.qdrant, "query_points"):
            response = self.qdrant.query_points(
                collection_name=settings.COLLECTION_NAME,
                query=query_vector,
                limit=top_k,
            )
            hits = response.points
        else:
            hits = self.qdrant.search(
                collection_name=settings.COLLECTION_NAME,
                query_vector=query_vector,
                limit=top_k,
            )

        results = []
        for rank, hit in enumerate(hits):
            results.append({
                "chunk_id": hit.id,
                "dense_rank": rank + 1,
                "dense_score": float(hit.score),
            })
        return results

    def sparse_bm25_search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Exact keyword/code search using BM25Okapi."""
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:
                chunk_id = self.chunks[int(idx)]["chunk_id"]
                results.append({
                    "chunk_id": chunk_id,
                    "bm25_rank": rank + 1,
                    "bm25_score": float(scores[idx]),
                })
        return results

    def reciprocal_rank_fusion(
        self, dense_res: List[Dict[str, Any]], sparse_res: List[Dict[str, Any]], k: int = 60
    ) -> List[Dict[str, Any]]:
        """Fuses Dense and Sparse rankings mathematically via RRF."""
        rrf_scores: Dict[int, float] = {}
        debug_info: Dict[int, Dict[str, Any]] = {}

        for item in dense_res:
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + item["dense_rank"]))
            debug_info.setdefault(cid, {})["dense_rank"] = item["dense_rank"]
            debug_info[cid]["dense_score"] = round(item["dense_score"], 4)

        for item in sparse_res:
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + item["bm25_rank"]))
            debug_info.setdefault(cid, {})["bm25_rank"] = item["bm25_rank"]
            debug_info[cid]["bm25_score"] = round(item["bm25_score"], 4)

        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        fused_results = []
        for cid in sorted_cids:
            chunk_data = self.chunk_map[cid].copy()
            chunk_data["rrf_score"] = round(rrf_scores[cid], 5)
            chunk_data["dense_rank"] = debug_info[cid].get("dense_rank", None)
            chunk_data["bm25_rank"] = debug_info[cid].get("bm25_rank", None)
            fused_results.append(chunk_data)

        return fused_results

    def retrieve_and_rerank(
        self, query: str, retrieve_top_k: int = 20, final_top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Full Pipeline: Dense + BM25 -> RRF Fusion -> Cross-Encoder Reranking.
        """
        dense_res = self.dense_search(query, top_k=retrieve_top_k)
        sparse_res = self.sparse_bm25_search(query, top_k=retrieve_top_k)
        fused_candidates = self.reciprocal_rank_fusion(dense_res, sparse_res)[:retrieve_top_k]

        if not fused_candidates:
            return []

        # Score candidate pairs [query, chunk_text] with Cross-Encoder
        pairs = [[query, c.get("contextual_text", c["text"])] for c in fused_candidates]
        rerank_scores = self.reranker.predict(pairs)

        for i, candidate in enumerate(fused_candidates):
            candidate["rerank_score"] = round(float(rerank_scores[i]), 4)

        # Sort by Cross-Encoder score descending
        reranked = sorted(fused_candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:final_top_n]

    def close(self):
        self.qdrant.close()

if __name__ == "__main__":
    retriever = HybridRetriever()
    test_query = "What are the different course types like VL, UE, KV and how does KUSSS registration work?"
    print(f"\nTesting Query: '{test_query}'\n")
    
    top_chunks = retriever.retrieve_and_rerank(test_query, final_top_n=3)
    for i, c in enumerate(top_chunks, 1):
        print(f"--- Rank #{i} | Rerank Score: {c['rerank_score']} | RRF: {c['rrf_score']} (Dense Rank: {c['dense_rank']}, BM25 Rank: {c['bm25_rank']}) ---")
        print(f"Source: {c['source']} (Page {c['page']}) | Section: {c['section']}")
        print(c.get("contextual_text", c["text"])[:350] + "...\n")
    
    retriever.close()