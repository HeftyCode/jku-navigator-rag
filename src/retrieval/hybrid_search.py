import re
import json
import threading
import unicodedata
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from nltk.stem.snowball import GermanStemmer, EnglishStemmer
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from src.config import settings

_DE_STEMMER = GermanStemmer()
_EN_STEMMER = EnglishStemmer()

STOPWORDS = {
    "the", "and", "what", "are", "is", "to", "in", "on", "onto", "of", "for", "am", "i", "my",
    "how", "does", "do", "can", "a", "an", "with", "between", "at", "by", "from", "or", "be",
    "der", "die", "das", "und", "ist", "was", "wie", "im", "für", "ein", "eine", "einer",
    "gibt", "es", "laut", "am", "den", "dem", "des", "sich", "auf", "mit", "von", "zu", "oder",
}


class HybridRetriever:
    def __init__(self, chunks_path: str = settings.CHUNKS_PATH):
        self.chunks_path = Path(chunks_path)
        self.chunks = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        self.chunk_map = {c["chunk_id"]: c for c in self.chunks}
        self._lock = threading.Lock()

        print("Initializing Multilingual Embedder, mMARCO Reranker, and Qdrant...")
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.reranker = CrossEncoder(settings.RERANKER_MODEL)
        self.qdrant = QdrantClient(path=settings.QDRANT_PATH)

        # Verify index consistency at startup
        if self.qdrant.collection_exists(settings.COLLECTION_NAME):
            qdrant_count = self.qdrant.count(collection_name=settings.COLLECTION_NAME).count
            if qdrant_count != len(self.chunks):
                raise RuntimeError(
                    f"Index mismatch: chunks.json has {len(self.chunks)} chunks, but Qdrant has {qdrant_count}. Re-run indexer.py!"
                )

        tokenized_corpus = [self._tokenize(c["contextual_text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Symmetric Bilingual Tokenizer with German & English Snowball Stemming:
        - Normalizes Unicode (NFKC) and collapses '§ 14' -> '§14'.
        - Preserves dotted course codes (326.001) and international characters (\\w).
        - Applies German + English Snowball stemmers symmetrically to both queries and documents
          so 'Hund' and 'Hunde' both stem to 'hund' without one-way length bugs.
        """
        text = unicodedata.normalize("NFKC", text.lower())
        text = re.sub(r"§\s*(\d+)", r"§\1", text)
        raw_tokens = re.findall(r"§?\w+(?:\.\d+)*", text)

        out: List[str] = []
        for t in raw_tokens:
            if t in STOPWORDS:
                continue
            out.append(t)
            if t.isalpha() and len(t) >= 4:
                de_stem = _DE_STEMMER.stem(t)
                en_stem = _EN_STEMMER.stem(t)
                if de_stem != t:
                    out.append(de_stem)
                if en_stem != t and en_stem != de_stem:
                    out.append(en_stem)
        return out

    def dense_search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Semantic vector search using Qdrant (thread-safe)."""
        with self._lock:
            query_vector = self.embedder.encode(query, normalize_embeddings=True).tolist()
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

        return [
            {
                "chunk_id": hit.id,
                "dense_rank": rank + 1,
                "dense_score": float(hit.score),
            }
            for rank, hit in enumerate(hits)
            if hit.id in self.chunk_map
        ]

    def sparse_bm25_search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Exact keyword/code/stem search using BM25Okapi."""
        tokens = self._tokenize(query)
        if not tokens:
            return []
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
        """Fuses Dense and Sparse rankings via Reciprocal Rank Fusion (RRF)."""
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
        self,
        search_query: str,
        rerank_query: Optional[str] = None,
        dense_query: Optional[str] = None,
        retrieve_top_k: int = 20,
        final_top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Decoupled Hybrid Pipeline:
        - Dense search uses natural query (`dense_query` or `rerank_query`).
        - Sparse BM25 uses bilingual expanded keywords (`search_query`).
        - Cross-Encoder Reranker scores against the natural question (`rerank_query`), NOT keyword soup!
        """
        natural_q = rerank_query or search_query
        dense_q = dense_query or natural_q

        dense_res = self.dense_search(dense_q, top_k=retrieve_top_k)
        sparse_res = self.sparse_bm25_search(search_query, top_k=retrieve_top_k)
        fused_candidates = self.reciprocal_rank_fusion(dense_res, sparse_res)[:retrieve_top_k]

        if not fused_candidates:
            return []

        pairs = [[natural_q, c["contextual_text"]] for c in fused_candidates]
        with self._lock:
            rerank_scores = self.reranker.predict(pairs)

        for i, candidate in enumerate(fused_candidates):
            candidate["rerank_score"] = round(float(rerank_scores[i]), 4)

        reranked = sorted(fused_candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:final_top_n]

    def close(self):
        self.qdrant.close()