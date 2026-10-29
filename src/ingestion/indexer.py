import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.config import settings

class JKUIndexer:
    def __init__(self, chunks_path: str = "data/processed/chunks.json"):
        self.chunks_path = Path(chunks_path)
        print(f"Loading Multilingual Embedding Model: {settings.EMBEDDING_MODEL}...")
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.client = QdrantClient(path=settings.QDRANT_PATH)

    def build_vector_index(self):
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Cannot find {self.chunks_path}. Run parser.py first!")

        chunks = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        print(f"Loaded {len(chunks)} chunks from {self.chunks_path}")

        # Prepend explicit [Source | Page | Section] header into the text before embedding
        texts_to_embed = []
        for c in chunks:
            clean_body = c["text"].strip()
            contextual_text = f"[{c['source']} | Page {c['page']} | Section: {c['section']}]\n{clean_body}"
            c["contextual_text"] = contextual_text
            texts_to_embed.append(contextual_text)

        # Save updated chunks with contextual_text back to chunks.json for BM25 to use
        self.chunks_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

        print("Generating dense vector embeddings (this takes ~15-30 seconds on CPU)...")
        embeddings = self.embedder.encode(texts_to_embed, show_progress_bar=True, normalize_embeddings=True)
        vector_dim = embeddings.shape[1]

        # Recreate collection cleanly in local Qdrant storage
        if self.client.collection_exists(settings.COLLECTION_NAME):
            self.client.delete_collection(settings.COLLECTION_NAME)

        self.client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )

        # Upload all points with metadata payloads
        points = [
            PointStruct(
                id=c["chunk_id"],
                vector=embeddings[i].tolist(),
                payload={
                    "chunk_id": c["chunk_id"],
                    "source": c["source"],
                    "page": c["page"],
                    "section": c["section"],
                    "text": c["contextual_text"],
                },
            )
            for i, c in enumerate(chunks)
        ]

        self.client.upsert(collection_name=settings.COLLECTION_NAME, points=points)
        self.client.close()
        print(f"Successfully indexed {len(points)} vectors (dim={vector_dim}) into {settings.QDRANT_PATH}!")

if __name__ == "__main__":
    indexer = JKUIndexer()
    indexer.build_vector_index()