"""
Company knowledgebase RAG -- separate from the past-questions RAG.

Same Qdrant project (QDRANT_URL / QDRANT_API_KEY), but its own
collection, so the two datasets never mix.

Two entry points:
  - ingest_knowledgebase(): reads knowledgebase.txt, chunks it, embeds
    it, and upserts it into Qdrant. Run this once now, and again
    whenever the file changes.
  - query_knowledgebase(): read-only search, meant to be called as a
    tool by an AI agent (see kb_tool.py for the tool-calling wrapper).
"""

import hashlib
import logging
import os
import uuid
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastembed import TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLLECTION = "company_knowledgebase"  # deliberately distinct from "pastQuestions"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384  # must match the model above

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Separate namespace from the past-questions store, so IDs never collide
# even if both stores are queried through the same client by mistake.
_ID_NAMESPACE = uuid.UUID("87654321-4321-8765-4321-876543218765")

_embedding_model: Optional[TextEmbedding] = None
_qdrant_client: Optional[QdrantClient] = None


def get_embedding_model() -> TextEmbedding:
    global _embedding_model
    if _embedding_model is None:
        cache_dir = Path.home() / ".cache" / "fastembed"
        cache_dir.mkdir(parents=True, exist_ok=True)
        _embedding_model = TextEmbedding(model_name=EMBED_MODEL_NAME, cache_dir=str(cache_dir))
    return _embedding_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    return [vector.tolist() for vector in get_embedding_model().embed(texts)]


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"]
        )
    return _qdrant_client


def ensure_collection() -> None:
    client = get_qdrant_client()
    if not client.collection_exists(COLLECTION):
        logger.info("Creating Qdrant collection '%s'", COLLECTION)
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=qmodels.VectorParams(size=EMBED_DIM, distance=qmodels.Distance.COSINE),
        )


def _chunk_id(chunk_text: str) -> str:
    """
    Deterministic ID derived from the chunk's own content. This means
    re-ingesting an unchanged file is a cheap no-op (same IDs, same
    vectors), and editing the file naturally produces new IDs for
    changed chunks -- old chunks just won't be in the new ID set.
    """
    digest = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(_ID_NAMESPACE, digest))


def ingest_knowledgebase(
    file_path: str = "knowledgebase.txt",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    batch_size: int = 64,
    prune_stale: bool = True,
) -> dict:
    """
    Read `file_path`, split it into chunks, embed them, and upsert them
    into Qdrant. If prune_stale is True, also deletes any previously
    ingested chunk whose content no longer appears in the current file
    (so edits/removals in knowledgebase.txt are reflected, not just
    additions).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"'{file_path}' not found. Upload/place it, then re-run ingestion.")

    ensure_collection()

    text = path.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    if not chunks:
        raise ValueError(f"No text extracted from '{file_path}'.")

    current_ids = {_chunk_id(chunk): chunk for chunk in chunks}

    client = get_qdrant_client()
    upserted = 0
    chunk_items = list(current_ids.items())
    for i in range(0, len(chunk_items), batch_size):
        batch = chunk_items[i : i + batch_size]
        vectors = embed_texts([chunk for _, chunk in batch])
        points = [
            qmodels.PointStruct(id=chunk_id, vector=vector, payload={"content": chunk})
            for (chunk_id, chunk), vector in zip(batch, vectors)
        ]
        client.upsert(collection_name=COLLECTION, points=points)
        upserted += len(points)

    deleted = 0
    if prune_stale:
        existing_ids = set()
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=COLLECTION, limit=256, offset=offset,
                with_payload=False, with_vectors=False,
            )
            existing_ids.update(p.id for p in points)
            if offset is None:
                break
        stale = existing_ids - set(current_ids.keys())
        if stale:
            client.delete(
                collection_name=COLLECTION,
                points_selector=qmodels.PointIdsList(points=list(stale)),
            )
            deleted = len(stale)

    result = {"chunks_in_file": len(chunks), "upserted": upserted, "deleted_stale": deleted}
    logger.info("Ingestion complete: %s", result)
    return result


def query_knowledgebase(query: str, k: int = 4) -> List[dict]:
    """
    Read-only similarity search against the company knowledgebase.
    This is the function an AI agent calls as a tool (see kb_tool.py).
    """
    client = get_qdrant_client()
    if not client.collection_exists(COLLECTION):
        raise RuntimeError(
            f"Collection '{COLLECTION}' doesn't exist yet. Run ingest_knowledgebase() first."
        )

    vector = embed_texts([query])[0]
    hits = client.query_points(
        collection_name=COLLECTION, query=vector, limit=k, with_payload=True
    ).points
    return [
        {"content": p.payload.get("content", ""), "score": round(float(p.score), 4)}
        for p in hits
    ]


if __name__ == "__main__":
    print(ingest_knowledgebase())
