"""
Qdrant-backed vector store for past-question search.

Two entry points, deliberately kept separate:

  - search() / search_with_scores():
        Read-only queries against the Qdrant collection. Safe to import
        and call on every request. By default (auto_sync=True) it also
        checks whether we're inside a configured SYNC_WINDOWS slot and
        haven't synced yet for this occurrence of it -- if so, it runs
        the Firestore sync first, then answers the query. This means
        the *first* request that lands in a window pays the sync's
        latency; every other request in the same window is a fast,
        sync-free no-op. No external scheduler required for this path.

  - sync_from_firestore():
        Pulls the current records from Firestore and makes the Qdrant
        collection match them (upserts new/changed docs, deletes ones
        that were removed or rejected). Called automatically by
        search_with_scores() per the above, and can also be run
        directly/on a schedule via sync_job.py if you want a guaranteed
        sync even on days with zero traffic in a window.

The "have we already synced this window" flag is stored in Firestore
(not in a Python variable), because on serverless platforms each
request can hit a fresh, stateless process with no memory of the last
one.

Env vars required:
  QDRANT_URL, QDRANT_API_KEY   (from your .env)
  FIREBASE_SERVICE_KEY         (path to your service account json;
                                 defaults to "service_key.json")
"""

import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import firebase_admin
from dotenv import load_dotenv
from fastembed import TextEmbedding
from firebase_admin import credentials, firestore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "pastQuestions"      # Qdrant collection name
FIRESTORE_COLLECTION = "pastQuestions"    # Firestore collection name
BASE_URL = "https://www.questohive.com/view-past-question/?id="

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384  # must match the model above -- change both together

# Fixed namespace so the same Firestore doc ID always maps to the same
# Qdrant point ID across runs (what makes re-syncing idempotent).
_ID_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")

# --------------------------------------------------------------------------
# Traffic-triggered sync windows
# --------------------------------------------------------------------------
# Each entry is (start_hour, end_hour) in 24h format, same calendar day,
# evaluated in SYNC_TIMEZONE. ASSUMPTION: two 1-hour windows 12h apart --
# edit these to match your actual traffic pattern / desired sync times.
SYNC_WINDOWS: List[Tuple[int, int]] = [
    (8, 9),    # 8:00-9:00 AM
    (20, 21),  # 8:00-9:00 PM
]
SYNC_TIMEZONE = timezone.utc  # switch to a local zoneinfo if you'd rather key off local time

# Firestore collection/doc used purely to remember "which window did we
# last sync for", so this survives across stateless serverless invocations.
_SYNC_STATE_COLLECTION = "syncMeta"
_SYNC_STATE_DOC = "lastSyncedWindow"

_embedding_model: Optional[TextEmbedding] = None
_qdrant_client: Optional[QdrantClient] = None
_firestore_client: Optional[firestore.Client] = None


# --------------------------------------------------------------------------
# Lazy singletons -- cheap to import, nothing heavy happens at import time
# --------------------------------------------------------------------------

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
        url = os.environ["QDRANT_URL"]
        api_key = os.environ["QDRANT_API_KEY"]
        _qdrant_client = QdrantClient(url=url, api_key=api_key)
    return _qdrant_client


def get_firestore_client(service_key_path: Optional[str] = None) -> firestore.Client:
    global _firestore_client
    if _firestore_client is not None:
        return _firestore_client

    service_key_path = service_key_path or os.environ.get("FIREBASE_SERVICE_KEY", "service_key.json")
    if not Path(service_key_path).exists():
        raise FileNotFoundError(
            f"Firebase service key not found at '{service_key_path}'. "
            "Set FIREBASE_SERVICE_KEY or pass service_key_path explicitly."
        )
    if not firebase_admin._apps:
        cred = credentials.Certificate(service_key_path)
        firebase_admin.initialize_app(cred)
    _firestore_client = firestore.client()
    return _firestore_client


def ensure_collection(collection: str = DEFAULT_COLLECTION) -> None:
    client = get_qdrant_client()
    if not client.collection_exists(collection):
        logger.info("Creating Qdrant collection '%s'", collection)
        client.create_collection(
            collection_name=collection,
            vectors_config=qmodels.VectorParams(size=EMBED_DIM, distance=qmodels.Distance.COSINE),
        )


# --------------------------------------------------------------------------
# Record formatting / IDs
# --------------------------------------------------------------------------

def _format_record(record: dict) -> str:
    faculty = record.get("faculty", "Click the link to confirm the faculty")
    department = record.get("department", "departments")
    course = record.get("course", "Click the link to confirm the course")
    level = record.get("level", "Click the link to confirm the level")
    year = record.get("year", "Click the link to confirm the session")
    pq_id = record.get("id", "This link is currently unavailable")
    uploaded_by = record.get("uploadedBy", "This past question was anonymously uploaded")
    return (
        f"Faculty: {faculty}\n"
        f"Department: {department}\n"
        f"Course: {course}\n"
        f"Level: {level}\n"
        f"Year: {year}\n"
        f"Link: {BASE_URL}{pq_id}\n"
        f"Uploaded by: {uploaded_by}"
    )


def _point_id(firestore_doc_id: str) -> str:
    """Deterministic Qdrant point ID from a Firestore doc ID."""
    return str(uuid.uuid5(_ID_NAMESPACE, firestore_doc_id))


# --------------------------------------------------------------------------
# Window tracking (so we sync at most once per window occurrence)
# --------------------------------------------------------------------------

def _current_window_key(now: Optional[datetime] = None) -> Optional[str]:
    """
    If `now` falls inside one of SYNC_WINDOWS, return a string that
    uniquely identifies *this occurrence* of that window, e.g.
    "2026-09-26-08". Returns None if we're outside every window.
    """
    now = now or datetime.now(SYNC_TIMEZONE)
    for start_hour, end_hour in SYNC_WINDOWS:
        if start_hour <= now.hour < end_hour:
            return f"{now.date().isoformat()}-{start_hour:02d}"
    return None


def _get_last_synced_window() -> Optional[str]:
    doc = get_firestore_client().collection(_SYNC_STATE_COLLECTION).document(_SYNC_STATE_DOC).get()
    return doc.to_dict().get("window_key") if doc.exists else None


def _set_last_synced_window(window_key: str) -> None:
    get_firestore_client().collection(_SYNC_STATE_COLLECTION).document(_SYNC_STATE_DOC).set(
        {"window_key": window_key, "synced_at": datetime.now(timezone.utc).isoformat()}
    )


def sync_if_due(now: Optional[datetime] = None) -> Optional[dict]:
    """
    Run sync_from_firestore() if -- and only if -- `now` falls inside a
    configured SYNC_WINDOWS slot AND we haven't already synced for this
    occurrence of that window. Otherwise a no-op. Returns the sync
    result dict, or None if no sync was needed.
    """
    window_key = _current_window_key(now)
    if window_key is None:
        return None  # not currently inside any sync window

    if _get_last_synced_window() == window_key:
        return None  # already synced during this window

    logger.info("Entering sync window '%s' -- running sync.", window_key)
    result = sync_from_firestore()
    _set_last_synced_window(window_key)
    return result


# --------------------------------------------------------------------------
# Sync: Firestore -> Qdrant  (triggered by sync_if_due, or run directly)
# --------------------------------------------------------------------------

def _fetch_firestore_records(fs_collection: str = FIRESTORE_COLLECTION) -> Dict[str, dict]:
    """Return {firestore_doc_id: record_dict} for all non-rejected docs."""
    db = get_firestore_client()
    try:
        docs = db.collection(fs_collection).stream()
    except Exception as exc:
        logger.exception("Failed to read collection '%s'", fs_collection)
        raise RuntimeError(f"Could not fetch '{fs_collection}' from Firestore") from exc

    records = {}
    for doc in docs:
        data = doc.to_dict() or {}
        if data.get("status") == "rejected":
            continue
        records[doc.id] = data
    return records


def _existing_point_ids(collection: str) -> set:
    client = get_qdrant_client()
    ids = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        ids.update(p.id for p in points)
        if offset is None:
            break
    return ids


def sync_from_firestore(
    collection: str = DEFAULT_COLLECTION,
    fs_collection: str = FIRESTORE_COLLECTION,
    batch_size: int = 64,
) -> dict:
    """
    Make the Qdrant collection match what's currently in Firestore:
    upsert new/changed records, delete ones that no longer exist (or
    were rejected). Idempotent -- safe to call repeatedly, e.g. every
    12 hours from sync_job.py.
    """
    ensure_collection(collection)

    records = _fetch_firestore_records(fs_collection)
    if not records:
        logger.warning("No usable documents found in Firestore collection '%s'", fs_collection)

    current_ids = {_point_id(doc_id) for doc_id in records}
    existing_ids = _existing_point_ids(collection)

    to_delete = existing_ids - current_ids
    if to_delete:
        get_qdrant_client().delete(
            collection_name=collection,
            points_selector=qmodels.PointIdsList(points=list(to_delete)),
        )

    doc_items = list(records.items())
    upserted = 0
    for i in range(0, len(doc_items), batch_size):
        batch = doc_items[i : i + batch_size]
        texts = [_format_record(record) for _, record in batch]
        vectors = embed_texts(texts)
        points = [
            qmodels.PointStruct(
                id=_point_id(doc_id),
                vector=vector,
                payload={"content": text, "firestore_id": doc_id, **record},
            )
            for (doc_id, record), text, vector in zip(batch, texts, vectors)
        ]
        get_qdrant_client().upsert(collection_name=collection, points=points)
        upserted += len(points)

    result = {"upserted": upserted, "deleted": len(to_delete), "total_in_firestore": len(records)}
    logger.info("Sync complete: %s", result)
    return result


# --------------------------------------------------------------------------
# Query: fast, read-only (this is what your main app imports)
# --------------------------------------------------------------------------

def search_with_scores(
    query: str, k: int = 3, collection: str = DEFAULT_COLLECTION, auto_sync: bool = True
) -> List[dict]:
    """
    Similarity search against the Qdrant collection. If auto_sync is
    True (default), first calls sync_if_due() -- this is a no-op unless
    we're inside a configured sync window and haven't synced yet for
    it, in which case this particular call will be slower while it
    refreshes the index. Pass auto_sync=False to skip that check
    entirely (e.g. for a health-check endpoint).
    """
    if auto_sync:
        try:
            sync_if_due()
        except Exception:
            logger.exception("Auto-sync check failed; answering with the existing index instead.")

    client = get_qdrant_client()
    if not client.collection_exists(collection):
        raise RuntimeError(
            f"Qdrant collection '{collection}' doesn't exist yet. "
            "Run `python sync_job.py` once to create it and populate it "
            "from Firestore before querying."
        )

    vector = embed_texts([query])[0]
    hits = client.query_points(
        collection_name=collection, query=vector, limit=k, with_payload=True
    ).points
    return [
        {
            "content": point.payload.get("content", ""),
            "score": round(float(point.score), 4),  # cosine similarity: higher = more relevant
        }
        for point in hits
    ]


def search(query: str, k: int = 3, collection: str = DEFAULT_COLLECTION, auto_sync: bool = True) -> str:
    results = search_with_scores(query, k=k, collection=collection, auto_sync=auto_sync)
    return "\n \n".join(r["content"] for r in results)
