"""
Standalone entry point that syncs Firestore -> Qdrant.

This does NOT run your main app -- it only keeps the Qdrant collection
up to date. Trigger it every 12 hours using ONE of the two options
below, depending on how you're deployed.
"""

import logging

from vector_store import sync_from_firestore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_sync():
    return sync_from_firestore()


if __name__ == "__main__":
    run_sync()

