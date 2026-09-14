"""Embedder factory with honest fallback.

Resolution order:
  1. The configured model (default BAAI/bge-base-en-v1.5) via fastembed.
  2. If the weights cannot be loaded (no network / blocked egress), fall
     back to the deterministic hashing embedder and log a WARNING so the
     degradation is visible. The rest of the pipeline is unaffected.
"""

from __future__ import annotations

import logging

from researchrag.config import Settings
from researchrag.embeddings.base import BaseEmbedder
from researchrag.embeddings.fastembed_embedder import FastEmbedder
from researchrag.embeddings.hash_embedder import HashingEmbedder

log = logging.getLogger(__name__)


def make_embedder(
    settings: Settings, prefer: str | None = None
) -> tuple[BaseEmbedder, bool]:
    """Return (embedder, fell_back)."""
    model = prefer or settings.embedding_model
    if model == "hashing":
        return HashingEmbedder(dim=768), False
    try:
        embedder = FastEmbedder(
            model_name=model,
            batch_size=settings.embedding_batch_size,
            query_instruction=settings.embedding_query_instruction
            if model.startswith("BAAI/bge")
            else "",
        )
        # force a load so download failures surface now, not at query time
        _ = embedder.dimension
        log.info("Dense embedder ready: %s (dim=%d)", model, embedder.dimension)
        return embedder, False
    except Exception as e:
        log.warning(
            "Could not load embedding model %s (%s). "
            "Falling back to the offline hashing embedder — semantic quality "
            "is degraded; connect to the network once to download the model.",
            model,
            type(e).__name__,
        )
        return HashingEmbedder(dim=768), True
