"""Publish rebuilt indexes only after every pipeline step succeeds."""

import json
import uuid
from pathlib import Path
from config import EMBEDDING_MODEL


def active_index_root(library_root):
    library_root = Path(library_root)
    manifest = library_root / "active_index.json"
    if not manifest.exists():
        return library_root  # Existing projects keep their original layout.
    name = json.loads(manifest.read_text(encoding="utf-8"))["version"]
    if not isinstance(name, str) or len(name) != 32 or any(c not in "0123456789abcdef" for c in name):
        raise ValueError("The active index configuration is invalid. Rebuild this library.")
    root = library_root / "indexes" / name
    if not root.is_dir():
        raise ValueError("The active index is missing. Rebuild this library.")
    return root


def validate_embedding_model(collection):
    model = (collection.metadata or {}).get("embedding_model")
    if model is None:
        # Original project indexes were generated with this model.
        model = "text-embedding-3-small"
    if model != EMBEDDING_MODEL:
        raise ValueError("The embedding model has changed. Use Rebuild Library Index before searching or embedding.")


def rebuild_library(library_root, library_id, library_type, api_key, openai_api_key):
    from zotero_ingest import ingest_library
    from chunk_papers import chunk_library
    from embed_chunks import embed_library

    root = Path(library_root)
    version = uuid.uuid4().hex
    staging = root / "indexes" / version
    staging.mkdir(parents=True, exist_ok=False)
    # A failed build is retained for diagnosis but never selected for searching.
    summary = ingest_library(library_id, library_type, api_key, staging / "papers")
    if summary["failed"]:
        raise ValueError("Some papers could not be downloaded. The previous index is unchanged; retry rebuilding.")
    chunk_library(staging / "papers", staging / "chunks")
    embedded = embed_library(staging / "chunks", staging / "chroma_db", openai_api_key)
    manifest = root / f".active-{version}.json"
    manifest.write_text(json.dumps({"version": version}), encoding="utf-8")
    manifest.replace(root / "active_index.json")
    return {**summary, **embedded}
