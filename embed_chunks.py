
import json
import time
from pathlib import Path
import chromadb
from openai import OpenAI
from config import EMBEDDING_MODEL
from library_storage import validate_embedding_model

BATCH_SIZE = 100


def embed_library(chunks_dir, chroma_dir, openai_api_key):
    """Embed a library's chunks into its own ChromaDB database."""

    chunks_dir = Path(chunks_dir)
    chroma_dir = Path(chroma_dir)

    if not chunks_dir.is_dir():
        raise FileNotFoundError(f"Chunks folder not found: {chunks_dir}")
    
    # use the researcher's OpenAI API key
    client = OpenAI(api_key=openai_api_key)

    # use the selected library's database
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = chroma_client.get_or_create_collection(
        "zotero_papers", metadata={"embedding_model": EMBEDDING_MODEL}
    )
    validate_embedding_model(collection)

    # load all chunks
    all_chunks = []

    for chunk_file in sorted(chunks_dir.glob("*.json")):
        with open(chunk_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        all_chunks.extend(chunks)
    print(f"Loaded {len(all_chunks)} chunks.")

    # skip chunks that are already embedded
    existing_ids = set(collection.get(include=[])["ids"])
    new_chunks = [
        chunk for chunk in all_chunks
        if chunk["chunk_id"] not in existing_ids
    ]
    print(f"Already embedded: {len(existing_ids)}")
    print(f"Remaining chunks: {len(new_chunks)}")

    # generate and store embeddings in batches
    for start in range(0, len(new_chunks), BATCH_SIZE):
        batch = new_chunks[start:start + BATCH_SIZE]
        texts = [chunk["text"] for chunk in batch]
        ids = [chunk["chunk_id"] for chunk in batch]

        metadatas = []

        for chunk in batch:
            metadatas.append({
                "zotero_key": chunk["zotero_key"],
                "title": chunk["title"],
                "authors": ", ".join(chunk["authors"]),
                "date": chunk["date"] or "",
                "doi": chunk["doi"] or "",
                "page": chunk["page"]
            })

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )

        embeddings = [item.embedding for item in response.data]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        print(
            f"Stored batch {start // BATCH_SIZE + 1}: "
            f"{len(batch)} chunks | "
            f"Collection total: {collection.count()}"
        )

        time.sleep(0.1)

    return {
        "total_chunks": len(all_chunks),
        "already_embedded": len(all_chunks) - len(new_chunks),
        "newly_embedded": len(new_chunks),
        "collection_total": collection.count()
    }
