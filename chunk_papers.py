
import json
import re
from pathlib import Path

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
MIN_CHUNK_SIZE = 100

def preprocess_text(text):
    """Clean common formatting issues from PDF extraction."""
    # join words split by hyphens across lines
    text = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])", "", text)
    # replace remaining whitespace with single spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text, chunk_size=400, overlap=50, min_chunk_size=100):
    """Split text into fixed-size overlapping word chunks."""
    if not 0 <= overlap < chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if not chunk_words:
            break
        if len(chunk_words) < min_chunk_size and chunks:
            previous_start = max(0, start - step)
            chunks[-1] = " ".join(words[previous_start:])
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks

def chunk_library(input_dir, output_dir):
    """Chunk all extracted papers in a specified library folder."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Paper folder not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    total_papers = 0
    total_chunks = 0
    skipped = 0
    for paper_file in input_dir.glob("*.json"):
        with open(paper_file, "r", encoding="utf-8") as f:
            paper = json.load(f)
        output_file = output_dir / f"{paper['zotero_key']}_chunks.json"
        if output_file.exists():
            skipped += 1
            continue
        print(f"Processing: {paper['title']}")
        all_chunks = []
        for page in paper["pages"]:
            page_number = page["page"]
            cleaned_text = preprocess_text(page["text"])
            page_chunks = chunk_text(
                cleaned_text,
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
                min_chunk_size=MIN_CHUNK_SIZE
            )
            for i, chunk in enumerate(page_chunks):
                all_chunks.append({
                    "chunk_id": f"{paper['zotero_key']}_p{page_number}_c{i}",
                    "zotero_key": paper["zotero_key"],
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "date": paper["date"],
                    "doi": paper["doi"],
                    "page": page_number,
                    "text": chunk
                })
        temporary = output_file.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output_file)
        total_papers += 1
        total_chunks += len(all_chunks)
        print(f"  Generated {len(all_chunks)} chunks.\n")
    summary = {
        "processed": total_papers,
        "chunks": total_chunks,
        "skipped": skipped
    }
    print("Chunking complete:", summary)
    return summary
