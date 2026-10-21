import json
import pymupdf
from pathlib import Path
from pyzotero import zotero
from zotero_connection import validate_library


def ingest_library(library_id, library_type, api_key, output_dir):
    """
    Download and extract PDFs from a specified Zotero library.
    Returns a summary of processed, skipped, and failed papers.
    """

    library_id = validate_library(library_id, library_type)

    # connect to the selected Zotero library
    library = zotero.Zotero(
        library_id,
        library_type,
        api_key
    )

    # create the selected output folder
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # retrieve all top-level items
    items = library.everything(library.top())
    print(f"Retrieved {len(items)} items from Zotero library.")

    processed = 0
    skipped = 0
    failed = 0

    for item in items:
        data = item["data"]
        item_key = item["key"]

        if data.get("itemType") == "attachment":
            continue

        title = data.get("title", "Untitled")

        output_file = output_dir / f"{item_key}.json"

        if output_file.exists():
            skipped += 1
            continue

        try:
            children = library.everything(library.children(item_key))

            pdfs = [
                child for child in children
                if child["data"].get("itemType") == "attachment"
                and child["data"].get("contentType") == "application/pdf"
            ]

            if not pdfs:
                skipped += 1
                print(f"No PDF: {title}")
                continue

            pdf = pdfs[0]
            pdf_key = pdf["key"]

            print(f"Processing: {title}")

            pdf_bytes = library.file(pdf_key)

            # extract text page by page
            pages = []

            with pymupdf.open(
                stream=pdf_bytes,
                filetype="pdf"
            ) as document:

                for page_number, page in enumerate(document, start=1):
                    text = page.get_text()

                    pages.append({
                        "page": page_number,
                        "text": text
                    })

            # extract authors
            authors = []

            for creator in data.get("creators", []):
                if creator.get("creatorType") == "author":
                    first = creator.get("firstName", "")
                    last = creator.get("lastName", "")
                    name = creator.get("name") or f"{first} {last}".strip()
                    authors.append(name)

            # create structured document
            paper = {
                "zotero_key": item_key,
                "pdf_key": pdf_key,
                "title": title,
                "authors": authors,
                "date": data.get("date", ""),
                "doi": data.get("DOI", ""),
                "pages": pages
            }

            # save as JSON
            temporary = output_file.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(paper, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(output_file)

            processed += 1
            print(f"Saved: {output_file}")

        except Exception:
            failed += 1
            print("A paper could not be downloaded or extracted; retry indexing.")

    summary = {
        "processed": processed,
        "skipped": skipped,
        "failed": failed
    }

    print("\n--- INGESTION SUMMARY ---")
    print(summary)

    return summary