import re
from pyzotero import zotero


def validate_library(library_id, library_type):
    library_id = str(library_id).strip()
    if library_type not in {"user", "group"}:
        raise ValueError("Choose a personal or group library.")
    if not re.fullmatch(r"[0-9]+", library_id) or int(library_id) <= 0:
        raise ValueError("The Zotero library ID must contain only digits and be greater than zero.")
    return library_id


def connect_zotero(library_id, library_type, api_key):
    """Connect to a Zotero library and verify access."""

    library_id = validate_library(library_id, library_type)
    library = zotero.Zotero(
        library_id,
        library_type,
        api_key
    )

    # test connection by requesting one item
    library.top(limit=1)

    return library