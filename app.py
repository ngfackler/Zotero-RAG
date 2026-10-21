import re
import streamlit as st
import html
from pathlib import Path
from datetime import datetime
from filelock import FileLock, Timeout
from config import bounded_history, validate_question
from export_safety import safe_markdown
from library_storage import active_index_root, rebuild_library

from zotero_connection import connect_zotero, validate_library
from zotero_ingest import ingest_library
from chunk_papers import chunk_library
from embed_chunks import embed_library
from rag_answer import (
    rewrite_question,
    identify_paper,
    retrieve_chunks,
    resolve_source_references,
    include_referenced_sources,
    format_context,
    generate_answer,
    validate_citations
)

st.set_page_config(
    page_title="Zotero Research Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("Zotero RAG Research Assistant")
st.write("Ask questions about the scientific papers in your Zotero library.")

# initialize conversation state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "source_registry" not in st.session_state:
    st.session_state["source_registry"] = {}

if "active_paper" not in st.session_state:
    st.session_state["active_paper"] = None

if "pending_papers" not in st.session_state:
    st.session_state["pending_papers"] = []
if "pending_question" not in st.session_state:
    st.session_state["pending_question"] = None

# interface to connect to Zotero and OpenAI
st.header("Connect Your Zotero Library")

library_type = st.radio(
    "Library type",
    ["user", "group"],
    format_func=lambda x: "Personal library" if x == "user" else "Group library"
)

library_id = st.text_input(
    "Zotero User ID" if library_type == "user" else "Zotero Group ID"
)

zotero_api_key = st.text_input(
    "Zotero API Key",
    type="password"
)

openai_api_key = st.text_input(
    "OpenAI API Key",
    type="password"
)

if st.button("Connect to Zotero"):
    if not all([
        library_id.strip(),
        zotero_api_key.strip(),
        openai_api_key.strip()
    ]):
        st.warning("Please complete all fields.")

    else:
        try:
            validate_library(library_id.strip(), library_type)
            library = connect_zotero(
                library_id.strip(),
                library_type,
                zotero_api_key.strip()
            )

            st.success("Successfully connected to Zotero!")
            # reset the conversation when connecting to a library
            st.session_state.pop("rag_results", None)
            st.session_state["messages"] = []
            st.session_state["source_registry"] = {}
            st.session_state["active_paper"] = None
            st.session_state["pending_papers"] = []
            st.session_state["pending_question"] = None
            st.session_state["connected_library"] = {
                "library_id": library_id.strip(),
                "library_type": library_type,
                "api_key": zotero_api_key.strip(),
                "openai_api_key": openai_api_key.strip()
            }
        except ValueError:
            st.error("Enter a valid numeric Zotero library ID and check the connection fields.")
        except Exception:
            st.error(
                "Could not connect to Zotero. "
                "Check your library ID, API key, and permissions."
            )

def reset_chat():
    st.session_state["messages"] = []
    st.session_state["source_registry"] = {}
    st.session_state["active_paper"] = None
    st.session_state["pending_papers"] = []
    st.session_state["pending_question"] = None


def report_error(error, action):
    # API exception strings may include sensitive request details; never display them.
    if isinstance(error, (ValueError, FileNotFoundError)):
        st.error(str(error))
    elif isinstance(error, Timeout):
        st.error("This library is busy. Wait for its current operation to finish and try again.")
    else:
        st.error(f"Could not {action}. Check your connection, API access and billing, and available disk space, then retry.")


if "connected_library" in st.session_state:
    st.subheader("Index Your Library & Generate Embeddings")
    connection = st.session_state["connected_library"]
    library_root = Path("data/libraries") / connection["library_type"] / connection["library_id"]
    operation = None
    if st.button("Index My Library"):
        operation = "index"
    if st.button("Chunk My Papers"):
        operation = "chunk"
    if st.button("Generate Embeddings"):
        operation = "embed"
    st.caption("To refresh replaced or removed papers, rebuild the full index. Your previous index is kept until the rebuild succeeds. Rebuilding sends paper text to OpenAI again and incurs API usage charges.")
    if st.button("Rebuild Library Index"):
        operation = "rebuild"

    if operation:
        try:
            library_root.mkdir(parents=True, exist_ok=True)
            with FileLock(str(library_root / ".index.lock"), timeout=0):
                with st.spinner("Processing your library..."):
                    index_root = active_index_root(library_root) if operation != "rebuild" else None
                    if operation == "index":
                        summary = ingest_library(connection["library_id"], connection["library_type"], connection["api_key"], index_root / "papers")
                        if summary["failed"]:
                            st.warning(f"Processed {summary['processed']} papers; {summary['failed']} failed. Retry indexing before chunking and embedding.")
                        else:
                            st.success(f"Processed {summary['processed']} papers. Skipped {summary['skipped']}.")
                    elif operation == "chunk":
                        summary = chunk_library(index_root / "papers", index_root / "chunks")
                        st.success(f"Generated {summary['chunks']} chunks from {summary['processed']} papers. Skipped {summary['skipped']}.")
                    elif operation == "embed":
                        summary = embed_library(index_root / "chunks", index_root / "chroma_db", connection["openai_api_key"])
                        reset_chat()
                        st.success(f"Embedded {summary['newly_embedded']} new chunks. Total: {summary['collection_total']}.")
                    else:
                        summary = rebuild_library(library_root, connection["library_id"], connection["library_type"], connection["api_key"], connection["openai_api_key"])
                        reset_chat()
                        st.success(f"Rebuilt the index: {summary['processed']} papers and {summary['collection_total']} passages. Started a new chat.")
        except Exception as error:
            report_error(error, "update the library index")

# Export the conversation and cited sources as safe HTML.

def export_conversation_html(messages):
    """Export the conversation as a styled HTML document."""

    def format_text(text):
        return safe_markdown(text)

    references = {}
    conversation = []

    for message in messages:
        role = "You" if message["role"] == "user" else "Assistant"
        role_class = "user" if message["role"] == "user" else "assistant"

        conversation.append(
            f'<section class="message {role_class}">'
            f'<div class="role">{role}</div>'
            f'<div class="content">{format_text(message["content"])}</div>'
            '</section>'
        )

        if message["role"] == "assistant":
            for source_id, source in message.get("sources", {}).items():
                references[source_id] = source

    reference_html = ""

    if references:
        items = []

        for source_id in sorted(
            references,
            key=lambda sid: int(sid[1:])
        ):
            source = references[source_id]

            items.append(
                '<div class="reference">'
                f'<strong>[{html.escape(source_id)}]</strong> '
                f'{html.escape(str(source["title"]))}<br>'
                f'<span>Authors: {html.escape(str(source["authors"]))}</span><br>'
                f'<span>PDF page: {html.escape(str(source["page"]))}</span>'
                '</div>'
            )

        reference_html = (
            '<section class="references">'
            '<h2>References</h2>'
            + "\n".join(items)
            + '</section>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Zotero Research Conversation</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        max-width: 850px;
        margin: 50px auto;
        padding: 0 24px;
        line-height: 1.7;
        color: #253047;
        background: #ffffff;
    }}
    h1 {{ font-size: 30px; margin-bottom: 8px; }}
    h2 {{ margin-top: 28px; }}
    .subtitle {{ color: #64748b; margin-bottom: 40px; }}
    .message {{
        padding: 22px 26px;
        margin: 18px 0;
        border-radius: 12px;
    }}
    .user {{ background: #eef4fc; }}
    .assistant {{ background: #f7f8fa; }}
    .role {{
        font-weight: 700;
        font-size: 14px;
        margin-bottom: 12px;
        color: #475569;
    }}
    .content > :first-child {{ margin-top: 0; }}
    .content > :last-child {{ margin-bottom: 0; }}
    .references {{
        border-top: 1px solid #d9dee7;
        margin-top: 48px;
        padding-top: 20px;
    }}
    .reference {{
        margin: 14px 0;
        padding-bottom: 12px;
        border-bottom: 1px solid #edf0f4;
    }}
    .reference span {{ color: #64748b; font-size: 14px; }}
    pre {{
        overflow-x: auto;
        padding: 16px;
        background: #e9edf3;
        border-radius: 8px;
    }}
    code {{ overflow-wrap: anywhere; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d9dee7; padding: 8px; }}
    @media print {{
        body {{ margin: 20px auto; }}
        .message {{ break-inside: avoid; }}
    }}
</style>
</head>
<body>
<h1>Zotero Research Assistant</h1>
<p class="subtitle">Conversation exported on {html.escape(datetime.now().strftime("%B %d, %Y"))}</p>
{''.join(conversation)}
{reference_html}
</body>
</html>"""



# sidebar settings
st.sidebar.header("Retrieval Settings")

top_k = st.sidebar.slider(
    "Number of retrieved passages",
    min_value=3,
    max_value=20,
    value=8
)

# start a new conversation
if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()


# download the current conversation

# Download the current conversation as HTML
if st.session_state["messages"]:

    html_text = export_conversation_html(
        st.session_state["messages"]
    )

    filename = (
        "zotero_conversation_"
        + datetime.now().strftime("%Y-%m-%d_%H-%M")
        + ".html"
    )

    st.sidebar.download_button(
        label="📥 Download Conversation",
        data=html_text,
        file_name=filename,
        mime="text/html",
        use_container_width=True
    )

st.subheader("Research Chat")

# display the full conversation
for message in st.session_state["messages"]:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "results" in message:

            results = message["results"]
            sources = message["sources"]

            with st.expander("Retrieved Evidence"):

                for source_id, source in sources.items():
                    st.markdown(
                        f"**[{source_id}] {source['title']} "
                        f"— PDF page {source['page']}**"
                    )
                    st.write(f"**Authors:** {source['authors']}")
                    st.write(source["text"])

# chat input
question = st.chat_input(
    "Ask a question about your library...",
    disabled="connected_library" not in st.session_state
)

if question:

    connection = st.session_state["connected_library"]

    library_root = (
        Path("data/libraries")
        / connection["library_type"]
        / connection["library_id"]
    )

    # capture the full conversation before adding the new question
    chat_history = bounded_history([
        {
            "role": message["role"],
            "content": message["content"]
        }
        for message in st.session_state["messages"]
    ])

    with st.chat_message("user"):
        st.markdown(question)

    
    with st.chat_message("assistant"):

        try:
            validate_question(question)
            library_root.mkdir(parents=True, exist_ok=True)
            with FileLock(str(library_root / ".index.lock"), timeout=0), st.spinner("Searching your Zotero library..."):

                chroma_dir = active_index_root(library_root) / "chroma_db"

                # Resolve a numbered/ordinal reply to a pending clarification.
                pending = st.session_state["pending_papers"]
                selection = question.strip().lower().rstrip(".!")
                ordinal = {"first": 0, "second": 1, "third": 2,
                           "fourth": 3, "fifth": 4, "last": len(pending) - 1}
                match = re.fullmatch(r"(?:the\s+)?(first|second|third|fourth|fifth|last)(?:\s+one)?", selection)
                index = ordinal[match.group(1)] if match and pending else None
                if pending and index is None and re.fullmatch(r"(?:option\s*)?\d+", selection):
                    index = int(re.search(r"\d+", selection).group()) - 1

                if pending and index is not None and 0 <= index < len(pending):
                    paper_match = {"status": "matched", "papers": [pending[index]]}
                    question_for_answer = (
                        st.session_state["pending_question"] or question
                    )
                else:
                    paper_match = identify_paper(
                        question=question,
                        chat_history=chat_history,
                        chroma_dir=chroma_dir,
                        openai_api_key=connection["openai_api_key"],
                        active_paper=st.session_state["active_paper"]
                    )
                    question_for_answer = question

                if paper_match["status"] == "matched":
                    st.session_state["active_paper"] = paper_match["papers"][0]
                    st.session_state["pending_papers"] = []
                elif paper_match["status"] == "ambiguous":
                    st.session_state["pending_question"] = question
                    st.session_state["pending_papers"] = paper_match["papers"]
                    st.session_state["active_paper"] = None
                elif paper_match["status"] == "general":
                    st.session_state["active_paper"] = None
                    st.session_state["pending_papers"] = []
                else:
                    st.session_state["active_paper"] = None
                    st.session_state["pending_papers"] = []

                if paper_match["status"] == "ambiguous":
                    options = "\n".join(
                        f"{i}. **{p['title']}** ({p['date']})"
                        for i, p in enumerate(paper_match["papers"], 1)
                    )
                    answer = (
                        "I found multiple matching papers. "
                        "Which one do you mean?\n\n" + options
                    )

                elif paper_match["status"] == "not_found":
                    answer = (
                        "I couldn't identify that paper in your indexed "
                        "Zotero library. Could you provide its title "
                        "or authors?"
                    )

                else:
                    # General question or one identified paper.
                    search_question = rewrite_question(
                        question=question_for_answer,
                        chat_history=chat_history,
                        openai_api_key=connection["openai_api_key"]
                    )

                    if paper_match["status"] == "matched":
                        search_question = f"{paper_match['papers'][0]['title'][:300]}: {search_question}"

                    zotero_keys = None
                    if paper_match["status"] == "matched":
                        zotero_keys = paper_match["papers"][0]["zotero_keys"]

                    results = retrieve_chunks(
                        question=search_question,
                        chroma_dir=chroma_dir,
                        openai_api_key=connection["openai_api_key"],
                        top_k=top_k,
                        zotero_keys=zotero_keys
                    )

                    # Preserve exact source-reference functionality.
                    referenced_chunks = resolve_source_references(
                        question=question,
                        source_registry=st.session_state["source_registry"],
                        chroma_dir=chroma_dir
                    )

                    results = include_referenced_sources(
                        results=results,
                        referenced_chunks=referenced_chunks
                    )

                    context, sources = format_context(
                        results=results,
                        source_registry=st.session_state["source_registry"]
                    )

                    answer = generate_answer(
                        question=question_for_answer,
                        context=context,
                        openai_api_key=connection["openai_api_key"],
                        chat_history=chat_history
                    )

                    citation_check = validate_citations(answer, sources)

            st.markdown(answer)

            if paper_match["status"] in {"general", "matched"}:

                if citation_check["invalid_ids"]:
                    st.warning(
                        "Unknown citations: "
                        + ", ".join(citation_check["invalid_ids"])
                    )

                with st.expander("Retrieved Evidence"):
                    for source_id, source in sources.items():
                        st.markdown(
                            f"**[{source_id}] {source['title']} "
                            f"— PDF page {source['page']}**"
                        )
                        st.write(f"**Authors:** {source['authors']}")
                        st.write(source["text"])

            if paper_match["status"] != "ambiguous":
                st.session_state["pending_question"] = None

            # Save both normal answers and clarification turns.
            assistant_message = {
                "role": "assistant",
                "content": answer
            }

            if paper_match["status"] in {"general", "matched"}:
                assistant_message["results"] = results
                assistant_message["sources"] = sources

            st.session_state["messages"].extend([
                {
                    "role": "user",
                    "content": question
                },
                assistant_message
            ])
            st.rerun()

        except Exception as error:
            report_error(error, "generate an answer")
