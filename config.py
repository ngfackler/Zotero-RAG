"""Shared model choices and request size limits."""

EMBEDDING_MODEL = "text-embedding-3-small"
ANSWER_MODEL = "gpt-4.1-mini"
MAX_HISTORY_MESSAGES = 12
MAX_HISTORY_CHARS = 16000
MAX_QUESTION_CHARS = 6000
MAX_CATALOG_PAPERS = 80
MAX_CATALOG_CHARS = 24000
MAX_CONTEXT_CHARS = 60000
MAX_OUTPUT_TOKENS = 3000


def bounded_history(messages):
    """Keep recent complete messages within a character budget."""
    selected = []
    remaining = MAX_HISTORY_CHARS
    for message in reversed((messages or [])[-MAX_HISTORY_MESSAGES:]):
        content = str(message.get("content", ""))
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        if len(content) > remaining:
            break
        selected.append({"role": role, "content": content})
        remaining -= len(content)
    return list(reversed(selected))


def validate_question(question):
    if not question.strip():
        raise ValueError("Enter a question first.")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"Please shorten your question to {MAX_QUESTION_CHARS} characters.")
    return question
