"""
Citation Formatter — converts raw LangChain source documents into
human-readable, deduplicated citation objects for display in the UI.
"""


def format_citations(source_docs: list) -> list:
    """
    Convert a list of LangChain Document objects into citation dicts.

    Deduplicates by (source_file, page_number) — if multiple chunks
    came from the same page, only the first (most relevant) is shown.

    Returns a list of dicts with keys:
      - document: filename
      - page: page number (int)
      - excerpt: first 220 chars of the chunk, cleaned up
    """
    seen = set()
    citations = []

    for doc in source_docs:
        meta = doc.metadata
        source_file = meta.get("source_file", "Unknown Document")
        page = meta.get("page_number", "?")

        dedup_key = f"{source_file}::{page}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        raw_text = doc.page_content.strip()
        # Clean up whitespace artifacts from PDF extraction
        excerpt = " ".join(raw_text.split())[:220]
        if len(raw_text) > 220:
            excerpt += "..."

        citations.append({
            "document": source_file,
            "page": page,
            "excerpt": excerpt,
        })

    return citations


def format_chat_history_for_agent(messages: list) -> str:
    """Flatten Streamlit message dicts into a plain string for the agent."""
    if not messages:
        return "No prior conversation."
    lines = []
    for msg in messages[-8:]:  # Last 4 exchanges
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:300]  # Truncate long messages
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
