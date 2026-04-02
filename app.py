"""
Smart Research Assistant
RAG-powered document Q&A with agent fallback and multi-query retrieval.
"""

import os
import uuid
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.document_processor import DocumentProcessor
from core.vector_store import VectorStoreManager
from core.rag_chain import RAGChain
from core.agent import ResearchAgent
from core.memory import create_memory
from utils.citations import format_citations, format_chat_history_for_agent

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.stApp { background: #0f1117; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #13151e;
    border-right: 1px solid #1f2335;
}

/* ── Chat messages ── */
.msg-wrapper { margin: 10px 0; }
.msg-user {
    background: #1a2744;
    border: 1px solid #2a3f6f;
    border-radius: 16px 16px 4px 16px;
    padding: 14px 18px;
    margin-left: 15%;
    color: #e0e8ff;
    line-height: 1.6;
}
.msg-ai {
    background: #151922;
    border: 1px solid #1f2b40;
    border-left: 3px solid #4f8ef7;
    border-radius: 4px 16px 16px 16px;
    padding: 14px 18px;
    margin-right: 8%;
    color: #d0d8e8;
    line-height: 1.7;
}
.msg-label {
    font-size: 0.72em;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 6px;
    opacity: 0.6;
}
.label-user { color: #7eb8f7; }
.label-ai   { color: #4f8ef7; }

/* ── Badges ── */
.badge {
    display: inline-block;
    font-size: 0.72em;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin: 6px 4px 0 0;
    letter-spacing: 0.04em;
}
.badge-doc  { background: #0d2240; color: #7eb8f7; border: 1px solid #2a5090; }
.badge-web  { background: #0d2e14; color: #6fcf97; border: 1px solid #1e6b38; }
.badge-memo { background: #2a1f0d; color: #f6c66e; border: 1px solid #6b4a1e; }

/* ── Citation cards ── */
.citation-card {
    background: #101520;
    border: 1px solid #1f2b40;
    border-left: 3px solid #2a5090;
    border-radius: 8px;
    padding: 12px 14px;
    margin: 6px 0;
    font-size: 0.85em;
}
.citation-title {
    color: #7eb8f7;
    font-weight: 600;
    margin-bottom: 5px;
}
.citation-excerpt {
    color: #8899bb;
    font-style: italic;
    line-height: 1.5;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 80px 20px;
    color: #3a4560;
}
.empty-state h2 { font-size: 1.8em; margin-bottom: 10px; color: #4a5580; }
.empty-state p  { font-size: 1em; }

/* ── Uploaded doc list ── */
.doc-pill {
    display: inline-block;
    background: #0d1a30;
    border: 1px solid #1f3560;
    border-radius: 20px;
    padding: 5px 12px;
    font-size: 0.82em;
    color: #7eb8f7;
    margin: 3px 2px;
}

/* ── Progress / status ── */
.status-bar {
    background: #151922;
    border: 1px solid #1f2b40;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.85em;
    color: #8899bb;
    margin: 8px 0;
}

/* ── Mobile responsiveness ── */
@media (max-width: 768px) {
    .msg-user {
        margin-left: 2%;
    }
    .msg-ai {
        margin-right: 2%;
    }
    .empty-state {
        padding: 40px 10px;
    }
    .empty-state h2 {
        font-size: 1.3em;
    }
    .badge {
        font-size: 0.68em;
        padding: 2px 8px;
    }
    .citation-card {
        padding: 10px 12px;
        font-size: 0.8em;
    }
    .doc-pill {
        font-size: 0.78em;
        padding: 4px 10px;
    }
}
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ───────────────────────────────────────────────
def init_session():
    defaults = {
        "session_id": str(uuid.uuid4())[:8],
        "messages": [],
        "memory": create_memory(),
        "rag_chain": None,
        "docs_uploaded": [],        # list of filenames
        "doc_chunk_counts": {},     # filename → chunk count
        "processing_done": False,
        "total_chunks": 0,
        "vs_manager": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# Agent is cached at module level (stateless, no session data)
@st.cache_resource
def get_agent():
    return ResearchAgent()

agent = get_agent()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Research Assistant")
    st.markdown("---")

    st.markdown("### 📂 Upload Documents")

    MAX_FILES = 5
    MAX_PAGES = 10
    already_loaded = len(st.session_state.docs_uploaded)
    slots_left = MAX_FILES - already_loaded

    if slots_left <= 0:
        st.warning("Maximum 5 documents already loaded. Clear session to start over.")
        uploaded_files = []
    else:
        st.caption(f"Up to {slots_left} more PDF(s) · max {MAX_PAGES} pages each")
        uploaded_files = st.file_uploader(
            label="Drop PDFs here",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

    # Live validation feedback before processing
    if uploaded_files:
        new_files = [f for f in uploaded_files
                     if f.name not in st.session_state.docs_uploaded]
        duplicate_files = [f for f in uploaded_files
                           if f.name in st.session_state.docs_uploaded]

        if duplicate_files:
            st.info(f"{len(duplicate_files)} file(s) already loaded and will be skipped.")

        total_after = already_loaded + len(new_files)
        if total_after > MAX_FILES:
            st.error(
                f"Too many files: {total_after} total would exceed the 5-file limit. "
                f"You can add at most {slots_left} more file(s)."
            )

    process_btn = st.button(
        "⚡ Process Documents",
        type="primary",
        use_container_width=True,
        disabled=(not uploaded_files),
    )

    if process_btn and uploaded_files:
        # Filter to new (non-duplicate) files only
        new_files = [f for f in uploaded_files
                     if f.name not in st.session_state.docs_uploaded]

        if not new_files:
            st.info("All selected files are already loaded.")
        elif already_loaded + len(new_files) > MAX_FILES:
            st.error(
                f"Cannot process: would exceed the {MAX_FILES}-file limit. "
                f"You have {already_loaded} file(s) loaded. "
                f"Clear session or select fewer files."
            )
        else:
            with st.spinner("Initializing..."):
                processor = DocumentProcessor()
                if st.session_state.vs_manager is None:
                    st.session_state.vs_manager = VectorStoreManager()

            all_chunks = []
            errors = []
            progress_bar = st.progress(0, text="Processing documents...")

            for i, f in enumerate(new_files):
                progress_bar.progress(
                    i / len(new_files),
                    text=f"Processing {f.name}..."
                )
                try:
                    chunks = processor.process(f)
                    all_chunks.extend(chunks)
                    st.session_state.doc_chunk_counts[f.name] = len(chunks)
                    st.session_state.docs_uploaded.append(f.name)
                except ValueError as e:
                    errors.append(str(e))

            if errors:
                for err in errors:
                    st.error(err)

            if all_chunks:
                progress_bar.progress(0.9, text="Building vector index...")
                namespace = st.session_state.session_id
                st.session_state.vs_manager.upsert(all_chunks, namespace)
                retriever = st.session_state.vs_manager.get_retriever(namespace)
                st.session_state.rag_chain = RAGChain(
                    retriever=retriever,
                    memory=st.session_state.memory,
                )
                st.session_state.total_chunks += len(all_chunks)
                st.session_state.processing_done = True
                progress_bar.progress(1.0, text="Done!")
                st.success(
                    f"Indexed {len(all_chunks)} chunks from "
                    f"{len(new_files) - len(errors)} file(s)."
                )
            else:
                progress_bar.progress(1.0, text="No valid content indexed.")
                if not errors:
                    st.warning("No content could be extracted from the uploaded files.")

    # ── Loaded documents list ──
    if st.session_state.docs_uploaded:
        st.markdown("### 📄 Loaded Documents")
        for doc in st.session_state.docs_uploaded:
            chunks = st.session_state.doc_chunk_counts.get(doc, "?")
            st.markdown(
                f'<div class="doc-pill">📄 {doc} <span style="opacity:0.5">({chunks} chunks)</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div class="status-bar">Total indexed: '
            f'<strong>{st.session_state.total_chunks}</strong> chunks '
            f'· Session: <code>{st.session_state.session_id}</code></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Clear session ──
    if st.button("🗑️ Clear Session", use_container_width=True):
        if st.session_state.vs_manager and st.session_state.docs_uploaded:
            with st.spinner("Cleaning up..."):
                st.session_state.vs_manager.delete_namespace(
                    st.session_state.session_id
                )
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ How it works")
    st.caption(
        "1. Upload PDFs → they're chunked & embedded\n"
        "2. Ask a question → top chunks are retrieved\n"
        "3. Gemini generates an answer with citations\n"
        "4. If docs don't have the answer → web search fallback\n"
        "5. Multi-query expansion improves retrieval recall"
    )


# ── Main Chat Area ─────────────────────────────────────────────────────────────
st.markdown("## 💬 Research Chat")

# Display messages or empty state
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <h2>Upload documents to begin</h2>
        <p>Ask any question about your PDFs — research papers, reports, manuals, contracts.<br>
        The assistant will cite its sources and fall back to web search when needed.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-wrapper">
                <div class="msg-label label-user">You</div>
                <div class="msg-user">{msg["content"]}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-wrapper">
                <div class="msg-label label-ai">Assistant</div>
                <div class="msg-ai">{msg["content"]}</div>
            </div>""", unsafe_allow_html=True)

            # Source badges
            badges = ""
            if msg.get("web_search"):
                badges += '<span class="badge badge-web">🌐 Web Search</span>'
            elif msg.get("citations"):
                badges += f'<span class="badge badge-doc">📄 {len(msg["citations"])} source(s)</span>'
            if badges:
                st.markdown(badges, unsafe_allow_html=True)

            # Citation expander
            if msg.get("citations"):
                with st.expander(f"View sources ({len(msg['citations'])})"):
                    for c in msg["citations"]:
                        st.markdown(f"""
                        <div class="citation-card">
                            <div class="citation-title">📄 {c['document']} — Page {c['page']}</div>
                            <div class="citation-excerpt">"{c['excerpt']}"</div>
                        </div>""", unsafe_allow_html=True)


# ── Chat Input ─────────────────────────────────────────────────────────────────
# st.chat_input only returns a value once per submission then clears itself,
# preventing the re-trigger loop that st.text_input causes on st.rerun().
user_input = st.chat_input("Ask a question about your documents...")

if user_input and user_input.strip():
    question = user_input.strip()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})

    if st.session_state.rag_chain is None:
        answer = "Please upload and process at least one document first."
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "citations": [],
            "web_search": False,
        })
    else:
        with st.spinner("Searching documents..."):
            result = st.session_state.rag_chain.query(question)

        if result["needs_web_search"]:
            with st.spinner("Searching the web..."):
                chat_history_str = format_chat_history_for_agent(
                    st.session_state.messages
                )
                web_answer = agent.search(question, chat_history_str)

            full_answer = (
                "The uploaded documents don't contain enough information to answer this. "
                "Here's what I found on the web:\n\n" + web_answer
            )
            # Save web search Q&A to memory too
            st.session_state.memory.chat_memory.add_user_message(question)
            st.session_state.memory.chat_memory.add_ai_message(full_answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "citations": [],
                "web_search": True,
            })
        else:
            citations = format_citations(result["source_docs"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "citations": citations,
                "web_search": False,
            })

    st.rerun()
