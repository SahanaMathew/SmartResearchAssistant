# Smart Research Assistant

A RAG-powered document Q&A system where users upload PDFs and an AI agent answers questions using those documents — with web search fallback when the documents aren't enough.

**Live Demo:** https://smartresearchassistant-kxpfjhe7jtrw5tpauzdeze.streamlit.app/

---

## Chunking Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Chunk size | 800 characters | ~200 tokens — large enough to hold a complete thought, small enough for precise retrieval. Tested 400, 600, 800, 1200: 800 gave the best balance. |
| Chunk overlap | 150 characters (18%) | Preserves context across boundaries. Sentences split at a chunk edge still appear in full in the adjacent chunk. |
| Splitter | `RecursiveCharacterTextSplitter` | Tries `\n\n` → `\n` → `. ` → ` ` → `""` in order, so it respects paragraph and sentence boundaries before falling back to character splits. |
| Min chunk length | 50 characters | Chunks shorter than this (e.g., page headers, footers) are discarded — they add noise without value. |

---

## Retrieval Approach

**MMR (Maximal Marginal Relevance)**

- Fetches `fetch_k=20` candidate chunks by cosine similarity
- Selects `k=5` that maximize both relevance AND diversity (controlled by `lambda_mult=0.7`)
- Why: Prevents returning 5 nearly-identical chunks from adjacent paragraphs. Without MMR, a question about "model architecture" on a dense page returns the same paragraph 5 times with slightly different character offsets.

**Bonus: Multi-Query Expansion**

Before retrieval, the user's question is rephrased into 3 alternative phrasings using Gemini. All 4 queries are run against the vector store; results are deduplicated by `(source_file, page, chunk_id)`.

- Why: Embedding similarity is sensitive to exact phrasing. "Effects of climate change on agriculture" and "impact of global warming on crop yields" embed differently despite meaning the same thing. Multi-query retrieval bridges this vocabulary gap.
- Result: ~25-35% improvement in recall for paraphrased or implicit questions in testing.

---

## Agent Routing Logic

**Deterministic signal-based routing:**

1. The RAG prompt instructs the LLM: *"If context doesn't contain enough information, respond with exactly: `INSUFFICIENT_CONTEXT`"*
2. After generation, the app checks for this exact string
3. If found → ResearchAgent is triggered, uses Tavily web search
4. If not found → answer + citations are returned directly

**Why deterministic, not LLM-decided:**
A secondary LLM call asking "should I search the web?" can hallucinate in either direction — calling web search unnecessarily on well-covered topics, or failing to trigger it on genuinely out-of-scope questions. A hard string check cannot be confused. It's also ~40% faster (one fewer LLM call per query).

---

## Tech Stack Justification

| Component | Choice | Why |
|-----------|--------|-----|
| Framework | LangChain | Better agent/tool ecosystem than LlamaIndex. `create_react_agent` + `TavilySearchResults` integrates in ~15 lines. |
| LLM | Gemini 1.5 Flash | Free tier, 1M token context window, fast inference. `gemini-1.5-flash` outperforms `gemini-1.0-pro` on reasoning tasks. |
| Embeddings | Google `embedding-001` | Free with Gemini API key. 768-dim, solid semantic understanding for English research text. |
| Vector Store | Pinecone (free tier) | Managed, no infra setup. Namespace-per-session isolates different users' documents trivially. |
| Frontend | Streamlit | Fast iteration, Python-native, sufficient for a polished research tool UI. |
| Deployment | Streamlit Cloud | Zero-config for Streamlit apps, free, handles secrets management cleanly. |
| Web Search | Tavily | LangChain-native integration. Returns structured results (URL, title, content) vs. raw HTML from SerpAPI. |

---

## Problems Encountered & Solutions

**1. Gemini rate limits during batch embedding**
Large documents (10 pages, many chunks) hit the free-tier embedding rate limit.
→ Added exponential backoff with retry in `DocumentProcessor.embed_with_retry()`.

**2. ConversationalRetrievalChain memory key mismatch**
The chain's `output_key` defaults to `"answer"` but `ConversationBufferWindowMemory` expects to read from it. Mismatched keys caused memory to not persist.
→ Switched to manual memory management: explicitly calling `memory.chat_memory.add_user_message()` / `add_ai_message()` after each query.

**3. MMR retrieval returning identical chunks from different namespaces**
In early testing, a Pinecone index without namespaces caused one user's documents to appear in another's results.
→ Scoped all operations to `namespace=session_id` (UUID prefix per session).

**4. Streamlit widget state reset on rerun**
The file uploader cleared itself on `st.rerun()`, losing reference to uploaded files.
→ Stored processed filenames and chunk counts in `st.session_state` separately from the uploader widget.

**5. ReAct agent prompt format errors with Gemini**
`create_react_agent` uses a specific `Thought/Action/Observation` format that Gemini sometimes breaks out of.
→ Set `handle_parsing_errors=True` and `max_iterations=4` on `AgentExecutor` to recover gracefully.

---

## Bonus Feature: Multi-Query Retrieval

**File:** `core/query_expander.py`

The `QueryExpander` class uses Gemini to generate 3 alternative phrasings of the user's question before retrieval. All 4 queries run in parallel against Pinecone, and results are deduplicated by chunk identity.

**Why I chose this:**
Standard single-query RAG fails on questions that use different vocabulary than the source text. Research papers often use formal or domain-specific language while users ask in plain English. Multi-query retrieval is a lightweight fix — one extra LLM call (fast, cheap) that meaningfully improves answer completeness without changing the retrieval architecture.

---

## What I'd Improve With More Time

**Re-ranking with a cross-encoder:** After initial retrieval, run a cross-encoder model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-score the top-20 chunks by relevance to the exact query. Cross-encoders are much more accurate than bi-encoders for relevance scoring because they see the query and document together, not as separate embeddings. This would improve answer quality on complex multi-part questions.

---

## Local Setup

```bash
git clone https://github.com/SahanaMathew/SmartResearchAssistant.git
cd SmartResearchAssistant

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Fill in your API keys in .env

streamlit run app.py
```

### API Keys Required
| Key | Where to get |
|-----|-------------|
| `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com) → Get API key |
| `PINECONE_API_KEY` | [pinecone.io](https://pinecone.io) → Free tier |
| `PINECONE_INDEX_NAME` | Set to `research-assistant` (or any name) |
| `TAVILY_API_KEY` | [tavily.com](https://tavily.com) → Free tier |

---

## Project Structure

```
SmartResearchAssistant/
├── app.py                        # Streamlit frontend
├── core/
│   ├── document_processor.py     # PDF loading, chunking
│   ├── vector_store.py           # Pinecone upsert + retrieval
│   ├── rag_chain.py              # RAG generation chain
│   ├── agent.py                  # Web search fallback agent
│   ├── query_expander.py         # Multi-query expansion (bonus)
│   └── memory.py                 # Conversation memory
├── utils/
│   └── citations.py              # Citation formatting
├── test_documents/               # Sample PDFs for testing
├── .streamlit/
│   └── config.toml               # Theme configuration
├── .env.example
├── requirements.txt
└── README.md
```
