"""
RAG Chain — retrieval-augmented generation with conversation memory.

Agent routing logic:
- The LLM is instructed to respond with the exact token "INSUFFICIENT_CONTEXT"
  if the retrieved chunks don't contain enough information to answer.
- This is deterministic string-matching routing, not LLM-decided routing.
  Deterministic routing is more reliable — an LLM asked "should I search the web?"
  can hallucinate either way. A hard string match cannot.
- When INSUFFICIENT_CONTEXT is detected, the caller (app.py) triggers the
  ResearchAgent which uses Tavily web search as a fallback.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from core.query_expander import QueryExpander


INSUFFICIENT_CONTEXT_SIGNAL = "INSUFFICIENT_CONTEXT"

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""You are a precise research assistant. Answer questions strictly from the provided document context.

DOCUMENT CONTEXT:
{context}

CONVERSATION HISTORY:
{chat_history}

USER QUESTION: {question}

INSTRUCTIONS:
1. Answer ONLY using information found in the document context above.
2. If the context contains relevant information, give a clear, well-structured answer.
3. If the context does NOT contain enough information to answer the question, respond with exactly this token and nothing else: INSUFFICIENT_CONTEXT
4. Never invent facts, statistics, or claims not found in the context.
5. Keep your answer focused and relevant to the question.

ANSWER:""",
)


class RAGChain:
    def __init__(self, retriever, memory):
        self.retriever = retriever
        self.memory = memory
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            convert_system_message_to_human=True,
        )
        self.expander = QueryExpander(llm=self.llm)
        self.prompt = RAG_PROMPT
        self.parser = StrOutputParser()

    def _format_chat_history(self) -> str:
        messages = self.memory.chat_memory.messages
        if not messages:
            return "No previous conversation."
        formatted = []
        for msg in messages[-6:]:  # Last 3 exchanges
            if isinstance(msg, HumanMessage):
                formatted.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted.append(f"Assistant: {msg.content}")
        return "\n".join(formatted)

    def _format_context(self, docs: list) -> str:
        if not docs:
            return "No relevant context found."
        parts = []
        for doc in docs:
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page_number", "?")
            parts.append(
                f"[Source: {source}, Page {page}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(parts)

    def query(self, question: str) -> dict:
        """
        Run multi-query retrieval, then generate an answer.
        Returns answer text, source docs, and whether web search is needed.
        """
        # Multi-query retrieval (bonus feature)
        source_docs = self.expander.multi_retrieve(question, self.retriever)

        context = self._format_context(source_docs)
        chat_history = self._format_chat_history()

        prompt_value = self.prompt.format(
            context=context,
            question=question,
            chat_history=chat_history,
        )

        response = self.llm.invoke(prompt_value)
        answer = self.parser.invoke(response)

        needs_web = INSUFFICIENT_CONTEXT_SIGNAL in answer

        # Only save to memory if we have a real answer
        if not needs_web:
            self.memory.chat_memory.add_user_message(question)
            self.memory.chat_memory.add_ai_message(answer)

        return {
            "answer": answer if not needs_web else "",
            "source_docs": source_docs,
            "needs_web_search": needs_web,
        }
