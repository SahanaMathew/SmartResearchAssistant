"""
Conversation Memory — session-scoped, window-limited chat history.

Uses ConversationBufferWindowMemory with k=10 (last 10 exchanges).
This prevents context overflow for long sessions while still enabling
follow-up questions like "tell me more about that" to resolve correctly.
"""

from langchain.memory import ConversationBufferWindowMemory


def create_memory() -> ConversationBufferWindowMemory:
    return ConversationBufferWindowMemory(
        k=10,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
