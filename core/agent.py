"""
Research Agent — web search fallback using Tavily.

This agent is only invoked when the RAG chain returns INSUFFICIENT_CONTEXT.
It uses LangChain's ReAct agent pattern with Tavily as the search tool.

The agent:
1. Receives the original question + recent chat history for context
2. Uses Tavily to search the web (max 3 results to stay concise)
3. Synthesizes a response that clearly indicates it came from web search
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate


AGENT_PROMPT_TEMPLATE = """You are a helpful research assistant performing a web search because the uploaded documents did not contain enough information to answer the user's question.

You have access to the following tools:
{tools}

Tool names: {tool_names}

Use this format strictly:
Question: the input question you must answer
Thought: your reasoning about what to search for
Action: the tool to use (must be one of [{tool_names}])
Action Input: the search query
Observation: the result from the tool
... (you can repeat Thought/Action/Observation up to 3 times)
Thought: I now have enough information to answer
Final Answer: your complete answer based on web search results

Previous conversation:
{chat_history}

Question: {input}
{agent_scratchpad}"""


class ResearchAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            convert_system_message_to_human=True,
        )
        self.search_tool = TavilySearchResults(
            max_results=3,
            search_depth="advanced",
        )
        self.tools = [self.search_tool]

        prompt = PromptTemplate(
            input_variables=["input", "chat_history", "agent_scratchpad",
                             "tools", "tool_names"],
            template=AGENT_PROMPT_TEMPLATE,
        )

        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )

        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            max_iterations=4,
            handle_parsing_errors=True,
            return_intermediate_steps=False,
        )

    def search(self, question: str, chat_history: str) -> str:
        """
        Perform web search and return a synthesized answer.
        Returns a fallback message if the agent fails.
        """
        try:
            result = self.executor.invoke({
                "input": question,
                "chat_history": chat_history or "No prior conversation.",
            })
            return result.get("output", "I was unable to find relevant information on the web.")
        except Exception as e:
            return (
                f"I attempted a web search but encountered an error: {str(e)}. "
                "Please try rephrasing your question."
            )
