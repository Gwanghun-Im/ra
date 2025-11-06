"""RAG Agent Implementation"""

import os
import logging
from typing import Any, Dict, List, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END

from mcp_custom.tools.mcp_tools import (
    search_knowledge_base,
    rag_query_tool,
    get_rag_stats_tool,
)

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the RAG Agent"""

    messages: List[Any]
    query: str
    rag_result: Dict[str, Any] | None
    next_action: str


class RAGAgent:
    """RAG Agent for knowledge base queries using LangGraph"""

    def __init__(self):
        """Initialize the RAG Agent"""
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY")
        )

        # Define available RAG tools
        self.tools = [
            search_knowledge_base,
            rag_query_tool,
            get_rag_stats_tool,
        ]

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Build the graph
        self.graph = self._build_graph()

        logger.info("RAG Agent initialized")

    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("rag_processor", self._rag_processor_node)
        workflow.add_node("tools", self._tools_node)
        workflow.add_node("synthesize", self._synthesize_node)

        # Add edges
        workflow.set_entry_point("rag_processor")
        workflow.add_conditional_edges(
            "rag_processor",
            self._should_continue,
            {"continue": "tools", "synthesize": "synthesize", "end": END},
        )
        workflow.add_edge("tools", "rag_processor")
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    async def _rag_processor_node(self, state: AgentState) -> AgentState:
        """Main RAG processing node"""
        messages = state["messages"]

        # Add system message if not present
        if not any(isinstance(m, SystemMessage) for m in messages):
            system_msg = SystemMessage(
                content="""You are a specialized RAG (Retrieval-Augmented Generation) Agent.

Your responsibilities:
- Search and retrieve relevant information from the knowledge base
- Process queries to find the most relevant context
- Provide accurate answers based on retrieved documents
- Report knowledge base statistics when requested

Always:
1. Use the appropriate tool for the query type
2. Retrieve relevant context before answering
3. Cite sources when providing information
4. Be clear when information is not available in the knowledge base
5. Provide concise and accurate responses

Available tools:
- search_knowledge_base: Search for relevant documents in the knowledge base
- rag_query_tool: Retrieve context and answer queries using RAG
- get_rag_stats_tool: Get statistics about the knowledge base
"""
            )
            messages = [system_msg] + messages

        # Get LLM response asynchronously
        response = await self.llm_with_tools.ainvoke(messages)

        # Update state
        return {**state, "messages": messages + [response]}

    async def _tools_node(self, state: AgentState) -> AgentState:
        """Execute tool calls using async HTTP calls to Docker MCP servers"""
        messages = state["messages"]
        last_message = messages[-1]

        # Create a mapping of tool names to tool functions
        tools_by_name = {tool.name: tool for tool in self.tools}

        # Execute each tool call
        tool_messages = []
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name in tools_by_name:
                    tool = tools_by_name[tool_name]
                    try:
                        # Execute the tool asynchronously (calls Docker MCP server via HTTP)
                        result = await tool.ainvoke(tool_args)
                        tool_messages.append(
                            ToolMessage(
                                content=str(result), tool_call_id=tool_call["id"], name=tool_name
                            )
                        )
                    except Exception as e:
                        logger.error(f"Error executing tool {tool_name}: {e}")
                        tool_messages.append(
                            ToolMessage(
                                content=f"Error: {str(e)}",
                                tool_call_id=tool_call["id"],
                                name=tool_name,
                            )
                        )

        return {**state, "messages": messages + tool_messages}

    def _should_continue(self, state: AgentState) -> str:
        """Determine next step in the workflow"""
        last_message = state["messages"][-1]

        # If LLM called tools, execute them
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"

        # Check if we have results to synthesize
        content = last_message.content.lower() if hasattr(last_message, "content") else ""
        if any(keyword in content for keyword in ["based on", "according to", "found"]):
            return "synthesize"

        return "end"

    async def _synthesize_node(self, state: AgentState) -> AgentState:
        """Synthesize final response from RAG results"""
        messages = state["messages"]

        # Create synthesis prompt
        synthesis_prompt = HumanMessage(
            content="""Based on the retrieved information, provide a clear and concise answer.

If sources were found:
- Summarize the relevant information
- Include key facts and insights
- Cite the sources used

If no relevant information was found:
- Clearly state that the information is not available in the knowledge base
- Suggest what type of information might help

Keep your response focused and professional."""
        )

        messages.append(synthesis_prompt)
        final_response = await self.llm.ainvoke(messages)

        return {
            **state,
            "messages": messages + [final_response],
            "rag_result": {"content": final_response.content, "query": state["query"]},
        }

    async def process_request(self, query: str) -> Dict[str, Any]:
        """
        Process RAG query request

        Args:
            query: User's search query or question

        Returns:
            Agent response with retrieved information
        """
        logger.info(f"Processing RAG query: {query}")

        # Initialize state
        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "rag_result": None,
            "next_action": "start",
        }

        # Run the graph
        final_state = await self.graph.ainvoke(initial_state)

        # Extract final response
        last_message = final_state["messages"][-1]
        response_content = (
            last_message.content if hasattr(last_message, "content") else str(last_message)
        )

        return {
            "response": response_content,
            "query": query,
            "rag_result": final_state.get("rag_result"),
            "message_count": len(final_state["messages"]),
        }

    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities"""
        return [
            "knowledge_base_search",
            "document_retrieval",
            "rag_query",
            "kb_statistics",
            "semantic_search",
        ]


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        agent = RAGAgent()

        # Test request
        result = await agent.process_request(
            "What are the key principles of portfolio diversification?"
        )

        print("Agent Response:")
        print(result["response"])

    asyncio.run(main())
