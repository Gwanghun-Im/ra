"""General Agent Implementation"""

import os
import logging
from typing import Any, Dict, List, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END

from mcp_custom.tools.mcp_tools import (
    get_stock_price_tool,
    get_company_financials_tool,
    get_portfolio_tool,
    calculate_returns_tool,
    analyze_risk_tool,
    search_tavily_tool,
)
from a2a.client import A2AClient

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the General Agent"""

    messages: List[Any]
    user_id: str
    next_action: str


class GeneralAgent:
    """General Agent using LangGraph"""

    def __init__(self):
        """Initialize the General Agent"""
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY")
        )

        # No tools for general agent (simple Q&A)
        self.llm_with_tools = self.llm

        # Build the graph
        self.graph = self._build_graph()

        logger.info("General Agent initialized")

    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("advisor", self._advisor_node)

        # Add edges
        workflow.set_entry_point("advisor")
        workflow.add_edge("advisor", END)

        return workflow.compile()

    async def _advisor_node(self, state: AgentState) -> AgentState:
        """Main advisor reasoning node"""
        messages = state["messages"]

        # Add system message if not present
        if not any(isinstance(m, SystemMessage) for m in messages):
            system_msg = SystemMessage(
                content="""You are a helpful financial assistant.

Answer the user's general question in a friendly and informative way.
If the question is about investments, portfolios, or financial advice,
suggest that they rephrase their question more specifically."""
            )
            messages = [system_msg] + messages

        # Get LLM response asynchronously
        response = await self.llm_with_tools.ainvoke(messages)

        # Update state
        return {**state, "messages": messages + [response]}

    async def process_request(self, user_message: str, user_id: str = "default") -> Dict[str, Any]:
        """
        Process user request

        Args:
            user_message: User's question or request
            user_id: User identifier

        Returns:
            Agent response with recommendations
        """
        logger.info(f"Processing request for user {user_id}: {user_message}")

        # Initialize state
        initial_state: AgentState = {
            "messages": [HumanMessage(content=user_message)],
            "user_id": user_id,
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
            "user_id": user_id,
            "analysis_result": final_state.get("analysis_result"),
            "message_count": len(final_state["messages"]),
        }

    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities"""
        return [
            "general_conversation",
            "question_answering",
            "information_retrieval",
            "task_guidance",
        ]


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        agent = GeneralAgent()

        # Test request
        result = await agent.process_request(
            "Analyze my portfolio and suggest some improvements", user_id="test_user"
        )

        print("Agent Response:")
        print(result["response"])

    asyncio.run(main())
