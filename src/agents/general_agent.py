"""General Agent Implementation"""

import os
import logging
from typing import Any, Dict, List, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the General Agent"""

    messages: List[Any]
    user_id: str
    context_id: str | None
    next_action: str


class GeneralAgent:
    """General Agent using LangGraph"""

    def __init__(self):
        """Initialize the General Agent (synchronous part)"""
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY")
        )

        # No tools for general agent (simple Q&A)
        self.llm_with_tools = self.llm

        # Build the graph
        self.graph = self._build_graph()

        logger.info("General Agent initialized")

    @classmethod
    async def create(cls):
        """Factory method to create and fully initialize GeneralAgent"""
        instance = cls()
        logger.info("General Agent fully initialized")
        return instance

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

    async def process_request(
        self, user_message: str, user_id: str = "default", context_id: str | None = None
    ) -> Dict[str, Any]:
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
            "context_id": context_id,
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
            "message": final_state.get("message"),
            "message_count": len(final_state["messages"]),
        }
