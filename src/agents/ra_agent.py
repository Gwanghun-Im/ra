"""Robo Advisor Agent Implementation"""

import os
import logging
from typing import Any, Dict, List, TypedDict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END

from src.mcp_custom.tools.mcp_tools import (
    get_stock_price_tool,
    get_market_indicators_tool,
    get_company_financials_tool,
    get_portfolio_tool,
    calculate_returns_tool,
    analyze_risk_tool,
    search_tavily_tool,
)
from src.a2a_agents.client import A2AClientManager

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the Robo Advisor Agent"""

    messages: List[Any]
    user_id: str
    context_id: str | None
    portfolio_data: Dict[str, Any] | None
    market_data: Dict[str, Any] | None
    analysis_result: Dict[str, Any] | None
    next_action: str


class RoboAdvisorAgent:
    """Robo Advisor Agent using LangGraph"""

    def __init__(self):
        """Initialize the Robo Advisor Agent (synchronous part)"""
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY")
        )

        # A2A client will be initialized asynchronously
        self.a2a_client: Optional[A2AClientManager] = None

        # Define available tools (RAG tools removed, now using A2A)
        self.tools = [
            get_stock_price_tool,
            get_market_indicators_tool,
            get_company_financials_tool,
            get_portfolio_tool,
            calculate_returns_tool,
            analyze_risk_tool,
            search_tavily_tool,
        ]

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Build the graph
        self.graph = self._build_graph()

        logger.info("Robo Advisor Agent initialized (A2A client pending async init)")

    @classmethod
    async def create(cls):
        """Factory method to create and fully initialize RoboAdvisorAgent"""
        instance = cls()
        # Initialize A2A client asynchronously
        instance.a2a_client = await A2AClientManager.create(remote_agent_names=["rag_agent"])
        logger.info("Robo Advisor Agent fully initialized with A2A RAG integration")
        return instance

    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("advisor", self._advisor_node)
        workflow.add_node("tools", self._tools_node)
        workflow.add_node("analyze", self._analyze_node)

        # Add edges
        workflow.set_entry_point("advisor")
        workflow.add_conditional_edges(
            "advisor",
            self._should_continue,
            {"continue": "tools", "analyze": "analyze", "end": END},
        )
        workflow.add_edge("tools", "advisor")
        workflow.add_edge("analyze", END)

        return workflow.compile()

    async def _advisor_node(self, state: AgentState) -> AgentState:
        """Main advisor reasoning node"""
        messages = state["messages"]

        # Add system message if not present
        if not any(isinstance(m, SystemMessage) for m in messages):
            system_msg = SystemMessage(
                content="""You are a professional Robo Advisor AI agent.

Your responsibilities:
- Analyze user portfolios and provide investment recommendations
- Assess market conditions and identify opportunities
- Evaluate risk levels and suggest diversification strategies
- Provide clear, actionable investment advice

Always:
1. Gather necessary data using available tools
2. Perform thorough analysis
3. Provide evidence-based recommendations
4. Consider risk tolerance and investment goals
5. Explain your reasoning clearly

Available tools:
- search_tavily: Get market news and sentiment
- get_market_indicators: Get key market indicators
- get_stock_price: Get current stock prices
- get_company_financials: Get company financial metrics
- get_portfolio: Retrieve user portfolio
- calculate_returns: Calculate portfolio performance
- analyze_risk: Analyze portfolio risk metrics

For knowledge base queries, research questions, or information retrieval:
- Request information from the RAG Agent via A2A protocol
- The RAG Agent specializes in searching and retrieving information from the knowledge base
- When you need to look up investment concepts, research topics, or historical data,
  indicate that you need RAG information and it will be fetched via A2A communication
"""
            )
            messages = [system_msg] + messages

        # Get LLM response asynchronously
        response = await self.llm_with_tools.ainvoke(messages)

        # Update state
        return {**state, "messages": messages + [response]}

    async def _call_rag_agent(self, query: str, context_id: str) -> str:
        """
        Call RAG Agent via A2A protocol

        Args:
            query: RAG query to send

        Returns:
            RAG Agent response
        """
        try:
            logger.info(f"Calling RAG Agent via A2A with query: {query}")
            result = await self.a2a_client.send_message(
                agent_name="rag_agent",
                message=query,
                context_id=context_id,
            )

            # Extract response from Task object
            content = "No response from RAG Agent"
            if hasattr(result, "artifacts") and result.artifacts:
                for artifact in result.artifacts:
                    if hasattr(artifact, "parts"):
                        for part in artifact.parts:
                            if hasattr(part, "text"):
                                content = part.text
                                break

            logger.info("Successfully received response from RAG Agent")
            return content

        except Exception as e:
            logger.error(f"Error calling RAG Agent via A2A: {e}")
            return f"Error retrieving information from knowledge base: {str(e)}"

    async def _tools_node(self, state: AgentState) -> AgentState:
        """Execute tool calls using async HTTP calls to Docker MCP servers"""
        messages = state["messages"]
        last_message = messages[-1]
        context_id = state["context_id"]

        # Check if the message contains RAG-related keywords
        content = last_message.content.lower() if hasattr(last_message, "content") else ""
        rag_keywords = [
            "search knowledge",
            "look up",
            "find information",
            "research",
            "what is",
            "explain",
            "define",
            "tell me about",
        ]

        needs_rag = any(keyword in content for keyword in rag_keywords)

        tool_messages = []

        # If RAG is needed, call RAG Agent via A2A
        if needs_rag:
            rag_response = await self._call_rag_agent(content, context_id=context_id)
            tool_messages.append(HumanMessage(content=f"[RAG Agent Response]: {rag_response}"))

        # Create a mapping of tool names to tool functions
        tools_by_name = {tool.name: tool for tool in self.tools}

        # Execute each tool call
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

        # Check if we have enough data to analyze
        content = last_message.content.lower() if hasattr(last_message, "content") else ""
        if "analysis" in content or "recommendation" in content:
            return "analyze"

        return "end"

    async def _analyze_node(self, state: AgentState) -> AgentState:
        """Final analysis and recommendation node"""
        messages = state["messages"]

        # Create analysis prompt
        analysis_prompt = HumanMessage(
            content="""Based on all the gathered information, provide:

1. **Portfolio Assessment**: Current status and performance
2. **Risk Analysis**: Risk level and diversification quality
3. **Market Opportunities**: Potential investments based on market conditions
4. **Recommendations**: Specific actionable advice with rationale
5. **Next Steps**: What the user should do

Format your response clearly with sections and bullet points."""
        )

        messages.append(analysis_prompt)
        final_response = await self.llm.ainvoke(messages)

        return {
            **state,
            "messages": messages + [final_response],
            "analysis_result": {"content": final_response.content, "timestamp": "now"},
        }

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
            "portfolio_data": None,
            "market_data": None,
            "analysis_result": None,
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
