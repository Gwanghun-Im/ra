"""Robo Advisor Agent Implementation"""

import os
from typing import Any, Dict, List, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
import logging

from src.mcp_custom.tools.mcp_tools import (
    get_stock_price_tool,
    get_market_news_tool,
    get_company_financials_tool,
    get_portfolio_tool,
    calculate_returns_tool,
    analyze_risk_tool,
    search_tavily_tool,
    search_knowledge_base,
    rag_query_tool,
    get_rag_stats_tool,
)

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the Robo Advisor Agent"""

    messages: List[Any]
    user_id: str
    portfolio_data: Dict[str, Any] | None
    market_data: Dict[str, Any] | None
    analysis_result: Dict[str, Any] | None
    next_action: str


class RoboAdvisorAgent:
    """Robo Advisor Agent using LangGraph"""

    def __init__(self):
        """Initialize the Robo Advisor Agent"""
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0.7, api_key=os.getenv("OPENAI_API_KEY")
        )

        # Define available tools
        self.tools = [
            get_stock_price_tool,
            # get_market_news_tool,
            get_company_financials_tool,
            get_portfolio_tool,
            calculate_returns_tool,
            analyze_risk_tool,
            search_tavily_tool,
            search_knowledge_base,
            rag_query_tool,
            get_rag_stats_tool,
        ]

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Build the graph
        self.graph = self._build_graph()

        logger.info("Robo Advisor Agent initialized")

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
- get_stock_price: Get current stock prices
- get_company_financials: Get company financial metrics
- get_portfolio: Retrieve user portfolio
- calculate_returns: Calculate portfolio performance
- analyze_risk: Analyze portfolio risk metrics
- search_tavily: Get market news and sentiment
- search_knowledge_base: Search RAG knowledge base for relevant documents
- rag_query_tool: Retrieve relevant context from knowledge base for answering queries
- get_rag_stats_tool: Get statistics about the knowledge base
- [deprecated]get_market_news: Get market news and sentiment
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

    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities"""
        return [
            "portfolio_analysis",
            "investment_recommendation",
            "risk_assessment",
            "market_research",
            "performance_calculation",
        ]


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        agent = RoboAdvisorAgent()

        # Test request
        result = await agent.process_request(
            "Analyze my portfolio and suggest some improvements", user_id="test_user"
        )

        print("Agent Response:")
        print(result["response"])

    asyncio.run(main())
