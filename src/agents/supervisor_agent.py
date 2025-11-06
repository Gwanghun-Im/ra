"""Supervisor Agent - Main Orchestrator"""

import os
from typing import Any, Dict, List, TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
import logging

from a2a.client import A2AClient

# from agents.robo_advisor_agent import RoboAdvisorAgent

logger = logging.getLogger(__name__)


class SupervisorState(TypedDict):
    """State for Supervisor Agent"""

    messages: List[Any]
    user_id: str
    task_type: str
    delegated_to: str | None
    response: str | None
    next_step: str


class SupervisorAgent:
    """Supervisor Agent that orchestrates tasks between agents"""

    def __init__(self):
        """Initialize Supervisor Agent"""
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0.3, api_key=os.getenv("OPENAI_API_KEY")
        )

        # Initialize A2A client for agent communication
        self.a2a_client = A2AClient()

        # Agent discovery will happen on first use (lazy initialization)
        self._agents_discovered = False

        # Use A2A for robo advisor communication instead of direct instantiation
        # self.robo_advisor = RoboAdvisorAgent()

        # Build workflow graph
        self.graph = self._build_graph()

        logger.info("Supervisor Agent initialized")

    def _build_graph(self) -> StateGraph:
        """Build supervisor workflow"""
        workflow = StateGraph(SupervisorState)

        # Add nodes
        workflow.add_node("classify", self._classify_task)
        workflow.add_node("route", self._route_task)
        workflow.add_node("finalize", self._finalize_response)

        # Add edges
        workflow.set_entry_point("classify")
        workflow.add_edge("classify", "route")
        workflow.add_conditional_edges(
            "route", self._determine_next, {"finalize": "finalize", "retry": "classify"}
        )
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def _classify_task(self, state: SupervisorState) -> SupervisorState:
        """Classify the user's request"""
        messages = state["messages"]

        classification_prompt = SystemMessage(
            content="""You are a task classifier. 
        
Analyze the user's message and classify it into one of these categories:
- portfolio_analysis: Questions about portfolio performance, holdings, returns
- investment_advice: Requests for investment recommendations or advice
- risk_assessment: Questions about portfolio risk and diversification
- market_research: General market information or stock research
- general: Other general questions

Respond with ONLY the category name, nothing else."""
        )

        classification = self.llm.invoke([classification_prompt] + messages)
        task_type = classification.content.strip().lower()

        logger.info(f"Task classified as: {task_type}")

        return {**state, "task_type": task_type}

    async def _route_task(self, state: SupervisorState) -> SupervisorState:
        """Route task to appropriate agent via A2A protocol"""
        task_type = state["task_type"]
        user_message = state["messages"][-1].content
        user_id = state["user_id"]

        # Ensure agents are discovered
        if not self._agents_discovered:
            logger.info("Discovering agents...")
            await self.a2a_client.discover_agents()
            self._agents_discovered = True
            logger.info(f"Discovered agents: {list(self.a2a_client.agents.keys())}")

        # Investment-related tasks go to Robo Advisor via A2A
        investment_tasks = [
            "portfolio_analysis",
            "investment_advice",
            "risk_assessment",
            "market_research",
        ]

        if task_type in investment_tasks:
            logger.info(f"Routing to Robo Advisor Agent via A2A")

            try:
                # Send task to robo advisor via A2A protocol
                result = await self.a2a_client.send_task(
                    agent_name="robo_advisor",
                    message=user_message,
                    task_id=f"task_{user_id}_{task_type}",
                    context={"user_id": user_id, "task_type": task_type},
                )

                # Extract response from A2A message format
                if "message" in result:
                    response = result["message"].get("content", "")
                else:
                    response = result.get("response", str(result))

                delegated_to = "robo_advisor (A2A)"

            except Exception as e:
                logger.error(f"Error communicating with Robo Advisor via A2A: {e}")
                response = f"죄송합니다. Robo Advisor와의 통신 중 오류가 발생했습니다: {str(e)}"
                delegated_to = "supervisor (error)"

        else:
            try:
                # Send task to robo advisor via A2A protocol
                result = await self.a2a_client.send_task(
                    agent_name="general_agent",
                    message=user_message,
                    task_id=f"task_{user_id}_{task_type}",
                    context={"user_id": user_id, "task_type": task_type},
                )

                # Extract response from A2A message format
                if "message" in result:
                    response = result["message"].get("content", "")
                else:
                    response = result.get("response", str(result))

                delegated_to = "general_agent (A2A)"

            except Exception as e:
                logger.error(f"Error communicating with General Agent via A2A: {e}")
                response = f"죄송합니다. general Agent와의 통신 중 오류가 발생했습니다: {str(e)}"
                delegated_to = "supervisor (error)"

        return {
            **state,
            "response": response,
            "delegated_to": delegated_to,
            "next_step": "finalize",
        }

    def _determine_next(self, state: SupervisorState) -> Literal["finalize", "retry"]:
        """Determine next step after routing"""
        if state.get("response"):
            return "finalize"
        return "retry"

    def _finalize_response(self, state: SupervisorState) -> SupervisorState:
        """Finalize and format the response"""
        response = state["response"]
        delegated_to = state["delegated_to"]

        # Add metadata to response
        final_response = f"{response}\n\n---\n*Processed by: {delegated_to}*"

        return {**state, "response": final_response}

    async def process_request(self, user_message: str, user_id: str = "default") -> Dict[str, Any]:
        """
        Process user request through supervisor workflow

        Args:
            user_message: User's message
            user_id: User identifier

        Returns:
            Response dictionary
        """
        logger.info(f"Supervisor processing request from user {user_id}")

        # Initialize state
        initial_state: SupervisorState = {
            "messages": [HumanMessage(content=user_message)],
            "user_id": user_id,
            "task_type": "",
            "delegated_to": None,
            "response": None,
            "next_step": "classify",
        }

        # Run workflow
        final_state = await self.graph.ainvoke(initial_state)

        return {
            "response": final_state["response"],
            "task_type": final_state["task_type"],
            "delegated_to": final_state["delegated_to"],
            "user_id": user_id,
        }

    async def get_available_agents(self) -> List[Dict[str, Any]]:
        """Get list of available agents via A2A discovery"""
        agents = [
            {
                "name": "supervisor",
                "type": "orchestrator",
                "description": "Main supervisor agent that routes tasks",
            },
        ]

        # Discover A2A agents
        try:
            discovered_agents = await self.a2a_client.discover_agents()
            for agent_card in discovered_agents:
                agents.append(
                    {
                        "name": agent_card.get("name", "unknown"),
                        "type": "a2a_agent",
                        "description": agent_card.get("description", ""),
                        "capabilities": agent_card.get("capabilities", []),
                        "service_url": agent_card.get("service_url", ""),
                    }
                )
        except Exception as e:
            logger.warning(f"Could not discover A2A agents: {e}")

        return agents


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        supervisor = SupervisorAgent()

        # Test requests
        test_queries = [
            "애플 주식의 현재 가격이 얼마인가요?",
            "내 포트폴리오를 분석하고 개선 방안을 제안해주세요",
            "현재 내 포트폴리오는 얼마나 위험한가요?",
            "복리란 무엇인가요?",
        ]

        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}")

            result = await supervisor.process_request(query, user_id="test")

            print(f"\nTask Type: {result['task_type']}")
            print(f"Delegated To: {result['delegated_to']}")
            print(f"\nResponse:\n{result['response']}")

    asyncio.run(main())
