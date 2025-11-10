"""Supervisor Agent - Main Orchestrator"""

import os
from typing import Any, Dict, List, TypedDict, Literal, Optional
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from a2a.types import Message, Task, TaskState

from src.a2a_agents.client import A2AClientManager
from src.a2a_agents.message_utils import extract_text_from_message
from src.task_management import RedisTaskManager, TaskType, TaskPriority

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class SupervisorState(TypedDict):
    """State for Supervisor Agent"""

    messages: List[Any]
    user_id: str
    context_id: str | None
    task_type: str
    delegated_to: str | None
    response: str | None
    next_step: str


class SupervisorAgent:
    """Supervisor Agent that orchestrates tasks between agents"""

    def __init__(self):
        """Initialize Supervisor Agent (synchronous part)"""
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0.3, api_key=os.getenv("OPENAI_API_KEY")
        )

        # A2A client will be initialized asynchronously
        self.a2a_client: Optional[A2AClientManager] = None

        # Redis Task Manager will be initialized asynchronously
        self.task_manager: Optional[RedisTaskManager] = None

        # Agent discovery will happen on first use (lazy initialization)
        self._agents_discovered = False

        # Build workflow graph
        self.graph = self._build_graph()

        logger.info("Supervisor Agent initialized (A2A client pending async init)")

    @classmethod
    async def create(cls):
        """Factory method to create and fully initialize SupervisorAgent"""
        instance = cls()
        # Initialize A2A client asynchronously
        instance.a2a_client = await A2AClientManager.create(
            remote_agent_names=["robo_advisor", "general_agent"]
        )

        # Initialize Redis Task Manager
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        instance.task_manager = await RedisTaskManager.create(redis_url=redis_url)

        logger.info("Supervisor Agent fully initialized with A2A client and Task Manager")
        return instance

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
        context_id = state["context_id"]

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
                # Send task to robo advisor via A2A protocol with streaming
                result = None
                async for response in self.a2a_client.send_message(
                    agent_name="robo_advisor",
                    message=user_message,
                    context_id=context_id,
                    streaming=True,
                ):
                    result = response  # Keep updating with latest response

                delegated_to = "robo_advisor (A2A)"

            except Exception as e:
                logger.error(f"Error communicating with Robo Advisor via A2A: {e}")
                response = f"죄송합니다. Robo Advisor와의 통신 중 오류가 발생했습니다: {str(e)}"
                delegated_to = "supervisor (error)"
                result = response

        else:
            try:
                # Send task to general agent via A2A protocol with streaming
                result = None
                async for response in self.a2a_client.send_message(
                    agent_name="general_agent",
                    message=user_message,
                    context_id=context_id,
                    streaming=True,
                ):
                    result = response  # Keep updating with latest response

                logger.info(f"result: {result}")
                delegated_to = "general_agent (A2A)"

            except Exception as e:
                logger.error(f"Error communicating with General Agent via A2A: {e}")
                response = f"죄송합니다. general Agent와의 통신 중 오류가 발생했습니다: {str(e)}"
                delegated_to = "supervisor (error)"
                result = response

        return {
            **state,
            "response": result,
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

    async def process_request(
        self, user_message: str, user_id: str = "default", context_id: str | None = None
    ) -> Dict[str, Any]:
        """
        Process user request through supervisor workflow

        Args:
            user_message: User's message
            user_id: User identifier
            context_id: Optional context ID for conversation tracking

        Returns:
            Response dictionary
        """
        logger.info(f"Supervisor processing request from user {user_id}")

        # Initialize state
        initial_state: SupervisorState = {
            "messages": [HumanMessage(content=user_message)],
            "user_id": user_id,
            "context_id": context_id,
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

    async def process_request_stream(
        self,
        message: Message,
        user_id: str = "default",
        context_id: str | None = None,
        task_id: str | None = None,
    ):
        """
        Process user request through supervisor workflow with streaming support

        Args:
            user_message: User's message
            user_id: User identifier
            context_id: Optional context ID for conversation tracking

        Yields:
            Streaming responses from delegated agents
        """
        logger.info(f"Supervisor processing request from user {user_id} with streaming")

        # Classify task
        messages = [HumanMessage(content=extract_text_from_message(message))]
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
        task_type_str = classification.content.strip().lower()
        logger.info(f"Task classified as: {task_type_str}")

        # Convert string to TaskType enum
        task_type_map = {
            "portfolio_analysis": TaskType.PORTFOLIO_ANALYSIS,
            "investment_advice": TaskType.INVESTMENT_ADVICE,
            "risk_assessment": TaskType.RISK_ASSESSMENT,
            "market_research": TaskType.MARKET_RESEARCH,
            "general": TaskType.GENERAL,
        }
        task_type = task_type_map.get(task_type_str, TaskType.GENERAL)

        # Route task and stream responses
        investment_tasks = [
            "portfolio_analysis",
            "investment_advice",
            "risk_assessment",
            "market_research",
        ]

        if task_type_str in investment_tasks:
            logger.info(f"Routing to Robo Advisor Agent via A2A with streaming")
            agent_name = "robo_advisor"
        else:
            logger.info(f"Routing to General Agent via A2A with streaming")
            agent_name = "general_agent"

        # Generate unique task ID
        import uuid

        # task_id = f"task_{uuid.uuid4().hex[:12]}"

        # Create task in Redis (if task_manager is available)
        if self.task_manager:
            try:
                # Create A2A Message object
                from a2a.types import TextPart, Role

                # Create task in Redis
                await self.task_manager.create_task(
                    task_id=task_id,
                    message=message,
                    user_id=user_id,
                    context_id=context_id,
                    task_type=task_type,
                    priority=TaskPriority.NORMAL,
                    assigned_agent=agent_name,
                )
                logger.info(f"Task {task_id} created and tracked in Redis")

            except Exception as e:
                logger.warning(f"Could not track task in Redis: {e}")
                # Continue even if Redis task creation fails

        try:
            # Stream responses from delegated agent
            async for response in self.a2a_client.send_message(
                agent_name=agent_name,
                message=extract_text_from_message(message),
                context_id=context_id,
                streaming=True,
            ):
                yield response

            # Update task to COMPLETED state (if task_manager available)
            if self.task_manager:
                try:
                    await self.task_manager.update_task(
                        task_id=task_id,
                        state=TaskState.completed,
                        message="Task completed successfully",
                    )
                    logger.info(f"Task {task_id} marked as completed in Redis")
                except Exception as e:
                    logger.warning(f"Could not update task status in Redis: {e}")

        except Exception as e:
            logger.error(f"Error communicating with {agent_name} via A2A: {e}")

            # Update task to FAILED state (if task_manager available)
            if self.task_manager:
                try:
                    await self.task_manager.update_task(
                        task_id=task_id,
                        state=TaskState.failed,
                        message=f"Task failed: {str(e)}",
                    )
                    logger.info(f"Task {task_id} marked as failed in Redis")
                except Exception as update_error:
                    logger.warning(f"Could not update task status in Redis: {update_error}")

            raise

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
