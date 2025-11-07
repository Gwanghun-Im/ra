"""A2A Server Implementation for Robo Advisor Agent"""

import os
import uvicorn
import logging
from datetime import datetime
from collections.abc import Iterable
from a2a.server.apps import A2AStarletteApplication
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Message,
    TextPart,
)
from agents.ra_agent import RoboAdvisorAgent

logger = logging.getLogger(__name__)

# Get service URL from environment or use default
SERVICE_URL = os.getenv("A2A_SERVICE_URL", "http://localhost:8100")

SUPPORTED_CONTENT_MIME_TYPES = ["text/plain", "text/markdown", "application/json"]

# Global robo advisor instance
robo_advisor: RoboAdvisorAgent | None = None


class RoboAdvisorAgentExecutor(AgentExecutor):
    """Agent Executor implementation for Robo Advisor Agent"""

    def __init__(self, agent: RoboAdvisorAgent):
        self.agent = agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the robo advisor agent's logic"""
        try:
            # Extract message content
            message = context.message
            content = ""
            for part in message.parts:
                if hasattr(part, "text"):
                    content += part.text

            # Extract user_id from context metadata if available
            user_id = "default"
            if context.metadata and isinstance(context.metadata, dict):
                user_id = context.metadata.get("user_id", "default")

            logger.info(f"Robo Advisor processing task {context.task_id}: {content}")

            # Process with Robo Advisor
            result = await self.agent.process_request(content, user_id)

            # Create response message
            response_message = Message(
                role="agent", parts=[TextPart(type="text", text=result["response"])]
            )

            # Enqueue the final message
            await event_queue.enqueue_message(response_message)

        except Exception as e:
            logger.error(f"Error processing task {context.task_id}: {e}")

            # Create error message
            error_message = Message(
                role="agent", parts=[TextPart(type="text", text=f"Error: {str(e)}")]
            )

            # Enqueue error message
            await event_queue.enqueue_message(error_message)
            raise

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the agent's execution"""
        logger.info(f"Cancelling task {context.task_id}")
        # For now, we don't have a specific cancellation logic
        # The task will be marked as cancelled by the framework


def create_agent_card() -> AgentCard:
    """Create Agent Card for A2A discovery"""

    skills: Iterable[AgentSkill] = [
        AgentSkill(
            id="portfolio_analysis",
            name="portfolio_analysis",
            description="Analyze investment portfolios and provide insights",
            tags=["portfolio", "analysis"],
        ),
        AgentSkill(
            id="investment_recommendation",
            name="investment_recommendation",
            description="Provide personalized investment recommendations",
            tags=["investment", "recommendation", "personalized"],
        ),
        AgentSkill(
            id="risk_assessment",
            name="risk_assessment",
            description="Assess investment risk based on user profile",
            tags=["risk", "assessment", "investment"],
        ),
        AgentSkill(
            id="market_research",
            name="market_research",
            description="Conduct market research and trend analysis",
            tags=["market", "research", "trends"],
        ),
    ]

    capabilities = AgentCapabilities(
        streaming=False,
        push_notifications=False,
    )

    return AgentCard(
        protocol_version="0.3.0",
        name="robo_advisor",
        description="AI-powered investment advisory and portfolio analysis agent using LangGraph and MCP servers",
        url=SERVICE_URL,
        version="1.0.0",
        preferred_transport="JSONRPC",
        default_input_modes=SUPPORTED_CONTENT_MIME_TYPES,
        default_output_modes=SUPPORTED_CONTENT_MIME_TYPES,
        capabilities=capabilities,
        skills=list(skills),
    )


async def initialize_robo_advisor() -> RoboAdvisorAgent:
    """Initialize the Robo Advisor Agent"""
    logger.info("Initializing Robo Advisor Agent...")
    agent = await RoboAdvisorAgent.create()
    logger.info("Robo Advisor Agent initialized successfully")
    return agent


def create_app():
    """Create and configure the A2A Starlette application"""
    import asyncio

    # Initialize robo advisor synchronously in the main thread
    global robo_advisor
    if robo_advisor is None:
        robo_advisor = asyncio.run(initialize_robo_advisor())

    # Create agent executor
    agent_executor = RoboAdvisorAgentExecutor(robo_advisor)

    # Create task store
    task_store = InMemoryTaskStore()

    # Create request handler
    http_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=task_store,
    )

    # Create agent card
    agent_card = create_agent_card()

    # Create A2A Starlette application
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=http_handler,
    )

    # Build and return the Starlette app
    return a2a_app.build()


def run_server(host: str = "0.0.0.0", port: int = 8100):
    """Run the A2A server"""
    logger.info(f"Starting Robo Advisor A2A Server on {host}:{port}")
    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
