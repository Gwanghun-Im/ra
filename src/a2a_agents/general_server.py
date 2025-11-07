"""A2A Server Implementation for General Agent"""

import os
import uvicorn
import logging
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
from agents.general_agent import GeneralAgent

logger = logging.getLogger(__name__)

# Get service URL from environment or use default
SERVICE_URL = os.getenv("A2A_SERVICE_URL", "http://localhost:8102")

SUPPORTED_CONTENT_MIME_TYPES = ["text/plain", "text/markdown", "application/json"]

# Global general agent instance
general_agent: GeneralAgent | None = None


class GeneralAgentExecutor(AgentExecutor):
    """Agent Executor implementation for General Agent"""

    def __init__(self, agent: GeneralAgent):
        self.agent = agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the general agent's logic"""
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

            logger.info(f"General Agent processing task {context.task_id}: {content}")

            # Process with General Agent
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
            id="general_conversation",
            name="general_conversation",
            description="Engage in general conversations and answer questions",
            tags=["conversation", "chat", "general"],
            examples=[
                "Hello, how can you assist me today?",
            ],
        )
    ]

    capabilities = AgentCapabilities(
        streaming=False,
        push_notifications=False,
    )

    return AgentCard(
        protocol_version="0.3.0",
        name="general_agent",
        version="1.0.0",
        description="General purpose conversational agent for question answering and task guidance",
        url=SERVICE_URL,
        preferred_transport="JSONRPC",
        default_input_modes=SUPPORTED_CONTENT_MIME_TYPES,
        default_output_modes=SUPPORTED_CONTENT_MIME_TYPES,
        capabilities=capabilities,
        skills=list(skills),
    )


def initialize_general_agent() -> GeneralAgent:
    """Initialize the General Agent"""
    logger.info("Initializing General Agent...")
    agent = GeneralAgent()
    logger.info("General Agent initialized successfully")
    return agent


def create_app():
    """Create and configure the A2A Starlette application"""
    # Initialize general agent
    global general_agent
    if general_agent is None:
        general_agent = initialize_general_agent()

    # Create agent executor
    agent_executor = GeneralAgentExecutor(general_agent)

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


def run_server(host: str = "0.0.0.0", port: int = 8102):
    """Run the A2A server"""
    logger.info(f"Starting General Agent A2A Server on {host}:{port}")
    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
