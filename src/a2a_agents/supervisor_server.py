"""A2A Server Implementation for Supervisor Agent"""

import os
import logging
from collections.abc import Iterable
from contextlib import asynccontextmanager
import uvicorn
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
)
from src.agents.supervisor_agent import SupervisorAgent
from src.a2a_agents.message_utils import extract_text_from_message, enqueue_response_as_artifact

logger = logging.getLogger(__name__)

# Get service URL from environment or use default
SERVICE_URL = os.getenv("A2A_SERVICE_URL", "http://localhost:8099")

SUPPORTED_CONTENT_MIME_TYPES = ["text/plain", "text/markdown", "application/json"]

# Global supervisor instance
supervisor: SupervisorAgent | None = None


class SupervisorAgentExecutor(AgentExecutor):
    """Agent Executor implementation for Supervisor Agent"""

    def __init__(self, agent: SupervisorAgent):
        self.agent = agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the supervisor agent's logic"""
        try:
            # Extract message content using utility function
            message: Message = context.message
            content = extract_text_from_message(message)

            # Extract user_id from context metadata if available
            user_id = "default"
            if context.metadata and isinstance(context.metadata, dict):
                user_id = context.metadata.get("user_id", "default")

            logger.info(f"Supervisor processing task {context.task_id}: {content}")

            # Process with Supervisor
            result = await self.agent.process_request(content, user_id, context.context_id)

            # Enqueue response as artifact
            await enqueue_response_as_artifact(
                event_queue=event_queue,
                task_id=context.task_id,
                context_id=message.context_id or context.task_id,
                response_text=result["response"],
                final=True,
            )

        except Exception as e:
            logger.error(f"Error processing task {context.task_id}: {e}")

            # Enqueue error as artifact
            await enqueue_response_as_artifact(
                event_queue=event_queue,
                task_id=context.task_id,
                context_id=context.message.context_id or context.task_id,
                response_text=f"Error: {str(e)}",
                final=True,
            )
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
            id="task_orchestration",
            name="task_orchestration",
            description="Orchestrate tasks among multiple agents",
            tags=[
                "agent_routing",
                "multi_agent_coordination",
                "task_classification",
                "workflow_management",
            ],
        )
    ]

    capabilities = AgentCapabilities(
        streaming=False,
        push_notifications=False,
    )

    return AgentCard(
        protocol_version="0.3.0",
        name="supervisor",
        description="Main orchestrator agent that routes tasks to specialized agents",
        url=SERVICE_URL,
        version="1.0.0",
        preferred_transport="JSONRPC",
        default_input_modes=SUPPORTED_CONTENT_MIME_TYPES,
        default_output_modes=SUPPORTED_CONTENT_MIME_TYPES,
        capabilities=capabilities,
        skills=list(skills),
    )


async def initialize_supervisor() -> SupervisorAgent:
    """Initialize the Supervisor Agent"""
    logger.info("Initializing Supervisor Agent...")
    agent = await SupervisorAgent.create()
    logger.info("Supervisor Agent initialized successfully")
    return agent


@asynccontextmanager
async def lifespan(app):
    """Lifespan context manager for async initialization and cleanup"""
    global supervisor

    # Startup: Initialize the supervisor agent
    logger.info("Starting up Supervisor Agent...")
    supervisor = await initialize_supervisor()

    yield

    # Shutdown: Clean up resources if needed
    logger.info("Shutting down Supervisor Agent...")
    supervisor = None


def create_app():
    """Create and configure the A2A Starlette application"""
    global supervisor

    # Create agent executor with None initially (will be set in lifespan)
    # We need to use a wrapper to access the global supervisor
    class LazyAgentExecutor(AgentExecutor):
        async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
            if supervisor is None:
                raise RuntimeError("Supervisor agent not initialized")
            executor = SupervisorAgentExecutor(supervisor)
            await executor.execute(context, event_queue)

        async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
            if supervisor is None:
                raise RuntimeError("Supervisor agent not initialized")
            executor = SupervisorAgentExecutor(supervisor)
            await executor.cancel(context, event_queue)

    agent_executor = LazyAgentExecutor()

    # Create task store
    task_store = InMemoryTaskStore()

    # Create request handler
    http_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=task_store,
    )

    # Create agent card
    agent_card = create_agent_card()

    # Create A2A Starlette application with lifespan
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=http_handler,
    )

    # Build and return the Starlette app with lifespan
    app = a2a_app.build()
    app.router.lifespan_context = lifespan

    return app


def run_server(host: str = "0.0.0.0", port: int = 8099):
    """Run the A2A server"""
    logger.info(f"Starting Supervisor Agent A2A Server on {host}:{port}")
    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 서버 실행
    run_server()
