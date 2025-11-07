"""A2A Server Implementation for RAG Agent"""

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
from agents.rag_agent import RAGAgent

logger = logging.getLogger(__name__)

# Get service URL from environment or use default
SERVICE_URL = os.getenv("A2A_SERVICE_URL", "http://localhost:8101")

SUPPORTED_CONTENT_MIME_TYPES = ["text/plain", "text/markdown", "application/json"]

# Global RAG agent instance
rag_agent: RAGAgent | None = None


class RAGAgentExecutor(AgentExecutor):
    """Agent Executor implementation for RAG Agent"""

    def __init__(self, agent: RAGAgent):
        self.agent = agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the RAG agent's logic"""
        try:
            # Extract message content
            message = context.message
            content = ""
            for part in message.parts:
                if hasattr(part, "text"):
                    content += part.text

            logger.info(f"RAG Agent processing task {context.task_id}: {content}")

            # Process with RAG Agent
            result = await self.agent.process_request(content)

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
            id="document_retrieval",
            name="document_retrieval",
            description="Retrieve documents from knowledge bases using semantic search",
            tags=["rag", "retrieval", "knowledge-base"],
        )
    ]

    capabilities = AgentCapabilities(
        streaming=False,
        push_notifications=False,
    )

    return AgentCard(
        protocol_version="0.3.0",
        name="rag_agent",
        description="Specialized agent for knowledge base search and retrieval-augmented generation",
        url=SERVICE_URL,
        version="1.0.0",
        preferred_transport="JSONRPC",
        default_input_modes=SUPPORTED_CONTENT_MIME_TYPES,
        default_output_modes=SUPPORTED_CONTENT_MIME_TYPES,
        capabilities=capabilities,
        skills=list(skills),
    )


def initialize_rag_agent() -> RAGAgent:
    """Initialize the RAG Agent"""
    logger.info("Initializing RAG Agent...")
    agent = RAGAgent()
    logger.info("RAG Agent initialized successfully")
    return agent


def create_app():
    """Create and configure the A2A Starlette application"""
    # Initialize RAG agent
    global rag_agent
    if rag_agent is None:
        rag_agent = initialize_rag_agent()

    # Create agent executor
    agent_executor = RAGAgentExecutor(rag_agent)

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


def run_server(host: str = "0.0.0.0", port: int = 8101):
    """Run the A2A server"""
    logger.info(f"Starting RAG Agent A2A Server on {host}:{port}")
    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
