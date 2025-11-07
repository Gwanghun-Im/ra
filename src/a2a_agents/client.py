"""A2A Client Implementation"""

import httpx
import os
import re
import json
import uuid
from typing import Dict, Any, List, Protocol, runtime_checkable
from collections.abc import Callable
import logging
import yaml
from pathlib import Path
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import (
    AgentCard,
    SendMessageRequest,
    MessageSendParams,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)

logger = logging.getLogger(__name__)

TaskCallbackArg = Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
TaskUpdateCallback = Callable[[TaskCallbackArg, AgentCard], Task]


@runtime_checkable
class GraphContext(Protocol):
    """Protocol for LangGraph node execution context.

    This protocol defines the interface for the context object passed to
    send_message when used within a LangGraph node function. The context
    provides access to the graph's state dictionary.

    Attributes:
        state: Dictionary containing the current graph state, which may include:
            - task_id (str, optional): Unique identifier for the task
            - context_id (str, optional): Context identifier for message grouping
            - active_agent (str, optional): Name of the currently active agent
            - input_message_metadata (dict, optional): Message metadata with optional message_id
    """

    @property
    def state(self) -> Dict[str, Any]:
        """Access the graph's state dictionary."""
        ...


class RemoteAgentConnections:
    """A class to hold the connections to the remote agents."""

    def __init__(self, agent_card: AgentCard, agent_url: str):
        print(f"agent_card: {agent_card}")
        print(f"agent_url: {agent_url}")
        self._httpx_client = httpx.AsyncClient(timeout=30)
        self.agent_client = A2AClient(self._httpx_client, agent_card, url=agent_url + "/a2a")
        self.card = agent_card

    def get_agent(self) -> AgentCard:
        return self.card

    async def send_message(self, message_request: SendMessageRequest) -> SendMessageResponse:
        return await self.agent_client.send_message(message_request)


class A2AClientManager:
    """Client for Agent-to-Agent (A2A) communication"""

    def __init__(
        self,
        config_path: str = "config/a2a_config.yaml",
        task_callback: TaskUpdateCallback | None = None,
    ):
        """
        Initialize A2A Client

        Args:
            config_path: Path to A2A configuration file
        """
        self.config = self._load_config(config_path)
        # Unified agent registry combining config and discovered agent cards
        self.task_callback = task_callback
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: dict[str, dict[str, Any]] = {}

        # Get timeout from config or use default
        timeout = self.config.get("a2a", {}).get("communication", {}).get("timeout", 60.0)
        self.http_client = httpx.AsyncClient(timeout=timeout)

        # Check if running in Docker environment
        self.is_docker = os.getenv("IS_DOCKER", "false").lower() == "true"

        logger.info(f"A2A Client initialized (Docker: {self.is_docker}, timeout: {timeout}s)")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load A2A configuration"""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r") as f:
                return yaml.safe_load(f)
        return {"a2a": {"agents": {}}}

    async def _async_init_components(self, remote_agent_names: list[str]) -> None:
        """Asynchronous part of initialization."""
        # Use a single httpx.AsyncClient for all card resolutions for efficiency
        # logger.info(f"self.config: {self.config}")
        async with httpx.AsyncClient(timeout=30) as client:
            for agent_name in remote_agent_names:
                address = (
                    self.config["a2a"]["agents"].get(agent_name, {}).get("docker_url", "")
                    if self.is_docker
                    else self.config["a2a"]["agents"].get(agent_name, {}).get("local_url", "")
                )
                card_resolver = A2ACardResolver(client, address)  # Constructor is sync
                try:
                    card = await card_resolver.get_agent_card()  # get_agent_card is async

                    remote_connection = RemoteAgentConnections(agent_card=card, agent_url=address)
                    self.remote_agent_connections[card.name] = remote_connection
                    self.cards[card.name] = card
                except httpx.ConnectError as e:
                    print(f"ERROR: Failed to get agent card from {address}: {e}")
                except Exception as e:  # Catch other potential errors
                    print(f"ERROR: Failed to initialize connection for {address}: {e}")

        # Populate self.agents dictionary with discovered agents
        for agent_detail_dict in self.list_remote_agents():
            agent_name = agent_detail_dict["name"]
            self.agents[agent_name] = {**agent_detail_dict, "discovered": True}

    async def send_task(
        self,
        agent_name: str,
        message: str,
        task_id: str | None = None,
        context_id: str | None = None,
        metadata: dict | None = None,
    ):
        """Send a task to a remote agent (simplified interface for direct use).

        Args:
            agent_name: The name of the agent to send the task to.
            message: The message/task to send.
            task_id: Optional task ID. If not provided, one will be generated.
            context_id: Optional context ID (threadId). If not provided, one will be generated.
            metadata: Optional metadata dictionary.

        Returns:
            Task object from the agent response.
        """
        if agent_name not in self.remote_agent_connections:
            raise ValueError(f"Agent {agent_name} not found")

        client = self.remote_agent_connections[agent_name]
        if not client:
            raise ValueError(f"Client not available for {agent_name}")

        message_id = task_id or str(uuid.uuid4())
        thread_id = context_id or str(uuid.uuid4())

        # A2A Protocol v0.3.0 compliant structure
        # messageId, thread are separate from message object
        payload = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": message}],
                "messageId": message_id,
            },
            "thread": {"threadId": thread_id},
        }

        # Add metadata if provided
        if metadata:
            payload["metadata"] = metadata

        message_request = SendMessageRequest(
            id=message_id, params=MessageSendParams.model_validate(payload)
        )
        send_response: SendMessageResponse = await client.send_message(
            message_request=message_request
        )

        logger.info(f"send_response: {send_response.model_dump_json(exclude_none=True, indent=2)}")

        if not isinstance(send_response.root, SendMessageSuccessResponse):
            logger.error("received non-success response")
            return None

        if not isinstance(send_response.root.result, Task):
            logger.error("received non-task response")
            return None

        return send_response.root.result

    async def send_message(self, agent_name: str, task: str, context: GraphContext):
        """Sends a task to remote agent (LangGraph context version).

        This method is designed to be used within LangGraph node functions where
        a context object with state management is available.

        Args:
            agent_name: The name of the agent to send the task to.
            task: The comprehensive conversation context summary
                and goal to be achieved regarding user inquiry and purchase request.
            context: The LangGraph execution context containing the state dictionary.

        Returns:
            Task object from the agent response, or None if the request fails.
        """
        if agent_name not in self.remote_agent_connections:
            raise ValueError(f"Agent {agent_name} not found")
        state = context.state
        state["active_agent"] = agent_name
        client = self.remote_agent_connections[agent_name]

        if not client:
            raise ValueError(f"Client not available for {agent_name}")
        # Extract task/context IDs from state
        task_id = state.get("task_id", str(uuid.uuid4()))
        context_id = state.get("context_id", str(uuid.uuid4()))

        # Extract metadata
        metadata = {}
        if "input_message_metadata" in state:
            metadata.update(**state["input_message_metadata"])

        message_id = metadata.get("message_id", task_id)

        # A2A Protocol v0.3.0 compliant structure
        # messageId and thread are separate from message object
        payload = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": task}],
            },
            "messageId": message_id,
            "thread": {"threadId": context_id},
        }

        # Add metadata if available
        if metadata:
            payload["metadata"] = metadata

        message_request = SendMessageRequest(
            id=message_id, params=MessageSendParams.model_validate(payload)
        )
        send_response: SendMessageResponse = await client.send_message(
            message_request=message_request
        )
        print(
            "send_response",
            send_response.model_dump_json(exclude_none=True, indent=2),
        )

        if not isinstance(send_response.root, SendMessageSuccessResponse):
            print("received non-success response. Aborting get task ")
            return None

        if not isinstance(send_response.root.result, Task):
            print("received non-task response. Aborting get task ")
            return None

        return send_response.root.result

    def list_remote_agents(self):
        """List the available remote agents you can use to delegate the task."""
        if not self.cards:
            return []

        remote_agent_info = []
        for card in self.cards.values():
            print(f"Found agent card: {card.model_dump(exclude_none=True)}")
            print("=" * 100)
            remote_agent_info.append({"name": card.name, "description": card.description})
        return remote_agent_info

    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    @classmethod
    async def create(
        cls,
        remote_agent_names: list[str],
        task_callback: TaskUpdateCallback | None = None,
        config_path: str = "config/a2a_config.yaml",
    ) -> "A2AClientManager":
        """Create and asynchronously initialize an instance of the A2AClientManager."""
        instance = cls(config_path=config_path, task_callback=task_callback)
        await instance._async_init_components(remote_agent_names)
        return instance

    def get_agent_capabilities(self, agent_name: str) -> List[str]:
        """
        Get capabilities of an agent

        Args:
            agent_name: Name of the agent

        Returns:
            List of agent capabilities
        """
        if agent_name in self.agents:
            return self.agents[agent_name].get("capabilities", [])
        return []

    def get_all_agents(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all agents in the registry

        Returns:
            Dictionary of all agents with their information
        """
        return self.agents.copy()

    def get_discovered_agents(self) -> Dict[str, Dict[str, Any]]:
        """
        Get only the agents that have been successfully discovered

        Returns:
            Dictionary of discovered agents
        """
        return {name: info for name, info in self.agents.items() if info.get("discovered", False)}


# Example usage
if __name__ == "__main__":
    None
