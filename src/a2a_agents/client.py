"""A2A Client Implementation"""

import os
import uuid
import logging
from pathlib import Path
from collections.abc import Callable
from typing import Any, List, Optional, Dict

import httpx
import yaml
from a2a.client import A2ACardResolver, A2AClientError, ClientConfig, ClientFactory
from a2a.client.auth.credentials import CredentialService
from a2a.types import (
    Role,
    AgentCard,
    Message,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    TextPart,
    TransportProtocol,
)

logger = logging.getLogger(__name__)

TaskCallbackArg = Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
TaskUpdateCallback = Callable[[TaskCallbackArg, AgentCard], Task]


class RemoteAgentConnections:
    """A class to hold the connections to the remote agents."""

    def __init__(
        self,
        agent_card: AgentCard,
        agent_url: str,
        streaming: bool = False,
        credential_service: Optional[CredentialService] = None,
    ):
        print(f"agent_card: {agent_card}")
        print(f"agent_url: {agent_url}")
        self._httpx_client = httpx.AsyncClient(timeout=30)

        # A2A 클라이언트 설정
        config = ClientConfig(
            streaming=streaming,
            polling=not streaming,
            httpx_client=self._httpx_client,
            supported_transports=[
                TransportProtocol.jsonrpc,
                TransportProtocol.http_json,
            ],
            accepted_output_modes=[
                "text/plain",
                "text/markdown",
                "application/json",
                "text/event-stream",
            ],
            use_client_preference=True,
        )

        factory = ClientFactory(config=config)

        # 인터셉터 추가 (인증이 필요한 경우)
        # - 토큰 기반 인증 등 환경에서 자동 헤더 주입을 지원합니다.
        interceptors = []
        if credential_service:
            from a2a.client.auth.interceptor import AuthInterceptor

            interceptors.append(AuthInterceptor(credential_service))
            logger.debug("Auth interceptor added")

        self.agent_client = factory.create(card=agent_card, interceptors=interceptors)
        self.card = agent_card

    def get_agent(self) -> AgentCard:
        return self.card

    async def send_message(self, message_request: Message):
        # send_message returns an async generator, yield responses in real-time
        async for response in self.agent_client.send_message(message_request):
            yield response


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

                    # 도커 외부에서 접속하는 경우 agent card의 URL을 로컬 주소로 override
                    # agent card에는 도커 내부 호스트명이 포함되어 있을 수 있지만,
                    # 호스트 머신에서 접속할 때는 localhost를 사용해야 함
                    if not self.is_docker:
                        card.url = address

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

    async def send_message(
        self,
        agent_name: str,
        message: str,
        metadata: dict[str, Any] | None = None,
        context_id: str | None = None,
        streaming: bool = False,
    ):
        """Send a task to a remote agent.

        Args:
            agent_name: The name of the agent to send the task to.
            message: The message/task to send.
            task_id: Optional task ID. If not provided, one will be generated.
            context_id: Optional context ID (threadId). If not provided, one will be generated.
            metadata: Optional metadata dictionary.
            streaming: If True, yields responses in real-time. If False, returns only the final result.

        Yields (if streaming=True):
            ClientEvent (tuple[Task, UpdateEvent]) or Message objects from the agent response.

        Returns (if streaming=False):
            Final ClientEvent (tuple[Task, UpdateEvent]) or Message object.
        """
        if agent_name not in self.remote_agent_connections:
            raise ValueError(f"Agent {agent_name} not found")

        client = self.remote_agent_connections[agent_name]
        if not client:
            raise ValueError(f"Client not available for {agent_name}")

        message_id = str(uuid.uuid4())

        # Create Message object using the imported Message class
        # Note: Use snake_case field names as per A2A spec
        message_request = Message(
            role=Role.user,
            parts=[TextPart(text=message)],  # TextPart has 'text' field, 'kind' defaults to 'text'
            message_id=message_id,  # snake_case, not messageId
            context_id=context_id,  # optional context_id
            metadata=metadata,
        )
        try:
            if streaming:
                # Stream responses in real-time
                async for response in client.send_message(message_request=message_request):
                    logger.info(f"send_response (streaming): {response}")
                    yield response
            else:
                # Collect all responses and return only the final one
                final_response = None
                async for response in client.send_message(message_request=message_request):
                    final_response = response

                logger.info(f"send_response (final): {final_response}")
                yield final_response
        except A2AClientError as e:
            logger.error(f"A2A Client error when sending message to agent {agent_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error sending message to agent {agent_name}: {e}")
            raise

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
