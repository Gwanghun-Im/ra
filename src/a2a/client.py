"""A2A Client Implementation"""

import httpx
import os
import re
from typing import Dict, Any, List
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class A2AClient:
    """Client for Agent-to-Agent (A2A) communication"""

    def __init__(self, config_path: str = "config/a2a_config.yaml"):
        """
        Initialize A2A Client

        Args:
            config_path: Path to A2A configuration file
        """
        self.config = self._load_config(config_path)
        # Unified agent registry combining config and discovered agent cards
        self.agents: Dict[str, Dict[str, Any]] = {}
        self._initialize_agents_from_config()

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

    def _initialize_agents_from_config(self):
        """Initialize agents registry from config file"""
        agents_config = self.config.get("a2a", {}).get("agents", {})
        for agent_name, agent_config in agents_config.items():
            if agent_config.get("enabled", True):
                self.agents[agent_name] = {
                    "name": agent_config.get("name", agent_name),
                    "description": agent_config.get("description", ""),
                    "version": agent_config.get("version", "1.0.0"),
                    "url": agent_config.get("url", ""),
                    "card_url": agent_config.get("card_url", ""),
                    "local_card_url": agent_config.get("local_card_url"),
                    "a2a_endpoint": agent_config.get("a2a_endpoint", ""),
                    "capabilities": agent_config.get("capabilities", []),
                    "modalities": agent_config.get("modalities", []),
                    "discovered": False,  # Not yet discovered
                    "service_url": None,  # Will be populated after discovery
                }

    def _get_agent_url(self, agent_name: str, url_type: str = "card") -> str | None:
        """
        Get the appropriate URL for an agent based on environment (Docker vs local).

        Args:
            agent_name: Name of the agent
            url_type: Type of URL to retrieve ("card", "service", "a2a")

        Returns:
            Appropriate URL based on environment, or None if not available
        """
        if agent_name not in self.agents:
            return None

        agent_info = self.agents[agent_name]

        # Determine which URL to use based on type
        if url_type == "card":
            docker_url = agent_info.get("card_url", "")
            local_url = agent_info.get("local_card_url")
        elif url_type == "service":
            # For service URL, use discovered service_url if available
            if agent_info.get("discovered"):
                return agent_info.get("service_url")
            docker_url = agent_info.get("url", "")
            local_url = agent_info.get("local_card_url")
            if local_url:
                # Extract base URL from local_card_url
                local_url = local_url.replace("/.well-known/agent.json", "")
        elif url_type == "a2a":
            docker_url = agent_info.get("a2a_endpoint", "")
            local_url = None  # Will be constructed from base URL
        else:
            return None

        # Choose URL based on environment
        if self.is_docker:
            return docker_url
        else:
            # Use local_url if available
            if local_url:
                if url_type == "a2a":
                    return f"{local_url}/a2a"
                return local_url

            # Fallback: convert Docker container URLs to localhost
            if docker_url:
                # Convert any container name pattern to localhost using regex
                # Matches: http://a2a_*:PORT -> http://localhost:PORT
                url = re.sub(
                    r"http://a2a_\w+:(\d+)", r"http://localhost:\1", docker_url
                )
                return url

        return None

    async def discover_agents(self) -> List[Dict[str, Any]]:
        """
        Discover available A2A agents by fetching their Agent Cards.
        Updates the unified agents registry with discovered information.

        Returns:
            List of discovered agents with their capabilities
        """
        discovered = []

        for agent_name in self.agents.keys():
            # Get card URL using centralized URL management
            card_url = self._get_agent_url(agent_name, url_type="card")

            if not card_url:
                logger.warning(f"No card URL available for agent {agent_name}")
                continue

            try:
                response = await self.http_client.get(card_url)
                response.raise_for_status()

                agent_card = response.json()
                # Update unified registry with discovered information
                self.agents[agent_name].update(
                    {
                        "discovered": True,
                        "service_url": agent_card.get("service_url"),
                        "agent_card": agent_card,  # Store full agent card
                    }
                )
                # Update capabilities from agent card if present
                if "capabilities" in agent_card:
                    self.agents[agent_name]["capabilities"] = agent_card["capabilities"]
                discovered.append(agent_card)
                logger.info(f"Discovered agent: {agent_name}")
            except httpx.HTTPStatusError as e:
                logger.warning(f"Failed to discover agent {agent_name}: HTTP {e.response.status_code}")
            except httpx.RequestError as e:
                logger.warning(f"Failed to discover agent {agent_name}: {type(e).__name__}")
            except Exception as e:
                logger.warning(f"Failed to discover agent {agent_name}: {e}")

        logger.debug(f"Discovery complete. Discovered {len(discovered)}/{len(self.agents)} agents")
        return discovered

    async def send_task(
        self,
        agent_name: str,
        message: str,
        task_id: str | None = None,
        context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Send a task to a remote A2A agent

        Args:
            agent_name: Name of the target agent
            message: Task message/instruction
            task_id: Optional task identifier
            context: Optional context information

        Returns:
            Task response from the agent
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found in configuration")

        agent_info = self.agents[agent_name]

        # Discover agent if not yet discovered
        if not agent_info.get("discovered"):
            logger.debug(f"Agent '{agent_name}' not yet discovered, initiating discovery...")
            await self.discover_agents()

        if not agent_info.get("discovered"):
            raise ValueError(f"Agent '{agent_name}' could not be discovered. Check if the agent is running.")

        # Get A2A endpoint URL using centralized URL management
        agent_url = self._get_agent_url(agent_name, url_type="a2a")

        if not agent_url:
            raise ValueError(f"No A2A endpoint URL for agent '{agent_name}'")

        # Construct A2A message
        payload = {
            "jsonrpc": "2.0",
            "id": task_id or "task-1",
            "method": "task.create",
            "params": {
                "message": {
                    "role": "user",
                    "content": message,
                    "parts": [{"type": "text", "text": message}],
                },
                "context": context or {},
            },
        }

        logger.debug(f"Sending task to agent '{agent_name}' at {agent_url}")
        try:
            response = await self.http_client.post(agent_url, json=payload)
            response.raise_for_status()

            result = response.json()
            logger.info(f"Task sent successfully to '{agent_name}'")
            return result.get("result", {})

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error sending task to '{agent_name}': {e.response.status_code}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Network error sending task to '{agent_name}': {type(e).__name__}")
            raise
        except Exception as e:
            logger.error(f"Error sending task to '{agent_name}': {e}")
            raise

    async def get_task_status(self, agent_name: str, task_id: str) -> Dict[str, Any]:
        """
        Get status of a task

        Args:
            agent_name: Name of the agent
            task_id: Task identifier

        Returns:
            Task status information
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found in configuration")

        agent_info = self.agents[agent_name]

        if not agent_info.get("discovered"):
            raise ValueError(f"Agent '{agent_name}' has not been discovered yet. Call discover_agents() first.")

        # Get service URL using centralized URL management
        agent_url = self._get_agent_url(agent_name, url_type="service")

        if not agent_url:
            raise ValueError(f"No service URL for agent '{agent_name}'")

        payload = {
            "jsonrpc": "2.0",
            "id": "status-1",
            "method": "task.status",
            "params": {"task_id": task_id},
        }

        try:
            response = await self.http_client.post(f"{agent_url}/a2a", json=payload)
            response.raise_for_status()

            result = response.json()
            return result.get("result", {})

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error getting task status from '{agent_name}': {e.response.status_code}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Network error getting task status from '{agent_name}': {type(e).__name__}")
            raise
        except Exception as e:
            logger.error(f"Error getting task status from '{agent_name}': {e}")
            raise

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
    import asyncio

    async def main():
        # Using context manager for automatic cleanup
        async with A2AClient() as client:
            # Discover agents
            agents = await client.discover_agents()
            print(f"\nDiscovered {len(agents)} agents")

            # Get all discovered agents
            discovered = client.get_discovered_agents()
            print(f"Available agents: {list(discovered.keys())}")

            # Send a task (if agents are available)
            if discovered:
                agent_name = list(discovered.keys())[0]
                print(f"\nSending task to '{agent_name}'...")

                result = await client.send_task(
                    agent_name, "Analyze the tech sector", task_id="test-task-1"
                )
                print(f"Task result: {result}")

    asyncio.run(main())
