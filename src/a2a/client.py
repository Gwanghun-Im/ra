"""A2A Client Implementation"""
import httpx
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
        self.discovered_agents: Dict[str, Dict[str, Any]] = {}
        self.http_client = httpx.AsyncClient(timeout=60.0)
        
        logger.info("A2A Client initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load A2A configuration"""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        return {"a2a": {"agents": {}}}
    
    async def discover_agents(self) -> List[Dict[str, Any]]:
        """
        Discover available A2A agents by fetching their Agent Cards
        
        Returns:
            List of discovered agents with their capabilities
        """
        agents_config = self.config.get("a2a", {}).get("agents", {})
        discovered = []
        
        for agent_name, agent_config in agents_config.items():
            card_url = agent_config.get("card_url")
            if not card_url:
                continue
            
            try:
                response = await self.http_client.get(card_url)
                if response.status_code == 200:
                    agent_card = response.json()
                    self.discovered_agents[agent_name] = agent_card
                    discovered.append(agent_card)
                    logger.info(f"Discovered agent: {agent_name}")
            except Exception as e:
                logger.warning(f"Failed to discover agent {agent_name}: {e}")
        
        return discovered
    
    async def send_task(
        self,
        agent_name: str,
        message: str,
        task_id: str | None = None,
        context: Dict[str, Any] | None = None
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
        if agent_name not in self.discovered_agents:
            await self.discover_agents()
        
        if agent_name not in self.discovered_agents:
            raise ValueError(f"Agent {agent_name} not found")
        
        agent_card = self.discovered_agents[agent_name]
        agent_url = agent_card.get("service_url")
        
        if not agent_url:
            raise ValueError(f"No service URL for agent {agent_name}")
        
        # Construct A2A message
        payload = {
            "jsonrpc": "2.0",
            "id": task_id or "task-1",
            "method": "task.create",
            "params": {
                "message": {
                    "role": "user",
                    "content": message,
                    "parts": [
                        {
                            "type": "text",
                            "text": message
                        }
                    ]
                },
                "context": context or {}
            }
        }
        
        try:
            response = await self.http_client.post(
                f"{agent_url}/a2a",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Task sent successfully to {agent_name}")
                return result.get("result", {})
            else:
                raise Exception(f"A2A request failed: {response.status_code}")
        
        except Exception as e:
            logger.error(f"Error sending task to {agent_name}: {e}")
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
        if agent_name not in self.discovered_agents:
            raise ValueError(f"Agent {agent_name} not found")
        
        agent_card = self.discovered_agents[agent_name]
        agent_url = agent_card.get("service_url")
        
        payload = {
            "jsonrpc": "2.0",
            "id": "status-1",
            "method": "task.status",
            "params": {
                "task_id": task_id
            }
        }
        
        try:
            response = await self.http_client.post(
                f"{agent_url}/a2a",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("result", {})
            else:
                raise Exception(f"Status request failed: {response.status_code}")
        
        except Exception as e:
            logger.error(f"Error getting task status: {e}")
            raise
    
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()
    
    def get_agent_capabilities(self, agent_name: str) -> List[str]:
        """
        Get capabilities of a discovered agent
        
        Args:
            agent_name: Name of the agent
        
        Returns:
            List of agent capabilities
        """
        if agent_name in self.discovered_agents:
            return self.discovered_agents[agent_name].get("capabilities", [])
        return []


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        client = A2AClient()
        
        # Discover agents
        agents = await client.discover_agents()
        print("Discovered agents:", agents)
        
        # Send a task (if agents are available)
        if agents:
            agent_name = list(client.discovered_agents.keys())[0]
            result = await client.send_task(
                agent_name,
                "Analyze the tech sector",
                task_id="test-task-1"
            )
            print("Task result:", result)
        
        await client.close()
    
    asyncio.run(main())
