"""A2A Server Implementation for Supervisor Agent"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn
import logging
from datetime import datetime
import asyncio
from agents.supervisor_agent import SupervisorAgent

logger = logging.getLogger(__name__)

# Get service URL from environment or use default
SERVICE_URL = os.getenv("A2A_SERVICE_URL", "http://localhost:8099")

app = FastAPI(title="Supervisor Agent A2A Server")

# Initialize agent
supervisor = SupervisorAgent()

# Task storage
tasks: Dict[str, Dict[str, Any]] = {}


class A2AMessage(BaseModel):
    """A2A Message model"""

    role: str
    content: str
    parts: List[Dict[str, Any]]


class TaskCreateParams(BaseModel):
    """Parameters for task creation"""

    message: A2AMessage
    context: Optional[Dict[str, Any]] = {}


class A2ARequest(BaseModel):
    """A2A JSON-RPC Request"""

    jsonrpc: str = "2.0"
    id: str
    method: str
    params: Dict[str, Any]


@app.get("/.well-known/agent.json")
async def agent_card():
    """
    Serve Agent Card for A2A discovery
    Implements the A2A Agent Card specification

    Returns:
        Agent Card with capabilities and metadata
    """
    return {
        "schema_version": "1.0",
        "name": "Supervisor Agent",
        "version": "1.0.0",
        "description": "Main orchestrator agent that routes tasks to specialized agents",
        "service_url": SERVICE_URL,
        "a2a_endpoint": f"{SERVICE_URL}/a2a",
        "capabilities": [
            "task_orchestration",
            "agent_routing",
            "multi_agent_coordination",
            "task_classification",
            "workflow_management",
        ],
        "modalities": ["text", "structured_data"],
        "supported_methods": ["task.create", "task.status", "task.cancel", "agents.list"],
        "features": {
            "streaming": False,
            "artifacts": True,
            "context_awareness": True,
            "multi_turn": True,
            "agent_delegation": True,
        },
        "authentication": {
            "type": "none",
            "note": "For development only; use OAuth2 in production",
        },
        "rate_limits": {"requests_per_minute": 60, "concurrent_tasks": 20},
        "contact": {
            "support_url": "http://localhost:8099/support",
            "documentation_url": "http://localhost:8099/docs",
            "issues_url": "https://github.com/gwanghun-im/ra/issues",
        },
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "agent_type": "orchestrator",
            "domain": "multi_agent_coordination",
            "tags": ["supervisor", "orchestrator", "router", "multi-agent", "workflow"],
        },
    }


@app.post("/a2a")
async def a2a_endpoint(request: A2ARequest):
    """
    Main A2A endpoint for task communication

    Args:
        request: A2A JSON-RPC request

    Returns:
        JSON-RPC response
    """
    try:
        method = request.method
        params = request.params

        if method == "task.create":
            result = await handle_task_create(request.id, params)
        elif method == "task.status":
            result = await handle_task_status(params)
        elif method == "task.cancel":
            result = await handle_task_cancel(params)
        elif method == "agents.list":
            result = await handle_agents_list(params)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {method}")

        return {"jsonrpc": "2.0", "id": request.id, "result": result}

    except Exception as e:
        logger.error(f"Error handling A2A request: {e}")
        return {"jsonrpc": "2.0", "id": request.id, "error": {"code": -32603, "message": str(e)}}


async def handle_task_create(task_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle task creation

    Args:
        task_id: Task identifier
        params: Task parameters

    Returns:
        Task result
    """
    message_data = params.get("message", {})
    context = params.get("context", {})

    # Extract message content
    content = message_data.get("content", "")
    user_id = context.get("user_id", "default")

    logger.info(f"Supervisor processing task {task_id}: {content}")

    # Store task
    tasks[task_id] = {
        "id": task_id,
        "status": "processing",
        "created_at": datetime.now().isoformat(),
        "message": content,
    }

    try:
        # Process with Supervisor
        result = await supervisor.process_request(content, user_id)

        # Update task
        tasks[task_id].update(
            {"status": "completed", "completed_at": datetime.now().isoformat(), "result": result}
        )

        # Return response in A2A format
        return {
            "task_id": task_id,
            "status": "completed",
            "message": {
                "role": "assistant",
                "content": result["response"],
                "parts": [{"type": "text", "text": result["response"]}],
            },
            "artifacts": [
                {
                    "type": "routing_info",
                    "data": {
                        "task_type": result.get("task_type"),
                        "delegated_to": result.get("delegated_to"),
                    },
                }
            ],
        }

    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}")

        tasks[task_id].update(
            {"status": "failed", "error": str(e), "failed_at": datetime.now().isoformat()}
        )

        raise


async def handle_task_status(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get task status

    Args:
        params: Parameters containing task_id

    Returns:
        Task status information
    """
    task_id = params.get("task_id")

    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = tasks[task_id]

    return {
        "task_id": task_id,
        "status": task["status"],
        "created_at": task["created_at"],
        "result": task.get("result"),
    }


async def handle_task_cancel(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cancel a task

    Args:
        params: Parameters containing task_id

    Returns:
        Cancellation confirmation
    """
    task_id = params.get("task_id")

    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    tasks[task_id]["status"] = "cancelled"
    tasks[task_id]["cancelled_at"] = datetime.now().isoformat()

    return {"task_id": task_id, "status": "cancelled"}


async def handle_agents_list(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List available agents

    Args:
        params: Optional parameters

    Returns:
        List of available agents
    """
    agents = await supervisor.get_available_agents()

    return {"agents": agents, "count": len(agents)}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": "supervisor",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/capabilities")
async def get_capabilities():
    """Get agent capabilities"""
    return {
        "capabilities": [
            "task_orchestration",
            "agent_routing",
            "multi_agent_coordination",
            "task_classification",
            "workflow_management",
        ],
        "agent_name": "Supervisor Agent",
    }


@app.get("/agents")
async def list_agents():
    """List available agents"""
    agents = await supervisor.get_available_agents()
    return {"agents": agents, "count": len(agents)}


def run_server(host: str = "0.0.0.0", port: int = 8099):
    """Run the A2A server"""
    logger.info(f"Starting Supervisor Agent A2A Server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
