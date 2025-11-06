"""A2A Server Implementation for RAG Agent"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn
import logging
from datetime import datetime
import asyncio
from agents.rag_agent import RAGAgent

logger = logging.getLogger(__name__)

# Get service URL from environment or use default
SERVICE_URL = os.getenv("A2A_SERVICE_URL", "http://localhost:8101")

app = FastAPI(title="RAG Agent A2A Server")

# Initialize agent
rag_agent = RAGAgent()

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
        "name": "RAG Agent",
        "version": "1.0.0",
        "description": "Specialized agent for knowledge base search and retrieval-augmented generation",
        "service_url": SERVICE_URL,
        "a2a_endpoint": f"{SERVICE_URL}/a2a",
        "capabilities": rag_agent.get_capabilities(),
        "modalities": ["text", "structured_data"],
        "supported_methods": ["task.create", "task.status", "task.cancel"],
        "features": {
            "streaming": False,
            "artifacts": True,
            "context_awareness": True,
            "multi_turn": True,
        },
        "authentication": {
            "type": "none",
            "note": "For development only; use OAuth2 in production",
        },
        "rate_limits": {"requests_per_minute": 120, "concurrent_tasks": 20},
        "contact": {
            "support_url": "http://localhost:8101/support",
            "documentation_url": "http://localhost:8101/docs",
            "issues_url": "https://github.com/gwanghun-im/ra/issues",
        },
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "agent_type": "specialist",
            "domain": "knowledge_retrieval",
            "tags": ["rag", "knowledge-base", "search", "retrieval", "semantic-search"],
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

    logger.info(f"Processing RAG task {task_id}: {content}")

    # Store task
    tasks[task_id] = {
        "id": task_id,
        "status": "processing",
        "created_at": datetime.now().isoformat(),
        "message": content,
    }

    try:
        # Process with RAG Agent
        result = await rag_agent.process_request(content)

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
            "artifacts": [{"type": "rag_result", "data": result.get("rag_result", {})}],
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


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "agent": "rag_agent", "timestamp": datetime.now().isoformat()}


@app.get("/capabilities")
async def get_capabilities():
    """Get agent capabilities"""
    return {"capabilities": rag_agent.get_capabilities(), "agent_name": "RAG Agent"}


def run_server(host: str = "0.0.0.0", port: int = 8101):
    """Run the A2A server"""
    logger.info(f"Starting RAG Agent A2A Server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
