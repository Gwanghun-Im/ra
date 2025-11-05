"""MCP Server for RAG (Retrieval Augmented Generation) using FAISS"""

import asyncio
from typing import Any, Dict, List
import os
import sys
from pathlib import Path
from fastmcp import FastMCP
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from vector_db.faiss_manager import FAISSManager

# Initialize MCP server
mcp = FastMCP("RAG Server")

# Initialize FAISS manager (will be loaded when server starts)
faiss_manager = None


def get_faiss_manager():
    """Lazy initialization of FAISS manager"""
    global faiss_manager
    if faiss_manager is None:
        index_path = os.getenv("FAISS_INDEX_PATH", "./vector_db/embeddings")
        faiss_manager = FAISSManager(index_path=index_path)
    return faiss_manager


async def get_embeddings(texts: List[str]) -> np.ndarray:
    """
    Get embeddings from OpenAI API

    Args:
        texts: List of texts to embed

    Returns:
        Numpy array of embeddings
    """
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Get embeddings
        response = await client.embeddings.create(
            model="text-embedding-3-small",  # 1536 dimensions
            input=texts
        )

        # Extract embeddings
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype='float32')

    except Exception as e:
        raise Exception(f"Failed to get embeddings: {str(e)}")


@mcp.tool()
async def add_documents(
    documents: List[str],
    metadata_list: List[Dict[str, Any]] | None = None
) -> Dict[str, Any]:
    """
    Add documents to the RAG vector database

    Args:
        documents: List of text documents to add
        metadata_list: Optional list of metadata dictionaries for each document

    Returns:
        Dictionary with status and count of added documents
    """
    await asyncio.sleep(0.1)  # Simulate processing

    manager = get_faiss_manager()

    # Create default metadata if not provided
    if metadata_list is None:
        metadata_list = [
            {"doc_id": i, "text": doc[:200]}  # Store first 200 chars
            for i, doc in enumerate(documents)
        ]

    if len(documents) != len(metadata_list):
        raise ValueError("Number of documents and metadata must match")

    # Get embeddings
    embeddings = await get_embeddings(documents)

    # Add to FAISS
    manager.add_embeddings(embeddings, metadata_list)

    return {
        "status": "success",
        "added_count": len(documents),
        "total_vectors": manager.index.ntotal,
        "message": f"Successfully added {len(documents)} documents to RAG database"
    }


@mcp.tool()
async def search_documents(
    query: str,
    k: int = 5,
    include_distances: bool = True
) -> List[Dict[str, Any]]:
    """
    Search for relevant documents using semantic search

    Args:
        query: Search query text
        k: Number of results to return (default: 5)
        include_distances: Whether to include distance scores

    Returns:
        List of matching documents with metadata and optional distances
    """
    await asyncio.sleep(0.1)  # Simulate processing

    manager = get_faiss_manager()

    # Check if index has vectors
    if manager.index.ntotal == 0:
        return []

    # Get query embedding
    query_embedding = await get_embeddings([query])

    # Search
    results = manager.search(query_embedding, k=min(k, manager.index.ntotal))

    # Format results
    formatted_results = []
    for result in results:
        formatted = {
            "metadata": result["metadata"],
        }
        if include_distances:
            formatted["distance"] = result["distance"]
            formatted["similarity"] = 1 / (1 + result["distance"])  # Convert distance to similarity

        formatted_results.append(formatted)

    return formatted_results


@mcp.tool()
async def rag_query(
    query: str,
    k: int = 3,
    context_template: str | None = None
) -> Dict[str, Any]:
    """
    Perform RAG query: retrieve relevant documents and prepare context for LLM

    Args:
        query: User query
        k: Number of documents to retrieve (default: 3)
        context_template: Optional template for formatting context

    Returns:
        Dictionary with query, retrieved documents, and formatted context
    """
    await asyncio.sleep(0.1)

    # Search for relevant documents
    results = await search_documents(query, k=k, include_distances=True)

    if not results:
        return {
            "query": query,
            "retrieved_documents": [],
            "context": "No relevant documents found in the knowledge base.",
            "num_results": 0
        }

    # Default context template
    if context_template is None:
        context_template = """Based on the following relevant information from the knowledge base:

{context}

Please answer the user's query: {query}"""

    # Build context from retrieved documents
    context_parts = []
    for i, result in enumerate(results, 1):
        metadata = result["metadata"]
        text = metadata.get("text", metadata.get("content", str(metadata)))
        similarity = result.get("similarity", 0)

        context_parts.append(
            f"[Document {i}] (Relevance: {similarity:.2%})\n{text}"
        )

    context = "\n\n".join(context_parts)

    # Format with template
    formatted_context = context_template.format(
        context=context,
        query=query
    )

    return {
        "query": query,
        "retrieved_documents": results,
        "context": formatted_context,
        "num_results": len(results)
    }


@mcp.tool()
async def get_rag_stats() -> Dict[str, Any]:
    """
    Get statistics about the RAG database

    Returns:
        Dictionary with database statistics
    """
    manager = get_faiss_manager()
    stats = manager.get_stats()

    return {
        "total_documents": stats["total_vectors"],
        "embedding_dimension": stats["dimension"],
        "index_path": stats["index_path"],
        "metadata_count": stats["metadata_count"],
        "status": "healthy" if stats["total_vectors"] > 0 else "empty"
    }


@mcp.tool()
async def clear_rag_database() -> Dict[str, Any]:
    """
    Clear all documents from the RAG database

    Returns:
        Dictionary with status message
    """
    manager = get_faiss_manager()
    manager.clear()

    return {
        "status": "success",
        "message": "RAG database cleared successfully"
    }


@mcp.resource("rag://status")
async def rag_status() -> str:
    """Get current RAG database status"""
    stats = await get_rag_stats()
    return f"RAG Database Status: {stats['total_documents']} documents indexed"


# Run the server
if __name__ == "__main__":
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn

    port = int(os.getenv("MCP_SERVER_PORT", "8004"))

    # Create FastAPI app for HTTP API
    app = FastAPI(title="RAG MCP Server")

    class ToolCallRequest(BaseModel):
        name: str
        arguments: dict

    @app.post("/call-tool")
    async def call_tool_endpoint(request: ToolCallRequest):
        """HTTP endpoint to call MCP tools"""
        try:
            # Map tool names to their async functions
            tool_functions = {
                "add_documents": add_documents,
                "search_documents": search_documents,
                "rag_query": rag_query,
                "get_rag_stats": get_rag_stats,
                "clear_rag_database": clear_rag_database,
            }

            if request.name not in tool_functions:
                raise HTTPException(
                    status_code=404,
                    detail=f"Tool {request.name} not found. Available: {list(tool_functions.keys())}",
                )

            # Call the tool function with arguments
            tool_obj = tool_functions[request.name]
            if hasattr(tool_obj, 'fn'):
                # It's a FunctionTool, get the underlying async function
                result = await tool_obj.fn(**request.arguments)
            else:
                # It's a regular function
                result = await tool_obj(**request.arguments)

            # Return result in consistent format
            import json
            return {"content": [{"type": "text", "text": json.dumps(result)}]}
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health():
        """Health check endpoint"""
        manager = get_faiss_manager()
        stats = manager.get_stats()
        return {
            "status": "healthy",
            "server": "RAG MCP",
            "documents": stats["total_vectors"]
        }

    # Run FastAPI server
    print(f"Starting RAG MCP Server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
