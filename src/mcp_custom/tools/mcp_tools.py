"""MCP Tools Wrappers for LangChain - HTTP Client Version"""

import os
import asyncio
import json
from typing import Dict, Any, List, Literal
from langchain_core.tools import tool
import httpx

# MCP Server URLs from environment variables or defaults
MARKET_DATA_URL = os.getenv("MCP_MARKET_DATA_URL", "http://localhost:8001")
PORTFOLIO_URL = os.getenv("MCP_PORTFOLIO_URL", "http://localhost:8002")
TAVILY_URL = os.getenv("MCP_TAVILY_URL", "http://localhost:8003")
RAG_URL = os.getenv("MCP_RAG_URL", "http://localhost:8004")


async def call_mcp_tool(base_url: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
    """
    Call MCP server tool via HTTP

    Args:
        base_url: Base URL of MCP server
        tool_name: Name of the tool to call
        arguments: Tool arguments

    Returns:
        Tool result
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{base_url}/call-tool", json={"name": tool_name, "arguments": arguments}
        )
        response.raise_for_status()
        result = response.json()

        # Extract content from MCP response format
        if isinstance(result, dict) and "content" in result:
            content = result["content"]
            if isinstance(content, list) and len(content) > 0:
                # Return the text content
                return content[0].get("text", content[0])
            return content
        return result


@tool
async def get_stock_price_tool(symbol: str) -> Dict[str, Any]:
    """
    Get current stock price for a given symbol

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')

    Returns:
        Dictionary with current price, change, and timestamp
    """
    result = await call_mcp_tool(MARKET_DATA_URL, "get_stock_price", {"symbol": symbol})
    # Parse JSON string if needed
    if isinstance(result, str):

        return json.loads(result)
    return result


@tool
async def get_market_news_tool(symbol: str | None = None, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Get market news articles

    Args:
        symbol: Optional stock symbol to filter news
        limit: Maximum number of articles to return

    Returns:
        List of news articles
    """
    args = {"limit": limit}
    if symbol:
        args["symbol"] = symbol

    result = await call_mcp_tool(MARKET_DATA_URL, "get_market_news", args)
    # Parse JSON string if needed
    if isinstance(result, str):

        return json.loads(result)
    return result


@tool
async def get_company_financials_tool(symbol: str) -> Dict[str, Any]:
    """
    Get company financial metrics

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dictionary with key financial metrics
    """
    result = await call_mcp_tool(MARKET_DATA_URL, "get_company_financials", {"symbol": symbol})
    # Parse JSON string if needed
    if isinstance(result, str):

        return json.loads(result)
    return result


@tool
async def get_portfolio_tool(user_id: str = "default") -> Dict[str, Any]:
    """
    Get user's portfolio

    Args:
        user_id: User identifier

    Returns:
        Portfolio with positions and total value
    """
    result = await call_mcp_tool(PORTFOLIO_URL, "get_portfolio", {"user_id": user_id})
    # Parse JSON string if needed
    if isinstance(result, str):

        return json.loads(result)
    return result


@tool
async def add_position_tool(
    user_id: str, symbol: str, shares: float, purchase_price: float
) -> Dict[str, Any]:
    """
    Add a position to the portfolio

    Args:
        user_id: User identifier
        symbol: Stock ticker symbol
        shares: Number of shares to add
        purchase_price: Price per share at purchase

    Returns:
        Updated portfolio
    """
    result = await call_mcp_tool(
        PORTFOLIO_URL,
        "add_position",
        {"user_id": user_id, "symbol": symbol, "shares": shares, "purchase_price": purchase_price},
    )
    # Parse JSON string if needed
    if isinstance(result, str):

        return json.loads(result)
    return result


@tool
async def remove_position_tool(
    user_id: str, symbol: str, shares: float, sell_price: float
) -> Dict[str, Any]:
    """
    Remove (sell) a position from the portfolio

    Args:
        user_id: User identifier
        symbol: Stock ticker symbol
        shares: Number of shares to sell
        sell_price: Price per share at sale

    Returns:
        Updated portfolio with transaction details
    """
    result = await call_mcp_tool(
        PORTFOLIO_URL,
        "remove_position",
        {"user_id": user_id, "symbol": symbol, "shares": shares, "sell_price": sell_price},
    )
    # Parse JSON string if needed
    if isinstance(result, str):

        return json.loads(result)
    return result


@tool
async def calculate_returns_tool(user_id: str, current_prices: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculate portfolio returns

    Args:
        user_id: User identifier
        current_prices: Dictionary mapping symbols to current prices

    Returns:
        Portfolio performance metrics
    """
    result = await call_mcp_tool(
        PORTFOLIO_URL, "calculate_returns", {"user_id": user_id, "current_prices": current_prices}
    )
    # Parse JSON string if needed
    if isinstance(result, str):

        return json.loads(result)
    return result


@tool
async def analyze_risk_tool(user_id: str) -> Dict[str, Any]:
    """
    Analyze portfolio risk metrics

    Args:
        user_id: User identifier

    Returns:
        Risk analysis including diversification and concentration
    """
    result = await call_mcp_tool(PORTFOLIO_URL, "analyze_risk", {"user_id": user_id})
    # Parse JSON string if needed
    if isinstance(result, str):

        return json.loads(result)
    return result


@tool
async def search_tavily_tool(
    query: str,
    search_depth: Literal["basic", "advanced"] = "basic",
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] | None = None,
    time_range: Literal["day", "week", "month", "year"] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    days: int | None = None,
    include_domains: List[str] | None = None,
    exclude_domains: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Tavily Search API를 활용한 웹 검색

    실시간 웹 검색을 수행하여 최신 정보를 가져옵니다.
    다양한 필터링 옵션을 통해 정확하고 관련성 높은 결과를 제공합니다.

    Args:
        query: 검색할 키워드 또는 질문
        search_depth: 검색 깊이
            - "basic": 빠른 검색, 기본적인 결과
            - "advanced": 상세 검색, 더 많은 정보와 분석
        max_results: 반환할 최대 결과 수 (1-100 범위)
        topic: 검색 주제 필터
            - "general": 일반적인 웹 검색
            - "news": 뉴스 및 시사 정보
            - "finance": 금융 및 경제 정보
        time_range: 검색할 시간 범위
            - "day": 최근 하루
            - "week": 최근 일주일
            - "month": 최근 한 달
            - "year": 최근 일년
        start_date: 검색 시작 날짜 (YYYY-MM-DD 형식)
        end_date: 검색 종료 날짜 (YYYY-MM-DD 형식)
        days: 최근 며칠 이내의 결과만 검색 (정수)
        include_domains: 포함할 도메인 리스트
            예: ["wikipedia.org", "github.com"]
        exclude_domains: 제외할 도메인 리스트
            예: ["ads.com", "spam.com"]

    Returns:
        List[Dict[str, Any]]: 검색 결과 리스트
            각 결과는 다음 필드를 포함:
            - title: 페이지 제목
            - url: 페이지 URL
            - content: 페이지 내용 요약
            - score: 관련성 점수
            - published_date: 발행 날짜 (있는 경우)

    Raises:
        Exception: API 호출 실패 또는 네트워크 오류 시
    """
    result = await call_mcp_tool(
        TAVILY_URL,
        "search",
        {
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "topic": topic,
            "time_range": time_range,
            "start_date": start_date,
            "end_date": end_date,
            "days": days,
            "include_domains": include_domains,
            "exclude_domains": exclude_domains,
        },
    )
    # Parse JSON string if needed
    if isinstance(result, str):

        return json.loads(result)
    return result


# ============================================================================
# RAG (Retrieval Augmented Generation) Tools
# ============================================================================


@tool
async def add_documents_to_rag(
    documents: List[str], metadata_list: List[Dict[str, Any]] | None = None
) -> Dict[str, Any]:
    """
    Add documents to the RAG knowledge base

    Args:
        documents: List of text documents to add
        metadata_list: Optional list of metadata dictionaries for each document

    Returns:
        Dictionary with status and count of added documents
    """
    result = await call_mcp_tool(
        RAG_URL, "add_documents", {"documents": documents, "metadata_list": metadata_list}
    )
    if isinstance(result, str):

        return json.loads(result)
    return result


@tool
async def search_knowledge_base(
    query: str, k: int = 5, include_distances: bool = True
) -> List[Dict[str, Any]]:
    """
    Search the RAG knowledge base for relevant documents

    Args:
        query: Search query text
        k: Number of results to return (default: 5)
        include_distances: Whether to include distance scores

    Returns:
        List of matching documents with metadata and optional distances
    """
    result = await call_mcp_tool(
        RAG_URL,
        "search_documents",
        {"query": query, "k": k, "include_distances": include_distances},
    )
    if isinstance(result, str):

        return json.loads(result)
    return result


@tool
async def rag_query_tool(
    query: str, k: int = 3, context_template: str | None = None
) -> Dict[str, Any]:
    """
    Perform RAG query: retrieve relevant documents and prepare context

    This tool retrieves relevant documents from the knowledge base and formats them
    as context that can be used to answer the user's query.

    Args:
        query: User query
        k: Number of documents to retrieve (default: 3)
        context_template: Optional template for formatting context

    Returns:
        Dictionary with query, retrieved documents, and formatted context
    """
    result = await call_mcp_tool(
        RAG_URL,
        "rag_query",
        {"query": query, "k": k, "context_template": context_template},
    )
    if isinstance(result, str):

        return json.loads(result)
    return result


@tool
async def get_rag_stats_tool() -> Dict[str, Any]:
    """
    Get statistics about the RAG knowledge base

    Returns:
        Dictionary with database statistics including document count
    """
    result = await call_mcp_tool(RAG_URL, "get_rag_stats", {})
    if isinstance(result, str):

        return json.loads(result)
    return result


# Example test
if __name__ == "__main__":

    async def test_tools():
        # Test market data
        print("Testing get_stock_price_tool...")
        price = await get_stock_price_tool.ainvoke({"symbol": "AAPL"})
        print("Stock Price:", price)

        # Test portfolio
        print("\nTesting get_portfolio_tool...")
        portfolio = await get_portfolio_tool.ainvoke({"user_id": "test"})
        print("Portfolio:", portfolio)

    asyncio.run(test_tools())
