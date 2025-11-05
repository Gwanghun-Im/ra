"""MCP Server for Tavily Management"""

import os
from typing import List, Any, Literal, cast
import json
from fastmcp import FastMCP
from tavily import TavilyClient

# Initialize MCP server
mcp = FastMCP("Tavily Server")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "INSERT_YOUR_API_KEY")


@mcp.tool()
async def search(
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
) -> dict[str, Any]:
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

    # Tavily 클라이언트 초기화
    client = TavilyClient(api_key=TAVILY_API_KEY)

    # 기본 검색 파라미터 구성
    search_params = {
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,
    }

    search_params["topic"] = topic or "general"
    search_params["time_range"] = time_range or None
    search_params["start_date"] = start_date or None
    search_params["end_date"] = end_date or None
    search_params["days"] = days or None
    search_params["include_domains"] = include_domains or None
    search_params["exclude_domains"] = exclude_domains or None

    # Tavily API 호출 및 결과 반환
    results = client.search(**search_params)
    return cast(dict[str, Any], results)


# Run the server
if __name__ == "__main__":
    import os
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn

    port = int(os.getenv("MCP_SERVER_PORT", "8003"))

    # Create FastAPI app for HTTP API
    app = FastAPI(title="Tavily MCP Server")

    class ToolCallRequest(BaseModel):
        """HTTP endpoint to call MCP tools"""

        name: str
        arguments: dict

    @app.post("/call-tool")
    async def call_tool_endpoint(request: ToolCallRequest):
        """HTTP endpoint to call MCP tools"""
        try:
            # Map tool names to their async functions
            # The @mcp.tool() decorator registers these, but we can still call them directly
            tool_functions = {
                "search": search,
            }

            if request.name not in tool_functions:
                raise HTTPException(
                    status_code=404,
                    detail=f"Tool {request.name} not found. Available: {list(tool_functions.keys())}",
                )

            # Call the tool function with arguments
            # The @mcp.tool() decorator wraps functions in FunctionTool objects
            # Access the underlying function via .fn attribute
            tool_obj = tool_functions[request.name]
            if hasattr(tool_obj, "fn"):
                # It's a FunctionTool, get the underlying async function
                result = await tool_obj.fn(**request.arguments)
            else:
                # It's a regular function
                result = await tool_obj(**request.arguments)

            # Return result in consistent format
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
        return {"status": "healthy", "server": "Tavily MCP"}

    # Run FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=port)
