"""MCP Server for Market Data"""

import os
import asyncio
from datetime import datetime
from typing import Any, Dict, List
import yfinance as yf
import requests
import random
from fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

# FRED API 키 (https://fred.stlouisfed.org/api/api_key.html 에서 발급)
FRED_API_KEY = os.getenv("FRED_API_KEY", "YOUR_FRED_API_KEY_HERE")

# Initialize MCP server
mcp = FastMCP("Market Data Server")


@mcp.tool()
async def get_stock_price(symbol: str) -> Dict[str, Any]:
    """
    Get current stock price for a given symbol

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')

    Returns:
        Dictionary with current price, change, and timestamp
    """
    # Simulate API call delay
    await asyncio.sleep(0.5)

    # Mock data (in production, use real API like Alpha Vantage or Yahoo Finance)
    base_prices = {
        "AAPL": 175.50,
        "GOOGL": 140.25,
        "MSFT": 380.75,
        "TSLA": 245.30,
        "AMZN": 155.80,
        "NVDA": 495.60,
    }

    base_price = base_prices.get(symbol.upper(), 100.0)
    current_price = base_price * (1 + random.uniform(-0.05, 0.05))
    change = current_price - base_price
    change_percent = (change / base_price) * 100

    return {
        "symbol": symbol.upper(),
        "price": round(current_price, 2),
        "change": round(change, 2),
        "change_percent": round(change_percent, 2),
        "timestamp": datetime.now().isoformat(),
        "currency": "USD",
    }


@mcp.tool()
async def get_company_financials(symbol: str) -> Dict[str, Any]:
    """
    Get company financial metrics

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dictionary with key financial metrics
    """
    await asyncio.sleep(0.7)

    # Mock financial data
    return {
        "symbol": symbol.upper(),
        "market_cap": f"${random.randint(100, 3000)}B",
        "pe_ratio": round(random.uniform(15, 45), 2),
        "eps": round(random.uniform(2, 15), 2),
        "revenue": f"${random.randint(50, 500)}B",
        "profit_margin": f"{round(random.uniform(10, 30), 2)}%",
        "dividend_yield": f"{round(random.uniform(0, 4), 2)}%",
        "beta": round(random.uniform(0.8, 1.5), 2),
        "year_range": f"${random.randint(100, 150)}-${random.randint(200, 300)}",
        "last_updated": datetime.now().isoformat(),
    }


@mcp.tool()
async def search_stocks(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search for stocks by name or symbol

    Args:
        query: Search query string
        limit: Maximum number of results

    Returns:
        List of matching stocks with symbol, name, and exchange
    """
    await asyncio.sleep(0.4)

    # Mock search results
    all_stocks = [
        {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"},
        {"symbol": "GOOGL", "name": "Alphabet Inc.", "exchange": "NASDAQ"},
        {"symbol": "MSFT", "name": "Microsoft Corporation", "exchange": "NASDAQ"},
        {"symbol": "TSLA", "name": "Tesla, Inc.", "exchange": "NASDAQ"},
        {"symbol": "AMZN", "name": "Amazon.com Inc.", "exchange": "NASDAQ"},
        {"symbol": "NVDA", "name": "NVIDIA Corporation", "exchange": "NASDAQ"},
        {"symbol": "META", "name": "Meta Platforms Inc.", "exchange": "NASDAQ"},
        {"symbol": "BRK.B", "name": "Berkshire Hathaway Inc.", "exchange": "NYSE"},
    ]

    # Filter by query
    query_lower = query.lower()
    results = [
        stock
        for stock in all_stocks
        if query_lower in stock["symbol"].lower() or query_lower in stock["name"].lower()
    ]

    return results[:limit]


@mcp.tool(
    name="get_market_indicators",
    description="여러 마켓 지표를 한 번에 조회합니다.",
)
def get_market_indicators(indicators: list[str]) -> dict:
    """실시간 마켓 지표를 반환합니다."""
    result = {}
    mapping = {
        "vix": fetch_vix,
        "treasury_10y": fetch_treasury_10y,
        "usd_krw": fetch_usd_krw,
        "gold": fetch_gold,
        "crude_oil": fetch_crude_oil,
        "unemployment_rate": fetch_unemployment_rate,
        "gdp_growth": fetch_gdp_growth,
    }

    for ind in indicators:
        func = mapping.get(ind)
        if not func:
            result[ind] = {"error": "unknown indicator"}
            continue
        try:
            result[ind] = func()
        except Exception as e:
            result[ind] = {"error": str(e)}

    return {"timestamp": datetime.utcnow().isoformat() + "Z", "data": result}


@mcp.resource("market://status")
async def market_status() -> str:
    """Get current market status"""
    now = datetime.now()
    hour = now.hour

    # Simple market hours check (US Eastern Time approximation)
    if 9 <= hour < 16:
        status = "OPEN"
    else:
        status = "CLOSED"

    return f"Market Status: {status} at {now.strftime('%Y-%m-%d %H:%M:%S')}"


# ------------------------------------------------------------
# 실제 데이터 가져오는 함수들
# ------------------------------------------------------------
def fetch_vix() -> float:
    return yf.Ticker("^VIX").history(period="1d")["Close"].iloc[-1]


def fetch_treasury_10y() -> float:
    return yf.Ticker("^TNX").history(period="1d")["Close"].iloc[-1]


def fetch_usd_krw() -> float:
    return yf.Ticker("KRW=X").history(period="1d")["Close"].iloc[-1]


def fetch_gold() -> float:
    return yf.Ticker("GC=F").history(period="1d")["Close"].iloc[-1]


def fetch_crude_oil() -> float:
    return yf.Ticker("CL=F").history(period="1d")["Close"].iloc[-1]


def fetch_fred(series_id: str) -> float:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 1,
    }
    resp = requests.get(url, params=params).json()
    value = resp["observations"][0]["value"]
    return float(value) if value != "." else None


def fetch_unemployment_rate() -> float:
    return fetch_fred("UNRATE")  # 미국 실업률


def fetch_gdp_growth() -> float:
    return fetch_fred("A191RL1Q225SBEA")  # 전분기 대비 GDP 성장률


# Run the server
if __name__ == "__main__":
    import os
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn

    port = int(os.getenv("MCP_SERVER_PORT", "8001"))

    # Create FastAPI app for HTTP API
    app = FastAPI(title="Market Data MCP Server")

    class ToolCallRequest(BaseModel):
        name: str
        arguments: dict

    @app.post("/call-tool")
    async def call_tool_endpoint(request: ToolCallRequest):
        """HTTP endpoint to call MCP tools"""
        try:
            # Map tool names to their async functions
            # The @mcp.tool() decorator registers these, but we can still call them directly
            tool_functions = {
                "get_stock_price": get_stock_price,
                "get_company_financials": get_company_financials,
                "search_stocks": search_stocks,
                "get_market_indicators": get_market_indicators,
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
        return {"status": "healthy", "server": "Market Data MCP"}

    # Run FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=port)
