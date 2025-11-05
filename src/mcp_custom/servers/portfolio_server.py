"""MCP Server for Portfolio Management"""

from typing import Any, Dict
from datetime import datetime
import json
from fastmcp import FastMCP
import redis.asyncio as redis
import os

# Initialize MCP server
mcp = FastMCP("Portfolio Server")

# Redis client for portfolio storage
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = None


async def get_redis():
    """Get or create Redis client"""
    global redis_client
    if redis_client is None:
        redis_client = await redis.from_url(REDIS_URL, decode_responses=True)
    return redis_client


@mcp.tool()
async def get_portfolio(user_id: str = "default") -> Dict[str, Any]:
    """
    Get user's portfolio

    Args:
        user_id: User identifier

    Returns:
        Portfolio with positions and total value
    """
    r = await get_redis()

    portfolio_key = f"portfolio:{user_id}"
    portfolio_data = await r.get(portfolio_key)

    if portfolio_data:
        portfolio = json.loads(portfolio_data)
    else:
        # Initialize empty portfolio
        portfolio = {
            "user_id": user_id,
            "positions": [],
            "cash": 100000.0,  # Starting cash
            "total_value": 100000.0,
            "last_updated": datetime.now().isoformat(),
        }
        await r.set(portfolio_key, json.dumps(portfolio))

    return portfolio


@mcp.tool()
async def add_position(
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
    r = await get_redis()
    portfolio = await get_portfolio(user_id)

    # Calculate cost
    total_cost = shares * purchase_price

    if total_cost > portfolio["cash"]:
        return {
            "success": False,
            "error": "Insufficient cash balance",
            "required": total_cost,
            "available": portfolio["cash"],
        }

    # Check if position already exists
    existing_position = None
    for pos in portfolio["positions"]:
        if pos["symbol"] == symbol.upper():
            existing_position = pos
            break

    if existing_position:
        # Update existing position (average cost)
        total_shares = existing_position["shares"] + shares
        total_cost_basis = (
            existing_position["shares"] * existing_position["avg_price"] + shares * purchase_price
        )
        existing_position["shares"] = total_shares
        existing_position["avg_price"] = total_cost_basis / total_shares
    else:
        # Add new position
        portfolio["positions"].append(
            {
                "symbol": symbol.upper(),
                "shares": shares,
                "avg_price": purchase_price,
                "purchase_date": datetime.now().isoformat(),
            }
        )

    # Update cash and total value
    portfolio["cash"] -= total_cost
    portfolio["total_value"] = portfolio["cash"] + sum(
        pos["shares"] * pos["avg_price"] for pos in portfolio["positions"]
    )
    portfolio["last_updated"] = datetime.now().isoformat()

    # Save to Redis
    await r.set(f"portfolio:{user_id}", json.dumps(portfolio))

    return {
        "success": True,
        "portfolio": portfolio,
        "transaction": {
            "type": "BUY",
            "symbol": symbol.upper(),
            "shares": shares,
            "price": purchase_price,
            "total_cost": total_cost,
        },
    }


@mcp.tool()
async def remove_position(
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
        Updated portfolio
    """
    r = await get_redis()
    portfolio = await get_portfolio(user_id)

    # Find position
    position = None
    for pos in portfolio["positions"]:
        if pos["symbol"] == symbol.upper():
            position = pos
            break

    if not position:
        return {"success": False, "error": f"Position {symbol} not found in portfolio"}

    if position["shares"] < shares:
        return {
            "success": False,
            "error": f'Insufficient shares. Available: {position["shares"]}, Requested: {shares}',
        }

    # Calculate proceeds and gain/loss
    proceeds = shares * sell_price
    cost_basis = shares * position["avg_price"]
    gain_loss = proceeds - cost_basis

    # Update position
    position["shares"] -= shares

    # Remove position if all shares sold
    if position["shares"] == 0:
        portfolio["positions"].remove(position)

    # Update cash and total value
    portfolio["cash"] += proceeds
    portfolio["total_value"] = portfolio["cash"] + sum(
        pos["shares"] * pos["avg_price"] for pos in portfolio["positions"]
    )
    portfolio["last_updated"] = datetime.now().isoformat()

    # Save to Redis
    await r.set(f"portfolio:{user_id}", json.dumps(portfolio))

    return {
        "success": True,
        "portfolio": portfolio,
        "transaction": {
            "type": "SELL",
            "symbol": symbol.upper(),
            "shares": shares,
            "price": sell_price,
            "proceeds": proceeds,
            "cost_basis": cost_basis,
            "gain_loss": gain_loss,
            "gain_loss_percent": (gain_loss / cost_basis * 100) if cost_basis > 0 else 0,
        },
    }


@mcp.tool()
async def calculate_returns(user_id: str, current_prices: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculate portfolio returns

    Args:
        user_id: User identifier
        current_prices: Dictionary mapping symbols to current prices

    Returns:
        Portfolio performance metrics
    """
    portfolio = await get_portfolio(user_id)

    total_cost_basis = 100000.0  # Initial investment
    current_value = portfolio["cash"]

    position_returns = []
    for pos in portfolio["positions"]:
        symbol = pos["symbol"]
        current_price = current_prices.get(symbol, pos["avg_price"])

        position_value = pos["shares"] * current_price
        cost_basis = pos["shares"] * pos["avg_price"]
        gain_loss = position_value - cost_basis
        gain_loss_percent = (gain_loss / cost_basis * 100) if cost_basis > 0 else 0

        current_value += position_value

        position_returns.append(
            {
                "symbol": symbol,
                "shares": pos["shares"],
                "avg_price": pos["avg_price"],
                "current_price": current_price,
                "position_value": round(position_value, 2),
                "cost_basis": round(cost_basis, 2),
                "gain_loss": round(gain_loss, 2),
                "gain_loss_percent": round(gain_loss_percent, 2),
            }
        )

    total_return = current_value - total_cost_basis
    total_return_percent = total_return / total_cost_basis * 100

    return {
        "user_id": user_id,
        "initial_investment": total_cost_basis,
        "current_value": round(current_value, 2),
        "cash": round(portfolio["cash"], 2),
        "total_return": round(total_return, 2),
        "total_return_percent": round(total_return_percent, 2),
        "positions": position_returns,
        "calculated_at": datetime.now().isoformat(),
    }


@mcp.tool()
async def analyze_risk(user_id: str) -> Dict[str, Any]:
    """
    Analyze portfolio risk metrics

    Args:
        user_id: User identifier

    Returns:
        Risk analysis including diversification and concentration
    """
    portfolio = await get_portfolio(user_id)

    if not portfolio["positions"]:
        return {
            "risk_level": "LOW",
            "message": "Portfolio is 100% cash - no market risk",
            "diversification_score": 0,
            "recommendations": ["Consider investing in diversified assets"],
        }

    # Calculate concentration
    total_invested = sum(pos["shares"] * pos["avg_price"] for pos in portfolio["positions"])

    position_weights = []
    for pos in portfolio["positions"]:
        position_value = pos["shares"] * pos["avg_price"]
        weight = (position_value / total_invested * 100) if total_invested > 0 else 0
        position_weights.append({"symbol": pos["symbol"], "weight_percent": round(weight, 2)})

    # Simple risk scoring
    num_positions = len(portfolio["positions"])
    max_weight = max(pw["weight_percent"] for pw in position_weights) if position_weights else 0

    if num_positions >= 10 and max_weight < 20:
        risk_level = "LOW"
        diversification_score = 90
    elif num_positions >= 5 and max_weight < 30:
        risk_level = "MEDIUM"
        diversification_score = 70
    else:
        risk_level = "HIGH"
        diversification_score = 40

    recommendations = []
    if num_positions < 5:
        recommendations.append("Increase diversification by adding more positions")
    if max_weight > 30:
        recommendations.append(f"Reduce concentration in largest position")
    if portfolio["cash"] / portfolio["total_value"] < 0.05:
        recommendations.append("Maintain higher cash reserves for opportunities")

    return {
        "user_id": user_id,
        "risk_level": risk_level,
        "diversification_score": diversification_score,
        "num_positions": num_positions,
        "position_weights": position_weights,
        "cash_percent": round(portfolio["cash"] / portfolio["total_value"] * 100, 2),
        "recommendations": recommendations,
        "analyzed_at": datetime.now().isoformat(),
    }


# Run the server
if __name__ == "__main__":
    import os
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn

    port = int(os.getenv("MCP_SERVER_PORT", "8002"))

    # Create FastAPI app for HTTP API
    app = FastAPI(title="Portfolio MCP Server")

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
                "get_portfolio": get_portfolio,
                "add_position": add_position,
                "remove_position": remove_position,
                "calculate_returns": calculate_returns,
                "analyze_risk": analyze_risk,
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
            if hasattr(tool_obj, 'fn'):
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
        return {"status": "healthy", "server": "Portfolio MCP"}

    # Run FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=port)
