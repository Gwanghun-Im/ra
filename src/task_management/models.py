"""Task Management Extensions - Additional metadata for Redis tracking"""

from enum import Enum


class TaskType(str, Enum):
    """Type of task based on domain"""

    PORTFOLIO_ANALYSIS = "portfolio_analysis"
    INVESTMENT_ADVICE = "investment_advice"
    RISK_ASSESSMENT = "risk_assessment"
    MARKET_RESEARCH = "market_research"
    GENERAL = "general"


class TaskPriority(int, Enum):
    """Task priority levels (lower number = higher priority)"""

    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
