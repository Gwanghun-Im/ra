FROM python:3.12-slim

WORKDIR /app

# Build arguments for customization
ARG AGENT_TYPE=supervisor
ARG PORT=8099

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml /app/

# Install Python dependencies
RUN pip install --no-cache-dir uv && \
    uv pip install --system \
    langgraph>=0.6.2 \
    fastmcp>=2.11.0 \
    a2a-sdk>=0.3.0 \
    langchain>=0.3.0 \
    langchain-openai>=0.2.0 \
    fastapi>=0.115.0 \
    uvicorn>=0.30.0 \
    redis>=5.0.0 \
    httpx \
    python-dotenv \
    pyyaml

# Copy application files
COPY src/agents/ /app/src/agents/
COPY src/a2a_agents /app/src/a2a_agents
COPY src/mcp_custom/ /app/src/mcp_custom/
COPY config/ /app/config/

# Set Python path so modules can be imported
ENV PYTHONPATH=/app:/app/src
ENV AGENT_TYPE=${AGENT_TYPE}
ENV PORT=${PORT}

# Expose port
EXPOSE ${PORT}

# Run A2A server based on AGENT_TYPE
# Valid AGENT_TYPES: supervisor, ra_agent, rag_agent, general_agent
# Map AGENT_TYPE to actual server module names
# Using factory pattern with create_app()
CMD if [ "$AGENT_TYPE" = "supervisor" ]; then \
        python -m uvicorn src.a2a_agents.supervisor_server:create_app --factory --host 0.0.0.0 --port ${PORT}; \
    elif [ "$AGENT_TYPE" = "ra_agent" ]; then \
        python -m uvicorn src.a2a_agents.ra_server:create_app --factory --host 0.0.0.0 --port ${PORT}; \
    elif [ "$AGENT_TYPE" = "rag_agent" ]; then \
        python -m uvicorn src.a2a_agents.rag_server:create_app --factory --host 0.0.0.0 --port ${PORT}; \
    elif [ "$AGENT_TYPE" = "general_agent" ]; then \
        python -m uvicorn src.a2a_agents.general_server:create_app --factory --host 0.0.0.0 --port ${PORT}; \
    else \
        echo "Unknown AGENT_TYPE: $AGENT_TYPE"; \
        exit 1; \
    fi
