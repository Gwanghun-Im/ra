# A2A Agents Unified Dockerfile

This unified Dockerfile consolidates all A2A agent builds into a single, parameterized Dockerfile.

## Overview

Previously, there were 4 separate Dockerfiles:
- `Dockerfile.a2a_supervisor`
- `Dockerfile.a2a_ra_agent`
- `Dockerfile.a2a_rag_agent`
- `Dockerfile.a2a_general_agent`

Now they are unified into one `docker/a2a_agents/Dockerfile` that uses build arguments to differentiate between agent types.

## Agent Types

| Agent Type | Port | Module |
|------------|------|--------|
| `supervisor` | 8099 | `a2a_agents.supervisor_server` |
| `ra_agent` | 8100 | `a2a_agents.ra_agent_server` |
| `rag_agent` | 8101 | `a2a_agents.rag_agent_server` |
| `general_agent` | 8102 | `a2a_agents.general_agent_server` |

## Usage with Docker Compose (Recommended)

The root `docker-compose.yml` has been updated to use this unified Dockerfile:

```bash
# Build and start all services
docker-compose up --build

# Start specific agent
docker-compose up a2a_supervisor

# Rebuild specific agent
docker-compose build a2a_supervisor
docker-compose up -d a2a_supervisor
```

## Manual Docker Build

If you need to build images manually:

### Build Supervisor Agent
```bash
docker build \
  --build-arg AGENT_TYPE=supervisor \
  --build-arg PORT=8099 \
  -t a2a-supervisor:latest \
  -f docker/a2a_agents/Dockerfile \
  .
```

### Build Robo Advisor Agent
```bash
docker build \
  --build-arg AGENT_TYPE=ra_agent \
  --build-arg PORT=8100 \
  -t a2a-ra-agent:latest \
  -f docker/a2a_agents/Dockerfile \
  .
```

### Build RAG Agent
```bash
docker build \
  --build-arg AGENT_TYPE=rag_agent \
  --build-arg PORT=8101 \
  -t a2a-rag-agent:latest \
  -f docker/a2a_agents/Dockerfile \
  .
```

### Build General Agent
```bash
docker build \
  --build-arg AGENT_TYPE=general_agent \
  --build-arg PORT=8102 \
  -t a2a-general-agent:latest \
  -f docker/a2a_agents/Dockerfile \
  .
```

## Running Containers Manually

### Run Supervisor Agent
```bash
docker run -d \
  --name a2a-supervisor \
  -p 8099:8099 \
  --env-file .env \
  -e IS_DOCKER=true \
  a2a-supervisor:latest
```

### Run Robo Advisor Agent
```bash
docker run -d \
  --name a2a-ra-agent \
  -p 8100:8100 \
  --env-file .env \
  -e IS_DOCKER=true \
  a2a-ra-agent:latest
```

## Environment Variables

Required environment variables (set in `.env` file):
- `OPENAI_API_KEY` - OpenAI API key
- `TAVILY_API_KEY` - Tavily API key (for RA agent)
- `REDIS_URL` - Redis connection URL
- `IS_DOCKER` - Set to `true` when running in Docker

## Benefits of Unified Dockerfile

1. **Reduced Duplication**: Single source of truth for all A2A agents
2. **Easy Maintenance**: Update dependencies in one place
3. **Consistent Builds**: All agents use the same base image and dependencies
4. **Flexible Configuration**: Use build args to customize per agent
5. **Smaller Repository**: Less file clutter

## Migration from Old Dockerfiles

The old Dockerfiles can now be safely deleted:
- `docker/Dockerfile.a2a_supervisor`
- `docker/Dockerfile.a2a_ra_agent`
- `docker/Dockerfile.a2a_rag_agent`
- `docker/Dockerfile.a2a_general_agent`

All functionality is preserved in the new unified Dockerfile.
