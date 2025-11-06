# A2A (Agent-to-Agent) 아키텍처 가이드

## 개요

이 프로젝트는 A2A (Agent-to-Agent) 프로토콜을 사용하여 여러 AI 에이전트가 서로 통신하고 협력할 수 있도록 설계되었습니다.

## 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI                             │
│                   (사용자 인터페이스)                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Supervisor Agent                            │
│              (오케스트레이터 에이전트)                          │
│                                                               │
│  - 태스크 분류                                                │
│  - A2A 클라이언트를 통한 라우팅                                │
│  - 응답 통합                                                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ A2A Protocol
                         │ (JSON-RPC 2.0)
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Robo Advisor Agent (A2A Server)                 │
│                                                               │
│  Endpoints:                                                   │
│  - /.well-known/agent.json (Agent Card)                      │
│  - /a2a (A2A 태스크 엔드포인트)                               │
│  - /health (헬스 체크)                                        │
│  - /capabilities (능력 조회)                                  │
│                                                               │
│  Capabilities:                                                │
│  - portfolio_analysis                                         │
│  - investment_recommendation                                  │
│  - risk_assessment                                            │
│  - market_research                                            │
│  - performance_calculation                                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    MCP Servers                                │
│                                                               │
│  - Market Data Server (8001)                                  │
│  - Portfolio Server (8002)                                    │
│  - Tavily Search Server (8003)                                │
│  - RAG Server (8004)                                          │
└─────────────────────────────────────────────────────────────┘
```

## 주요 컴포넌트

### 1. Supervisor Agent
- **위치**: `src/agents/supervisor_agent.py`
- **역할**: 메인 오케스트레이터
- **기능**:
  - 사용자 요청 분류
  - A2A 클라이언트를 통해 적절한 에이전트로 라우팅
  - 에이전트 discovery 관리
  - 응답 통합 및 포맷팅

### 2. Robo Advisor Agent (A2A Server)
- **위치**: `src/a2a/server.py`
- **역할**: 투자 자문 전문 에이전트
- **프로토콜**: A2A (JSON-RPC 2.0)
- **포트**: 8100

#### Agent Card (/.well-known/agent.json)
Agent Card는 에이전트의 메타데이터와 능력을 설명하는 JSON 문서입니다:

```json
{
  "schema_version": "1.0",
  "name": "Robo Advisor Agent",
  "version": "1.0.0",
  "description": "AI-powered investment advisory and portfolio analysis agent",
  "service_url": "http://localhost:8100",
  "a2a_endpoint": "http://localhost:8100/a2a",
  "capabilities": [...],
  "modalities": ["text", "structured_data"],
  "supported_methods": ["task.create", "task.status", "task.cancel"]
}
```

### 3. A2A Client
- **위치**: `src/a2a/client.py`
- **역할**: A2A 프로토콜 클라이언트
- **기능**:
  - 에이전트 discovery
  - 태스크 전송
  - 태스크 상태 조회

## A2A 프로토콜

### 태스크 생성 (task.create)

```json
{
  "jsonrpc": "2.0",
  "id": "task-123",
  "method": "task.create",
  "params": {
    "message": {
      "role": "user",
      "content": "내 포트폴리오를 분석해주세요",
      "parts": [
        {
          "type": "text",
          "text": "내 포트폴리오를 분석해주세요"
        }
      ]
    },
    "context": {
      "user_id": "user123",
      "task_type": "portfolio_analysis"
    }
  }
}
```

### 응답 형식

```json
{
  "jsonrpc": "2.0",
  "id": "task-123",
  "result": {
    "task_id": "task-123",
    "status": "completed",
    "message": {
      "role": "assistant",
      "content": "포트폴리오 분석 결과...",
      "parts": [
        {
          "type": "text",
          "text": "포트폴리오 분석 결과..."
        }
      ]
    },
    "artifacts": [
      {
        "type": "analysis",
        "data": {...}
      }
    ]
  }
}
```

## 설정

### A2A 설정 파일 (config/a2a_config.yaml)

```yaml
a2a:
  agents:
    robo_advisor:
      name: "Robo Advisor Agent"
      url: "http://localhost:8100"
      card_url: "http://localhost:8100/.well-known/agent.json"
      a2a_endpoint: "http://localhost:8100/a2a"
      enabled: true

  communication:
    timeout: 60
    retry_attempts: 3
    retry_delay: 2

  task_management:
    max_concurrent_tasks: 10
    task_timeout: 300
    enable_async: true

  discovery:
    auto_discover: true
    discovery_interval: 300
    cache_agent_cards: true
```

## 사용 방법

### 1. Docker Compose로 실행

```bash
# 모든 서비스 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f a2a_robo_advisor

# 서비스 상태 확인
docker-compose ps

# Agent Card 확인
curl http://localhost:8100/.well-known/agent.json

# 헬스 체크
curl http://localhost:8100/health
```

### 2. Streamlit 앱 실행

```bash
# 로컬에서 실행
cd src/streamlit_app
streamlit run app.py
```

### 3. Python으로 직접 사용

```python
import asyncio
from src.agents.supervisor_agent import SupervisorAgent

async def main():
    supervisor = SupervisorAgent()

    # 요청 처리
    result = await supervisor.process_request(
        user_message="애플 주식의 현재 가격을 알려주세요",
        user_id="user123"
    )

    print(result["response"])
    print(f"Delegated to: {result['delegated_to']}")

asyncio.run(main())
```

### 4. A2A 클라이언트로 직접 통신

```python
import asyncio
from src.a2a.client import A2AClient

async def main():
    client = A2AClient()

    # 에이전트 discovery
    agents = await client.discover_agents()
    print("Discovered agents:", agents)

    # 태스크 전송
    result = await client.send_task(
        agent_name="robo_advisor",
        message="내 포트폴리오 리스크를 분석해주세요",
        task_id="task-456",
        context={"user_id": "user123"}
    )

    print(result)

    await client.close()

asyncio.run(main())
```

## 새로운 A2A 에이전트 추가하기

### 1. 에이전트 서버 생성

```python
from fastapi import FastAPI
from src.a2a.server import A2ARequest

app = FastAPI()

@app.get("/.well-known/agent.json")
async def agent_card():
    return {
        "schema_version": "1.0",
        "name": "My Custom Agent",
        "version": "1.0.0",
        "description": "Custom agent description",
        "service_url": "http://localhost:8200",
        "a2a_endpoint": "http://localhost:8200/a2a",
        "capabilities": ["custom_capability"],
        "modalities": ["text"],
        "supported_methods": ["task.create", "task.status", "task.cancel"]
    }

@app.post("/a2a")
async def a2a_endpoint(request: A2ARequest):
    # A2A 요청 처리
    pass
```

### 2. 설정 파일에 추가

```yaml
a2a:
  agents:
    my_custom_agent:
      name: "My Custom Agent"
      url: "http://localhost:8200"
      card_url: "http://localhost:8200/.well-known/agent.json"
      a2a_endpoint: "http://localhost:8200/a2a"
      enabled: true
```

### 3. Supervisor에서 라우팅 로직 추가

```python
# supervisor_agent.py에서
if task_type == "custom_task":
    result = await self.a2a_client.send_task(
        agent_name="my_custom_agent",
        message=user_message,
        ...
    )
```

## 테스트

### Agent Card 확인
```bash
curl http://localhost:8100/.well-known/agent.json | jq
```

### A2A 엔드포인트 테스트
```bash
curl -X POST http://localhost:8100/a2a \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "test-1",
    "method": "task.create",
    "params": {
      "message": {
        "role": "user",
        "content": "테스트 메시지",
        "parts": [{"type": "text", "text": "테스트 메시지"}]
      },
      "context": {"user_id": "test"}
    }
  }' | jq
```

## 문제 해결

### 에이전트 discovery 실패
- Agent Card URL이 올바른지 확인
- 에이전트 서버가 실행 중인지 확인
- 네트워크 연결 확인

### 태스크 실행 실패
- 로그 확인: `docker-compose logs -f a2a_robo_advisor`
- MCP 서버들이 정상 동작하는지 확인
- API 키가 올바르게 설정되었는지 확인

### 타임아웃 오류
- `config/a2a_config.yaml`에서 timeout 값 조정
- MCP 서버 응답 시간 확인

## 참고 자료

- [A2A Protocol Specification](https://github.com/google/a2a)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MCP Protocol](https://modelcontextprotocol.io/)
