# A2A 아키텍처 전환 변경 로그

## 개요

이 문서는 기존 에이전트 시스템을 A2A (Agent-to-Agent) 아키텍처로 전환한 주요 변경 사항을 설명합니다.

## 날짜

2025-11-06

## 주요 변경 사항

### 1. Supervisor Agent → A2A 클라이언트로 전환

**파일**: `src/agents/supervisor_agent.py`

#### 변경 내용:
- ✅ A2A 클라이언트 통합
- ✅ Robo Advisor와의 직접 통신을 A2A 프로토콜로 대체
- ✅ 에이전트 discovery 메커니즘 추가
- ✅ 비동기 에이전트 통신 구현

#### 변경된 메서드:
```python
# 이전
def __init__(self):
    self.robo_advisor = RoboAdvisorAgent()  # 직접 인스턴스화

# 이후
def __init__(self):
    self.a2a_client = A2AClient()  # A2A 클라이언트 사용
```

```python
# 이전
async def _route_task(self, state):
    result = await self.robo_advisor.process_request(user_message, user_id)

# 이후
async def _route_task(self, state):
    result = await self.a2a_client.send_task(
        agent_name="robo_advisor",
        message=user_message,
        task_id=f"task_{user_id}_{task_type}",
        context={"user_id": user_id, "task_type": task_type}
    )
```

```python
# 이전
def get_available_agents(self):
    return [...]  # 하드코딩된 리스트

# 이후
async def get_available_agents(self):
    discovered_agents = await self.a2a_client.discover_agents()
    # 동적으로 에이전트 발견
```

### 2. A2A 서버 개선

**파일**: `src/a2a/server.py`

#### 변경 내용:
- ✅ 강화된 Agent Card (/.well-known/agent.json)
- ✅ A2A 프로토콜 표준 준수
- ✅ 추가 메타데이터 및 기능 정보
- ✅ 헬스 체크 및 capability 엔드포인트

#### Agent Card 개선:
```json
{
  "schema_version": "1.0",
  "name": "Robo Advisor Agent",
  "version": "1.0.0",
  "description": "AI-powered investment advisory and portfolio analysis agent using LangGraph and MCP servers",
  "service_url": "http://localhost:8100",
  "a2a_endpoint": "http://localhost:8100/a2a",
  "capabilities": [...],
  "supported_methods": ["task.create", "task.status", "task.cancel"],
  "features": {
    "streaming": false,
    "artifacts": true,
    "context_awareness": true,
    "multi_turn": true
  },
  "authentication": {...},
  "rate_limits": {...},
  "metadata": {...}
}
```

### 3. A2A 설정 강화

**파일**: `config/a2a_config.yaml`

#### 추가된 설정:
- ✅ 통신 설정 (timeout, retry, connection pool)
- ✅ 태스크 관리 설정
- ✅ Discovery 설정
- ✅ 로깅 설정

```yaml
a2a:
  agents:
    robo_advisor:
      # 에이전트 메타데이터

  communication:
    timeout: 60
    retry_attempts: 3
    retry_delay: 2
    connection_pool_size: 10

  task_management:
    max_concurrent_tasks: 10
    task_timeout: 300
    enable_async: true
    task_queue_size: 100

  discovery:
    auto_discover: true
    discovery_interval: 300
    cache_agent_cards: true

  logging:
    level: "INFO"
    log_a2a_requests: true
    log_a2a_responses: true
```

### 4. Streamlit UI 업데이트

**파일**: `src/streamlit_app/app.py`

#### 변경 내용:
- ✅ 비동기 에이전트 discovery 호출
- ✅ A2A 에이전트 정보 표시 개선
- ✅ Service URL 표시 추가

```python
# 이전
agents = st.session_state.supervisor.get_available_agents()

# 이후
agents = run_async(st.session_state.supervisor.get_available_agents())
```

### 5. Docker Compose 개선

**파일**: `docker-compose.yml`

#### 변경 내용:
- ✅ 모든 MCP 서버에 대한 환경 변수 추가
- ✅ 헬스 체크 추가
- ✅ 서비스 간 의존성 명확화

```yaml
a2a_robo_advisor:
  environment:
    - MCP_MARKET_DATA_URL=http://mcp_market_data:8001
    - MCP_PORTFOLIO_URL=http://mcp_portfolio:8002
    - MCP_TAVILY_URL=http://mcp_tavily:8003
    - MCP_RAG_URL=http://mcp_rag:8004
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8100/health"]
    interval: 30s
    timeout: 10s
    retries: 3
```

### 6. 새로운 문서 및 테스트

#### 추가된 파일:
- ✅ `docs/A2A_GUIDE.md` - 상세한 A2A 아키텍처 가이드
- ✅ `tests/test_a2a.py` - A2A 통신 테스트 스위트
- ✅ `scripts/test_system.sh` - 시스템 통합 테스트 스크립트
- ✅ `CHANGELOG_A2A.md` - 이 변경 로그

## 아키텍처 비교

### 이전 아키텍처 (직접 통신)

```
Streamlit UI
    ↓
Supervisor Agent
    ↓ (직접 메서드 호출)
Robo Advisor Agent
    ↓
MCP Servers
```

### 새로운 A2A 아키텍처

```
Streamlit UI
    ↓
Supervisor Agent (A2A Client)
    ↓ (A2A Protocol - JSON-RPC 2.0)
Robo Advisor Agent (A2A Server)
    ↓
MCP Servers
```

## 장점

### 1. **느슨한 결합 (Loose Coupling)**
- Supervisor는 더 이상 Robo Advisor의 구현 세부 사항을 알 필요가 없음
- 에이전트를 독립적으로 배포하고 확장 가능

### 2. **표준화된 통신**
- JSON-RPC 2.0 기반 A2A 프로토콜
- 다른 A2A 호환 에이전트와 쉽게 통합 가능

### 3. **동적 Discovery**
- 에이전트를 런타임에 발견
- Agent Card를 통한 메타데이터 제공

### 4. **확장성**
- 새로운 A2A 에이전트를 쉽게 추가
- 마이크로서비스 아키텍처로 확장 가능

### 5. **에러 처리**
- 네트워크 장애에 대한 견고한 처리
- Retry 및 timeout 메커니즘

### 6. **관찰성 (Observability)**
- 각 A2A 요청/응답 로깅
- 서비스별 독립적인 모니터링

## 마이그레이션 가이드

### 기존 코드를 A2A로 마이그레이션하려면:

#### 1. 직접 에이전트 호출을 A2A 호출로 변경

```python
# Before
result = await self.robo_advisor.process_request(message, user_id)

# After
result = await self.a2a_client.send_task(
    agent_name="robo_advisor",
    message=message,
    context={"user_id": user_id}
)
```

#### 2. Agent Card 구현

각 에이전트에 `/.well-known/agent.json` 엔드포인트 추가

#### 3. A2A 설정 추가

`config/a2a_config.yaml`에 에이전트 정보 추가

#### 4. Docker 서비스 설정

`docker-compose.yml`에 서비스 추가 및 네트워크 설정

## 테스트 방법

### 1. 전체 시스템 테스트
```bash
./scripts/test_system.sh
```

### 2. Python A2A 테스트
```bash
python tests/test_a2a.py
```

### 3. 개별 엔드포인트 테스트
```bash
# Agent Card
curl http://localhost:8100/.well-known/agent.json

# Health Check
curl http://localhost:8100/health

# Capabilities
curl http://localhost:8100/capabilities

# A2A Task
curl -X POST http://localhost:8100/a2a \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"1","method":"task.create","params":{...}}'
```

## 호환성

- ✅ 기존 Streamlit UI와 호환
- ✅ 기존 MCP 서버와 호환
- ✅ 기존 LangGraph 워크플로우와 호환
- ✅ A2A 프로토콜 표준 준수

## 향후 개선 사항

1. **인증 및 권한 부여**
   - OAuth2 통합
   - API 키 관리

2. **스트리밍 지원**
   - 실시간 응답 스트리밍
   - Server-Sent Events (SSE)

3. **추가 에이전트**
   - 추가 전문 에이전트 구현
   - 에이전트 간 협업 패턴

4. **모니터링 및 메트릭**
   - Prometheus 메트릭
   - 분산 추적 (Distributed Tracing)

5. **에이전트 오케스트레이션**
   - 복잡한 멀티 에이전트 워크플로우
   - 에이전트 체이닝 및 병렬 실행

## 참고 자료

- [A2A Protocol Specification](https://github.com/google/a2a)
- [docs/A2A_GUIDE.md](docs/A2A_GUIDE.md)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)

## 기여자

- AI Assistant (Claude)
- 프로젝트 팀

## 라이센스

MIT
