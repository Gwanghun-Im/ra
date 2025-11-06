# RA - Robo Advisor Agent System

MCP와 A2A 프로토콜을 활용한 멀티 에이전트 투자 자문 시스템

## 🌟 프로젝트 개요

**RA (Robo Advisor)** 는 최신 AI 에이전트 기술을 활용한 지능형 투자 자문 시스템입니다.

### 주요 특징

- **A2A (Agent-to-Agent) 아키텍처**: 에이전트 간 표준화된 통신 프로토콜
- **MCP (Model Context Protocol)**: 외부 데이터 소스와의 통합
- **LangGraph**: 복잡한 에이전트 워크플로우 관리
- **벡터 DB (FAISS)**: RAG 기반 지식 검색
- **Docker**: 마이크로서비스 아키텍처

### 에이전트 구성

- **Supervisor Agent**: A2A 클라이언트로 동작하며 사용자 요청을 분류하고 적절한 에이전트로 라우팅
- **Robo Advisor Agent**: A2A 서버로 동작하며 투자 자문 및 포트폴리오 분석 수행
  - 포트폴리오 분석
  - 투자 추천
  - 리스크 평가
  - 시장 조사
  - 수익률 계산

## 기술 스택

- Python 3.12+
- uv (패키지 매니저)
- LangGraph ≥ 0.6.2
- FastMCP ≥ 2.11.0
- a2a-sdk ≥ 0.3.0
- FAISS (Vector DB)
- Docker & Docker Compose
- Streamlit (A2A Client UI)

## 📁 프로젝트 구조

```
ra/
├── src/
│   ├── agents/                      # AI 에이전트 구현
│   │   ├── supervisor_agent.py     # A2A 클라이언트 - 메인 오케스트레이터
│   │   └── robo_advisor_agent.py   # 투자 자문 에이전트 (LangGraph)
│   ├── a2a/                         # A2A 프로토콜 구현
│   │   ├── client.py               # A2A 클라이언트 (Supervisor용)
│   │   └── server.py               # A2A 서버 (Robo Advisor 노출)
│   ├── mcp_custom/                  # MCP 서버 및 도구
│   │   ├── servers/
│   │   │   ├── market_data_server.py   # 주식 시장 데이터
│   │   │   ├── portfolio_server.py     # 포트폴리오 관리
│   │   │   ├── tavily_server.py        # 웹 검색
│   │   │   └── rag_server.py           # RAG 지식 베이스
│   │   └── tools/
│   │       └── mcp_tools.py            # MCP 도구 래퍼
│   └── streamlit_app/               # Streamlit UI (A2A 클라이언트)
│       └── app.py
├── vector_db/                       # FAISS 벡터 DB
│   └── faiss_manager.py
├── config/
│   └── a2a_config.yaml             # A2A 에이전트 설정
├── docker/
│   ├── Dockerfile.a2a_agent        # A2A 서버 컨테이너
│   └── mcp_servers/                # MCP 서버 컨테이너들
├── docs/
│   └── A2A_GUIDE.md                # A2A 아키텍처 가이드
├── tests/
│   └── test_a2a.py                 # A2A 통신 테스트
├── docker-compose.yml               # 모든 서비스 오케스트레이션
└── main.py                          # CLI 엔트리포인트
```

## 🚀 설치 및 실행

### 1. 환경 변수 설정

`.env` 파일 생성:

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
REDIS_URL=redis://localhost:6379
```

### 2. UV 설치 (선택사항 - 로컬 개발용)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Docker Compose로 모든 서비스 실행

```bash
# 모든 서비스 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 특정 서비스 로그
docker-compose logs -f a2a_robo_advisor
```

### 4. A2A 에이전트 확인

```bash
# Agent Card 확인
curl http://localhost:8100/.well-known/agent.json | jq

# 헬스 체크
curl http://localhost:8100/health

# 에이전트 능력 조회
curl http://localhost:8100/capabilities
```

### 5. Streamlit UI 실행 (로컬)

```bash
cd src/streamlit_app
streamlit run app.py
```

또는:

```bash
uv run streamlit run src/streamlit_app/app.py
```

### 6. A2A 통신 테스트

```bash
# Python으로 테스트 실행
python tests/test_a2a.py

# 또는 uv로 실행
uv run python tests/test_a2a.py
```

## 📚 사용 예시

### Python에서 Supervisor 사용

```python
import asyncio
from src.agents.supervisor_agent import SupervisorAgent

async def main():
    supervisor = SupervisorAgent()

    result = await supervisor.process_request(
        user_message="애플 주식의 현재 가격을 알려주세요",
        user_id="user123"
    )

    print(result["response"])
    print(f"처리 에이전트: {result['delegated_to']}")

asyncio.run(main())
```

### A2A 클라이언트로 직접 통신

```python
import asyncio
from src.a2a.client import A2AClient

async def main():
    client = A2AClient()

    # 에이전트 발견
    agents = await client.discover_agents()
    print(f"발견된 에이전트: {len(agents)}개")

    # 태스크 전송
    result = await client.send_task(
        agent_name="robo_advisor",
        message="내 포트폴리오를 분석해주세요",
        task_id="task-123",
        context={"user_id": "user123"}
    )

    print(result)
    await client.close()

asyncio.run(main())
```

## 🏗️ A2A 아키텍처

자세한 A2A 아키텍처 가이드는 [docs/A2A_GUIDE.md](docs/A2A_GUIDE.md)를 참고하세요.

### A2A 통신 흐름

```
사용자 → Streamlit UI → Supervisor Agent (A2A Client)
                              ↓
                        A2A Protocol (JSON-RPC 2.0)
                              ↓
                   Robo Advisor Agent (A2A Server)
                              ↓
                        MCP Servers
                   (Market Data, Portfolio, RAG, Tavily)
```

## 🐳 Docker 서비스

| 서비스               | 포트     | 설명                  |
| -------------------- | -------- | --------------------- |
| Redis                | 6379     | 데이터 캐싱           |
| MCP Market Data      | 8001     | 주식 시장 데이터 서버 |
| MCP Portfolio        | 8002     | 포트폴리오 관리 서버  |
| MCP Tavily           | 8003     | 웹 검색 서버          |
| MCP RAG              | 8004     | RAG 지식 베이스 서버  |
| **A2A Robo Advisor** | **8100** | **A2A 에이전트 서버** |

## 🧪 테스트

```bash
# A2A 통신 테스트
python tests/test_a2a.py

# 서비스 헬스 체크
curl http://localhost:8100/health
curl http://localhost:8001/health
curl http://localhost:8002/health
```

## 📖 추가 문서

- [A2A 아키텍처 가이드](docs/A2A_GUIDE.md) - A2A 프로토콜 상세 설명
- [프로젝트 구조](PROJECT_STRUCTURE.md) - 상세 프로젝트 구조
- [빠른 시작](QUICKSTART.md) - 빠른 시작 가이드

## 라이센스

MIT
