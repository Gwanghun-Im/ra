# RA - Robo Advisor Agent System

MCP와 A2A 프로토콜을 활용한 멀티 에이전트 투자 자문 시스템

## 프로젝트 개요

- **Supervisor Agent**: 전체 시스템을 조율하고 사용자 요청을 라우팅
- **Robo Advisor Agent**: 투자 자문 및 포트폴리오 분석 수행

## 기술 스택

- Python 3.12+
- uv (패키지 매니저)
- LangGraph ≥ 0.6.2
- FastMCP ≥ 2.11.0
- a2a-sdk ≥ 0.3.0
- FAISS (Vector DB)
- Docker & Docker Compose
- Streamlit (A2A Client UI)

## 프로젝트 구조

```
RA/
├── agents/                      # 에이전트 구현
│   ├── supervisor_agent.py     # 메인 오케스트레이터
│   └── robo_advisor_agent.py   # 투자 자문 에이전트
├── mcp/                         # MCP 서버 및 도구
│   └── servers/
│       ├── market_data_server.py
│       └── portfolio_server.py
├── a2a/                         # A2A 통신 레이어
│   ├── client.py
│   ├── server.py
│   └── agent_cards/
├── vector_db/                   # FAISS 벡터 DB
│   ├── faiss_manager.py
│   └── embeddings/
├── streamlit_app/               # Streamlit UI
│   └── app.py
├── config/                      # 설정 파일
├── docker/                      # Docker 구성
├── tests/                       # 테스트
└── main.py                      # 엔트리포인트
```

## 설치 및 실행

### 1. UV 설치
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 프로젝트 의존성 설치
```bash
uv sync
```

### 3. Docker 서비스 실행
```bash
docker-compose up -d
```

### 4. 에이전트 시스템 실행
```bash
uv run python main.py
```

### 5. Streamlit UI 실행
```bash
uv run streamlit run streamlit_app/app.py
```

## 환경 변수

`.env` 파일 생성:
```env
OPENAI_API_KEY=your_api_key
ANTHROPIC_API_KEY=your_api_key
REDIS_URL=redis://localhost:6379
```

## 라이센스

MIT