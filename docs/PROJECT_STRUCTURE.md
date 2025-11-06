# RA Project Structure

```
RA/
├── README.md                          # 프로젝트 문서
├── pyproject.toml                     # UV 패키지 설정
├── docker-compose.yml                 # Docker 서비스 구성
├── setup.sh                           # 설치 스크립트
├── main.py                            # CLI 실행 파일
├── .env.example                       # 환경 변수 템플릿
│
├── agents/                            # 에이전트 구현
│   ├── __init__.py
│   ├── supervisor_agent.py           # 메인 오케스트레이터
│   └── robo_advisor_agent.py         # 투자 자문 에이전트
│
├── mcp/                               # MCP 통합
│   ├── servers/                      # MCP 서버들
│   │   ├── market_data_server.py    # 시장 데이터 도구
│   │   └── portfolio_server.py      # 포트폴리오 관리 도구
│   │
│   └── tools/                        # MCP 도구 래퍼
│       ├── __init__.py
│       └── mcp_tools.py              # LangChain 도구 래퍼
│
├── a2a/                               # A2A 통신 레이어
│   ├── __init__.py
│   ├── client.py                     # A2A 클라이언트
│   ├── server.py                     # A2A 서버
│   └── agent_cards/                  # Agent Cards (JSON)
│
├── vector_db/                         # FAISS 벡터 DB
│   ├── __init__.py
│   ├── faiss_manager.py              # FAISS 관리자
│   └── embeddings/                   # 임베딩 저장소
│
├── streamlit_app/                     # Streamlit UI
│   ├── __init__.py
│   └── app.py                        # 메인 UI
│
├── config/                            # 설정 파일
│   ├── mcp_config.yaml               # MCP 설정
│   └── a2a_config.yaml               # A2A 설정
│
├── docker/                            # Docker 파일
│   ├── Dockerfile.a2a_agent          # A2A 에이전트
│   └── mcp_servers/
│       ├── Dockerfile.market_data    # Market Data 서버
│       └── Dockerfile.portfolio      # Portfolio 서버
│
├── tests/                             # 테스트
│   ├── __init__.py
│   ├── test_agents/
│   ├── test_mcp/
│   └── test_a2a/
│
└── logs/                              # 로그 파일
    └── *.log
```

## 주요 구성 요소

### 1. Agents (에이전트)
- **supervisor_agent.py**: 전체 시스템 조율, 태스크 분류 및 라우팅
- **robo_advisor_agent.py**: 투자 자문 및 포트폴리오 분석

### 2. MCP (Model Context Protocol)
- **market_data_server.py**: 주가, 뉴스, 재무제표 조회
- **portfolio_server.py**: 포트폴리오 관리, 수익률 계산, 리스크 분석

### 3. A2A (Agent-to-Agent Protocol)
- **client.py**: 원격 에이전트 디스커버리 및 태스크 전송
- **server.py**: A2A 서버, Agent Card 제공

### 4. Vector DB (FAISS)
- **faiss_manager.py**: 문서 임베딩 저장 및 검색

### 5. UI (Streamlit)
- **app.py**: 대화형 웹 인터페이스

## 데이터 흐름

```
User Input (Streamlit/CLI)
    ↓
Supervisor Agent (LangGraph)
    ├── Task Classification
    ├── Routing Decision
    └── Response Finalization
    ↓
Robo Advisor Agent (LangGraph)
    ├── Tool Selection
    ├── MCP Tool Calls
    │   ├── Market Data Server
    │   └── Portfolio Server
    └── Analysis & Recommendation
    ↓
Response to User
```

## 프로토콜 사용

- **MCP**: Tools/Data access (Agent ↔ External Systems)
- **A2A**: Agent collaboration (Agent ↔ Agent)
- **LangGraph**: Internal workflow orchestration
