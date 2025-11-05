# RA (Robo Advisor) - 빠른 시작 가이드

## 📋 사전 요구사항

- Python 3.12 이상
- Docker & Docker Compose
- OpenAI API Key (또는 Anthropic API Key)

## 🚀 설치 및 실행

### 1. 프로젝트 클론 및 이동
```bash
cd RA
```

### 2. 환경 변수 설정
```bash
cp .env.example .env
```

`.env` 파일을 열어 API 키를 설정하세요:
```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### 3. 자동 설치 스크립트 실행
```bash
chmod +x setup.sh
./setup.sh
```

또는 수동 설치:

```bash
# UV 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync

# 디렉토리 생성
mkdir -p logs vector_db/embeddings

# Docker 서비스 시작
docker-compose up -d
```

### 4. 시스템 실행

#### Option A: CLI 인터페이스
```bash
uv run python main.py
```

#### Option B: Streamlit UI (권장)
```bash
uv run streamlit run streamlit_app/app.py
```

그런 다음 브라우저에서 http://localhost:8501 접속

#### Option C: A2A 서버만 실행
```bash
uv run python -m uvicorn a2a.server:app --host 0.0.0.0 --port 8100
```

## 💬 사용 예시

### CLI에서:
```
You: What's the current price of Apple stock?
Assistant: [Supervisor가 Robo Advisor에게 라우팅]
The current price of Apple (AAPL) is $175.50...

You: Analyze my portfolio
Assistant: [포트폴리오 분석 수행]
Your portfolio consists of...

You: agents
[사용 가능한 에이전트 목록 표시]
```

### Streamlit UI에서:
1. 좌측 사이드바에서 User ID 설정
2. 채팅 입력창에 질문 입력
3. Quick Actions 버튼 활용:
   - 📊 Analyze Portfolio
   - 💡 Get Recommendations
   - ⚠️ Risk Assessment

## 🧪 테스트

```bash
# 기본 테스트 실행
uv run pytest tests/test_basic.py -v

# 모든 테스트 실행
uv run pytest tests/ -v
```

## 🔧 문제 해결

### Docker 서비스 확인
```bash
docker-compose ps
docker-compose logs -f
```

### 특정 서비스 재시작
```bash
docker-compose restart mcp_market_data
docker-compose restart mcp_portfolio
```

### 로그 확인
```bash
tail -f logs/ra_system.log
```

## 📚 추가 정보

- MCP 서버:
  - Market Data: http://localhost:8001
  - Portfolio: http://localhost:8002

- A2A 서버:
  - Robo Advisor: http://localhost:8100
  - Agent Card: http://localhost:8100/.well-known/agent.json

- Redis:
  - 포트: 6379

## 🛑 종료

```bash
# Docker 서비스 중지
docker-compose down

# Docker 서비스 및 볼륨 삭제
docker-compose down -v
```

## 📖 더 알아보기

- [전체 문서](README.md)
- [프로젝트 구조](PROJECT_STRUCTURE.md)
- [MCP 문서](https://modelcontextprotocol.io)
- [A2A 프로토콜](https://a2aprotocol.ai)
