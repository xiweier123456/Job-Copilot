# Job Copilot

**AI-powered job search assistant for Chinese graduates, built on real recruitment data from publicly listed companies (2024–2026).**

Job Copilot combines RAG (Retrieval-Augmented Generation) with a multi-agent architecture to deliver semantic job search, resume matching, career path planning, and interview preparation — all grounded in ~93,600 real job postings.

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Milvus-2.4+-00A1EA?logo=milvus&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-0.2+-1C3C3C?logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/FastMCP-supported-cc785c?logo=anthropic&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

---

## Features

| Feature | Description |
|---------|-------------|
| Semantic Job Search | Natural language queries against 93K+ real JD embeddings via Milvus HNSW index |
| Resume Matching | Evaluate resume-to-JD fit with skill gap analysis and actionable improvement tips |
| Career Path Advisor | Personalized career direction recommendations based on academic background |
| Interview Prep | Role-specific questions, answer frameworks, and preparation checklists |
| Live Web Augmentation | Real-time search via Tavily API for current market signals, company pages, interview experiences, and community posts |
| Multi-Agent Orchestration | A main Deep Agent delegates to 4 specialized subagents: job search, resume, career, interview |
| Unified Tool Registry | `app/mcp/tool_registry.py` is the single source of truth for MCP tools, agent wrappers, subagent tool groups, and display metadata |
| Structured Tool Metadata | Chat responses and SSE events expose tool `display_name`, `description`, `category`, `requires_network`, `latency`, and `evidence_type` |
| Session Memory + History | Each chat turn is persisted with context, status, tool activity, and structured metadata for history replay |
| Split Model Strategy | DeepSeek handles RAG/service-side reasoning, while the deep-agent framework runs on Minimax |

## Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                        User Interfaces                       │
│   FastAPI REST + SSE chat      Vue frontend      MCP client │
└───────────────────────────────┬──────────────────────────────┘
                                │
                     ┌──────────▼──────────┐
                     │   Main Orchestrator │
                     │   (Deep Agent)      │
                     └───────┬─────┬───────┘
                             │     │
         ┌───────────────────┘     └───────────────────┐
         ▼                                             ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│ job-search     │ │ resume         │ │ career         │ │ interview      │
│ agent          │ │ agent          │ │ agent          │ │ agent          │
└───────┬────────┘ └───────┬────────┘ └───────┬────────┘ └───────┬────────┘
        │                  │                  │                  │
        └──────────────────┴──────────┬───────┴──────────────────┘
                                       ▼
                         ┌──────────────────────────────┐
                         │ Unified Tool Registry        │
                         │ search_jobs / Tavily tools   │
                         │ + metadata serialization     │
                         └──────────────┬───────────────┘
                                        ▼
                      ┌─────────────────────────────────────┐
                      │ RAG + Services                      │
                      │ Milvus retrieval + rerank + LLM     │
                      └─────────────────────────────────────┘
```

### Multi-Agent System

The core agent delegates job-related queries to 4 specialized subagents, each with its own prompt and tool set:

| Subagent | Responsibility | Tools |
|----------|---------------|-------|
| `job-search-agent` | Job retrieval, requirements analysis, market comparison | `search_jobs`, Tavily search/research/extract |
| `resume-agent` | Resume-JD matching, skill gap identification | `search_jobs`, Tavily search/research/extract |
| `career-agent` | Career direction recommendations, preparation planning | `search_jobs`, Tavily search/research |
| `interview-agent` | Interview preparation and real interview experiences | `search_jobs`, Tavily search/research/extract |

### RAG Pipeline

```text
CSV Data
→ Chunking (summary + sliding-window JD)
→ Embedding (bge-small-zh-v1.5)
→ Milvus HNSW Index
→ Semantic Retrieval with scalar filters (city, industry, education)
→ Cross-encoder rerank (bge-reranker-v2-m3)
→ LLM Generation with retrieved context
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web Framework | FastAPI + Uvicorn |
| Agent Framework | DeepAgents + LangGraph + LangChain |
| Vector Database | Milvus (HNSW index, cosine similarity) |
| Embeddings | BAAI/bge-small-zh-v1.5 |
| Reranker | BAAI/bge-reranker-v2-m3 (CrossEncoder) |
| LLM | DeepSeek for RAG/service-side flows, Minimax for DeepAgents runtime |
| MCP Server | FastMCP (stdio / HTTP / SSE transport) |
| Web Search | Tavily API |
| Validation | Pydantic v2 |

## Quick Start

### Prerequisites

- Python 3.10+
- Milvus running on `localhost:19530`
- DeepSeek-compatible service API key for service-side flows
- Minimax-compatible API key for agent runtime
- Optional: Tavily API key for web search
- Recommended for local reranker/tokenizer loading: `sentencepiece`

### 1. Install Dependencies

```bash
git clone https://github.com/your-username/job-copilot.git
cd job-copilot
pip install -r requirements.txt
pip install sentencepiece
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys and model paths:

```env
SERVICE_LLM_API_KEY=your-deepseek-api-key
SERVICE_LLM_BASE_URL=https://api.deepseek.com/v1
SERVICE_LLM_MODEL=deepseek-chat

AGENT_LLM_API_KEY=your-minimax-api-key
AGENT_LLM_BASE_URL=https://api.minimaxi.com/v1
AGENT_LLM_MODEL=MiniMax-M1

EMBEDDING_MODEL=E:/Study/model/bge-small-zh-v1.5
RERANKER_MODEL=E:/Study/model/BAAI-bge-reranker-v2-m3
ENABLE_RERANK=true

TAVILY_API_KEY=your-tavily-api-key
```

> If you use local model paths, make sure the embedding and reranker directories are complete and readable by the runtime.

### 3. Initialize Vector Database

```bash
python scripts/create_collection.py
python scripts/ingest_jobs.py --file data/上市公司招聘数据2026.csv --limit 1000
python scripts/ingest_jobs.py --file data/上市公司招聘数据2026.csv
```

### 4. Start the API Server

```bash
uvicorn app.main:app --reload --port 8001
```

API docs: **http://localhost:8001/docs**

### 5. Start the MCP Server (Optional)

```bash
python -m app.mcp.server
```

## API Reference

### Health Check

```http
GET /health
```

### Semantic Job Search

```http
GET /jobs/search?query=数据分析师&city=北京&top_k=5
```

### Resume Matching

```http
POST /resume/match
Content-Type: application/json

{
  "resume_text": "计算机科学硕士，熟悉 Python、SQL...",
  "job_query": "数据分析师",
  "city": "北京"
}
```

### Conversational Agent

```http
POST /chat
Content-Type: application/json

{
  "message": "数据分析师一般需要哪些技能？",
  "session_id": "user_001",
  "target_city": "北京",
  "job_direction": "数据分析",
  "user_profile": "统计学硕士，应届生，熟悉 Python 和 SQL",
  "resume_text": "...optional..."
}
```

**Returns:** reply text, used subagents, tool call summaries, structured `tool_calls`, source URLs, latency, and error info.

### Streaming Chat

```http
POST /chat/stream
Content-Type: application/json
```

SSE event types include:

- `status`
- `todo`
- `subagent`
- `tool`
- `final`
- `error`
- `stopped`

`tool` events and final payloads include structured metadata such as:

```json
{
  "name": "search_jobs_tool",
  "agent_name": "search_jobs_tool",
  "display_name": "岗位检索",
  "description": "从岗位数据库中检索代表性岗位样本，用于给出真实招聘证据。",
  "category": "job_db",
  "requires_network": false,
  "latency": "medium",
  "evidence_type": "job_postings",
  "status": "started"
}
```

### Chat History

```http
GET /chat/history?session_id=default
```

Returns expanded user/agent messages. Agent messages include:

- `meta.tool_calls`
- `activity.toolDetails`
- `activity.todos`
- `activity.subagents`

### Stop Streaming Run

```http
POST /chat/stop
Content-Type: application/json

{
  "run_id": "..."
}
```

### Clear Chat Session

```http
POST /chat/clear
Content-Type: application/json

{
  "session_id": "default"
}
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `search_jobs` | Semantic job search with city/industry filters and retrieval metadata |
| `tavily_search` | Fast web search for current market signals, articles, and interview experiences |
| `tavily_research` | Multi-source Tavily research summary with source list |
| `tavily_extract` | Extract clean content from one or more URLs |
| `batch_tavily_search` | Run multiple Tavily queries concurrently |

Tool metadata is centralized in [app/mcp/tool_registry.py](app/mcp/tool_registry.py).

## Project Structure

```text
job-copilot/
├── app/
│   ├── main.py                    # FastAPI entry point
│   ├── config.py                  # Pydantic settings
│   ├── api/
│   │   ├── health.py              # GET  /health
│   │   ├── jobs.py                # GET  /jobs/search
│   │   ├── resume.py              # POST /resume/match
│   │   └── chat.py                # /chat, /chat/stream, /chat/history, /chat/stop, /chat/clear
│   ├── agents/
│   │   ├── graph.py               # Deep agent orchestration, event normalization, final payload assembly
│   │   └── tools.py               # Compatibility exports for agent tool wrappers
│   ├── mcp/
│   │   ├── server.py              # FastMCP server entry
│   │   ├── tool_registry.py       # Unified tool registry + metadata serialization
│   │   └── tools/                 # Raw MCP tool implementations
│   ├── rag/
│   │   ├── chunker.py             # JD chunking
│   │   ├── embedder.py            # Local embedding model loader
│   │   ├── reranker.py            # CrossEncoder rerank
│   │   └── retriever.py           # Milvus retrieval + filters
│   ├── prompts/                   # Agent/system prompts
│   ├── schemas/                   # Pydantic schemas
│   └── services/                  # Business logic layer
├── scripts/
│   ├── create_collection.py       # Milvus collection setup
│   └── ingest_jobs.py             # CSV → embed → Milvus pipeline
├── data/                          # Recruitment datasets
├── outputs/                       # Generated memories / runtime artifacts
├── requirements.txt
└── .env.example
```

## Notes

- The frontend project used during development may live in a separate directory (for example `E:/code/demo_vue`).
- If `sentencepiece` is missing, local reranker/tokenizer loading may fail and retrieval will fall back to vector order.
- Structured tool metadata is exposed to the frontend through both SSE `tool` events and final `/chat` payloads.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Milvus](https://milvus.io/) — High-performance vector database
- [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) — Chinese text embedding model
- [DeepSeek](https://deepseek.com/) — LLM API provider
- [Tavily](https://tavily.com/) — AI-optimized web search API
- [FastMCP](https://github.com/jlowin/fastmcp) — Model Context Protocol framework
- [LangGraph](https://github.com/langchain-ai/langgraph) — Agent orchestration framework
