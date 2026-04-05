# Job Copilot 🤖

**AI-powered job search assistant for Chinese graduates, built on real recruitment data from publicly listed companies (2024–2026).**

Job Copilot combines RAG (Retrieval-Augmented Generation) with a multi-agent architecture to deliver semantic job search, resume matching, career path planning, and interview preparation — all grounded in ~93,600 real job postings.

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Milvus-2.4+-00A1EA?logo=milvus&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-0.2+-1C3C3C?logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/MCP-Claude_Desktop-cc785c?logo=anthropic&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Semantic Job Search** | Natural language queries against 93K+ real JD embeddings via Milvus HNSW index |
| **Resume Matching** | Evaluate resume-to-JD fit with skill gap analysis and actionable improvement tips |
| **Career Path Advisor** | Personalized career direction recommendations based on academic background |
| **Interview Prep** | Role-specific questions, answer frameworks, and preparation checklists |
| **Live Web Augmentation** | Real-time search via Tavily API — pulls job postings and interview experiences from 牛客网, 知乎, etc. |
| **MCP Integration** | Plug into Claude Desktop or any MCP-compatible client as a tool server |
| **Multi-Agent Orchestration** | 4 specialized subagents coordinated by a central orchestrator via DeepAgents + LangGraph |
| **Split Model Strategy** | DeepSeek handles RAG/service-side reasoning, while the deep-agent framework runs on Minimax |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interfaces                        │
│         FastAPI (/chat, /jobs, /resume)  │   MCP             │
└──────────────────────┬──────────────────────────────────────┘
                       │
              ┌────────▼────────┐
              │  Main Orchestrator  │
              │  (Deep Agent)    │
              └────────┬────────┘
        ┌──────┬───────┼───────┬──────┐
        ▼      ▼       ▼       ▼      ▼
   ┌────────┐┌────────┐┌────────┐┌────────┐
   │  Job   ││Resume  ││Career  ││Interview│
   │ Search ││ Match  ││  Path  ││  Prep  │
   │ Agent  ││ Agent  ││ Agent  ││ Agent  │
   └───┬────┘└───┬────┘└───┬────┘└───┬────┘
       │         │         │         │
  ┌────▼─────────▼─────────▼─────────▼────┐
  │           Shared Tool Layer           │
  │Milvus RAG+rerank│ LLM API │  Tavily   │
  └───────────────────────────────────────┘
```

### Multi-Agent System

The core agent delegates every job-related query to one of 4 specialized subagents, each with its own system prompt, tools, and output format:

| Subagent | Responsibility | Tools |
|----------|---------------|-------|
| `job-search-agent` | Job retrieval, requirements analysis, market comparison | `search_jobs`, Tavily |
| `resume-agent` | Resume-JD matching, skill gap identification | `search_jobs`, Tavily |
| `career-agent` | Career direction recommendations, preparation planning | `search_jobs`, Tavily |
| `interview-agent` | Interview preparation and real interview experiences | `search_jobs`, Tavily |

### RAG Pipeline

```
CSV Data → Chunking (summary + sliding-window JD) → Embedding (bge-small-zh-v1.5)
→ Milvus HNSW Index → Semantic Retrieval with scalar filters (city, industry, education) ->rerank ->LLM Self-Generation
→ LLM Generation with retrieved context
```

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Web Framework | **FastAPI** + Uvicorn |
| Agent Framework | **DeepAgents** + LangGraph + LangChain |
| Vector Database | **Milvus** (HNSW index, cosine similarity) |
| Embeddings | **BAAI/bge-small-zh-v1.5** (512-dim, local 
inference) |
| RERANK   | **BAAI-bge-reranker-v2-m3**|
| LLM | Split remote chat models: **DeepSeek** for RAG/service-side flows, **Minimax** for DeepAgents runtime |
| MCP Server | **FastMCP** (stdio / HTTP / SSE transport) |
| Web Search | **Tavily** API |
| Data Validation | **Pydantic** v2 |

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Milvus](https://milvus.io/docs/install_standalone-docker.md) running on `localhost:19530`
- An LLM API key (DeepSeek / Zhipu / Tongyi)
- (Optional) [Tavily](https://tavily.com/) API key for web search

### 1. Install Dependencies

```bash
git clone https://github.com/your-username/job-copilot.git
cd job-copilot
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and fill in your API keys:

```env
SERVICE_LLM_API_KEY=your-deepseek-api-key
SERVICE_LLM_BASE_URL=https://api.deepseek.com/v1
SERVICE_LLM_MODEL=deepseek-chat

AGENT_LLM_API_KEY=your-minimax-api-key
AGENT_LLM_BASE_URL=https://api.minimaxi.com/v1
AGENT_LLM_MODEL=MiniMax-M1

TAVILY_API_KEY=your-tavily-api-key     # optional
```

> The embedding model (`BAAI/bge-small-zh-v1.5`, ~1.3 GB) will be auto-downloaded on first run.

### 3. Initialize Vector Database

```bash
# Create Milvus collection with HNSW index
python scripts/create_collection.py

# Ingest data (test with 1000 rows first)
python scripts/ingest_jobs.py --file data/上市公司招聘数据2026.csv --limit 1000

# Full ingestion (~93K records)
python scripts/ingest_jobs.py --file data/上市公司招聘数据2026.csv
```

### 4. Start the Server

```bash
uvicorn app.main:app --reload --port 8000
```

API docs available at: **http://localhost:8000/docs**

### 5. (Optional) Start MCP Server

For Claude Desktop or other MCP-compatible clients:

```bash
python -m app.mcp.server
```

## 📡 API Reference

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

**Returns:** match score, skill gap analysis, and improvement suggestions.

### Conversational Agent

```http
POST /chat
Content-Type: application/json

{
  "message": "数据分析师一般需要哪些技能？",
  "session_id": "user_001"
}
```

**Returns:** reply text, subagents invoked, tool call summaries, source URLs, and latency.

## 🔌 MCP Tools

Connect to Claude Desktop or any MCP client to use these tools:

| Tool | Description |
|------|-------------|
| `search_jobs` | Semantic job search with city/industry filters and retrieval metadata |
| `tavily_search` | Tavily web search for current market signals, articles, and interview experiences |
| `tavily_research` | Multi-source Tavily research summary with source list |
| `tavily_extract` | Extract clean content from one or more concrete URLs |
| `batch_tavily_search` | Run multiple Tavily queries concurrently |

## 📁 Project Structure

```
job-copilot/
├── app/
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Pydantic Settings configuration
│   ├── api/                    # HTTP route handlers
│   │   ├── health.py           # GET  /health
│   │   ├── jobs.py             # GET  /jobs/search
│   │   ├── resume.py           # POST /resume/match
│   │   └── chat.py             # POST /chat
│   ├── agents/                 # Multi-agent orchestration
│   │   ├── graph.py            # Agent graph definition & system prompts
│   │   └── tools.py            # LangChain tool wrappers (search_jobs + Tavily)
│   ├── mcp/                    # MCP protocol server
│   │   ├── server.py           # FastMCP server entry
│   │   └── tools/              # MCP tool implementations
│   ├── rag/                    # RAG pipeline
│   │   ├── chunker.py          # Summary + sliding-window chunking
│   │   ├── embedder.py         # Local embedding (bge-small-zh-v1.5)
│   │   └── retriever.py        # Milvus vector search + scalar filters
│   ├── schemas/                # Pydantic request/response models
│   └── services/               # Business logic layer
│       ├── llm_client.py       # Async LLM client
│       ├── job_service.py      # Job search orchestration
│       ├── resume_service.py   # Resume analysis
│       ├── interview_service.py # Career Q&A
│       └── tavily_client.py    # Tavily web search client
├── scripts/
│   ├── create_collection.py    # Milvus collection setup
│   └── ingest_jobs.py          # CSV → embed → Milvus pipeline
├── data/                       # Raw recruitment datasets (2026)
├── langgraph.json              # LangGraph deployment config
├── requirements.txt
└── .env.example
```


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Milvus](https://milvus.io/) — High-performance vector database
- [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) — Chinese text embedding model
- [DeepSeek](https://deepseek.com/) — LLM API provider
- [Tavily](https://tavily.com/) — AI-optimized web search API
- [FastMCP](https://github.com/jlowin/fastmcp) — Model Context Protocol framework
- [LangGraph](https://github.com/langchain-ai/langgraph) — Agent orchestration framework
