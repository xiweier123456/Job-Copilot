from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


MODEL_ROOT = Path("E:/Study/model")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "job_chunks"
    milvus_chat_memory_collection: str = "chat_memory"

    # Embedding / Rerank
    embedding_model: str = str(MODEL_ROOT / "bge-small-zh-v1.5")
    embedding_dim: int = 512  # bge-small-zh-v1.5 输出维度
    hf_token: str = ""
    enable_rerank: bool = True
    reranker_model: str = str(MODEL_ROOT / "BAAI-bge-reranker-v2-m3")
    rerank_candidate_k: int = 20

    # Self-reflective retrieval
    enable_query_understanding: bool = True
    enable_retrieval_judge: bool = True
    retrieval_max_retry: int = 1
    retrieval_min_results: int = 3
    retrieval_min_rerank_score: float = 0.6
    retrieval_judge_top_k: int = 3

    # Service / RAG LLM (DeepSeek)
    service_llm_api_key: str = ""
    service_llm_base_url: str = "https://api.deepseek.com/v1"
    service_llm_model: str = "deepseek-chat"

    # Agent LLM (Minimax)
    agent_llm_api_key: str = ""
    agent_llm_base_url: str = "https://api.minimaxi.com/v1"
    agent_llm_model: str = "MiniMax-M2.7"

    # Search
    tavily_api_key: str = ""

    # MCP
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8000
    mcp_path: str = "/mcp"
    mcp_client_timeout: int = 30
    mcp_debug: bool = False
    mcp_tool_name_prefix: bool = False
    mcp_transport: str = "streamable-http"

    # 应用
    app_title: str = "Job Copilot"
    app_version: str = "0.1.0"
    debug: bool = False
    langsmith_api_key: str = ""


settings = Settings()
