from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


MODEL_ROOT = Path("E:/Study/model")


class ExternalMCPServiceConfig(BaseModel):
    """描述一个外部 MCP 服务的接入配置。"""

    name: str
    enabled: bool = True
    transport: Literal["streamable-http", "http", "sse", "stdio"] = "streamable-http"
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    timeout: int | None = None
    keep_alive: bool | None = None
    priority: int = 0
    prefix: str | None = None
    include_tools: list[str] = Field(default_factory=list)
    exclude_tools: list[str] = Field(default_factory=list)
    description: str | None = None

    @model_validator(mode="after")
    def validate_endpoint(self):
        """根据 transport 校验远端地址或本地命令是否完整。"""
        if self.transport == "stdio":
            if not self.command:
                raise ValueError("stdio 外部 MCP 服务必须提供 command")
        elif not self.url:
            raise ValueError("远程外部 MCP 服务必须提供 url")
        return self


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

    # Memory
    memory_enabled: bool = True
    memory_recall_top_k: int = 3
    memory_recall_score_threshold: float | None = None
    memory_recent_history_limit: int = 4
    memory_prompt_char_budget: int = 1200
    memory_save_user_raw: bool = False
    memory_min_user_chars: int = 12
    memory_min_assistant_chars: int = 80
    memory_context_summary_chars: int = 1200
    mem0_collection_name: str = "mem0_chat_memory"
    mem0_llm_provider: str = "deepseek"
    mem0_llm_model: str = ""
    mem0_embedder_provider: str = "huggingface"
    mem0_embedder_model: str = ""
    mem0_embedder_dims: int | None = None
    mem0_history_db_path: str = "outputs/mem0/history.db"

    # Context compression
    context_compression_enabled: bool = True
    context_compression_trigger_tokens: int = 6000
    context_compression_target_tokens: int = 2500
    context_compression_max_input_chars: int = 16000
    context_compression_fallback_recent_chars: int = 6000

    # LangGraph checkpoint
    checkpoint_backend: Literal["memory", "sqlite"] = "sqlite"
    checkpoint_sqlite_path: str = "outputs/checkpoints/langgraph.db"

    # Search
    tavily_api_key: str = ""

    # Redis cache / runtime state
    redis_enabled: bool = False
    redis_url: str = "redis://127.0.0.1:6379/0"
    redis_key_prefix: str = "jobcopilot"
    redis_default_ttl_seconds: int = 600
    redis_run_lock_ttl_seconds: int = 900
    tavily_cache_enabled: bool = True
    tavily_cache_ttl_seconds: int = 1800

    # Tool safety
    tool_security_enabled: bool = True
    tool_allowed_domains: str = ""
    tool_security_redact_inputs: bool = True
    tool_security_preview_chars: int = 160

    # MCP
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8000
    mcp_path: str = "/mcp"
    mcp_client_timeout: int = 30
    mcp_debug: bool = False
    mcp_tool_name_prefix: bool = False
    mcp_transport: str = "streamable-http"
    mcp_external_discovery_ttl_seconds: int = 60
    mcp_external_discovery_failure_ttl_seconds: int = 30
    external_mcp_services: list[ExternalMCPServiceConfig] = Field(default_factory=list)
    app_title: str = "Job Copilot"
    app_version: str = "0.1.0"
    debug: bool = False
    langsmith_api_key: str = ""


settings = Settings()
