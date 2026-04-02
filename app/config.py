from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Embedding
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512  # bge-small-zh-v1.5 输出维度
    hf_token: str = ""

    # LLM
    llm_api_key: str =""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"

    # Search
    # tavily_api_key: str = ""
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
