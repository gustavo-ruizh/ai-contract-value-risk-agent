from typing import Optional

try:
    from pydantic_settings import BaseSettings
    _PYDANTIC_V2 = True
except ImportError:
    from pydantic import BaseSettings  # type: ignore[no-redef]
    _PYDANTIC_V2 = False


class Settings(BaseSettings):
    app_name: str = "AI Contract Value & Risk Intelligence Agent"
    debug: bool = False

    # Default discount rate used when the request does not specify one
    default_discount_rate: float = 0.03

    # LLM provider keys — loaded from environment or .env file
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    # LLM model identifiers
    default_llm_model: str = "claude-sonnet-4-6"
    default_embedding_model: str = "text-embedding-3-small"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
