"""Configuration management with pydantic-settings."""

import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # LLM Configuration
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field("https://api.deepseek.com/v1", alias="DEEPSEEK_BASE_URL")
    deepseek_model: str = Field("deepseek-chat", alias="DEEPSEEK_MODEL")
    llm_temperature: float = Field(0.7, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(2048, alias="LLM_MAX_TOKENS")

    # Java Backend
    java_backend_url: str = Field("http://localhost:8080", alias="JAVA_BACKEND_URL")
    java_backend_timeout: int = Field(60, alias="JAVA_BACKEND_TIMEOUT")

    # MCTS Defaults
    mcts_max_iterations: int = Field(100, alias="MCTS_MAX_ITERATIONS")
    mcts_coverage_target: float = Field(0.95, alias="MCTS_COVERAGE_TARGET")
    mcts_exploration_coef: float = Field(1.414, alias="MCTS_EXPLORATION_COEF")
    mcts_beam_width: int = Field(5, alias="MCTS_BEAM_WIDTH")

    # Session Management
    session_ttl_minutes: int = Field(60, alias="SESSION_TTL_MINUTES")
    token_budget_default: int = Field(100000, alias="TOKEN_BUDGET_DEFAULT")

    # Server Configuration
    host: str = Field("0.0.0.0", alias="HOST")
    port: int = Field(8000, alias="PORT")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # Audit Configuration
    audit_dir: str = Field("./audit", alias="AUDIT_DIR")

    class Config:
        # Try multiple .env file locations
        env_file = [
            os.path.join(os.path.dirname(__file__), "..", ".env"),  # python-lats-server/.env
            ".env",  # Current directory
        ]
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()
