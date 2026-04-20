"""Configuration management for FinQA chatbot."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# Load environment variables
load_dotenv()


class VLLMConfig(BaseModel):
    """vLLM server configuration."""

    api_base: str = Field(default="http://localhost:8000/v1")
    model: str = Field(default="Qwen/Qwen2.5-32B-Instruct")
    port: int = Field(default=8000)
    tensor_parallel_size: int = Field(default=1)
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=0.1)

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO")
    format: str = Field(default="json")

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v_upper


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""

    path: Path = Field(default=Path("./data/vector_store"))
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)

    @field_validator("chunk_size", "chunk_overlap")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Value must be positive")
        return v


class PerformanceConfig(BaseModel):
    """Performance and monitoring configuration."""

    max_retrieval_docs: int = Field(default=5)
    gpu_monitoring_enabled: bool = Field(default=True)

    @field_validator("max_retrieval_docs")
    @classmethod
    def validate_max_docs(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_retrieval_docs must be positive")
        return v


class Config(BaseModel):
    """Main configuration class."""

    vllm: VLLMConfig
    logging: LoggingConfig
    vector_store: VectorStoreConfig
    performance: PerformanceConfig
    hf_token: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            vllm=VLLMConfig(
                api_base=os.getenv("VLLM_API_BASE", "http://localhost:8000/v1"),
                model=os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-32B-Instruct"),
                port=int(os.getenv("VLLM_PORT", "8000")),
                tensor_parallel_size=int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")),
                max_tokens=int(os.getenv("VLLM_MAX_TOKENS", "2048")),
                temperature=float(os.getenv("VLLM_TEMPERATURE", "0.1")),
            ),
            logging=LoggingConfig(
                level=os.getenv("LOG_LEVEL", "INFO"),
                format=os.getenv("LOG_FORMAT", "json"),
            ),
            vector_store=VectorStoreConfig(
                path=Path(os.getenv("VECTOR_STORE_PATH", "./data/vector_store")),
                embedding_model=os.getenv(
                    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
                ),
                chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            ),
            performance=PerformanceConfig(
                max_retrieval_docs=int(os.getenv("MAX_RETRIEVAL_DOCS", "5")),
                gpu_monitoring_enabled=os.getenv("GPU_MONITORING_ENABLED", "true").lower() == "true",
            ),
            hf_token=os.getenv("HF_TOKEN"),
        )


# Global config instance
config = Config.from_env()
