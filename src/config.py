"""Configuration management for the FinQA chatbot."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv() -> None:
        return None


load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default).lower()).lower() == "true"


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass
class VLLMConfig:
    api_base: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    port: int = 8000
    tensor_parallel_size: int = 1
    max_tokens: int = 2048
    temperature: float = 0.1
    timeout_seconds: int = 120


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "json"


@dataclass
class VectorStoreConfig:
    path: Path = Path("./data/vector_store")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class PerformanceConfig:
    max_retrieval_docs: int = 5
    gpu_monitoring_enabled: bool = True


@dataclass
class AgentConfig:
    max_retries: int = 1
    retrieval_top_k: int = 5
    dense_weight: float = 0.55
    sparse_weight: float = 0.45
    max_context_characters: int = 5000


@dataclass
class Config:
    vllm: VLLMConfig
    logging: LoggingConfig
    vector_store: VectorStoreConfig
    performance: PerformanceConfig
    agent: AgentConfig
    hf_token: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Config":
        config = cls(
            vllm=VLLMConfig(
                api_base=os.getenv("VLLM_API_BASE", "http://localhost:8000/v1"),
                api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
                model=os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
                port=_env_int("VLLM_PORT", 8000),
                tensor_parallel_size=_env_int("VLLM_TENSOR_PARALLEL_SIZE", 1),
                max_tokens=_env_int("VLLM_MAX_TOKENS", 2048),
                temperature=_env_float("VLLM_TEMPERATURE", 0.1),
                timeout_seconds=_env_int("VLLM_TIMEOUT_SECONDS", 120),
            ),
            logging=LoggingConfig(
                level=os.getenv("LOG_LEVEL", "INFO").upper(),
                format=os.getenv("LOG_FORMAT", "json"),
            ),
            vector_store=VectorStoreConfig(
                path=Path(os.getenv("VECTOR_STORE_PATH", "./data/vector_store")),
                embedding_model=os.getenv(
                    "EMBEDDING_MODEL",
                    "sentence-transformers/all-MiniLM-L6-v2",
                ),
                chunk_size=_env_int("CHUNK_SIZE", 512),
                chunk_overlap=_env_int("CHUNK_OVERLAP", 50),
            ),
            performance=PerformanceConfig(
                max_retrieval_docs=_env_int("MAX_RETRIEVAL_DOCS", 5),
                gpu_monitoring_enabled=_env_bool("GPU_MONITORING_ENABLED", True),
            ),
            agent=AgentConfig(
                max_retries=_env_int("AGENT_MAX_RETRIES", 1),
                retrieval_top_k=_env_int("AGENT_RETRIEVAL_TOP_K", 5),
                dense_weight=_env_float("AGENT_DENSE_WEIGHT", 0.55),
                sparse_weight=_env_float("AGENT_SPARSE_WEIGHT", 0.45),
                max_context_characters=_env_int("AGENT_MAX_CONTEXT_CHARACTERS", 5000),
            ),
            hf_token=os.getenv("HF_TOKEN"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not 0.0 <= self.vllm.temperature <= 2.0:
            raise ValueError("VLLM_TEMPERATURE must be between 0.0 and 2.0")
        if self.performance.max_retrieval_docs <= 0:
            raise ValueError("MAX_RETRIEVAL_DOCS must be positive")
        if self.agent.max_retries <= 0:
            raise ValueError("AGENT_MAX_RETRIES must be positive")
        if self.agent.retrieval_top_k <= 0:
            raise ValueError("AGENT_RETRIEVAL_TOP_K must be positive")
        if self.agent.max_context_characters <= 0:
            raise ValueError("AGENT_MAX_CONTEXT_CHARACTERS must be positive")
        if not 0.0 <= self.agent.dense_weight <= 1.0:
            raise ValueError("AGENT_DENSE_WEIGHT must be between 0.0 and 1.0")
        if not 0.0 <= self.agent.sparse_weight <= 1.0:
            raise ValueError("AGENT_SPARSE_WEIGHT must be between 0.0 and 1.0")


config = Config.from_env()
