"""Configuration management for SimuNet."""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    mongodb_url: str = Field(
        default="mongodb://localhost:27017",
        env="MONGODB_URL",
        description="MongoDB connection URL"
    )
    mongodb_database: str = Field(
        default="simu_net",
        env="MONGODB_DATABASE", 
        description="MongoDB database name"
    )
    redis_url: str = Field(
        default="redis://localhost:6379",
        env="REDIS_URL",
        description="Redis connection URL"
    )
    faiss_index_path: str = Field(
        default="./data/faiss_indices",
        env="FAISS_INDEX_PATH",
        description="Path to FAISS index files"
    )


class AgentSettings(BaseSettings):
    """Agent configuration settings."""
    
    max_agents: int = Field(
        default=1000,
        env="MAX_AGENTS",
        description="Maximum number of concurrent agents"
    )
    agent_tick_interval: float = Field(
        default=1.0,
        env="AGENT_TICK_INTERVAL",
        description="Agent processing interval in seconds"
    )
    user_agent_personas: list[str] = Field(
        default=["casual", "influencer", "bot", "activist"],
        description="Available user agent persona types"
    )


class MLSettings(BaseSettings):
    """Machine learning model settings."""
    
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL",
        description="Sentence transformer model for embeddings"
    )
    toxicity_model: str = Field(
        default="unitary/toxic-bert",
        env="TOXICITY_MODEL", 
        description="Model for toxicity detection"
    )
    misinformation_model: str = Field(
        default="roberta-base",
        env="MISINFORMATION_MODEL",
        description="Model for misinformation detection"
    )
    device: str = Field(
        default="cpu",
        env="ML_DEVICE",
        description="Device for ML inference (cpu/cuda)"
    )


class APISettings(BaseSettings):
    """API server configuration."""
    
    host: str = Field(
        default="0.0.0.0",
        env="API_HOST",
        description="API server host"
    )
    port: int = Field(
        default=8000,
        env="API_PORT",
        description="API server port"
    )
    debug: bool = Field(
        default=False,
        env="DEBUG",
        description="Enable debug mode"
    )
    cors_origins: list[str] = Field(
        default=["http://localhost:3000"],
        env="CORS_ORIGINS",
        description="Allowed CORS origins"
    )


class MonitoringSettings(BaseSettings):
    """Monitoring and metrics settings."""
    
    prometheus_port: int = Field(
        default=8001,
        env="PROMETHEUS_PORT",
        description="Prometheus metrics port"
    )
    log_level: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="Logging level"
    )
    metrics_enabled: bool = Field(
        default=True,
        env="METRICS_ENABLED",
        description="Enable metrics collection"
    )


class Settings(BaseSettings):
    """Main application settings."""
    
    environment: str = Field(
        default="development",
        env="ENVIRONMENT",
        description="Application environment"
    )
    
    # Sub-settings
    database: DatabaseSettings = DatabaseSettings()
    agents: AgentSettings = AgentSettings()
    ml: MLSettings = MLSettings()
    api: APISettings = APISettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }


# Global settings instance - will be initialized when needed
settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create global settings instance."""
    global settings
    if settings is None:
        settings = Settings()
    return settings