from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application
    app_name: str = "AURA Platform"
    app_version: str = "1.0.0"
    app_description: str = "AI-Powered Wellness Booking Platform"
    debug: bool = False
    environment: str = "development"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # CORS — restrict in production via CORS_ORIGINS env var
    cors_origins: List[str] = [
        "http://localhost:8501",
        "http://localhost:3000",
        "http://frontend:8501",
    ]
    cors_allow_credentials: bool = True

    # Database
    database_url: str = "sqlite:///./aura.db"
    database_pool_size: int = 5
    database_max_overflow: int = 10

    # Authentication
    secret_key: str = Field(
        default="change-this-in-production-minimum-32-characters",
        description="JWT signing key — set via SECRET_KEY env var in production",
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    # Set AUTH_ENABLED=true in production to require bearer tokens
    auth_enabled: bool = False

    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60

    # ML Model
    model_path: Optional[str] = None
    model_confidence_threshold: float = 0.70
    ml_enabled: bool = True

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # "json" | "text"

    # Validation limits
    max_message_length: int = 2000
    max_appointments_per_user: int = 20

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return upper

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        valid = {"development", "staging", "production"}
        lower = v.lower()
        if lower not in valid:
            raise ValueError(f"environment must be one of {valid}")
        return lower

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def effective_cors_origins(self) -> List[str]:
        """Returns ['*'] only in development, never in production."""
        if self.is_production:
            return self.cors_origins
        return self.cors_origins


settings = Settings()
