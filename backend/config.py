from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Application
    secret_key: str
    allowed_origins: List[str]

    # Admin account
    admin_email: str
    admin_password: str

    # Database
    postgres_user: str
    postgres_password: str
    postgres_db: str
    database_url: str

    # Anthropic
    anthropic_api_key: str

    # ML settings
    models_dir: str = "./saved_models"
    uploads_dir: str = "./uploads"
    tfidf_max_features: int = 10000

    @field_validator('allowed_origins', mode='before')
    def split_allowed_origins(cls, value):
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(',') if origin.strip()]
        return value

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()