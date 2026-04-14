from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "course_chunks"

    EMBEDDING_PROVIDER: str = "openai"
    OPENAI_API_KEY: str = ""
    EMBEDDING_DIM: int = 1536

    STORAGE_PATH: str = "./storage"
    MAX_FILE_SIZE_MB: int = 50
    MAX_PAGES: int = 300
    TOP_K_DEFAULT: int = 5
    TOP_K_MAX: int = 20

    RETRIEVAL_SCORE_THRESHOLD: float = 0.35

    ALLOWED_ORIGINS: str = "*"
    QDRANT_TIMEOUT: float = 30.0

    # OCR feature flag — set to false to skip Vision API and use native text only
    OCR_ENABLED: bool = True
    # Max seconds for a single OCR API call before giving up on that page
    OCR_TIMEOUT: float = 30.0

    class Config:
        env_file = ".env"


settings = Settings()
