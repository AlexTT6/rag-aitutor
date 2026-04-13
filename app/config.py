from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "course_chunks"

    # "openai" uses text-embedding-3-small (1536 dims).
    # "local" uses all-MiniLM-L6-v2 via sentence-transformers (384 dims).
    # Changing EMBEDDING_PROVIDER after the Qdrant collection is created requires
    # dropping and recreating the collection — vector dimensions must match.
    EMBEDDING_PROVIDER: str = "openai"
    OPENAI_API_KEY: str = ""

    # Fixed embedding output dimension. Must match the model named above.
    # openai/text-embedding-3-small -> 1536
    # sentence-transformers/all-MiniLM-L6-v2 -> 384
    EMBEDDING_DIM: int = 1536

    STORAGE_PATH: str = "./storage"
    MAX_FILE_SIZE_MB: int = 50
    MAX_PAGES: int = 300
    TOP_K_DEFAULT: int = 5
    TOP_K_MAX: int = 20

    RETRIEVAL_SCORE_THRESHOLD: float = 0.70
    MIN_EXTRACTABLE_RATIO: float = 0.50

    # CORS — comma-separated list of allowed origins for the React frontend
    # Example: "https://myapp.com,https://www.myapp.com"
    # Use "*" to allow all origins (not recommended for production)
    ALLOWED_ORIGINS: str = "*"

    # Qdrant client timeout in seconds
    QDRANT_TIMEOUT: float = 30.0

    class Config:
        env_file = ".env"


settings = Settings()
