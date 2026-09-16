from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Vector store
    vector_store_dir: str = "data/vector_store"
    collection_name: str = "retail_docs"
    embedding_model: str = "all-MiniLM-L6-v2"

    # YOLO
    yolo_model_path: str = "models/yolo_sku110k/best.pt"
    yolo_conf_threshold: float = 0.25

    # Ollama
    ollama_model: str = "qwen2.5:0.5b"
    ollama_host: str = "http://localhost:11434"

    # CORS
    frontend_origin: str = "http://localhost:8501"


settings = Settings()
