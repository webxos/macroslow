```python
from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List
import os

class MarkupConfig(BaseModel):
    """Configuration model for MARKUP Agent settings."""
    db_uri: str = Field(default="sqlite:///markup_logs.db", description="SQLAlchemy database URI")
    api_host: str = Field(default="0.0.0.0", description="API host address")
    api_port: int = Field(default=8000, description="API port number")
    quantum_enabled: bool = Field(default=False, description="Enable quantum integration")
    quantum_api_url: Optional[HttpUrl] = Field(default=None, description="Qiskit quantum API endpoint")
    visualization_theme: str = Field(default="dark", description="Visualization theme: 'dark' or 'light'")
    max_streams: int = Field(default=8, description="Maximum concurrent transformation streams")
    error_threshold: float = Field(default=0.5, description="Threshold for error detection")

    class Config:
        env_prefix = "MARKUP_"  # Environment variable prefix
        case_sensitive = False

    @classmethod
    def load_from_env(cls) -> 'MarkupConfig':
        """Load configuration from environment variables or defaults."""
        return cls(
            db_uri=os.getenv("MARKUP_DB_URI", "sqlite:///markup_logs.db"),
            api_host=os.getenv("MARKUP_API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("MARKUP_API_PORT", 8000)),
            quantum_enabled=os.getenv("MARKUP_QUANTUM_ENABLED", "false").lower() == "true",
            quantum_api_url=os.getenv("MARKUP_QUANTUM_API_URL", None),
            visualization_theme=os.getenv("MARKUP_VISUALIZATION_THEME", "dark"),
            max_streams=int(os.getenv("MARKUP_MAX_STREAMS", 8)),
            error_threshold=float(os.getenv("MARKUP_ERROR_THRESHOLD", 0.5))
        )