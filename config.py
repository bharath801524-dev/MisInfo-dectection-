"""
Mis-Info Verify - Configuration Module.

This module centralizes application settings, environment variables,
directory paths, and logging configuration.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Define base paths
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env file if it exists
ENV_PATH: Path = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()  # Fallback to default environment lookup


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration loaded from environment with sensible defaults."""

    app_name: str = "Mis-Info Verify"
    app_version: str = "0.2.0"
    app_env: str = os.getenv("APP_ENV", "development")
    debug: bool = os.getenv("APP_DEBUG", "false").lower() in ("true", "1", "yes")
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # Storage paths
    database_path: str = os.getenv("DATABASE_PATH", str(DATA_DIR / "mindmesh.db"))
    outputs_dir: str = str(OUTPUTS_DIR)

    # Search & Verification API Keys (Phase 2)
    tavily_api_key: str | None = os.getenv("TAVILY_API_KEY")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    # Legacy/Alternative Keys
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    serpapi_api_key: str | None = os.getenv("SERPAPI_API_KEY")

    # Google News RSS Configuration
    google_news_rss_enabled: bool = os.getenv("GOOGLE_NEWS_RSS_ENABLED", "true").lower() in ("true", "1", "yes")
    google_news_rss_hl: str = os.getenv("GOOGLE_NEWS_RSS_HL", "en-IN")
    google_news_rss_gl: str = os.getenv("GOOGLE_NEWS_RSS_GL", "IN")
    google_news_rss_ceid: str = os.getenv("GOOGLE_NEWS_RSS_CEID", "IN:en")
    google_news_rss_timeout: int = int(os.getenv("GOOGLE_NEWS_RSS_TIMEOUT", "8"))

    def has_tavily(self) -> bool:
        """Check if a Tavily API key is provided and non-empty."""
        return bool(self.tavily_api_key and self.tavily_api_key.strip() and not self.tavily_api_key.startswith("your_"))

    def has_gemini(self) -> bool:
        """Check if a Gemini API key is provided and non-empty."""
        return bool(self.gemini_api_key and self.gemini_api_key.strip() and not self.gemini_api_key.startswith("your_"))

    def has_google_news_rss(self) -> bool:
        """Check if Google News RSS evidence retrieval is enabled."""
        return self.google_news_rss_enabled


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure basic structured logging for the application.

    Args:
        level: Desired log level string (e.g. 'DEBUG', 'INFO', 'WARNING').

    Returns:
        Configured root logger instance.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("mindmesh")
    logger.setLevel(numeric_level)
    return logger


# Instantiate global configuration
config = AppConfig()
logger = setup_logging(config.log_level)
