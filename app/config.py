"""
Loads configuration from environment variables (.env file).

Why this file exists:
Every other module that needs an API key or a connection string imports
from here instead of calling os.getenv() directly. That way there is
exactly one place that knows about environment variables, and exactly
one place that fails loudly if something required is missing.
"""

import os
from dotenv import load_dotenv

# Reads the .env file in the project root and loads it into the
# process environment. If .env does not exist, this simply does nothing
# (it will not crash) — os.getenv() calls below will then return None.
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "./data/qdrant_storage")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "docchat_chunks")


def validate_config() -> None:
    """
    Checks that required configuration is present.

    Called once at startup (see main.py) so that a missing API key
    fails immediately with a clear message, instead of failing later
    inside a random request with a confusing error.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and "
            "add your Gemini API key before starting the server."
        )
