# config.py

import os
import sys
from dotenv import load_dotenv
from typing import List
import logging

# Configure logging for the configuration module
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)


class Config:
    """
    Configuration class to hold environment variables.
    """
    ATLAS_CONNECTION_STRING: str
    OPENAI_API_KEY: str
    DB_NAME: str
    COLLECTION_NAME: str
    VECTOR_INDEX_NAME: str
    TEXT_INDEX_NAME: str
    VECTOR_FIELD: str
    TEXT_FIELD: str
    VECTOR_WEIGHT: float
    TEXT_WEIGHT: float

    def __init__(self):
        load_dotenv()
        self.ATLAS_CONNECTION_STRING = os.getenv(
            'ATLAS_CONNECTION_STRING', '').strip('"')
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '').strip('"')
        self.DB_NAME = os.getenv('DB_NAME', 'sample_mflix')
        self.COLLECTION_NAME = os.getenv(
            'COLLECTION_NAME', 'movies_embedded_ada')
        self.VECTOR_INDEX_NAME = os.getenv('VECTOR_INDEX_NAME', 'vectorIndex')
        self.TEXT_INDEX_NAME = os.getenv('TEXT_INDEX_NAME', 'searchIndex')
        self.VECTOR_FIELD = os.getenv('VECTOR_FIELD', 'embedding')
        self.TEXT_FIELD = os.getenv('TEXT_FIELD', 'text')
        self.VECTOR_WEIGHT = float(os.getenv('VECTOR_WEIGHT', '0.5'))
        self.TEXT_WEIGHT = float(os.getenv('TEXT_WEIGHT', '0.5'))

        # Validate required variables
        required_vars = [
            'ATLAS_CONNECTION_STRING',
            'OPENAI_API_KEY',
            'DB_NAME',
            'COLLECTION_NAME',
            'VECTOR_INDEX_NAME',
            'TEXT_INDEX_NAME',
            'VECTOR_FIELD',
            'TEXT_FIELD'
        ]

        missing_vars: List[str] = [
            var for var in required_vars if not getattr(self, var)]
        if missing_vars:
            logger.error(
                f"Missing environment variables: {', '.join(missing_vars)}")
            sys.exit(1)
        else:
            logger.info(
                "All required environment variables loaded successfully.")
