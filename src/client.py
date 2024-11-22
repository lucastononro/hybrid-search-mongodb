# client.py

import sys
import json
import time
from typing import Any, Dict, List, Tuple
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
import openai
import logging

from config import Config

# Configure logging for the client module
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)


def check_collection(client: MongoClient, db_name: str, coll_name: str) -> None:
    """
    Check if the specified collection exists in the database.
    """
    try:
        collections = client[db_name].list_collection_names()
        if coll_name not in collections:
            raise Exception(
                f"Collection '{coll_name}' not found in database '{db_name}'.")
        logger.info(f"Collection '{coll_name}' found in database '{db_name}'.")
    except Exception as e:
        logger.error(f"Error checking collection: {e}")
        sys.exit(1)


def check_index(client: MongoClient, db_name: str, coll_name: str, index_name: str) -> None:
    """
    Check if the specified index exists in the collection.
    """
    try:
        collection = client[db_name][coll_name]
        indexes = list(collection.list_search_indexes(index_name))
        if not indexes:
            raise Exception(
                f"Index '{index_name}' not found in collection '{coll_name}'.")
        logger.info(f"Index '{index_name}' found in collection '{coll_name}'.")
    except Exception as e:
        logger.error(f"Error checking index: {e}")
        sys.exit(1)


class HybridSearchClient:
    """
    Client class to perform hybrid search (vector and text) on MongoDB using OpenAI embeddings.
    """

    def __init__(self, config: Config):
        self.config = config
        self.client = self._initialize_mongo_client()
        self.oai_client = self._initialize_openai()

    def _initialize_mongo_client(self) -> MongoClient:
        """
        Initialize MongoDB client and connect to the cluster.
        """
        try:
            client = MongoClient(self.config.ATLAS_CONNECTION_STRING)
            logger.info("Connected to MongoDB Atlas.")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            sys.exit(1)

    def _initialize_openai(self) -> openai.OpenAI:
        """
        Initialize OpenAI API with the provided API key.
        """
        return openai.OpenAI(api_key=self.config.OPENAI_API_KEY)

    def validate_setup(self) -> None:
        """
        Validate that the specified collection and indexes exist in MongoDB.
        """
        try:
            check_collection(self.client, self.config.DB_NAME,
                             self.config.COLLECTION_NAME)
            check_index(self.client, self.config.DB_NAME,
                        self.config.COLLECTION_NAME, self.config.VECTOR_INDEX_NAME)
            check_index(self.client, self.config.DB_NAME,
                        self.config.COLLECTION_NAME, self.config.TEXT_INDEX_NAME)
            logger.info(
                "MongoDB collection and indexes validated successfully.")
        except Exception as e:
            logger.error(f"Validation error: {e}")
            sys.exit(1)

    def get_embedding(self, text: str, model: str = "text-embedding-ada-002") -> List[float]:
        """
        Generate a vector embedding for a given text using OpenAI API.
        """
        text = text.replace("\n", " ")
        try:
            response = self.oai_client.embeddings.create(
                input=[text], model=model)
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding for text: {text}")
            return embedding
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            sys.exit(1)

    def set_pipeline_indices(
        self,
        pipeline: List[Dict[str, Any]],
        vector_index: str,
        text_index: str,
        collection_name: str
    ) -> List[Dict[str, Any]]:
        """
        Recursively set the appropriate vector and text index names in the aggregation pipeline.
        """
        updated_pipeline = []
        for stage in pipeline:
            if '$vectorSearch' in stage:
                stage['$vectorSearch']['index'] = vector_index
                updated_pipeline.append(stage)
                logger.debug(f"Set '$vectorSearch' index to '{vector_index}'.")
            elif '$search' in stage:
                stage['$search']['index'] = text_index
                updated_pipeline.append(stage)
                logger.debug(f"Set '$search' index to '{text_index}'.")
            elif '$unionWith' in stage:
                stage['$unionWith']['coll'] = collection_name
                stage['$unionWith']['pipeline'] = self.set_pipeline_indices(
                    stage['$unionWith']['pipeline'],
                    vector_index,
                    text_index,
                    collection_name
                )
                updated_pipeline.append(stage)
                logger.debug(
                    f"Set '$unionWith' collection to '{collection_name}'.")
            else:
                updated_pipeline.append(stage)
        return updated_pipeline

    def execute_query(
        self,
        db_name: str,
        coll_name: str,
        pipeline: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Execute the aggregation pipeline and return results along with execution time.
        """
        collection: Collection = self.client[db_name][coll_name]
        start_time: float = time.time()
        try:
            results = list(collection.aggregate(pipeline))
            logger.info("Aggregation pipeline executed successfully.")
        except Exception as e:
            logger.error(f"Error executing pipeline: {e}")
            sys.exit(1)
        elapsed_time: float = time.time() - start_time
        logger.info(f"Pipeline execution time: {elapsed_time:.2f} seconds.")
        return results, elapsed_time

    def hybrid_search(self, query_text: str) -> Tuple[List[Dict[str, Any]], float]:
        """
        Perform a hybrid search combining vector and text search.
        """
        # Generate query vector
        query_vector: List[float] = self.get_embedding(query_text)

        # Define the vector search pipeline
        vector_search_pipeline: List[Dict[str, Any]] = [
            {
                '$vectorSearch': {
                    'path': self.config.VECTOR_FIELD,
                    'queryVector': query_vector,
                    'numCandidates': 100,
                    'limit': 20
                }
            },
            {
                '$group': {
                    '_id': None,
                    'docs': {'$push': '$$ROOT'}
                }
            },
            {
                '$unwind': {
                    'path': '$docs',
                    'includeArrayIndex': 'rank'
                }
            },
            {
                '$addFields': {
                    'vs_score': {
                        '$multiply': [
                            self.config.VECTOR_WEIGHT,
                            {
                                '$divide': [
                                    1.0,
                                    {'$add': ['$rank', 60]}
                                ]
                            }
                        ]
                    }
                }
            },
            {
                '$project': {
                    'vs_score': 1,
                    '_id': '$docs._id',
                    f'{self.config.TEXT_FIELD}': f'$docs.{self.config.TEXT_FIELD}'
                }
            }
        ]

        # Define the text search pipeline
        text_search_pipeline: List[Dict[str, Any]] = [
            {
                '$search': {
                    'index': self.config.TEXT_INDEX_NAME,
                    'text': {
                        'query': query_text,
                        'path': self.config.TEXT_FIELD
                    }
                }
            },
            {
                '$limit': 20
            },
            {
                '$group': {
                    '_id': None,
                    'docs': {'$push': '$$ROOT'}
                }
            },
            {
                '$unwind': {
                    'path': '$docs',
                    'includeArrayIndex': 'rank'
                }
            },
            {
                '$addFields': {
                    'fts_score': {
                        '$multiply': [
                            self.config.TEXT_WEIGHT,
                            {
                                '$divide': [
                                    1.0,
                                    {'$add': ['$rank', 60]}
                                ]
                            }
                        ]
                    }
                }
            },
            {
                '$project': {
                    'fts_score': 1,
                    '_id': '$docs._id',
                    f'{self.config.TEXT_FIELD}': f'$docs.{self.config.TEXT_FIELD}'
                }
            }
        ]

        # Combine pipelines using $unionWith
        combined_pipeline: List[Dict[str, Any]] = vector_search_pipeline + [
            {
                '$unionWith': {
                    'coll': self.config.COLLECTION_NAME,
                    'pipeline': text_search_pipeline
                }
            },
            {
                '$group': {
                    '_id': f'${self.config.TEXT_FIELD}',
                    'vs_score': {'$max': '$vs_score'},
                    'fts_score': {'$max': '$fts_score'}
                }
            },
            {
                '$project': {
                    '_id': 1,
                    f'{self.config.TEXT_FIELD}': '$_id',
                    'vs_score': {'$ifNull': ['$vs_score', 0]},
                    'fts_score': {'$ifNull': ['$fts_score', 0]}
                }
            },
            {
                '$project': {
                    'score': {'$add': ['$vs_score', '$fts_score']},
                    '_id': 1,
                    f'{self.config.TEXT_FIELD}': 1,
                    'vs_score': 1,
                    'fts_score': 1
                }
            },
            {'$sort': {'score': -1}},
            {'$limit': 10}
        ]

        # Adjust pipeline indices
        updated_pipeline: List[Dict[str, Any]] = self.set_pipeline_indices(
            combined_pipeline,
            self.config.VECTOR_INDEX_NAME,
            self.config.TEXT_INDEX_NAME,
            self.config.COLLECTION_NAME
        )

        logger.debug(
            f"Updated Aggregation Pipeline: {json.dumps(updated_pipeline, indent=2)}")

        # Execute the query
        results, elapsed_time = self.execute_query(
            self.config.DB_NAME,
            self.config.COLLECTION_NAME,
            updated_pipeline
        )

        return results, elapsed_time
