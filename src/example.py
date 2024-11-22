# example.py

from typing import Any, Dict, List, Tuple
from client import HybridSearchClient
from config import Config
import logging

# Configure logging for the example module
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)


def main():
    """
    Example script to perform a hybrid search using HybridSearchClient.
    """
    # Initialize configuration
    config = Config()

    # Initialize HybridSearchClient
    client = HybridSearchClient(config)

    # Validate MongoDB setup
    client.validate_setup()

    while True:
        try:
            # Prompt user for search query
            query_text: str = input(
                "\nEnter your search query (or type 'exit' to quit): ").strip()
            if query_text.lower() == 'exit':
                logger.info("Exiting the search application.")
                break
            elif not query_text:
                logger.warning(
                    "Empty query provided. Please enter a valid search query.")
                continue

            # Perform hybrid search
            results: List[Dict[str, Any]]
            elapsed_time: float
            results, elapsed_time = client.hybrid_search(query_text)

            # Display the results
            print(
                f"\nQuery executed successfully in {elapsed_time:.2f} seconds.")
            print("\nSearch Results:")
            if results:
                for idx, result in enumerate(results, start=1):
                    text: str = result.get(config.TEXT_FIELD, 'N/A')
                    score: float = result.get('score', 0.0)
                    print(f"{idx}. {text} (Score: {score:.6f})")
            else:
                print("No results found.")
        except KeyboardInterrupt:
            logger.info(
                "\nKeyboard interrupt received. Exiting the application.")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            break


if __name__ == '__main__':
    main()
