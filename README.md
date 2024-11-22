# hybrid-search-mongodb

A simple hybrid search client for MongoDB using Python.

This project is a generalized Python implementation of the reciprocal rank fusion (RRF) hybrid search example provided in MongoDB's documentation. It combines vector search and text search capabilities to perform hybrid search operations on a MongoDB database.

Reference Node.js implementation for RRF: [MongoDB Reciprocal Rank Fusion Tutorial](https://www.mongodb.com/docs/atlas/atlas-vector-search/tutorials/reciprocal-rank-fusion/)

## Project Structure

```
hybrid-search-mongodb/
│
├── src/
│   ├── config.py          # Configuration settings for the client
│   ├── client.py          # Hybrid search client that performs vector + text search
│   └── example.py         # Example usage script that utilizes the hybrid search client
│
└── README.md              # This documentation
```

## Setup Instructions

### Prerequisites

- Python 3.8 or newer
- MongoDB Atlas cluster with Atlas Search enabled and configured
- OpenAI API access

### Installation

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd hybrid-search-mongodb
   ```

2. **Install Dependencies**

   Use `pip` to install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Configuration

To set up your environment, create a `.env` file in the root directory with the following variables:

```plaintext
# MongoDB Configuration
ATLAS_CONNECTION_STRING="mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?authSource=admin"
DB_NAME="test-vector-search"
COLLECTION_NAME="rag"
VECTOR_INDEX_NAME="liza_vector_search"
TEXT_INDEX_NAME="search_index"
VECTOR_FIELD="embedding"
TEXT_FIELD="text"

# OpenAI API Key
OPENAI_API_KEY="sk-your-openai-api-key"
```

### Project Files Overview

1. **src/config.py**: Loads environment variables and provides configuration settings for MongoDB and OpenAI API.

2. **src/client.py**: Implements the `HybridSearchClient` class, which is responsible for executing hybrid searches on MongoDB, combining both vector and text search results.

3. **src/example.py**: Demonstrates how to utilize the `HybridSearchClient` to perform a hybrid search for a user-specified query.

## Usage

### Hybrid Search Client

To use the hybrid search client, follow the steps below:

1. **Import the HybridSearchClient**:

   ```python
   from src.client import HybridSearchClient
   ```

2. **Initialize the Client**:

   ```python
   # Instantiate the hybrid search client
   client = HybridSearchClient()
   ```

3. **Execute a Hybrid Search**:

   ```python
   # Perform a search for a given query text
   query_text = "Find interesting research papers"
   results, elapsed_time = client.execute_hybrid_search(query_text)

   # Display the results
   print(f"\nQuery executed successfully in {elapsed_time:.2f} seconds.")
   print("\nSearch Results:")
   for result in results:
       print(f"- {result['text']} (Score: {result['score']:.6f})")
   ```

### Example Script

The script in `src/example.py` provides a full example of how to use the client:

```python
from src.client import HybridSearchClient

def main():
    client = HybridSearchClient()
    query_text = "Discover climate change articles"
    results, elapsed_time = client.execute_hybrid_search(query_text)
    
    print(f"\nQuery executed successfully in {elapsed_time:.2f} seconds.")
    print("\nSearch Results:")
    for result in results:
        print(f"- {result['text']} (Score: {result['score']:.6f})")

if __name__ == '__main__':
    main()
```

### Running the Example

Run the example script from the root of the project:

```bash
python src/example.py
```

## Important Notes

- Ensure your MongoDB Atlas instance has the relevant vector and text indexes created as per the MongoDB documentation.
- The `OpenAI API Key` is needed to generate vector embeddings using OpenAI's API. This key is loaded via the `.env` file.

## License

MIT License

## Acknowledgments

- MongoDB's reciprocal rank fusion implementation provided in [MongoDB Atlas documentation](https://www.mongodb.com/docs/atlas/atlas-vector-search/tutorials/reciprocal-rank-fusion/).
- OpenAI API for vector embedding generation.
