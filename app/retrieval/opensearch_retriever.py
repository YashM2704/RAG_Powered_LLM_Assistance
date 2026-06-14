from app.opensearch_client import client, INDEX_NAME
from langchain_huggingface import HuggingFaceEmbeddings

_embeddings = None


def get_embeddings():
    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5"
        )

    return _embeddings

def dense_search(query, k=10):

    query_vector = get_embeddings().embed_query(query)

    response = client.search(
        index=INDEX_NAME,
        body={
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": k
                    }
                }
            }
        }
    )

    return response["hits"]["hits"]
