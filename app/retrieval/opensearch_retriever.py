from app.opensearch_client import client, INDEX_NAME
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

def dense_search(query, k=10):

    query_vector = embeddings.embed_query(query)

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