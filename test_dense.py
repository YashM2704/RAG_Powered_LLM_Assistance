from app.opensearch_client import (
    client,
    INDEX_NAME
)

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

query = "Describe Principal Component Analysis PCA"

query_vector = embeddings.embed_query(
    query
)

response = client.search(
    index=INDEX_NAME,
    body={
        "size": 5,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": 5
                }
            }
        }
    }
)

print("\n===== DENSE RETRIEVAL RESULTS =====\n")

for hit in response["hits"]["hits"]:

    print("Score:", hit["_score"])

    print(
        "File:",
        hit["_source"]["source_file"]
    )

    print(
        hit["_source"]["content"][:300]
    )

    print("-" * 80)