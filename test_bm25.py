from app.opensearch_client import (
    client,
    INDEX_NAME
)

query = "What is Yash Mahajan's email?"

response = client.search(
    index=INDEX_NAME,
    body={
        "size": 5,
        "query": {
            "match": {
                "content": query
            }
        }
    }
)

print("\n===== BM25 RESULTS =====\n")

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