from app.opensearch_client import (
    client,
    INDEX_NAME
)


def bm25_search(
    query: str,
    k: int = 10
):

    response = client.search(
        index=INDEX_NAME,
        body={
            "size": k,
            "query": {
                "match": {
                    "content": query
                }
            }
        }
    )

    return response["hits"]["hits"]