from app.opensearch_client import (
    client,
    INDEX_NAME
)

if client.indices.exists(index=INDEX_NAME):
    client.indices.delete(index=INDEX_NAME)
    print("Index deleted")
else:
    print("Index not found")