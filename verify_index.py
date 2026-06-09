from app.opensearch_client import (
    client,
    INDEX_NAME
)

print(
    client.count(
        index=INDEX_NAME
    )
)