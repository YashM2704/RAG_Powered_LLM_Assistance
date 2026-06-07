# test_index.py

from app.opensearch_client import (
    client,
    create_index,
    INDEX_NAME
)

create_index()

print(
    client.indices.get(
        index=INDEX_NAME
    )
)