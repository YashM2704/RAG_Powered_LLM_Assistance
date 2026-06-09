from app.opensearch_client import (
    client,
    INDEX_NAME
)

def index_chunks(chunks, embeddings):

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    vectors = embeddings.embed_documents(
        texts
    )

    for chunk, vector in zip(
        chunks,
        vectors
    ):

        client.index(
            index=INDEX_NAME,
            body={
                "content": chunk.page_content,
                "source_file": chunk.metadata.get(
                    "source_file",
                    "unknown"
                ),
                "embedding": vector
            }
        )

    client.indices.refresh(
        index=INDEX_NAME
    )

    print(
        f"Indexed {len(chunks)} chunks into OpenSearch"
    )