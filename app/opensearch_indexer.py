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

    source_chunk_counts = {}

    for chunk, vector in zip(
        chunks,
        vectors
    ):
        source_file = chunk.metadata.get(
            "source_file",
            "unknown"
        )
        source_chunk_index = source_chunk_counts.get(
            source_file,
            0
        )
        source_chunk_counts[source_file] = source_chunk_index + 1

        client.index(
            index=INDEX_NAME,
            body={
                "content": chunk.page_content,
                "source_file": source_file,
                "source_chunk_index": source_chunk_index,
                "embedding": vector
            }
        )

    client.indices.refresh(
        index=INDEX_NAME
    )

    print(
        f"Indexed {len(chunks)} chunks into OpenSearch"
    )
