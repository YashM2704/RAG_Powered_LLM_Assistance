from app.retrieval.bm25_retriever import (
    bm25_search
)

from app.retrieval.opensearch_retriever import (
    dense_search
)


def hybrid_search(
    query: str,
    k: int = 10
):

    dense_docs = dense_search(
        query,
        k=k
    )

    bm25_docs = bm25_search(
        query,
        k=k
    )

    merged = {}

    for doc in dense_docs:

        doc_id = doc["_id"]

        merged[doc_id] = doc

    for doc in bm25_docs:

        doc_id = doc["_id"]

        if doc_id not in merged:

            merged[doc_id] = doc

    return list(
        merged.values()
    )