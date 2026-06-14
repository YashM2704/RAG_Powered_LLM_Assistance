from app.retrieval.metadata_retriever import (
    deduplicate_documents,
    filter_administrative_noise,
    format_document_wide_answer,
    is_document_wide_query,
    retrieve_by_document_reference,
)
import re

_llm = None
_reranker = None
_prompt = None

PROMPT_TEMPLATE = """
You are a Retrieval-Augmented AI Assistant.

You must answer ONLY from the provided context.

STRICT RULES:
1. Never use outside knowledge.
2. Never make assumptions.
3. If the answer is not found in the context, reply exactly:
   "I couldn't find that information in the documents."
4. If the user asks for questions, list them exactly as written.
5. If the context is unrelated to the question, return:
   "I couldn't find that information in the documents."
6. Do not provide generic explanations.
7. For emails, phone numbers, URLs, names,
   and identifiers, copy them exactly from
   the context without modification.
8. If the answer is a single value
   (email, phone number, URL, date),
   return only that value.

Context:
{context}

Question:
{question}

Answer:
"""


def get_llm():
    global _llm

    if _llm is None:
        from langchain_ollama import ChatOllama

        _llm = ChatOllama(
            model="llama3"
        )

    return _llm


def get_prompt():
    global _prompt

    if _prompt is None:
        from langchain_core.prompts import ChatPromptTemplate

        _prompt = ChatPromptTemplate.from_template(
            PROMPT_TEMPLATE
        )

    return _prompt


def get_reranker():
    global _reranker

    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    return _reranker


def run_hybrid_search(query: str, k: int = 10):
    from app.retrieval.hybrid_retriever import hybrid_search

    return hybrid_search(
        query,
        k=k
    )

def rerank_documents(question, docs):

    if not docs:
        return []

    pairs = [
        (
            question,
            doc["_source"]["content"]
        )
        for doc in docs
    ]

    scores = get_reranker().predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    print("\n========== RERANK SCORES ==========")

    for doc, score in ranked:
        print(
            f"{score:.4f} | "
            f"{doc['_source'].get('source_file', 'unknown')}"
        )

    return [doc for doc, _ in ranked]


def extract_email(docs):

    pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"

    for doc in docs:

        match = re.search(
            pattern,
            doc["_source"]["content"]
        )

        if match:
            return match.group(), doc

    return None


def ask_question(question: str):

    question_lower = question.lower()
    document_wide_query = is_document_wide_query(question)

    docs, matched_source_files = retrieve_by_document_reference(
        question
    )

    if matched_source_files:
        print(
            "\nMetadata filter matched source_file:",
            ", ".join(matched_source_files)
        )
    else:
        # OpenSearch hybrid search Retrieval
        docs = run_hybrid_search(
            question,
            k=10
        )

        docs = filter_administrative_noise(
            deduplicate_documents(docs)
        )

        docs = rerank_documents(
            question,
            docs
        )

        docs = docs[:4]

    if matched_source_files and document_wide_query:
        print("\n========== DOCUMENT-WIDE RETRIEVAL RESULTS ==========")

        for rank, doc in enumerate(docs, start=1):
            print(f"\nChunk: {rank}")
            print(
                "Source File:",
                doc["_source"].get(
                    "source_file",
                    "unknown"
                )
            )
            print(doc["_source"]["content"])
            print("-" * 80)

        return {
            "answer": format_document_wide_answer(docs),
            "sources": [
                {
                    "file": doc["_source"].get(
                        "source_file",
                        "unknown"
                    ),
                    "content": doc["_source"].get(
                        "content",
                        ""
                    )
                }
                for doc in docs
            ]
        }

    # Exact email extraction
    if "email" in question_lower:

        email_match = extract_email(docs)

        if email_match:

            email, source_doc = email_match

            return {
                "answer": email,
                "sources": [
                    {
                        "file": source_doc["_source"].get(
                            "source_file",
                            "unknown"
                        ),
                        "content": source_doc["_source"].get(
                            "content",
                            ""
                        )[:300]
                    }
                ]
            }

    print("\n========== RETRIEVAL RESULTS ==========")

    displayed_docs = docs[:10]

    for rank, doc in enumerate(displayed_docs, start=1):

        print(f"\nRank: {rank}")

        print(
            "Source File:",
            doc["_source"].get(
                "source_file",
                "unknown"
            )
        )

        print(
            doc["_source"]["content"][:300]
        )

        print("-" * 80)

    if len(docs) > len(displayed_docs):
        print(
            f"... {len(docs) - len(displayed_docs)} more chunks omitted "
            "from debug output"
        )

    context = "\n\n".join(
        doc["_source"]["content"]
        for doc in docs
    )

    chain = get_prompt() | get_llm()

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return {
        "answer": response.content,
        "sources": [
            {
                "file": doc["_source"].get(
                    "source_file",
                    "unknown"
                ),
                "content": doc["_source"].get(
                    "content",
                    ""
                )[:300]
            }
            for doc in docs
        ]
    }


if __name__ == "__main__":

    while True:

        q = input("\nAsk a question: ")

        if q.lower() == "exit":
            break

        result = ask_question(q)

        print("\n========== ANSWER ==========\n")

        print(result["answer"])

        print("\n========== SOURCES ==========\n")

        for source in result["sources"]:

            print(
                f"File: {source['file']}"
            )

            print(
                source["content"][:150]
            )

            print("-" * 50)
