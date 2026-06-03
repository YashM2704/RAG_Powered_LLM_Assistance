from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

CHROMA_DB_DIR = "chroma_db"

# Load embeddings once
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatOllama(
    model="llama3"
)


def get_vector_store():

    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )


prompt = ChatPromptTemplate.from_template(
    """
You are a document question-answering assistant.

Use ONLY the provided context.

Rules:
- Answer only from the retrieved context.
- Do not use outside knowledge.
- Do not invent information.
- If the answer is not present, respond:
  "I couldn't find that information in the documents."
- If asked to list questions, extract them exactly as written.

Context:
{context}

Question:
{question}

Answer:
"""
)


def ask_question(question: str):

    vector_store = get_vector_store()

    # MMR Retrieval
    docs = vector_store.max_marginal_relevance_search(
        query=question,
        k=8,
        fetch_k=20
    )

    print("\n========== MMR RETRIEVAL RESULTS ==========")

    for rank, doc in enumerate(docs, start=1):

        print(f"\nRank: {rank}")

        print(
            "Source File:",
            doc.metadata.get(
                "source_file",
                "Unknown File"
            )
        )

        print(
            doc.page_content[:300]
        )

        print("-" * 80)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return {
        "answer": response.content,
        "sources": [
            {
                "file": doc.metadata.get(
                    "source_file",
                    "Unknown File"
                ),
                "page": doc.metadata.get(
                    "page",
                    "unknown"
                ),
                "content": doc.page_content[:300]
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
                f"Page: {source['page']}"
            )

            print(
                source["content"][:150]
            )

            print("-" * 50)