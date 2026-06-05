from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder
import re
CHROMA_DB_DIR = "chroma_db"

# Load embeddings once
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

llm = ChatOllama(
    model="llama3"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
def get_vector_store():

    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )


prompt = ChatPromptTemplate.from_template(
    """
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
)


def ask_question(question: str):

    question_lower = question.lower()
    vector_store = get_vector_store()

    # MMR Retrieval
    docs = vector_store.max_marginal_relevance_search(
        query=question,
        k=10,
        fetch_k=20
    )
    
    docs = rerank_documents(
    question,
    docs
    )
    
    docs = docs[:4]

    if "email" in question_lower:
        email_match = extract_email(docs)
        if email_match:
            email, source_doc = email_match
            return {
                "answer": email,
                "sources": [
                    {
                        "file": source_doc.metadata.get(
                            "source_file",
                            "Unknown File"
                        ),
                        "page": source_doc.metadata.get(
                            "page",
                            "unknown"
                        ),
                        "content": source_doc.page_content[:300]
                    }
                ]
            }

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

def rerank_documents(question, docs):

    pairs = [
        (question, doc.page_content)
        for doc in docs
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    print("\n========== RERANK SCORES ==========")

    for doc, score in ranked:
        print(
            f"{score:.4f} | "
            f"{doc.metadata.get('source_file')}"
        )

    return [doc for doc, _ in ranked]

def extract_email(docs):

    pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"

    for doc in docs:

        match = re.search(
            pattern,
            doc.page_content
        )

        if match:
            return match.group(), doc

    return None

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
