from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

import os
import shutil

PDF_FOLDER = "data/documents"
CHROMA_DB_DIR = "chroma_db"


def load_documents():
    documents = []

    for file in os.listdir(PDF_FOLDER):

        if not file.endswith(".pdf"):
            continue

        print(f"\nFILE: {file}")

        pdf_path = os.path.join(PDF_FOLDER, file)

        loader = UnstructuredPDFLoader(
            pdf_path,
            mode="single"
        )

        docs = loader.load()

        print("Pages:", len(docs))

        for i, doc in enumerate(docs):
            print(
                f"Page {i + 1} chars:",
                len(doc.page_content)
            )

            # IMPORTANT: Store filename metadata
            doc.metadata["source_file"] = file

        documents.extend(docs)

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )

    return splitter.split_documents(documents)


def create_vector_store(chunks):

    # Remove empty chunks
    chunks = [
        chunk
        for chunk in chunks
        if chunk.page_content.strip()
    ]

    # Remove metadata Chroma cannot store
    chunks = filter_complex_metadata(chunks)

    # Keep only simple metadata types
    for chunk in chunks:
        chunk.metadata = {
            k: v
            for k, v in chunk.metadata.items()
            if isinstance(v, (str, int, float, bool))
        }

    print(f"\nValid Chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )


def ingest_documents():

    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)

    docs = load_documents()

    print("\n===== DOCUMENT CONTENT =====")

    for doc in docs[:3]:
        print(doc.page_content[:500])

    chunks = split_documents(docs)

    print("\nDocuments:", len(docs))
    print("Chunks:", len(chunks))

    create_vector_store(chunks)

    return {
        "documents": len(docs),
        "chunks": len(chunks)
    }


if __name__ == "__main__":

    result = ingest_documents()

    print(
        f"\nIndexed {result['documents']} pages into "
        f"{result['chunks']} chunks"
    )