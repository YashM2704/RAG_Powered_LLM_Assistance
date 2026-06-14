import contextlib
import io
import unittest
from unittest import mock

from app.retrieval import rag


def make_hit(source_file, content, source_chunk_index):
    return {
        "_id": f"{source_file}-{source_chunk_index}",
        "_source": {
            "source_file": source_file,
            "content": content,
            "source_chunk_index": source_chunk_index,
        },
    }


class RagDocumentWideTests(unittest.TestCase):
    def test_document_wide_assignment_query_bypasses_ranking_and_llm(self):
        docs = [
            make_hit(
                "ML Unit - 2 Assignment.pdf",
                "1. Explain support vector machines.",
                1
            ),
            make_hit(
                "ML Unit - 2 Assignment.pdf",
                "2. What is clustering?",
                2
            ),
            make_hit(
                "ML Unit - 2 Assignment.pdf",
                "3. Describe the k-means algorithm.",
                3
            ),
            make_hit(
                "ML Unit - 2 Assignment.pdf",
                "4. Compare supervised and unsupervised learning.",
                4
            ),
            make_hit(
                "ML Unit - 2 Assignment.pdf",
                "5. Apply Naive Bayes classification.",
                5
            ),
        ]

        with mock.patch.object(
            rag,
            "retrieve_by_document_reference",
            return_value=(
                docs,
                [
                    "ML Unit - 2 Assignment.pdf"
                ]
            )
        ) as metadata_retrieval, mock.patch.object(
            rag,
            "run_hybrid_search",
            side_effect=AssertionError("hybrid search should not run")
        ) as hybrid_search, mock.patch.object(
            rag,
            "rerank_documents",
            side_effect=AssertionError("reranking should not run")
        ) as rerank, mock.patch.object(
            rag,
            "get_llm",
            side_effect=AssertionError("LLM should not run")
        ) as get_llm:
            with contextlib.redirect_stdout(io.StringIO()):
                result = rag.ask_question(
                    "Return all question chunks from Assignment 02"
                )

        metadata_retrieval.assert_called_once()
        hybrid_search.assert_not_called()
        rerank.assert_not_called()
        get_llm.assert_not_called()

        expected_answer = "\n\n".join(
            doc["_source"]["content"]
            for doc in docs
        )

        self.assertEqual(result["answer"], expected_answer)
        self.assertEqual(len(result["sources"]), 5)


if __name__ == "__main__":
    unittest.main()
