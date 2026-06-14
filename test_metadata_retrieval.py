import unittest

from app.retrieval.metadata_retriever import (
    detect_document_references,
    is_document_wide_query,
    retrieve_by_document_reference,
)


SOURCE_FILES = [
    "ML Unit - 1 Assignment.pdf",
    "ML Unit - 2 Assignment.pdf",
    "Yash_Mahajan_Resume(1).pdf",
]


def make_hit(source_file, content, source_chunk_index):
    return {
        "_id": f"{source_file}-{source_chunk_index}",
        "_source": {
            "source_file": source_file,
            "content": content,
            "source_chunk_index": source_chunk_index,
        },
    }


class RecordingSearchClient:
    def __init__(self, hits):
        self.hits = hits
        self.search_calls = []
        self.scroll_calls = []
        self.clear_scroll_calls = []

    def search(self, **kwargs):
        self.search_calls.append(kwargs)

        return {
            "_scroll_id": "scroll-1",
            "hits": {
                "hits": self.hits
            },
        }

    def scroll(self, **kwargs):
        self.scroll_calls.append(kwargs)

        return {
            "_scroll_id": "scroll-1",
            "hits": {
                "hits": []
            },
        }

    def clear_scroll(self, **kwargs):
        self.clear_scroll_calls.append(kwargs)


class MetadataRetrievalTests(unittest.TestCase):
    def assert_terms_filter(self, client, expected_source_file):
        body = client.search_calls[0]["body"]
        terms_filter = body["query"]["bool"]["filter"][0]["terms"]

        self.assertEqual(
            terms_filter,
            {
                "source_file": [
                    expected_source_file
                ]
            }
        )

    def assert_contents(self, docs, expected_contents):
        self.assertEqual(
            [
                doc["_source"]["content"]
                for doc in docs
            ],
            expected_contents
        )

    def test_detects_assignment_no_02(self):
        matches = detect_document_references(
            "Return all question chunks from Assignment 02",
            SOURCE_FILES
        )

        self.assertEqual(
            matches,
            [
                "ML Unit - 2 Assignment.pdf"
            ]
        )

    def test_detects_document_wide_query_phrases(self):
        queries = [
            "all questions from Assignment 02",
            "list questions from Assignment 02",
            "show assignment 02",
            "full assignment 02",
            "return all question chunks from Assignment 02",
        ]

        for query in queries:
            with self.subTest(query=query):
                self.assertTrue(
                    is_document_wide_query(query)
                )

    def test_assignment_no_02_returns_questions_without_admin_chunks(self):
        hits = [
            make_hit(
                "ML Unit - 2 Assignment.pdf",
                "Assignment No: 02\nDate of Submission: 2024-01-01\n"
                "Submitted by: Student",
                0
            ),
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
                "Describe the k-means algorithm with an example.",
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
            make_hit(
                "ML Unit - 2 Assignment.pdf",
                "a a Explain feature extraction techniques.",
                6
            ),
        ]
        client = RecordingSearchClient(hits)

        docs, matched_source_files = retrieve_by_document_reference(
            "Return all question chunks from Assignment 02",
            source_files=SOURCE_FILES,
            search_client=client
        )

        self.assertEqual(
            matched_source_files,
            [
                "ML Unit - 2 Assignment.pdf"
            ]
        )
        self.assert_terms_filter(
            client,
            "ML Unit - 2 Assignment.pdf"
        )
        self.assertEqual(len(docs), 6)
        self.assert_contents(
            docs,
            [
                "1. Explain support vector machines.",
                "2. What is clustering?",
                "Describe the k-means algorithm with an example.",
                "4. Compare supervised and unsupervised learning.",
                "5. Apply Naive Bayes classification.",
                "a a Explain feature extraction techniques.",
            ]
        )

    def test_assignment_no_01_returns_questions_without_admin_chunks(self):
        hits = [
            make_hit(
                "ML Unit - 1 Assignment.pdf",
                "Assignment No: 01\nRoll No: 123\nDate of Submission: 2024",
                0
            ),
            make_hit("ML Unit - 1 Assignment.pdf", "1. Define PCA.", 1),
            make_hit("ML Unit - 1 Assignment.pdf", "2. Explain regression.", 2),
            make_hit(
                "ML Unit - 1 Assignment.pdf",
                "3 Explain dimensionality reduction.",
                3
            ),
            make_hit(
                "ML Unit - 1 Assignment.pdf",
                "4. Calculate covariance matrix.",
                4
            ),
            make_hit(
                "ML Unit - 1 Assignment.pdf",
                "Evaluate model accuracy using suitable metrics.",
                5
            ),
        ]
        client = RecordingSearchClient(hits)

        docs, matched_source_files = retrieve_by_document_reference(
            "List all questions from Assignment No 01",
            source_files=SOURCE_FILES,
            search_client=client
        )

        self.assertEqual(
            matched_source_files,
            [
                "ML Unit - 1 Assignment.pdf"
            ]
        )
        self.assert_terms_filter(
            client,
            "ML Unit - 1 Assignment.pdf"
        )
        self.assertEqual(len(docs), 5)
        self.assert_contents(
            docs,
            [
                "1. Define PCA.",
                "2. Explain regression.",
                "3 Explain dimensionality reduction.",
                "4. Calculate covariance matrix.",
                "Evaluate model accuracy using suitable metrics.",
            ]
        )

    def test_resume_project_query_prioritizes_projects_section(self):
        hits = [
            make_hit(
                "Yash_Mahajan_Resume(1).pdf",
                "Professional Summary\nBackend developer with Python.",
                0
            ),
            make_hit(
                "Yash_Mahajan_Resume(1).pdf",
                "Projects\nRAG Assistant - Built retrieval over PDFs.",
                1
            ),
            make_hit(
                "Yash_Mahajan_Resume(1).pdf",
                "Resume Parser - Extracted structured resume data.",
                2
            ),
            make_hit(
                "Yash_Mahajan_Resume(1).pdf",
                "Skills\nPython, FastAPI, OpenSearch.",
                3
            ),
        ]
        client = RecordingSearchClient(hits)

        docs, matched_source_files = retrieve_by_document_reference(
            "What projects are mentioned in the resume?",
            source_files=SOURCE_FILES,
            search_client=client
        )

        self.assertEqual(
            matched_source_files,
            [
                "Yash_Mahajan_Resume(1).pdf"
            ]
        )
        self.assert_terms_filter(
            client,
            "Yash_Mahajan_Resume(1).pdf"
        )
        self.assert_contents(
            docs,
            [
                "Projects\nRAG Assistant - Built retrieval over PDFs.",
                "Resume Parser - Extracted structured resume data.",
            ]
        )

    def test_no_duplicate_sources_returned(self):
        hits = [
            make_hit(
                "ML Unit - 2 Assignment.pdf",
                "1. Explain support vector machines.",
                1
            ),
            make_hit(
                "ML Unit - 2 Assignment.pdf",
                "1. Explain support vector machines.",
                2
            ),
            make_hit(
                "ML Unit - 2 Assignment.pdf",
                "2. What is clustering?",
                3
            ),
        ]
        client = RecordingSearchClient(hits)

        docs, _ = retrieve_by_document_reference(
            "List all questions from Assignment No 02",
            source_files=SOURCE_FILES,
            search_client=client
        )

        returned_sources = [
            (
                doc["_source"]["source_file"],
                doc["_source"]["content"]
            )
            for doc in docs
        ]

        self.assertEqual(
            returned_sources,
            list(dict.fromkeys(returned_sources))
        )
        self.assert_contents(
            docs,
            [
                "1. Explain support vector machines.",
                "2. What is clustering?",
            ]
        )

    def test_normal_semantic_question_does_not_use_metadata_filter(self):
        docs, matched_source_files = retrieve_by_document_reference(
            "Describe Principal Component Analysis PCA",
            source_files=SOURCE_FILES,
            search_client=RecordingSearchClient([])
        )

        self.assertEqual(matched_source_files, [])
        self.assertEqual(docs, [])


if __name__ == "__main__":
    unittest.main()
