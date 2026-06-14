import os
import re
import hashlib
from typing import List, Optional, Set, Tuple


SOURCE_FILE_AGG_SIZE = 1000
METADATA_BATCH_SIZE = 500
DEFAULT_INDEX_NAME = "documents"
RESUME_SECTION_HEADINGS = (
    "professional summary",
    "summary",
    "projects",
    "skills",
    "experience",
    "education",
    "certifications",
    "achievements",
)

DOCUMENT_WIDE_QUERY_PATTERNS = (
    r"\ball\s+questions?\b",
    r"\blist\s+(?:all\s+)?questions?\b",
    r"\bshow\s+(?:me\s+)?(?:the\s+)?assignment\b",
    r"\bfull\s+assignment\b",
    r"\bcomplete\s+assignment\b",
    r"\breturn\s+all\s+question\s+chunks?\b",
)


def _resolve_search_client(search_client):
    if search_client is not None:
        return search_client

    from app.opensearch_client import client

    return client


def _resolve_index_name(index_name: Optional[str]) -> str:
    if index_name is not None:
        return index_name

    try:
        from app.opensearch_client import INDEX_NAME

        return INDEX_NAME
    except ModuleNotFoundError:
        return DEFAULT_INDEX_NAME


def normalize_identifier(value: str) -> str:
    normalized = re.sub(
        r"[^a-z0-9]+",
        " ",
        value.lower()
    )

    return re.sub(
        r"\s+",
        " ",
        normalized
    ).strip()


def _content(doc: dict) -> str:
    return doc.get("_source", {}).get("content", "")


def _source_file(doc: dict) -> str:
    return doc.get("_source", {}).get("source_file", "")


def _content_fingerprint(text: str) -> str:
    normalized = re.sub(
        r"\s+",
        " ",
        text.strip().lower()
    )

    return hashlib.sha1(
        normalized.encode("utf-8")
    ).hexdigest()


def deduplicate_documents(docs: List[dict]) -> List[dict]:
    seen = set()
    unique_docs = []

    for doc in docs:
        source = doc.get("_source", {})
        content = source.get("content", "")

        if not content.strip():
            continue

        key = (
            source.get("source_file", ""),
            _content_fingerprint(content)
        )

        if key in seen:
            continue

        seen.add(key)
        unique_docs.append(doc)

    return unique_docs


def _line_normalized_values(text: str) -> List[str]:
    return [
        normalize_identifier(line)
        for line in text.splitlines()
        if line.strip()
    ]


def _has_resume_section_heading(text: str, heading: str) -> bool:
    normalized_lines = _line_normalized_values(text)

    return any(
        line == heading
        or line.startswith(f"{heading} ")
        for line in normalized_lines
    )


def _is_project_chunk(text: str) -> bool:
    normalized = normalize_identifier(text)

    return (
        "project" in normalized
        or bool(
            re.search(
                r"\b(github|deployed|built|developed|implemented)\b",
                normalized
            )
        )
    )


def _is_question_chunk(text: str) -> bool:
    stripped = text.strip()
    normalized = normalize_identifier(stripped)

    if not stripped:
        return False

    question_patterns = [
        r"(?im)^\s*q(?:uestion)?\.?\s*\d+\s*[\).:-]?\s+\S+",
        r"(?im)^\s*\d+\s*[\).:-]\s+(?:what|why|how|explain|describe|define|discuss|compare|differentiate|write|state|list|derive|implement|apply|calculate|evaluate|solve|perform|use|create|build)\b",
        r"(?im)^\s*\d+\s+(?:what|why|how|explain|describe|define|discuss|compare|differentiate|write|state|list|derive|implement|apply|calculate|evaluate|solve|perform|use|create|build)\b",
        r"(?im)^\s*(?:what|why|how|explain|describe|define|discuss|compare|differentiate|write|state|list|derive|implement|apply|calculate|evaluate|solve|perform|use|create|build)\b.{20,}",
        r"(?im)^\s*(?:[a-z]\s+){1,3}(?:what|why|how|explain|describe|define|discuss|compare|differentiate|write|state|list|derive|implement|apply|calculate|evaluate|solve|perform|use|create|build)\b.{20,}",
        r"(?im)^\s*\d+\s*[\).:-].*\?",
    ]

    if any(
        re.search(pattern, stripped)
        for pattern in question_patterns
    ):
        return True

    return (
        "?" in stripped
        and bool(
            re.search(
                r"\b(what|why|how|explain|describe|define|discuss|compare|differentiate|write|state|list)\b",
                normalized
            )
        )
    )


def _is_administrative_chunk(doc: dict) -> bool:
    text = _content(doc)
    normalized = normalize_identifier(text)

    if not normalized:
        return True

    if _is_question_chunk(text) or _is_project_chunk(text):
        return False

    admin_patterns = [
        r"\bpage\s+\d+\s+of\s+\d+\b",
        r"\bassignment\s+no\b",
        r"\bdate\s+of\s+submission\b",
        r"\bsubmitted\s+(?:by|to)\b",
        r"\broll\s+no\b",
        r"\benrollment\s+no\b",
        r"\bname\s+of\s+(?:student|faculty)\b",
        r"\bdepartment\s+of\b",
        r"\bacademic\s+year\b",
    ]
    matches = sum(
        1
        for pattern in admin_patterns
        if re.search(pattern, normalized)
    )

    if matches >= 2:
        return True

    return (
        matches == 1
        and len(normalized.split()) <= 30
    )


def filter_administrative_noise(docs: List[dict]) -> List[dict]:
    return [
        doc
        for doc in docs
        if not _is_administrative_chunk(doc)
    ]


def _resume_section_for_chunk(text: str, current_section: Optional[str]) -> Optional[str]:
    normalized_lines = _line_normalized_values(text)
    detected_section = current_section

    for line in normalized_lines:
        for heading in RESUME_SECTION_HEADINGS:
            if line == heading or line.startswith(f"{heading} "):
                detected_section = heading

    return detected_section


def _annotate_resume_sections(docs: List[dict]) -> List[Tuple[dict, Optional[str]]]:
    annotated = []
    current_sections = {}

    for doc in sorted(
        docs,
        key=lambda item: (
            _source_file(item),
            item.get("_source", {}).get("source_chunk_index", 0)
        )
    ):
        source_file = _source_file(doc)
        current_section = current_sections.get(source_file)
        section = _resume_section_for_chunk(
            _content(doc),
            current_section
        )
        current_sections[source_file] = section
        annotated.append(
            (
                doc,
                section
            )
        )

    return annotated


def _is_resume_query(query: str, docs: List[dict]) -> bool:
    query_normalized = normalize_identifier(query)

    return (
        "resume" in query_normalized
        or any(
            "resume" in normalize_identifier(_source_file(doc))
            for doc in docs
        )
    )


def _is_assignment_query(query: str, docs: List[dict]) -> bool:
    return (
        bool(_assignment_numbers(normalize_identifier(query)))
        or any(
            "assignment" in normalize_identifier(_source_file(doc))
            for doc in docs
        )
    )


def is_document_wide_query(query: str) -> bool:
    query_normalized = normalize_identifier(query)

    return any(
        re.search(pattern, query_normalized)
        for pattern in DOCUMENT_WIDE_QUERY_PATTERNS
    )


def _is_project_query(query: str) -> bool:
    query_normalized = normalize_identifier(query)

    return "project" in query_normalized


def _is_question_query(query: str) -> bool:
    query_normalized = normalize_identifier(query)

    return (
        "question" in query_normalized
        or "questions" in query_normalized
        or is_document_wide_query(query)
    )


def apply_section_aware_filters(
    query: str,
    docs: List[dict]
) -> List[dict]:
    docs = deduplicate_documents(docs)
    docs = filter_administrative_noise(docs)

    if not docs:
        return []

    if _is_resume_query(query, docs) and _is_project_query(query):
        annotated = _annotate_resume_sections(docs)
        project_section_docs = [
            doc
            for doc, section in annotated
            if section == "projects"
        ]

        if project_section_docs:
            return project_section_docs

        project_docs = [
            doc
            for doc in docs
            if _is_project_chunk(_content(doc))
        ]

        if project_docs:
            return project_docs

    if _is_assignment_query(query, docs) and _is_question_query(query):
        question_docs = docs

        if question_docs:
            return question_docs

    return docs


def format_document_wide_answer(docs: List[dict]) -> str:
    return "\n\n".join(
        _content(doc).strip()
        for doc in docs
        if _content(doc).strip()
    )


def list_source_files(
    search_client=None,
    index_name: Optional[str] = None
) -> List[str]:
    search_client = _resolve_search_client(search_client)
    index_name = _resolve_index_name(index_name)

    response = search_client.search(
        index=index_name,
        body={
            "size": 0,
            "aggs": {
                "source_files": {
                    "terms": {
                        "field": "source_file",
                        "size": SOURCE_FILE_AGG_SIZE
                    }
                }
            }
        }
    )

    buckets = response.get(
        "aggregations",
        {}
    ).get(
        "source_files",
        {}
    ).get(
        "buckets",
        []
    )

    return [
        bucket["key"]
        for bucket in buckets
    ]


def _assignment_numbers(query: str) -> Set[int]:
    patterns = [
        r"\bassignment\s*(?:no|number|#|:)?\s*0*(\d+)\b",
        r"\bunit\s*0*(\d+)\s*assignment\b",
    ]
    numbers = set()

    for pattern in patterns:
        for match in re.finditer(
            pattern,
            query
        ):
            numbers.add(
                int(match.group(1))
            )

    return numbers


def _numbers(value: str) -> Set[int]:
    return {
        int(number)
        for number in re.findall(
            r"\d+",
            value
        )
    }


def detect_document_references(
    query: str,
    source_files: List[str]
) -> List[str]:
    query_normalized = normalize_identifier(query)
    assignment_numbers = _assignment_numbers(query_normalized)
    matches = []

    for source_file in source_files:
        file_normalized = normalize_identifier(source_file)
        stem = os.path.splitext(source_file)[0]
        stem_normalized = normalize_identifier(stem)

        is_full_name_match = (
            file_normalized
            and file_normalized in query_normalized
        )
        is_stem_match = (
            stem_normalized
            and stem_normalized in query_normalized
        )
        is_resume_match = (
            "resume" in query_normalized
            and "resume" in file_normalized
        )
        is_assignment_match = (
            bool(assignment_numbers)
            and "assignment" in file_normalized
            and bool(assignment_numbers & _numbers(file_normalized))
        )

        if (
            is_full_name_match
            or is_stem_match
            or is_resume_match
            or is_assignment_match
        ):
            matches.append(source_file)

    return list(dict.fromkeys(matches))


def search_by_source_files(
    source_files: List[str],
    search_client=None,
    index_name: Optional[str] = None,
    batch_size: int = METADATA_BATCH_SIZE
) -> List[dict]:
    if not source_files:
        return []

    search_client = _resolve_search_client(search_client)
    index_name = _resolve_index_name(index_name)

    body = {
        "size": batch_size,
        "query": {
            "bool": {
                "filter": [
                    {
                        "terms": {
                            "source_file": source_files
                        }
                    }
                ]
            }
        },
        "sort": [
            {
                "_doc": "asc"
            }
        ]
    }

    response = search_client.search(
        index=index_name,
        body=body,
        scroll="1m",
        size=batch_size
    )

    scroll_id = response.get("_scroll_id")
    hits = response.get("hits", {}).get("hits", [])
    documents = list(hits)

    try:
        while hits and scroll_id:
            response = search_client.scroll(
                scroll_id=scroll_id,
                scroll="1m"
            )
            scroll_id = response.get(
                "_scroll_id",
                scroll_id
            )
            hits = response.get("hits", {}).get("hits", [])
            documents.extend(hits)
    finally:
        if scroll_id:
            search_client.clear_scroll(
                scroll_id=scroll_id
            )

    return sorted(
        documents,
        key=lambda doc: (
            doc["_source"].get("source_file", ""),
            doc["_source"].get("source_chunk_index", 0)
        )
    )


def retrieve_by_document_reference(
    query: str,
    source_files: Optional[List[str]] = None,
    search_client=None,
    index_name: Optional[str] = None
) -> Tuple[List[dict], List[str]]:
    index_name = _resolve_index_name(index_name)

    if source_files is None:
        source_files = list_source_files(
            search_client=search_client,
            index_name=index_name
        )

    matching_source_files = detect_document_references(
        query,
        source_files
    )

    if not matching_source_files:
        return [], []

    docs = search_by_source_files(
        matching_source_files,
        search_client=search_client,
        index_name=index_name
    )

    return (
        apply_section_aware_filters(
            query,
            docs
        ),
        matching_source_files
    )
