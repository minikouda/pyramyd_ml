from __future__ import annotations

import pytest


@pytest.fixture()
def toy_docs() -> list[dict]:
    # Minimal doc schema used across src/* modules.
    return [
        {
            "id": "0",
            "text": "Great work life balance and supportive team.",
            "meta": {"company": "A", "rating": 4.6, "salary_median": 120000, "locations": ["SF"]},
        },
        {
            "id": "1",
            "text": "High pay but long hours. Fast growth.",
            "meta": {"company": "B", "rating": 3.9, "salary_median": 180000, "locations": ["NY"]},
        },
        {
            "id": "2",
            "text": "Average comp, excellent culture, flexible schedule.",
            "meta": {"company": "C", "rating": 4.2, "salary_median": 130000, "locations": ["SF"]},
        },
    ]


@pytest.fixture()
def toy_results(toy_docs: list[dict]) -> list[dict]:
    # Results mimic retrieval output (id/score/text/meta).
    return [
        {"id": toy_docs[0]["id"], "score": 0.2, "text": toy_docs[0]["text"], "meta": toy_docs[0]["meta"]},
        {"id": toy_docs[1]["id"], "score": 0.9, "text": toy_docs[1]["text"], "meta": toy_docs[1]["meta"]},
        {"id": toy_docs[2]["id"], "score": 0.5, "text": toy_docs[2]["text"], "meta": toy_docs[2]["meta"]},
    ]
