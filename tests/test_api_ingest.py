import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_ingestion
from app.ingestion.service import IngestionService
from app.main import app


TEXT_A = (
    "Asana is a project-management tool. It ships list, board, calendar, timeline "
    "and Gantt-style views, plus workload and goal features. Pricing starts at a "
    "free plan for individuals. "
) * 4
TEXT_B = (
    "Trello is a board-based project-management tool from Atlassian. Work happens "
    "on cards that move left to right across columns. It offers Power-Ups for "
    "automation and integration. "
) * 4


@pytest.fixture()
def client() -> TestClient:
    service = IngestionService()
    service.persist = None
    app.dependency_overrides[get_ingestion] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_ingestion, None)


def test_ingest_text_documents_returns_aggregate(client: TestClient) -> None:
    response = client.post(
        "/tenants/acme/ingest",
        json={
            "documents": [
                {"text": TEXT_A, "source": "manual-1", "brand": "Asana"},
                {"text": TEXT_B, "source": "manual-2", "brand": "Trello"},
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["documents_stored"] == 2
    assert body["chunks_stored"] > 0
    assert body["errors"] == []


def test_ingest_mixed_url_and_text_documents(client: TestClient) -> None:
    response = client.post(
        "/tenants/globex/ingest",
        json={
            "documents": [
                {"text": TEXT_A, "source": "manual-1", "brand": "Asana"},
                {"text": TEXT_B, "brand": "Trello"},
            ]
        },
    )
    assert response.status_code == 200
    assert response.json()["documents_stored"] == 2


def test_ingest_empty_documents_list_is_422(client: TestClient) -> None:
    response = client.post("/tenants/acme/ingest", json={"documents": []})
    assert response.status_code == 422


def test_ingest_url_and_text_together_is_422(client: TestClient) -> None:
    response = client.post(
        "/tenants/acme/ingest",
        json={
            "documents": [
                {"url": "https://example.com/a", "text": TEXT_A, "brand": "Asana"}
            ]
        },
    )
    assert response.status_code == 422


def test_ingest_missing_brand_is_422(client: TestClient) -> None:
    response = client.post(
        "/tenants/acme/ingest",
        json={"documents": [{"text": TEXT_A, "source": "manual-1"}]},
    )
    assert response.status_code == 422


def test_tenants_are_isolated_across_requests(client: TestClient) -> None:
    for tenant in ("acme", "globex"):
        response = client.post(
            f"/tenants/{tenant}/ingest",
            json={"documents": [{"text": TEXT_A, "source": "s", "brand": "Asana"}]},
        )
        assert response.status_code == 200
        assert response.json()["documents_stored"] == 1