"""Tests for the optional shared-password gate (app/main.py). No LLM calls
involved, so these just exercise routing/auth and run fully offline."""

import base64

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app


def _basic_auth_header(password: str, username: str = "anyone") -> dict:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def test_no_password_configured_allows_requests(monkeypatch):
    monkeypatch.setattr(main_module.settings, "app_password", "")
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_is_always_open_even_with_password_set(monkeypatch):
    monkeypatch.setattr(main_module.settings, "app_password", "s3cret")
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200


def test_protected_route_rejects_missing_credentials(monkeypatch):
    monkeypatch.setattr(main_module.settings, "app_password", "s3cret")
    client = TestClient(app)
    resp = client.get("/analysis/some-job-id")
    assert resp.status_code == 401
    assert "Basic" in resp.headers.get("www-authenticate", "")


def test_protected_route_rejects_wrong_password(monkeypatch):
    monkeypatch.setattr(main_module.settings, "app_password", "s3cret")
    client = TestClient(app)
    resp = client.get("/analysis/some-job-id", headers=_basic_auth_header("wrong"))
    assert resp.status_code == 401


def test_protected_route_accepts_correct_password_any_username(monkeypatch):
    monkeypatch.setattr(main_module.settings, "app_password", "s3cret")
    client = TestClient(app)
    resp = client.get("/analysis/nonexistent-job-id", headers=_basic_auth_header("s3cret", username="whoever"))
    # 404 (job not found) proves the auth check passed and the route ran.
    assert resp.status_code == 404
