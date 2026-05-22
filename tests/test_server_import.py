"""Tests for server import and health endpoint."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_server_import_and_health():
    try:
        from fastapi.testclient import TestClient
    except Exception:
        print("  [SKIP] fastapi chưa cài")
        return
    from server.app import app

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert "cuda" in data
    print("  [OK] test_server_import_and_health")


def run_all():
    print("Đang chạy test_server_import...")
    test_server_import_and_health()
    print("test_server_import: TẤT CẢ ĐỀU PASS\n")


if __name__ == "__main__":
    run_all()
