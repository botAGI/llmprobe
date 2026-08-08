"""Tests proving the mock server behaviors are real.

Hermetic: no network, no real inference server. Uses FastAPI's in-process
TestClient against the scripted mock in tests/mocks/server.py.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.mocks.server import make_mock_server, _split_words, EMBED_DIM


def _make_input(num_words: int, last_word: str) -> str:
    words = [f"tok{i}" for i in range(num_words - 1)] + [last_word]
    return " ".join(words)


def test_props_shape_omits_n_batch_and_n_ubatch() -> None:
    server = make_mock_server(max_tokens=64, behavior="honest")
    client = TestClient(server)
    resp = client.get("/props")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_slots" in data
    assert "default_generation_settings" in data
    assert "n_ctx" in data["default_generation_settings"]
    assert "build_info" in data
    assert "n_batch" not in data
    assert "n_ubatch" not in data
    assert data["total_slots"] == 1


def test_tokenize_one_token_per_word() -> None:
    server = make_mock_server(max_tokens=16, behavior="honest")
    client = TestClient(server)
    resp = client.post("/tokenize", json={"content": "hello brave new world"})
    assert resp.status_code == 200
    assert resp.json()["tokens"] == ["hello", "brave", "new", "world"]


def test_models_lists_mock() -> None:
    server = make_mock_server(max_tokens=16, behavior="honest")
    client = TestClient(server)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert {"id": "mock"} in resp.json()["data"]


@pytest.mark.parametrize(
    "behavior, should_differ",
    [("honest", True), ("silent_truncation", False)],
)
def test_embedding_tail_detection(behavior: str, should_differ: bool) -> None:
    server = make_mock_server(max_tokens=64, behavior=behavior)
    client = TestClient(server)

    n = 64 + 50
    a = _make_input(n, "FIRSTDIVERGENT")
    b = _make_input(n, "SECONDDIVERGENT")

    ra = client.post("/v1/embeddings", json={"input": a, "model": "mock"})
    rb = client.post("/v1/embeddings", json={"input": b, "model": "mock"})
    assert ra.status_code == 200
    assert rb.status_code == 200

    va = ra.json()["data"][0]["embedding"]
    vb = rb.json()["data"][0]["embedding"]
    assert len(va) == EMBED_DIM
    assert len(vb) == EMBED_DIM
    assert (va == vb) is (not should_differ)


def test_honest_embedding_uses_full_input() -> None:
    server = make_mock_server(max_tokens=8, behavior="honest")
    client = TestClient(server)
    resp = client.post(
        "/v1/embeddings",
        json={"input": "one two three four", "model": "mock"},
    )
    assert resp.status_code == 200
    assert len(resp.json()["data"][0]["embedding"]) == EMBED_DIM


def test_hard_error_500s_on_overflow() -> None:
    server = make_mock_server(max_tokens=64, behavior="hard_error")
    client = TestClient(server)
    resp = client.post(
        "/v1/embeddings",
        json={"input": _make_input(64 + 1, "boom"), "model": "mock"},
    )
    assert resp.status_code == 500


def test_hard_error_ok_under_limit() -> None:
    server = make_mock_server(max_tokens=64, behavior="hard_error")
    client = TestClient(server)
    resp = client.post(
        "/v1/embeddings",
        json={"input": _make_input(64, "ok"), "model": "mock"},
    )
    assert resp.status_code == 200


def test_chat_completions_echoes_first_word() -> None:
    server = make_mock_server(max_tokens=16, behavior="honest")
    client = TestClient(server)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock",
            "messages": [{"role": "user", "content": "hello world from mock"}],
        },
    )
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    assert content == "hello"


def test_chat_completions_silent_truncation_drops_tail() -> None:
    server = make_mock_server(max_tokens=8, behavior="silent_truncation")
    client = TestClient(server)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock",
            "messages": [{"role": "user", "content": "a b c d e f g h i j"}],
        },
    )
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    assert len(_split_words(resp.json()["choices"][0]["message"]["content"])) == 1


def test_make_mock_server_rejects_unknown_behavior() -> None:
    with pytest.raises(ValueError):
        make_mock_server(max_tokens=8, behavior="nonsense")
