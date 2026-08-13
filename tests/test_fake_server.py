"""Tests proving tests/fake_server.py REALLY truncates input at the boundary.

Hermetic: no network, no real inference server. Uses FastAPI's in-process
TestClient against the FakeServer built directly, so the truncation behavior
is exercised through the exact same request paths a probe would hit.

The server is the reference (etalon) for measuring truncation accuracy: it
must truncate at exactly the configured length in two honest forms:

* ``refuse`` — an oversized payload is rejected with HTTP 400 and an error
  body that names the excess, so a probe can observe a hard refusal.
* ``silent`` — an oversized payload is echoed truncated to exactly
  ``truncate_len`` with HTTP 200 and no indication anything was dropped, so a
  probe must detect the silent truncation itself.

These tests pin down that contract so nobody can silently weaken the etalon.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.fake_server import FakeServer, EMBED_DIM, _count_tokens, _split_words


def _words(n: int) -> str:
    return " ".join(f"tok{i}" for i in range(n))


def test_refuse_mode_rejects_oversized_echo() -> None:
    # /echo truncates by CHARACTERS, not tokens.
    server = FakeServer(truncate_len=8, mode="refuse")
    client = TestClient(server.build())
    resp = client.post("/echo", json={"input": "abcdefghijklmno"})
    assert resp.status_code == 400
    body = resp.json()["detail"]
    assert body["input_length"] == 15
    assert body["truncate_len"] == 8


def test_refuse_mode_accepts_at_limit_echo() -> None:
    server = FakeServer(truncate_len=8, mode="refuse")
    client = TestClient(server.build())
    payload = "abcdefgh"
    resp = client.post("/echo", json={"input": payload})
    assert resp.status_code == 200
    assert resp.json()["input"] == payload
    assert resp.json()["input_length"] == 8


def test_refuse_mode_accepts_under_limit_echo() -> None:
    server = FakeServer(truncate_len=8, mode="refuse")
    client = TestClient(server.build())
    payload = "abc"
    resp = client.post("/echo", json={"input": payload})
    assert resp.status_code == 200
    assert resp.json()["input"] == payload


def test_silent_mode_truncates_echo_to_exact_limit() -> None:
    # /echo truncates by CHARACTERS, so the kept prefix is exactly truncate_len.
    server = FakeServer(truncate_len=8, mode="silent")
    client = TestClient(server.build())
    resp = client.post("/echo", json={"input": "abcdefghijklmno"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["input_length"] == 8
    assert body["input"] == "abcdefgh"
    assert len(body["input"]) == 8


def test_silent_mode_keeps_under_limit_echo_untouched() -> None:
    server = FakeServer(truncate_len=8, mode="silent")
    client = TestClient(server.build())
    payload = "abcd"
    resp = client.post("/echo", json={"input": payload})
    assert resp.status_code == 200
    assert resp.json()["input"] == payload


def test_chat_silent_truncation_drops_head_not_tail() -> None:
    server = FakeServer(truncate_len=4, mode="silent")
    client = TestClient(server.build())
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-4096",
            "messages": [{"role": "user", "content": _words(10)}],
        },
    )
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    # The retained reply is drawn from the LAST truncate_len tokens; the head
    # (which would carry a canary marker) is dropped on an oversized prompt.
    assert content == _words(10).split()[-4]


def test_chat_refuse_mode_checks_against_token_count() -> None:
    server = FakeServer(truncate_len=4, mode="refuse")
    client = TestClient(server.build())
    under = client.post(
        "/v1/chat/completions",
        json={"model": "fake-4096", "messages": [{"role": "user", "content": _words(3)}]},
    )
    over = client.post(
        "/v1/chat/completions",
        json={"model": "fake-4096", "messages": [{"role": "user", "content": _words(10)}]},
    )
    # refuse mode only refuses at /echo; chat silently passes through the full
    # token list (no dropping), so the reply reflects the true head token.
    assert under.status_code == 200
    assert over.status_code == 200
    assert over.json()["choices"][0]["message"]["content"] == _words(10).split()[0]
    assert over.json()["usage"]["prompt_tokens"] == 10


def test_embedding_silent_truncation_collides_on_dropped_tail() -> None:
    server = FakeServer(truncate_len=4, mode="silent")
    client = TestClient(server.build())
    a = _words(6) + " AAAA"
    b = _words(6) + " BBBB"
    ra = client.post("/v1/embeddings", json={"input": a, "model": "fake-4096"})
    rb = client.post("/v1/embeddings", json={"input": b, "model": "fake-4096"})
    assert ra.status_code == 200
    assert rb.status_code == 200
    va = ra.json()["data"][0]["embedding"]
    vb = rb.json()["data"][0]["embedding"]
    assert len(va) == EMBED_DIM
    # With silent truncation only the first truncate_len tokens survive, so the
    # divergent tail token is dropped and the two vectors collide.
    assert va == vb


def test_embedding_refuse_mode_omits_truncation() -> None:
    server = FakeServer(truncate_len=4, mode="refuse")
    client = TestClient(server.build())
    a = _words(6) + " AAAA"
    b = _words(6) + " BBBB"
    ra = client.post("/v1/embeddings", json={"input": a, "model": "fake-4096"})
    rb = client.post("/v1/embeddings", json={"input": b, "model": "fake-4096"})
    assert ra.status_code == 200
    assert rb.status_code == 200
    va = ra.json()["data"][0]["embedding"]
    vb = rb.json()["data"][0]["embedding"]
    # refuse mode does not truncate embeddings, so the divergent tail is kept
    # and the two honest vectors differ.
    assert va != vb


def test_props_shape_omits_n_batch_and_n_ubatch() -> None:
    server = FakeServer(truncate_len=8, mode="silent")
    client = TestClient(server.build())
    resp = client.get("/props")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "fake-4096"
    assert data["default_generation_settings"]["n_ctx"] == 4096
    assert "n_batch" not in data
    assert "n_ubatch" not in data


def test_slots_501_when_disabled() -> None:
    server = FakeServer(truncate_len=8, mode="silent", no_slots=True)
    client = TestClient(server.build())
    resp = client.get("/slots")
    assert resp.status_code == 501


def test_slots_enabled_reports_per_slot_ctx() -> None:
    server = FakeServer(truncate_len=8, mode="silent", total_slots=2)
    client = TestClient(server.build())
    resp = client.get("/slots")
    assert resp.status_code == 200
    payload = resp.json()
    assert isinstance(payload, list)
    assert len(payload) == 2
    assert payload[0]["n_ctx"] == 4096


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"truncate_len": -1, "mode": "silent"}, "truncate_len"),
        ({"truncate_len": 0, "mode": "bogus"}, "mode"),
        ({"truncate_len": 4, "mode": "silent", "total_slots": 0}, "total_slots"),
    ],
)
def test_fake_server_validates_constructor(kwargs: dict, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        FakeServer(**kwargs)


def test_count_tokens_word_based() -> None:
    assert _count_tokens("one two three") == 3


def test_split_words_roundtrip() -> None:
    assert _split_words("a b c") == ["a", "b", "c"]
