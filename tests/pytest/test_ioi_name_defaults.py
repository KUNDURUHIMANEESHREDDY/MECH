"""IOI name selection is deterministic and self-describing.

`POST /gpt2/ioi` used `random.choice` for absent names: two identical
requests ran different experiments. The response echoed the names, so a
careful caller could reproduce -- a careless one could not, and nothing
recorded which case it was.
"""

from __future__ import annotations


def _client():
    from fastapi.testclient import TestClient
    from backend.core import auth as auth_mod
    from backend.main import app

    return TestClient(app, raise_server_exceptions=False,
                      headers=auth_mod.auth_headers())


def test_bare_calls_get_the_same_pair_twice():
    first = _client().post("/api/gpt2/ioi", json={}).json()
    second = _client().post("/api/gpt2/ioi", json={}).json()
    assert (first["io_name"], first["subj_name"]) == (
        second["io_name"], second["subj_name"])
    assert first["io_name"] != first["subj_name"]
    assert first["io_name_defaulted"] is True


def test_explicit_names_are_not_marked_defaulted():
    body = _client().post(
        "/api/gpt2/ioi", json={"io_name": "John", "subj_name": "Mary"}).json()
    assert body["io_name"] == "John"
    assert body["subj_name"] == "Mary"
    assert body["io_name_defaulted"] is False


def test_non_string_names_are_client_errors():
    r = _client().post("/api/gpt2/ioi", json={"io_name": 42})
    assert r.status_code == 400
    r = _client().post("/api/gpt2/ioi", json={"subj_name": ["Mary"]})
    assert r.status_code == 400


def test_identical_names_get_a_distinct_subject():
    body = _client().post(
        "/api/gpt2/ioi", json={"io_name": "John", "subj_name": "John"}).json()
    assert body["subj_name"] != "John"
    assert body["io_name_defaulted"] is True
