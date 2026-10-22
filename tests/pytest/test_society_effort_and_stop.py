"""The cost budget and the stop check of a Society run.

Two things were wrong with a Society run before this:

* its cost was fixed by the layers below (ten prompts, five interaction
  prompts, every head of GPT-2 screened), so a caller could not ask for a
  cheaper run, and a run cost minutes of CPU whatever the caller wanted; and
* a cancelled or timed-out run kept sweeping every remaining candidate,
  because nothing observed the stop request where the work happens -- the
  worker thread cannot be killed in CPython, so the timeout freed the
  admission slot while the thread kept burning CPU.

These tests pin the fixes: the run owner decides the effort, the effort is
reported with the result, and a stop is observed between measurements.
"""

from __future__ import annotations

import threading

import pytest

from backend.agents.society import ResearchSocietyV2
from backend.interpretability.discovery.budget import (
    DiscoveryBudget,
    current_budget,
    effective_effort,
    reset_budget,
    set_budget,
    stop_requested,
)


# --------------------------------------------------------------------------
# the effort
# --------------------------------------------------------------------------

def test_effort_presets_are_bounded_and_ordered():
    from backend.api import dispatcher as core

    quick = core._SOCIETY_EFFORTS["quick"]
    standard = core._SOCIETY_EFFORTS["standard"]
    full = core._SOCIETY_EFFORTS["full"]

    # A cheaper run is a smaller sample, never a bigger one.
    assert quick["n_prompts"] < standard["n_prompts"] <= full["n_prompts"]
    assert quick["n_interaction_prompts"] <= standard["n_interaction_prompts"]
    # The standard preset is the sample the platform treats as adequate for a
    # validation claim; a quick run must not be able to claim one.
    from backend.interpretability.discovery.live_discovery import (
        MIN_PROMPTS_FOR_INTERACTION,
        MIN_PROMPTS_FOR_VALIDATION,
    )

    assert standard["n_prompts"] >= MIN_PROMPTS_FOR_VALIDATION
    assert standard["n_interaction_prompts"] >= MIN_PROMPTS_FOR_INTERACTION
    assert quick["n_prompts"] < MIN_PROMPTS_FOR_VALIDATION


def test_unknown_effort_is_refused_with_the_known_names():
    from fastapi import HTTPException

    from backend.api import dispatcher as core

    with pytest.raises(HTTPException) as excinfo:
        core._society_effort({"effort": "as-fast-as-possible"})
    assert excinfo.value.status_code == 400
    assert "quick" in str(excinfo.value.detail)


def test_the_run_owner_decides_the_effort():
    token = set_budget(DiscoveryBudget(n_prompts=2, n_interaction_prompts=1,
                                       max_heads=4))
    try:
        # The budget wins over the caller's own numbers: it is the run owner's
        # decision, and the layers below carry defaults for direct calls.
        assert effective_effort(10, 5, 10) == (2, 1, 4)
    finally:
        reset_budget(token)
    # With no budget, the caller's numbers stand.
    assert effective_effort(10, 5, 10) == (10, 5, 10)


def test_the_run_response_names_the_effort():
    from fastapi.testclient import TestClient

    from backend.api import dispatcher as core
    from backend.core import auth as auth_mod
    from backend.main import app

    client = TestClient(app, raise_server_exceptions=False)
    headers = auth_mod.auth_headers()

    response = client.post("/api/society/run",
                           json={"goal": "effort in the response",
                                 "effort": "quick"},
                           headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body.get("status") == "started", body
    assert body.get("effort") == core._SOCIETY_EFFORTS["quick"], body

    # The run must not be left occupying a slot: cancel it straight away.
    client.post(f"/api/society/runs/{body['runId']}/cancel", json={},
                headers=headers)


# --------------------------------------------------------------------------
# the stop check
# --------------------------------------------------------------------------

def test_a_stopped_measurement_raises_before_the_model_is_touched():
    from backend.interpretability.discovery import live_measure as lm

    token = set_budget(DiscoveryBudget(n_prompts=1, n_interaction_prompts=1,
                                       max_heads=1,
                                       should_stop=lambda: True))
    try:
        with pytest.raises(lm.LiveStopped):
            lm.ablate("some prompt", 1, 2, {(0, 0)})
    finally:
        reset_budget(token)


def test_the_stop_check_is_observed_between_measurements():
    state = {"calls": 0}

    def should_stop() -> bool:
        state["calls"] += 1
        return state["calls"] > 1

    token = set_budget(DiscoveryBudget(n_prompts=1, n_interaction_prompts=1,
                                       max_heads=1,
                                       should_stop=should_stop))
    try:
        from backend.interpretability.discovery import live_measure as lm

        assert stop_requested() is False
        with pytest.raises(lm.LiveStopped):
            lm.ablate("some prompt", 1, 2, {(0, 0)})
    finally:
        reset_budget(token)
    assert current_budget() is None


def test_a_cancelled_stage_is_not_a_successful_stage():
    # A stage that was stopped measured nothing, so it must not count as a
    # success in the Society's own bookkeeping.
    assert ResearchSocietyV2._failed({"status": "cancelled"}) is True
    assert ResearchSocietyV2._failed({"status": "completed"}) is False


def test_the_cancel_route_asks_the_worker_to_stop():
    from fastapi.testclient import TestClient

    from backend.api import dispatcher as core
    from backend.core import auth as auth_mod
    from backend.main import app

    client = TestClient(app, raise_server_exceptions=False)
    headers = auth_mod.auth_headers()

    stop = threading.Event()
    run_id = "r_test_stop_event"
    with core._society_lock:
        core._society_runs[run_id] = {
            "run_id": run_id, "status": "running", "goal": "stop me",
            "model_name": "gpt2", "events": [], "result": None,
            "cancel_requested": False, "pipeline_started": True,
            "stop": stop, "effort": {}, "created": 0.0, "deadline": None,
            "timeout_s": 600, "worker": None, "reason": "",
            "started_at": None, "finished_at": None,
            "queue": __import__("queue").Queue(),
        }
    try:
        response = client.post(f"/api/society/runs/{run_id}/cancel", json={},
                               headers=headers)
        assert response.status_code == 200, response.text
        assert response.json().get("outcome") == "cancel_requested"
        # The worker must be able to see the request: the stop event is set.
        assert stop.is_set(), "the cancel route did not ask the worker to stop"
    finally:
        with core._society_lock:
            core._society_runs.pop(run_id, None)
