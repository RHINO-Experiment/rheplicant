from __future__ import annotations

import pytest
import yaml

from .test_document import BASE


@pytest.fixture
def client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from rheplicant.gui.api import create_app

    return TestClient(create_app())


def create_session(client):
    response = client.post("/api/sessions", json={"yaml_text": BASE})
    assert response.status_code == 201
    return response.json()


def test_session_routes_expose_projection_and_durable_state(client):
    created = create_session(client)

    assert created["revision"] == 0
    assert created["dirty"] is False
    assert created["validation_stale"] is True
    assert created["can_undo"] is False
    assert len(created["document"]["nodes"]) == 33
    assert len(created["document"]["forms"]["sections"]) == 12
    assert created["document"]["forms"]["sections"][-1]["disabled"] is True

    fetched = client.get(f"/api/sessions/{created['session_id']}")
    assert fetched.status_code == 200
    assert fetched.json() == created


def test_yaml_node_undo_and_redo_routes_are_revision_checked(client):
    created = create_session(client)
    session_id = created["session_id"]

    edited = client.patch(
        f"/api/sessions/{session_id}/nodes/gain",
        json={
            "expected_revision": 0,
            "enabled": True,
            "settings": {"type": "GainOperator", "gain": 1.25},
        },
    )
    assert edited.status_code == 200
    assert edited.json()["revision"] == 1
    assert edited.json()["dirty"] is True

    conflict = client.put(
        f"/api/sessions/{session_id}/yaml",
        json={"expected_revision": 0, "yaml_text": BASE},
    )
    assert conflict.status_code == 409
    assert "current revision is 1" in conflict.json()["detail"]

    undone = client.post(
        f"/api/sessions/{session_id}/undo",
        json={"expected_revision": 1},
    )
    assert undone.status_code == 200
    assert undone.json()["document"]["yaml_text"] == BASE
    assert undone.json()["can_redo"] is True
    assert undone.json()["validation_stale"] is True

    redone = client.post(
        f"/api/sessions/{session_id}/redo",
        json={"expected_revision": 2},
    )
    assert redone.status_code == 200
    assert redone.json()["revision"] == 3
    assert redone.json()["validation_stale"] is True


def test_load_and_save_are_distinct_explicit_user_actions(client):
    created = create_session(client)
    session_id = created["session_id"]
    edited_yaml = BASE.replace("gain: 1.0", "gain: 2.0")

    edited = client.put(
        f"/api/sessions/{session_id}/yaml",
        json={"expected_revision": 0, "yaml_text": edited_yaml},
    ).json()
    assert edited["dirty"] is True

    saved = client.post(
        f"/api/sessions/{session_id}/save",
        json={"expected_revision": 1},
    ).json()
    assert saved["dirty"] is False
    assert saved["revision"] == 2

    loaded_yaml = BASE.replace("gain: 1.0", "gain: 3.0")
    loaded = client.post(
        f"/api/sessions/{session_id}/load",
        json={"expected_revision": 2, "yaml_text": loaded_yaml},
    ).json()
    assert loaded["document"]["yaml_text"] == loaded_yaml
    assert loaded["dirty"] is False
    assert loaded["validation_stale"] is True
    assert loaded["can_undo"] is False


def test_session_route_refusals_do_not_overwrite_the_current_document(client):
    created = create_session(client)
    session_id = created["session_id"]

    invalid = client.put(
        f"/api/sessions/{session_id}/yaml",
        json={"expected_revision": 0, "yaml_text": "model: []\n"},
    )
    assert invalid.status_code == 422

    missing_undo = client.post(
        f"/api/sessions/{session_id}/undo",
        json={"expected_revision": 0},
    )
    assert missing_undo.status_code == 422

    fetched = client.get(f"/api/sessions/{session_id}").json()
    assert fetched["revision"] == 0
    assert fetched["document"]["yaml_text"] == BASE


def test_unknown_session_is_404_and_payloads_are_closed(client):
    missing = client.get("/api/sessions/not-there")
    assert missing.status_code == 404

    created = create_session(client)
    response = client.post(
        f"/api/sessions/{created['session_id']}/save",
        json={"expected_revision": 0, "surprise": True},
    )
    assert response.status_code == 422


def test_graph_editor_routes_commit_many_order_compose_placement_and_snapshot(client):
    document = """\
model:
  gain: {type: GainOperator, gain: 1.0}
  flagging: {type: FlaggingOperator, threshold: 4.0}
  filters:
    - {type: FourierBandFilter, name: first}
    - {type: SkySpaceFilter, name: second}
variants:
  alternate:
    model: {}
runs: []
"""
    created = client.post("/api/sessions", json={"yaml_text": document}).json()
    session_id = created["session_id"]

    many = client.put(
        f"/api/sessions/{session_id}/nodes/filters/many",
        json={
            "expected_revision": 0,
            "entries": [
                {"type": "FourierBandFilter", "name": "first"},
                {"type": "SkySpaceFilter", "name": "second"},
                {"type": "DelayFilter", "name": "third"},
            ],
            "variant": None,
        },
    )
    assert many.status_code == 200

    moved = client.post(
        f"/api/sessions/{session_id}/nodes/filters/move",
        json={"expected_revision": 1, "from_index": 2, "to_index": 0},
    )
    assert moved.status_code == 200
    assert [
        item["name"]
        for item in yaml.safe_load(moved.json()["document"]["yaml_text"])["model"]["filters"]
    ] == ["third", "first", "second"]

    composed = client.put(
        f"/api/sessions/{session_id}/nodes/gain/compose",
        json={
            "expected_revision": 2,
            "compose": "cascade",
            "stages": [
                {"name": "lna", "type": "GainOperator", "gain": 0.5},
                {"name": "post", "type": "GainOperator", "gain": 2.0},
            ],
        },
    )
    assert composed.status_code == 200

    placed = client.put(
        f"/api/sessions/{session_id}/nodes/cw_tone/placement",
        json={
            "expected_revision": 3,
            "at": ["noise", "emi"],
            "settings": {"python": "pkg:Tone", "amplitude": 1.0},
        },
    )
    assert placed.status_code == 200
    placed_model = yaml.safe_load(placed.json()["document"]["yaml_text"])["model"]
    assert placed_model["emi"]["at"] == ["noise", "emi"]

    snapped = client.put(
        f"/api/sessions/{session_id}/nodes/flagging/snapshot-before",
        json={"expected_revision": 4, "snapshot_name": "raw"},
    )
    assert snapped.status_code == 200
    snapped_document = yaml.safe_load(snapped.json()["document"]["yaml_text"])
    assert snapped_document["model"]["flagging"]["snapshot_before"] == "raw"
    assert snapped_document["outputs"]["write"]["aux"]["keys"] == ["snapshot/raw"]

    conflict = client.post(
        f"/api/sessions/{session_id}/nodes/filters/move",
        json={"expected_revision": 2, "from_index": 0, "to_index": 1},
    )
    assert conflict.status_code == 409
