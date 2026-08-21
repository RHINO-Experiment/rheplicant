from __future__ import annotations

import pytest
import yaml

from .test_document import BASE


@pytest.fixture
def client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from tools.config_gui_spike.react_api import create_app

    return TestClient(create_app())


def test_snapshot_endpoint_returns_the_live_canvas_contract(client):
    response = client.post("/api/snapshot", json={"yaml_text": BASE})
    assert response.status_code == 200
    body = response.json()
    assert len(body["nodes"]) == 33
    assert len(body["walk_order"]) == 33
    assert body["walk_order"][24] == "gain"
    assert 'data-node-id="gain"' in body["svg"]


def test_node_edit_endpoint_returns_transformed_yaml(client):
    response = client.patch(
        "/api/nodes/gain",
        json={
            "yaml_text": BASE,
            "enabled": True,
            "settings": {"type": "GainOperator", "gain": 3.0},
        },
    )
    assert response.status_code == 200
    assert yaml.safe_load(response.json()["yaml_text"])["model"]["gain"]["gain"] == 3.0


@pytest.mark.parametrize("node_id", ["missing", "astro_sum", "receiver_input"])
def test_api_preserves_config_refusals_as_422(node_id, client):
    response = client.patch(
        f"/api/nodes/{node_id}",
        json={"yaml_text": BASE, "enabled": True, "settings": {"type": "X"}},
    )
    assert response.status_code == 422
    assert "operator slot" in response.json()["detail"]


def test_invalid_yaml_is_a_client_error_not_a_server_traceback(client):
    response = client.post("/api/snapshot", json={"yaml_text": "model: []\n"})
    assert response.status_code == 422
    assert response.json() == {"detail": "model: must be a mapping."}
