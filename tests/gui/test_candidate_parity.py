from __future__ import annotations

import pytest

from .test_document import BASE


def test_both_candidates_return_byte_identical_yaml_for_the_same_interaction():
    pytest.importorskip("panel")
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from tools.config_gui_spike.panel_app import apply_node_edit
    from tools.config_gui_spike.react_api import create_app

    settings = {"type": "GainOperator", "gain": 1.875}
    panel_result = apply_node_edit(
        BASE, "gain", enabled=True, settings=settings
    ).yaml_text
    response = TestClient(create_app()).patch(
        "/api/nodes/gain",
        json={
            "yaml_text": BASE,
            "enabled": True,
            "settings": settings,
        },
    )
    assert response.status_code == 200
    assert response.json()["yaml_text"] == panel_result
