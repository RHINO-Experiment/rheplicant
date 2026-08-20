from __future__ import annotations

import json

import pytest
import yaml

from tests.config.preflight_helpers import preflight_document
from tests.config.test_config_document import synthetic_document

from .test_document import BASE


@pytest.fixture
def client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from rheplicant.gui.api import create_app

    return TestClient(create_app())


@pytest.fixture
def job_client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from rheplicant.gui.api import create_app

    def runner(kind, yaml_text):
        return {"kind": kind, "yaml_text": yaml_text}

    return TestClient(create_app(job_runner=runner))


@pytest.fixture
def real_job_client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from rheplicant.gui.api import create_app
    from rheplicant.gui.jobs import execute_job

    return TestClient(create_app(job_runner=execute_job))


def create_session(client, yaml_text=BASE):
    response = client.post("/api/sessions", json={"yaml_text": yaml_text})
    assert response.status_code == 201
    return response.json()


def test_starter_route_returns_a_valid_bounded_file_free_document(client):
    from rheplicant.gui.starter import STARTER_YAML

    response = client.get("/api/starter")

    assert response.status_code == 200
    assert response.json()["yaml_text"] == STARTER_YAML


def complete_priced_yaml(tmp_path):
    document = synthetic_document()
    document["defaults"] = ["rhino_v1"]
    document["observation"]["pointing"] = {"materialise": []}
    document["outputs"] = {
        "dir": str(tmp_path / "priced-api"),
        "clobber": False,
        "write": {"arrays": {"format": "npz"}},
    }
    return yaml.safe_dump(document, sort_keys=False)


def wait_for_terminal_job(client, session_id, kind):
    for _ in range(5):
        jobs = client.get(f"/api/sessions/{session_id}").json()["jobs"]
        matches = [row for row in jobs if row["kind"] == kind]
        if matches and matches[-1]["status"] in {"succeeded", "refused", "error"}:
            return matches[-1]
    raise AssertionError(f"{kind} job did not reach a terminal state")


def test_session_routes_expose_projection_and_durable_state(client):
    created = create_session(client)

    assert created["revision"] == 0
    assert created["dirty"] is False
    assert created["validation_stale"] is False
    assert created["jobs"] == []
    assert created["document"]["validation"]["run_blocked"] is True
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
    assert undone.json()["validation_stale"] is False

    redone = client.post(
        f"/api/sessions/{session_id}/redo",
        json={"expected_revision": 2},
    )
    assert redone.status_code == 200
    assert redone.json()["revision"] == 3
    assert redone.json()["validation_stale"] is False


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
    assert loaded["validation_stale"] is False
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


def test_api_serializes_the_complete_attributed_ledger_and_preset_diff(client):
    document = preflight_document(
        defaults=["rhino_v1"],
        model={"ghost": {}},
        variants={"bad": {"model": {"phantom": {}}}},
    )
    response = client.post(
        "/api/sessions",
        json={"yaml_text": yaml.safe_dump(document, sort_keys=False)},
    )

    assert response.status_code == 201
    validation = response.json()["document"]["validation"]
    assert [(row["check"], row["attribution"]) for row in validation["findings"]] == [
        ("A2", "base"),
        ("A2", "variant:bad"),
    ]
    assert validation["selected_presets"] == ["rhino_v1"]
    assert validation["preset_changes"]
    assert validation["run_blocked"] is True


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


def test_priced_actions_return_job_ids_and_refresh_to_completed_results(job_client):
    document = preflight_document(variants={})
    created = job_client.post(
        "/api/sessions",
        json={"yaml_text": yaml.safe_dump(document, sort_keys=False)},
    ).json()
    session_id = created["session_id"]

    response = job_client.post(
        f"/api/sessions/{session_id}/jobs",
        json={"expected_revision": 0, "kind": "validate"},
    )
    assert response.status_code == 202
    submitted = response.json()
    assert len(submitted["jobs"]) == 1
    assert submitted["jobs"][0]["job_id"]
    assert submitted["jobs"][0]["status"] == "queued"

    refreshed = job_client.get(f"/api/sessions/{session_id}").json()
    assert refreshed["jobs"][0]["status"] == "succeeded"
    assert refreshed["jobs"][0]["result"]["kind"] == "validate"


def test_real_priced_jobs_with_outputs_reach_succeeded(real_job_client, tmp_path):
    session = create_session(real_job_client, complete_priced_yaml(tmp_path))

    for kind in ("validate", "preview_forward"):
        submitted = real_job_client.post(
            f"/api/sessions/{session['session_id']}/jobs",
            json={"expected_revision": session["revision"], "kind": kind},
        )
        assert submitted.status_code == 202
        terminal = wait_for_terminal_job(
            real_job_client, session["session_id"], kind
        )
        assert terminal["status"] == "succeeded"
        assert terminal["stale"] is False


def test_every_explicit_action_returns_a_distinct_job_id(job_client):
    document = preflight_document(variants={})
    created = job_client.post(
        "/api/sessions",
        json={"yaml_text": yaml.safe_dump(document, sort_keys=False)},
    ).json()
    session_id = created["session_id"]
    ids = []

    for kind in ("validate", "preview_forward", "run", "compare", "benchmark"):
        response = job_client.post(
            f"/api/sessions/{session_id}/jobs",
            json={"expected_revision": 0, "kind": kind},
        )
        assert response.status_code == 202
        ids.append(response.json()["jobs"][-1]["job_id"])

    assert len(ids) == len(set(ids)) == 5


def test_job_submission_is_revision_checked_before_scheduling(job_client):
    document = preflight_document(variants={})
    created = job_client.post(
        "/api/sessions",
        json={"yaml_text": yaml.safe_dump(document, sort_keys=False)},
    ).json()
    session_id = created["session_id"]

    response = job_client.post(
        f"/api/sessions/{session_id}/jobs",
        json={"expected_revision": 8, "kind": "validate"},
    )
    assert response.status_code == 409
    assert job_client.get(f"/api/sessions/{session_id}").json()["jobs"] == []


def test_editing_marks_completed_priced_results_stale(job_client):
    document = preflight_document(variants={})
    text = yaml.safe_dump(document, sort_keys=False)
    created = job_client.post("/api/sessions", json={"yaml_text": text}).json()
    session_id = created["session_id"]
    job_client.post(
        f"/api/sessions/{session_id}/jobs",
        json={"expected_revision": 0, "kind": "validate"},
    )
    assert job_client.get(f"/api/sessions/{session_id}").json()["jobs"][0]["stale"] is False

    changed = yaml.safe_load(text)
    changed["runtime"]["seed"] = 7
    edited = job_client.put(
        f"/api/sessions/{session_id}/yaml",
        json={
            "expected_revision": 0,
            "yaml_text": yaml.safe_dump(changed, sort_keys=False),
        },
    ).json()
    assert edited["jobs"][0]["stale"] is True


def test_text_refusal_blocks_every_execution_job_but_not_the_free_projection(client):
    created = create_session(client)
    session_id = created["session_id"]

    for kind in ("validate", "preview_forward", "run", "compare", "benchmark"):
        response = client.post(
            f"/api/sessions/{session_id}/jobs",
            json={"expected_revision": 0, "kind": kind},
        )
        assert response.status_code == 422
        assert "text-level refusals" in response.json()["detail"]
    fetched = client.get(f"/api/sessions/{session_id}").json()
    assert len(fetched["document"]["nodes"]) == 33
    assert fetched["jobs"] == []


def test_output_product_and_report_routes_are_revision_checked(client):
    created = create_session(client)
    session_id = created["session_id"]
    assert len(created["outputs"]["products"]) == 22
    assert created["outputs"]["requested_yaml"] == BASE

    product = client.put(
        f"/api/sessions/{session_id}/outputs/products/chains",
        json={
            "expected_revision": 0,
            "enabled": True,
            "format": "netcdf",
            "runs": ["forward"],
            "keys": [],
            "themes": [],
        },
    )
    assert product.status_code == 200
    assert product.json()["revision"] == 1
    selected = next(
        row for row in product.json()["outputs"]["products"] if row["name"] == "chains"
    )
    assert selected["enabled"] is True
    assert selected["format"] == "netcdf"
    assert selected["runs"] == ["forward"]

    report = client.put(
        f"/api/sessions/{session_id}/outputs/report",
        json={
            "expected_revision": 1,
            "enabled": True,
            "rows": ["forward"],
            "columns": ["seconds"],
            "reference": None,
            "relative": [],
            "formats": ["json"],
        },
    )
    assert report.status_code == 200
    assert report.json()["outputs"]["report"]["rows"] == ["forward"]

    conflict = client.put(
        f"/api/sessions/{session_id}/outputs/products/arrays",
        json={
            "expected_revision": 0,
            "enabled": True,
            "format": "npz",
            "runs": [],
            "keys": [],
            "themes": [],
        },
    )
    assert conflict.status_code == 409


def test_completed_job_audit_link_is_identity_bound(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from rheplicant.gui.api import create_app

    target = tmp_path / "result"
    target.mkdir(mode=0o700)
    marker_id = "12345678-1234-4123-8123-123456789abc"
    marker = target / ".rheplicant-results.json"
    marker.write_text(
        '{"format_version":1,"run_directory_id":"' + marker_id + '"}\n',
        encoding="utf-8",
    )
    marker.chmod(0o600)
    resolved = target / "config.resolved.yaml"
    resolved.write_text("schema_version: 1\n", encoding="utf-8")
    resolved.chmod(0o600)
    identity = target.stat()

    def runner(_kind, _yaml_text):
        return {
            "output": {
                "target_path": str(target),
                "marker_id": marker_id,
                "target_device": identity.st_dev,
                "target_inode": identity.st_ino,
                "audit_files": ["config.resolved.yaml"],
            }
        }

    browser = TestClient(create_app(job_runner=runner))
    document = preflight_document(variants={})
    created = browser.post(
        "/api/sessions",
        json={"yaml_text": yaml.safe_dump(document, sort_keys=False)},
    ).json()
    session_id = created["session_id"]
    submitted = browser.post(
        f"/api/sessions/{session_id}/jobs",
        json={"expected_revision": 0, "kind": "run"},
    ).json()
    job_id = submitted["jobs"][0]["job_id"]
    refreshed = browser.get(f"/api/sessions/{session_id}").json()
    assert refreshed["jobs"][0]["status"] == "succeeded"

    link = browser.get(
        f"/api/sessions/{session_id}/jobs/{job_id}/artifacts/config.resolved.yaml"
    )
    assert link.status_code == 200
    assert link.content == b"schema_version: 1\n"

    replaced = {
        "format_version": 1,
        "run_directory_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
    }
    marker.write_text(json.dumps(replaced), encoding="utf-8")
    assert browser.get(
        f"/api/sessions/{session_id}/jobs/{job_id}/artifacts/config.resolved.yaml"
    ).status_code == 409
