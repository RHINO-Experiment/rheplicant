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


def test_full_session_carries_exact_yaml_digest(client):
    from rheplicant.gui.jobs import yaml_digest

    session = create_session(client, BASE)

    assert session["yaml_digest"] == yaml_digest(
        session["document"]["yaml_text"]
    )


def test_get_jobs_returns_identity_without_full_document(client):
    from rheplicant.gui.jobs import yaml_digest

    session = create_session(client, BASE)

    response = client.get(f"/api/sessions/{session['session_id']}/jobs")

    assert response.status_code == 200
    assert response.json() == {
        "session_id": session["session_id"],
        "revision": session["revision"],
        "yaml_digest": yaml_digest(session["document"]["yaml_text"]),
        "jobs": [],
    }
    assert "document" not in response.json()
    assert "outputs" not in response.json()


def test_get_jobs_returns_the_projected_submitted_job(job_client):
    from rheplicant.gui.jobs import yaml_digest

    yaml_text = yaml.safe_dump(preflight_document(variants={}), sort_keys=False)
    session = create_session(job_client, yaml_text)
    submitted = job_client.post(
        f"/api/sessions/{session['session_id']}/jobs",
        json={"expected_revision": 0, "kind": "validate"},
    )
    assert submitted.status_code == 202
    submitted_job_id = submitted.json()["jobs"][0]["job_id"]

    response = job_client.get(f"/api/sessions/{session['session_id']}/jobs")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"session_id", "revision", "yaml_digest", "jobs"}
    assert body["session_id"] == session["session_id"]
    assert body["revision"] == 0
    assert body["yaml_digest"] == yaml_digest(yaml_text)
    assert len(body["jobs"]) == 1
    assert {
        key: body["jobs"][0][key]
        for key in (
            "job_id",
            "session_id",
            "kind",
            "status",
            "revision",
            "yaml_digest",
            "stale",
        )
    } == {
        "job_id": submitted_job_id,
        "session_id": session["session_id"],
        "kind": "validate",
        "status": "succeeded",
        "revision": 0,
        "yaml_digest": yaml_digest(yaml_text),
        "stale": False,
    }


def test_get_jobs_returns_404_for_an_unknown_session(client):
    response = client.get("/api/sessions/unknown-session/jobs")

    assert response.status_code == 404


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


def test_session_field_transition_uses_existing_revision_history_and_noop_identity():
    from rheplicant.gui.session import (
        RevisionConflict,
        new_session,
        set_session_field,
    )

    original = new_session(BASE)
    changed = set_session_field(
        original,
        "runtime.jax_enable_x64",
        False,
        expected_revision=0,
    )

    assert changed.revision == 1
    assert changed.cursor == 1
    assert changed.history[0] == BASE
    assert changed.dirty is True
    assert changed.validation_stale is False
    assert changed.can_undo is True
    assert yaml.safe_load(changed.yaml_text)["runtime"]["jax_enable_x64"] is False

    same = set_session_field(
        changed,
        "runtime.jax_enable_x64",
        False,
        expected_revision=1,
    )
    assert same is changed

    with pytest.raises(RevisionConflict):
        set_session_field(
            changed,
            "model.gain",
            {},
            expected_revision=0,
        )


def test_session_field_route_returns_the_complete_updated_projection(client):
    created = create_session(client)
    session_id = created["session_id"]

    response = client.patch(
        f"/api/sessions/{session_id}/fields",
        json={
            "expected_revision": 0,
            "path": "runtime.jax_enable_x64",
            "value": False,
            "remove": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["revision"] == 1
    assert body["dirty"] is True
    assert body["validation_stale"] is False
    assert set(body) == {
        "session_id",
        "revision",
        "yaml_digest",
        "dirty",
        "validation_stale",
        "can_undo",
        "can_redo",
        "jobs",
        "outputs",
        "document",
    }
    assert len(body["document"]["nodes"]) == 33
    assert len(body["document"]["forms"]["sections"]) == 12
    assert "base_diagram" in body["document"]
    assert "validation" in body["document"]
    assert body["outputs"]["requested_yaml"] == body["document"]["yaml_text"]
    assert yaml.safe_load(body["document"]["yaml_text"])["runtime"][
        "jax_enable_x64"
    ] is False


def test_session_field_route_preserves_exact_yaml_and_revision_on_noop(client):
    created = create_session(client)
    session_id = created["session_id"]

    response = client.patch(
        f"/api/sessions/{session_id}/fields",
        json={
            "expected_revision": 0,
            "path": "runtime.jax_enable_x64",
            "value": True,
            "remove": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["revision"] == 0
    assert body["document"]["yaml_text"] == BASE
    assert body["dirty"] is False
    assert body["can_undo"] is False


def test_session_field_route_is_closed_and_preserves_state_after_conflict_or_refusal(
    client,
):
    created = create_session(client)
    session_id = created["session_id"]
    route = f"/api/sessions/{session_id}/fields"

    changed = client.patch(
        route,
        json={
            "expected_revision": 0,
            "path": "runtime.jax_enable_x64",
            "value": False,
        },
    )
    assert changed.status_code == 200
    accepted_yaml = changed.json()["document"]["yaml_text"]

    conflict = client.patch(
        route,
        json={
            "expected_revision": 0,
            "path": "runtime.jax_enable_x64",
            "value": True,
        },
    )
    assert conflict.status_code == 409

    generic = client.patch(
        route,
        json={"expected_revision": 1, "path": "model.gain", "value": {}},
    )
    assert generic.status_code == 422
    assert "Edit this value in YAML" in generic.json()["detail"]

    bool_as_int = client.patch(
        route,
        json={"expected_revision": 1, "path": "runtime.seed", "value": True},
    )
    assert bool_as_int.status_code == 422

    extra = client.patch(
        route,
        json={
            "expected_revision": 1,
            "path": "runtime.seed",
            "value": 4,
            "surprise": True,
        },
    )
    assert extra.status_code == 422

    fetched = client.get(f"/api/sessions/{session_id}").json()
    assert fetched["revision"] == 1
    assert fetched["document"]["yaml_text"] == accepted_yaml


@pytest.mark.parametrize(
    ("yaml_text", "path", "value"),
    [
        (BASE, "model.foregrounds.type", "ForegroundOperator"),
        (
            "schema_version: 1\nresources:\n  beams:\n    horn.dot:\n"
            "      format: inline\nmodel: {}\nruns: []\n",
            "resources.beams.horn.dot.format",
            "gaussian",
        ),
    ],
)
def test_session_field_route_refuses_hidden_or_ambiguous_paths_without_state_change(
    client,
    yaml_text: str,
    path: str,
    value: object,
):
    created = create_session(client, yaml_text)
    session_id = created["session_id"]

    refused = client.patch(
        f"/api/sessions/{session_id}/fields",
        json={
            "expected_revision": 0,
            "path": path,
            "value": value,
            "remove": False,
        },
    )

    assert refused.status_code == 422
    fetched = client.get(f"/api/sessions/{session_id}")
    assert fetched.status_code == 200
    assert fetched.json()["revision"] == 0
    assert fetched.json()["document"]["yaml_text"] == yaml_text


def test_session_field_route_reprojects_candidate_before_store_install(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    import rheplicant.gui.session as session_module
    from rheplicant.gui.api import create_app

    browser = TestClient(create_app(), raise_server_exceptions=False)
    created = create_session(browser)
    session_id = created["session_id"]
    monkeypatch.setattr(
        session_module,
        "set_form_value",
        lambda *_args, **_kwargs: "model: []\n",
    )

    refused = browser.patch(
        f"/api/sessions/{session_id}/fields",
        json={
            "expected_revision": 0,
            "path": "runtime.seed",
            "value": 4,
            "remove": False,
        },
    )

    assert refused.status_code == 422
    fetched = browser.get(f"/api/sessions/{session_id}")
    assert fetched.status_code == 200
    assert fetched.json()["revision"] == 0
    assert fetched.json()["document"]["yaml_text"] == BASE


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
    assert len(created["outputs"]["products"]) == 23
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


def test_formal_job_api_projects_closed_unsafe_output_refusal(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from rheplicant.gui.api import create_app
    from rheplicant.gui.jobs import execute_job
    from rheplicant.gui.outputs import OutputState

    monkeypatch.setattr(
        "rheplicant.gui.outputs._inspection_state",
        lambda _inspection: OutputState(
            "blocked_unsafe",
            "Closed API unsafe output projection.",
        ),
    )

    dispatched = []

    def refused(_command, _source, *, stdout, stderr):
        dispatched.append(True)
        stderr.write("formal API refusal without an output classification")
        return 2

    browser = TestClient(create_app(job_runner=lambda kind, text: execute_job(
        kind,
        text,
        dispatcher=refused,
    )))
    document = preflight_document(variants={})
    target = tmp_path / "unsafe-api-result"
    document["outputs"] = {"dir": str(target), "stdout": "none"}
    created = browser.post(
        "/api/sessions",
        json={"yaml_text": yaml.safe_dump(document, sort_keys=False)},
    ).json()
    session_id = created["session_id"]

    submitted = browser.post(
        f"/api/sessions/{session_id}/jobs",
        json={"expected_revision": 0, "kind": "run"},
    )
    assert submitted.status_code == 202
    found = browser.get(f"/api/sessions/{session_id}/jobs").json()["jobs"][0]
    assert found["status"] == "refused"
    assert found["message"] == "Closed API unsafe output projection."
    assert dispatched == []
    assert found["result"] == {
        "output": {
            "state": "blocked_unsafe",
            "state_message": "Closed API unsafe output projection.",
            "target_path": str(target),
        }
    }


# --- Finding 7: the API refuses an identical active job with a bounded 422 ---


def test_an_identical_active_job_is_refused_by_the_api_with_a_bounded_422():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from rheplicant.gui.api import SessionStore, create_app

    store = SessionStore()
    text = yaml.safe_dump(preflight_document(variants={}), sort_keys=False)
    session_id, session = store.create(text)
    store.jobs.submit(session_id, "validate", session.revision, session.yaml_text)
    client = TestClient(create_app(session_store=store))

    response = client.post(
        f"/api/sessions/{session_id}/jobs",
        json={"expected_revision": session.revision, "kind": "validate"},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "'validate'" in detail
    assert f"revision {session.revision}" in detail
    assert "schema_version" not in detail
    assert len(detail) <= 200
    assert len(client.get(f"/api/sessions/{session_id}").json()["jobs"]) == 1


def test_two_concurrent_api_submissions_of_one_action_create_exactly_one_job():
    """Two live HTTP submissions overlap: the first job is still running."""
    import threading

    pytest.importorskip("fastapi")
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from rheplicant.gui.api import create_app

    started = threading.Event()
    release = threading.Event()

    def runner(kind, _yaml_text):
        started.set()
        assert release.wait(timeout=30)
        return {"kind": kind}

    client = TestClient(create_app(job_runner=runner))
    text = yaml.safe_dump(preflight_document(variants={}), sort_keys=False)
    session_id = client.post("/api/sessions", json={"yaml_text": text}).json()[
        "session_id"
    ]
    outcome: dict[str, object] = {}

    def submit_first():
        outcome["first"] = client.post(
            f"/api/sessions/{session_id}/jobs",
            json={"expected_revision": 0, "kind": "run"},
        ).status_code

    worker = threading.Thread(target=submit_first)
    worker.start()
    try:
        assert started.wait(timeout=30)
        second = client.post(
            f"/api/sessions/{session_id}/jobs",
            json={"expected_revision": 0, "kind": "run"},
        )
    finally:
        release.set()
        worker.join(timeout=30)

    assert outcome["first"] == 202
    assert second.status_code == 422
    assert "already running" in second.json()["detail"]
    jobs = client.get(f"/api/sessions/{session_id}").json()["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["status"] == "succeeded"


# --- Finding 8: a megabyte failure cannot become a megabyte response --------


def test_a_megabyte_job_failure_leaves_the_api_response_bounded():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from _rheplicant_bootstrap.gui_limits import (
        MAX_TEXT_CHARACTERS,
        TRUNCATION_MARKER,
    )

    # Pinned literal, not the constant: an emptied constant must not make the
    # presence assertion below pass by accident.
    marker = "…[truncated]"
    assert TRUNCATION_MARKER == marker
    from rheplicant.gui.api import create_app

    def runner(_kind, _yaml_text):
        raise RuntimeError("x" * 2_000_000)

    client = TestClient(create_app(job_runner=runner))
    text = yaml.safe_dump(preflight_document(variants={}), sort_keys=False)
    session_id = client.post("/api/sessions", json={"yaml_text": text}).json()[
        "session_id"
    ]

    submitted = client.post(
        f"/api/sessions/{session_id}/jobs",
        json={"expected_revision": 0, "kind": "validate"},
    )
    assert submitted.status_code == 202

    response = client.get(f"/api/sessions/{session_id}/jobs")
    row = response.json()["jobs"][0]
    assert row["status"] == "error"
    assert row["message"].startswith("RuntimeError: ")
    assert len(row["message"]) <= MAX_TEXT_CHARACTERS
    assert marker in row["message"]
    assert len(response.content) < 100_000
    assert len(client.get(f"/api/sessions/{session_id}").content) < 1_000_000


def test_a_submission_that_fails_after_the_insert_leaves_no_queued_ghost(
    job_client, monkeypatch
):
    """Nothing between the insert and the response may strand a ``queued`` row.

    ``submit_job`` inserts the row as ``queued`` and the background task is
    registered only after the response body is built.  ``queued`` is one of
    ``_ACTIVE_STATUSES`` and no route cancels or deletes a job, so a row that
    never reaches ``run`` refuses every identical resubmission for the life of
    the process -- the same defect already closed for ``running``, on the half
    that was left open.
    """
    from rheplicant.gui import api

    document = preflight_document(variants={})
    created = job_client.post(
        "/api/sessions",
        json={"yaml_text": yaml.safe_dump(document, sort_keys=False)},
    ).json()
    session_id = created["session_id"]
    real_session_body = api._session_body
    failures = {"left": 1}

    def failing_session_body(store, wanted_id, session):
        if failures["left"]:
            failures["left"] -= 1
            raise RuntimeError("projecting the session body failed")
        return real_session_body(store, wanted_id, session)

    monkeypatch.setattr(api, "_session_body", failing_session_body)

    with pytest.raises(RuntimeError, match="projecting the session body failed"):
        job_client.post(
            f"/api/sessions/{session_id}/jobs",
            json={"expected_revision": 0, "kind": "validate"},
        )

    # The row the failed submission inserted must already be terminal, so the
    # identical resubmission below is a new question rather than a duplicate.
    stranded = job_client.get(f"/api/sessions/{session_id}/jobs").json()["jobs"]
    assert len(stranded) == 1
    assert stranded[0]["status"] == "error"
    assert stranded[0]["message"]

    for _ in range(2):
        again = job_client.post(
            f"/api/sessions/{session_id}/jobs",
            json={"expected_revision": 0, "kind": "validate"},
        )
        assert again.status_code == 202, again.json()
