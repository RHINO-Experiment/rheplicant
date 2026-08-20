from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from types import SimpleNamespace

import pytest

from _rheplicant_bootstrap import execution_environment as environment
from _rheplicant_bootstrap.audit import AuditTrace
from _rheplicant_bootstrap.errors import ConfigError


@dataclass(frozen=True, slots=True)
class _ProcessStub:
    runtime: object
    plugins: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SourceStub:
    layered_document: object
    base_dir: str


@dataclass(frozen=True, slots=True)
class _PreparedConfigStub:
    process: _ProcessStub
    source: _SourceStub
    layers: object
    layer_origins: object
    layer_deletions: object


def _prepared_config_stub(
    tmp_path, *, plugins: tuple[str, ...] = ()
) -> _PreparedConfigStub:
    return _PreparedConfigStub(
        process=_ProcessStub(runtime=object(), plugins=plugins),
        source=_SourceStub(layered_document={}, base_dir=str(tmp_path)),
        layers=(object(), object()),
        layer_origins={"base": object()},
        layer_deletions={"variant": (object(),)},
    )


class _RecordingRuntime:
    def __init__(self, events: list[tuple[object, ...]]) -> None:
        self.events = events

    def verify(self, *, boundary: str) -> None:
        name = boundary.removeprefix("plugin ").strip("'")
        self.events.append((f"verify:{name}",))


class _RecordingTrace:
    def __init__(self, events: list[tuple[object, ...]]) -> None:
        self.events = events

    def record_runtime(self, _row: object) -> None:
        self.events.append(("runtime_audit",))

    def boundary_completed(self, stage: str) -> None:
        self.events.append((f"{stage}_boundary",))

    def record_plugin(self, name: str) -> None:
        self.events.append((f"plugin_audit:{name}",))

    def record_input(self, *_args: object) -> None:
        raise AssertionError("the fake orchestration captures no input")

    @staticmethod
    def snapshot() -> SimpleNamespace:
        return SimpleNamespace(python_targets=())


class _RecordingCapture:
    def __init__(
        self,
        _root: object,
        *,
        on_verified: object,
        events: list[tuple[object, ...]],
    ) -> None:
        self.on_verified = on_verified
        self.events = events
        self.close_calls = 0
        self.events.append(("capture",))

    def close(self) -> None:
        self.close_calls += 1


def _event_names(events: list[tuple[object, ...]]) -> list[object]:
    return [row[0] for row in events]


def test_prepared_environment_orders_runtime_plugins_and_all_layer_metadata(
    tmp_path, monkeypatch
):
    events: list[tuple[object, ...]] = []
    prepared = _prepared_config_stub(
        tmp_path, plugins=("tests.fake_a", "tests.fake_b")
    )
    orchestration = SimpleNamespace(
        prepare_document=lambda document, **kwargs: (
            events.append(("prepare_document", document, kwargs)),
            "prepared-document",
        )[1]
    )

    def establish_runtime(_requested, *, import_main):
        del import_main
        events.append(("runtime",))
        return _RecordingRuntime(events), orchestration

    def import_plugin(name: str) -> str:
        events.append((f"plugin:{name}",))
        return name

    monkeypatch.setattr(environment, "establish_runtime", establish_runtime)
    monkeypatch.setattr(environment, "runtime_audit_row", lambda _session: {})
    monkeypatch.setattr(environment, "import_plugin", import_plugin)
    monkeypatch.setattr(environment, "plugin_audit_row", lambda name: name)
    monkeypatch.setattr(
        environment,
        "CaptureService",
        lambda root, *, on_verified: _RecordingCapture(
            root, on_verified=on_verified, events=events
        ),
    )

    found = environment.prepare_execution_environment(
        prepared,
        trace=_RecordingTrace(events),
        stderr=StringIO(),
        warning_written=False,
    )
    capture = found.capture
    try:
        assert _event_names(events) == [
            "runtime",
            "runtime_audit",
            "runtime_boundary",
            "plugin:tests.fake_a",
            "plugin_audit:tests.fake_a",
            "verify:tests.fake_a",
            "plugin:tests.fake_b",
            "plugin_audit:tests.fake_b",
            "verify:tests.fake_b",
            "plugins_boundary",
            "capture",
            "prepare_document",
        ]
        call = next(row for row in events if row[0] == "prepare_document")
        assert call[2]["layers"] is prepared.layers
        assert call[2]["layer_origins"] is prepared.layer_origins
        assert call[2]["layer_deletions"] is prepared.layer_deletions
        assert capture.close_calls == 0
    finally:
        found.close()
    assert capture.close_calls == 1


def test_prepared_execution_close_is_idempotent():
    capture = _RecordingCapture(
        object(), on_verified=object(), events=[]
    )
    row = environment.PreparedExecution(
        runtime=object(),
        orchestration=object(),
        document=object(),
        capture=capture,
        trace=AuditTrace(),
        warning_written=False,
    )

    row.close()
    row.close()

    assert capture.close_calls == 1


def test_prepare_failure_closes_capture(tmp_path, monkeypatch):
    events: list[tuple[object, ...]] = []
    capture = _RecordingCapture(
        object(), on_verified=object(), events=events
    )

    def raising_prepare(*_args, **_kwargs):
        raise ConfigError("probe refusal")

    orchestration = SimpleNamespace(prepare_document=raising_prepare)

    def establish_runtime(_requested, *, import_main):
        del import_main
        return _RecordingRuntime(events), orchestration

    monkeypatch.setattr(environment, "establish_runtime", establish_runtime)
    monkeypatch.setattr(
        environment,
        "runtime_audit_row",
        lambda _session: {
            "requested": {"jax_enable_x64": False, "platform": "auto"},
            "actual": {"jax_enable_x64": False, "backend": "cpu"},
            "prior_environment": {
                "jax_enable_x64": None,
                "jax_platforms": None,
            },
        },
    )
    monkeypatch.setattr(
        environment,
        "CaptureService",
        lambda _root, *, on_verified: capture,
    )

    with pytest.raises(ConfigError, match="probe refusal"):
        environment.prepare_execution_environment(
            _prepared_config_stub(tmp_path),
            trace=AuditTrace(),
            stderr=StringIO(),
            warning_written=False,
        )

    assert capture.close_calls == 1
