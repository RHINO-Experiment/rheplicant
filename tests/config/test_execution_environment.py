from __future__ import annotations

import os
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


class TestCleanupDoesNotReplaceTheFailure:
    """A7.8. ``prepare_execution_environment``'s unwind called
    ``capture.close()`` bare, and ``close()`` can raise: ``_remove`` forgives
    only ``FileNotFoundError`` and the closing ``shutil.rmtree`` forgives
    nothing. A failure there REPLACED the exception being handled, so a
    permissions problem during cleanup would be reported in place of the
    fault the user has to act on.

    The original now propagates and the cleanup failure travels with it as a
    note -- the half that a bare ``contextlib.suppress`` would have thrown
    away.
    """

    def test_the_original_exception_survives_a_failing_close(self):
        class Boom(RuntimeError):
            """The real failure, the one that must reach the caller."""

        class ExplodingCapture:
            def close(self):
                raise OSError("cleanup could not remove the capture root")

        capture = ExplodingCapture()

        # The shape of the code under test, isolated: an in-flight exception
        # and a cleanup that fails during the unwind.
        def unwind():
            try:
                raise Boom("the fault worth reporting")
            except BaseException as error:
                try:
                    capture.close()
                except BaseException as cleanup:  # noqa: BLE001
                    error.add_note(
                        f"capture cleanup also failed and was not the original "
                        f"fault: {type(cleanup).__name__}: {cleanup}"
                    )
                raise

        with pytest.raises(Boom) as caught:
            unwind()

        notes = getattr(caught.value, "__notes__", [])
        assert any("cleanup also failed" in note for note in notes), notes
        assert any("OSError" in note for note in notes), notes

    def test_the_source_still_wraps_close_rather_than_calling_it_bare(self):
        """The test above pins the SHAPE; this pins that the shape is the one
        shipped. Reading the source is the only way to tell here -- the real
        function needs a whole prepared config, a runtime session and a
        plugin walk before it can reach its own unwind, and a fixture that
        heavy would be testing the fixture."""
        import inspect

        source = inspect.getsource(environment.prepare_execution_environment)
        assert "except BaseException as error:" in source
        assert "add_note" in source
        # The bare call is what this replaced; it must not come back.
        assert "\n        capture.close()\n        raise" not in source

    def test_a_capture_service_that_cannot_be_built_leaves_no_directory(
        self, tmp_path, monkeypatch
    ):
        """The other half: ``mkdtemp`` runs before the object that owns the
        directory exists, so anything raised in between leaks it -- nobody
        else knows the path."""
        import tempfile

        made = []
        real_mkdtemp = tempfile.mkdtemp

        def recording_mkdtemp(*args, **kwargs):
            path = real_mkdtemp(*args, **kwargs, dir=str(tmp_path))
            made.append(path)
            return path

        monkeypatch.setattr(environment.tempfile, "mkdtemp", recording_mkdtemp)

        def exploding_service(*args, **kwargs):
            raise RuntimeError("the service could not be constructed")

        monkeypatch.setattr(environment, "CaptureService", exploding_service)

        with pytest.raises(RuntimeError, match="could not be constructed"):
            environment._capture_service(trace=None)

        assert made, "the probe must have seen a mkdtemp to be evidence"
        assert not any(os.path.exists(path) for path in made), made
