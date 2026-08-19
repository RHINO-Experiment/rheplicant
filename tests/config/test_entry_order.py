from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace


def test_bootstrap_import_does_not_import_the_main_package_or_jax():
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, _rheplicant_bootstrap, _rheplicant_bootstrap.cli; "
            "assert 'rheplicant' not in sys.modules; "
            "assert 'jax' not in sys.modules; assert 'jaxlib' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr


def test_plugin_fact_is_recorded_before_post_import_runtime_verification(
    tmp_path, monkeypatch
):
    import _rheplicant_bootstrap.entry as entry

    events = []

    class Session:
        def verify(self, *, boundary):
            events.append(f"verify:{boundary}")

    class Orchestration:
        @staticmethod
        def prepare_document(*_args, **_kwargs):
            events.append("prepare_document")
            return object()

    class Trace:
        def record_runtime(self, _row):
            events.append("record_runtime")

        def boundary_completed(self, stage):
            events.append(f"boundary:{stage}")

        def record_plugin(self, _row):
            events.append("record_plugin")

        def record_input(self, *_args):
            raise AssertionError("the fake orchestration captures no input")

        @staticmethod
        def snapshot():
            return SimpleNamespace(python_targets=())

    def establish(_requested, *, import_main):
        events.append("runtime_environment")
        return Session(), import_main()

    monkeypatch.setattr(entry, "establish_runtime", establish)
    monkeypatch.setattr(
        entry.importlib,
        "import_module",
        lambda name: events.append(f"main_import:{name}") or Orchestration(),
    )
    monkeypatch.setattr(entry, "runtime_audit_row", lambda _session: {})
    monkeypatch.setattr(
        entry,
        "import_plugin",
        lambda name: events.append(f"import:{name}") or object(),
    )
    monkeypatch.setattr(entry, "plugin_audit_row", lambda _record: {})
    prepared = SimpleNamespace(
        process=SimpleNamespace(runtime=object(), plugins=("trusted.plugin",)),
        source=SimpleNamespace(layered_document={}, base_dir=str(tmp_path)),
        layers=(),
        layer_origins={},
        layer_deletions={},
    )
    result = entry._prepare_document(
        prepared,
        trace=Trace(),
        stderr=SimpleNamespace(write=lambda _text: None, flush=lambda: None),
        warning_written=False,
    )
    result[3].close()
    assert events.count("main_import:rheplicant.config.orchestration") == 1
    assert events.index("runtime_environment") < events.index(
        "main_import:rheplicant.config.orchestration"
    )
    assert events.index("record_plugin") < events.index("verify:plugin 'trusted.plugin'")
