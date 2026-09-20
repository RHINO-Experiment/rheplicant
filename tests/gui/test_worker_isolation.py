"""The GUI's scientific worker does not import from the server's directory.

``python -m`` puts the current directory first on ``sys.path``, so a worker
started that way would import a module file lying in whatever directory the
GUI server was started from ahead of the installed package a document's
``plugins:`` or ``python:`` target names. The ``rheplicant`` console script
does not do this, and the worker is started with ``-P`` so that it does not
either.
"""

from __future__ import annotations

import sys

import pytest

from _rheplicant_bootstrap.errors import ConfigError
from rheplicant.gui import jobs
from rheplicant.gui.starter import STARTER_YAML

MARKER = "rheplicant_gui_cwd_shadow_marker"


class _Stop(Exception):
    pass


def test_the_worker_is_started_with_safe_path(monkeypatch) -> None:
    seen: list[list[str]] = []

    def record(argv, **_):
        seen.append(list(argv))
        raise _Stop

    monkeypatch.setattr(jobs, "_drained_run", record)
    with pytest.raises(_Stop):
        jobs.run_priced_validation("schema_version: 1\n")
    assert seen == [
        [sys.executable, "-P", "-m", "_rheplicant_bootstrap.gui_worker", "validate"]
    ]


def test_a_module_in_the_server_cwd_is_not_importable_in_the_worker(
    tmp_path, monkeypatch
) -> None:
    """The real worker, started from a directory holding a module a document
    names under ``plugins:``. The module only records that it was imported."""
    sentinel = tmp_path / "imported.txt"
    (tmp_path / f"{MARKER}.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(sentinel)!r}).write_text('imported')\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigError) as excinfo:
        jobs.run_priced_validation(f"plugins:\n  - {MARKER}\n" + STARTER_YAML)

    assert not sentinel.exists()
    assert f"importing {MARKER!r} raised ModuleNotFoundError" in str(excinfo.value)
