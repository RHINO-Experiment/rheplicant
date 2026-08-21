from __future__ import annotations

import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pytest

from _rheplicant_bootstrap import gui_child

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UV = "uv"


CommandArgument = str | os.PathLike[str]


def _run(
    arguments: Sequence[CommandArgument],
    *,
    cwd: Path = PROJECT_ROOT,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"command failed: {arguments!r}\n{completed.stdout}\n{completed.stderr}"
    )
    return completed


def _single(directory: Path, suffix: str) -> Path:
    rows = tuple(path for path in directory.iterdir() if path.name.endswith(suffix))
    assert len(rows) == 1, rows
    return rows[0]


def build_distributions(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("config-distributions")
    direct = root / "direct"
    sdist = root / "sdist"
    derived = root / "from-sdist"
    _run([UV, "build", "--wheel", "--out-dir", str(direct), "--clear"])
    _run([UV, "build", "--sdist", "--out-dir", str(sdist), "--clear"])
    archive = _single(sdist, ".tar.gz")
    _run(
        [
            UV,
            "build",
            str(archive),
            "--wheel",
            "--out-dir",
            str(derived),
            "--clear",
        ]
    )
    return {
        "direct-wheel": _single(direct, ".whl"),
        "sdist-wheel": _single(derived, ".whl"),
        "root": root,
    }


@dataclass(frozen=True)
class Install:
    python: Path
    command: Path
    gui_command: Path
    cwd: Path
    env: dict[str, str]

    def run(
        self, arguments: Sequence[str], *, input: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [os.fspath(self.command), *arguments],
            cwd=self.cwd,
            env=self.env,
            input=input,
            check=False,
            capture_output=True,
            text=True,
        )

    def python_run(self, program: str) -> subprocess.CompletedProcess[str]:
        return _run(
            [os.fspath(self.python), "-c", program],
            cwd=self.cwd,
            env=self.env,
        )


class InstallFactory(Protocol):
    def __call__(
        self,
        source: Path,
        *,
        editable: bool = False,
        extras: tuple[str, ...] = (),
    ) -> Install: ...


def fresh_install_factory(tmp_path: Path) -> InstallFactory:
    counter = 0

    def install(
        source: Path,
        *,
        editable: bool = False,
        extras: tuple[str, ...] = (),
    ) -> Install:
        nonlocal counter
        counter += 1
        root = tmp_path / f"install-{counter}"
        venv = root / "venv"
        cwd = root / "cwd"
        cwd.mkdir(parents=True)
        _run([UV, "venv", "--clear", str(venv)])
        arguments = [
            UV,
            "pip",
            "install",
            "--python",
            os.fspath(venv / "bin/python"),
        ]
        if editable:
            arguments.append("--editable")
        requirement = os.fspath(source)
        if extras:
            requirement += f"[{','.join(extras)}]"
        arguments.append(requirement)
        _run(arguments)
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        return Install(
            venv / "bin/python",
            venv / "bin/rheplicant",
            venv / "bin/rheplicant-gui",
            cwd,
            env,
        )

    return install


def wait_for_url(
    url: str, process: subprocess.Popen[str], *, timeout: float
) -> None:
    deadline = time.monotonic() + timeout
    while True:
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(f"GUI exited with {returncode}")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                ready = response.status == 200
        except (OSError, urllib.error.URLError):
            ready = False
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(f"GUI exited with {returncode}")
        if ready:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"GUI did not become ready at {url}")
        time.sleep(0.1)


#: How long a TERMed process group may take to drain before the survivors are
#: killed.  Kept here rather than taken from ``gui_child`` so a test can shorten
#: the wait it is measuring; the sweep itself is the production one.
_GROUP_GRACE_SECONDS = gui_child.GROUP_GRACE_SECONDS

# The GUI server sweeps its own job workers by process group -- see
# ``_rheplicant_bootstrap.gui_child`` -- and this sweeps the server the same
# way.  The reasoning about which groups are ours to signal, why the pid must be
# type-tested rather than caught, and why ``EPERM`` is swallowed, lives there
# once; a second copy would be a second behaviour the first time one of them was
# corrected.  These names stay bound in this module because the tests that pin
# the sweep substitute them here.
_process_group = gui_child.process_group
_signal_group = gui_child.signal_group
_stop_child = gui_child.stop_child


def _reap_group(group: int | None) -> None:
    """Wait a bounded time for a TERMed group to drain, then kill the rest."""
    gui_child.reap_group(group, grace_seconds=_GROUP_GRACE_SECONDS)


def _stop_process(process: subprocess.Popen[str]) -> None:
    """Stop the server AND everything it spawned.

    The server spawns a scientific worker for the life of every job -- see
    ``_rheplicant_bootstrap.gui_child.drained_run`` -- so signalling only the
    direct child leaves that worker holding its outputs, memory and CPU long
    after the test has reported green.  The whole group is the unit of cleanup.
    """
    gui_child.stop_process_group(
        process,
        # Named through this module rather than passed straight through, so a
        # test that substitutes ``_stop_child`` still substitutes what runs.
        stop=lambda child: _stop_child(child),
        grace_seconds=_GROUP_GRACE_SECONDS,
    )


@contextmanager
def running_gui(install: Install) -> Iterator[str]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    stdout_path = install.cwd / "rheplicant-gui.stdout.log"
    stderr_path = install.cwd / "rheplicant-gui.stderr.log"
    with (
        stdout_path.open("w+", encoding="utf-8") as stdout,
        stderr_path.open("w+", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(
            [
                os.fspath(install.gui_command),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ],
            cwd=install.cwd,
            env=install.env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            # The server's own children -- one scientific worker per job -- are
            # only reachable at teardown as a GROUP, and a group that is ours to
            # signal is one we created. Without this the server would share
            # pytest's group and there would be nothing safe to signal.
            start_new_session=True,
        )
        base_url = f"http://127.0.0.1:{port}"
        primary_error: BaseException | None = None
        try:
            wait_for_url(base_url + "/api/starter", process, timeout=30)
            yield base_url
        except Exception as error:
            primary_error = error
            stdout.flush()
            stderr.flush()
            enriched_error = RuntimeError(
                f"{error}\nstdout:\n{stdout_path.read_text()}"
                f"\nstderr:\n{stderr_path.read_text()}"
            )
            primary_error = enriched_error
            raise enriched_error from error
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                _stop_process(process)
            except Exception:
                if primary_error is None:
                    raise
