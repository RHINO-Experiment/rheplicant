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

#: Names the bayesmith checkout directly, for a machine where it is not a
#: sibling of rheplicant's main checkout. Unset or empty, it is found from git.
BAYESMITH_VARIABLE = "RHEPLICANT_BAYESMITH_CHECKOUT"

#: Variables that point git at a repository other than the one its working
#: directory is in. A hook or a mutation script can leave one set, and the
#: lookup below has to answer for the project root it was given.
_GIT_REDIRECTS = frozenset({"GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR"})


@dataclass(frozen=True)
class BayesmithCheckout:
    """The sibling bayesmith checkout, for the guards that read its pages.

    Nothing installs from it. bayesmith 0.10.0 is on PyPI (2026-10-02), so
    the fresh-venv installs below take the range `pyproject.toml` declares
    from the index. Until then they installed a wheel from this checkout's
    `runs/t004/dist` after checking it against a release manifest, and the
    day that wheel was rebuilt at the release tag without the manifest, six
    installs failed on a hash while the package was unchanged.

    `basis` says how `path` was chosen, so a skip can say where it looked.
    """

    path: Path
    basis: str


def _main_checkout(project_root: Path, environ: Mapping[str, str]) -> Path:
    """The root of the main working tree that `project_root` belongs to.

    A linked worktree, which every `.claude/worktrees/<name>` is, has its own
    root and shares the main checkout's `.git`; `--git-common-dir` names that
    directory, and its parent is the main root. Measured 2026-09-23: taking
    `project_root.parent` instead put bayesmith under `.claude/worktrees/`,
    and every fresh-venv install skipped with a message saying this machine
    had no release while the release was on disk.

    Where there is no main working tree to find, `project_root` is the answer:
    no git, a directory that is not the top of its repository (an unpacked
    sdist inside some other checkout), a bare repository or a submodule. In
    the last two the common directory is not named `.git`.
    """
    env = {key: value for key, value in environ.items() if key not in _GIT_REDIRECTS}
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel", "--git-common-dir"],
            cwd=project_root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return project_root
    lines = completed.stdout.splitlines()
    if completed.returncode != 0 or len(lines) != 2:
        return project_root
    # The common directory is relative from the main checkout and absolute
    # from a worktree; joining onto the root reads both.
    top, common = (Path(project_root, line).resolve() for line in lines)
    if top != project_root.resolve() or common.name != ".git":
        return project_root
    return common.parent


def locate_bayesmith(project_root: Path, environ: Mapping[str, str]) -> BayesmithCheckout:
    """Where the bayesmith checkout is expected, and why there.

    `RHEPLICANT_BAYESMITH_CHECKOUT` wins when set. Otherwise bayesmith is the
    sibling of rheplicant's MAIN checkout, which is one directory whether the
    suite runs from that checkout or from any worktree of it.
    """
    named = environ.get(BAYESMITH_VARIABLE, "")
    if named:
        return BayesmithCheckout(
            Path(named).expanduser().resolve(), f"named by ${BAYESMITH_VARIABLE}"
        )
    main = _main_checkout(project_root, environ)
    return BayesmithCheckout(
        main.parent / "bayesmith", f"the sibling of rheplicant's main checkout, {main}"
    )


BAYESMITH = locate_bayesmith(PROJECT_ROOT, os.environ)
BAYESMITH_CHECKOUT = BAYESMITH.path


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
    """Installs into fresh venvs, every dependency resolved from the index.

    bayesmith included: the wheel under test declares its range and the
    resolver finds it, which is what a user's `pip install` does.
    """
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


def wait_for_url(url: str, process: subprocess.Popen[str], *, timeout: float) -> None:
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
                f"{error}\nstdout:\n{stdout_path.read_text()}\nstderr:\n{stderr_path.read_text()}"
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
