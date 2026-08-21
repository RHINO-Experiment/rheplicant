from __future__ import annotations

import contextlib
import json
import os
import pathlib
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from unittest.mock import MagicMock, call, patch

import pytest

from tests.config import wheel_support

PROJECT_ROOT = wheel_support.PROJECT_ROOT

_POSIX_ONLY = pytest.mark.skipif(
    os.name != "posix", reason="sessions, process groups and killpg are POSIX"
)


@pytest.fixture(scope="session")
def packaged_distributions(tmp_path_factory):
    return wheel_support.build_distributions(tmp_path_factory)


@pytest.fixture
def packaged_install(tmp_path):
    return wheel_support.fresh_install_factory(tmp_path)


def test_wait_for_url_rejects_a_200_after_the_owned_process_exits():
    response = MagicMock(status=200)
    response.__enter__.return_value = response
    process = MagicMock()
    process.poll.side_effect = [None, 17]

    with patch.object(wheel_support.urllib.request, "urlopen", return_value=response):
        with pytest.raises(RuntimeError, match="GUI exited with 17"):
            wheel_support.wait_for_url(
                "http://127.0.0.1:8123/api/starter", process, timeout=30
            )


def test_wait_for_url_reports_an_early_owned_process_exit():
    process = MagicMock()
    process.poll.return_value = 9
    process.returncode = 9

    with patch.object(
        wheel_support.urllib.request,
        "urlopen",
        side_effect=urllib.error.URLError("not ready"),
    ):
        with pytest.raises(RuntimeError, match="GUI exited with 9"):
            wheel_support.wait_for_url(
                "http://127.0.0.1:8123/api/starter", process, timeout=30
            )


def test_wait_for_url_stops_at_the_bounded_deadline():
    url = "http://127.0.0.1:8123/api/starter"
    process = MagicMock()
    process.poll.return_value = None

    with (
        patch.object(
            wheel_support.urllib.request,
            "urlopen",
            side_effect=urllib.error.URLError("not ready"),
        ) as urlopen,
        patch.object(
            wheel_support.time,
            "monotonic",
            side_effect=[100.0, 100.0, 130.0],
        ),
        patch.object(wheel_support.time, "sleep") as sleep,
    ):
        with pytest.raises(
            TimeoutError,
            match="GUI did not become ready at " + url,
        ):
            wheel_support.wait_for_url(url, process, timeout=30)

    assert urlopen.call_args_list == [
        call(url, timeout=1),
        call(url, timeout=1),
    ]
    sleep.assert_called_once_with(0.1)


def test_stop_process_is_a_noop_after_an_early_exit():
    process = MagicMock()
    process.poll.return_value = 0

    wheel_support._stop_process(process)

    process.terminate.assert_not_called()
    process.wait.assert_not_called()


def test_stop_process_terminates_and_waits_normally():
    process = MagicMock()
    process.poll.return_value = None

    wheel_support._stop_process(process)

    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=10)
    process.kill.assert_not_called()


def test_stop_process_kills_after_the_terminate_wait_times_out():
    process = MagicMock()
    process.poll.side_effect = [None, None]
    process.wait.side_effect = [subprocess.TimeoutExpired("gui", 10), 0]

    wheel_support._stop_process(process)

    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert process.wait.call_args_list == [
        ((), {"timeout": 10}),
        ((), {"timeout": 5}),
    ]


def test_stop_process_tolerates_an_exit_race_during_terminate():
    process = MagicMock()
    process.poll.return_value = None
    process.terminate.side_effect = ProcessLookupError("already exited")

    wheel_support._stop_process(process)

    process.wait.assert_not_called()
    process.kill.assert_not_called()


def test_stop_process_skips_kill_when_process_exits_after_timeout():
    process = MagicMock()
    process.poll.side_effect = [None, 0]
    process.wait.side_effect = subprocess.TimeoutExpired("gui", 10)

    wheel_support._stop_process(process)

    process.terminate.assert_called_once_with()
    process.kill.assert_not_called()


def test_stop_process_tolerates_an_exit_race_during_kill():
    process = MagicMock()
    process.poll.side_effect = [None, None]
    process.wait.side_effect = subprocess.TimeoutExpired("gui", 10)
    process.kill.side_effect = ProcessLookupError("already exited")

    wheel_support._stop_process(process)

    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=10)


def test_running_gui_uses_the_starter_boundary_and_bounded_cleanup(tmp_path):
    install = wheel_support.Install(
        python=tmp_path / "python",
        command=tmp_path / "rheplicant",
        gui_command=tmp_path / "rheplicant-gui",
        cwd=tmp_path,
        env={"PATH": "/bin"},
    )
    probe = MagicMock()
    probe.__enter__.return_value = probe
    probe.getsockname.return_value = ("127.0.0.1", 43123)
    process = MagicMock()

    with (
        patch.object(wheel_support.socket, "socket", return_value=probe),
        patch.object(wheel_support.subprocess, "Popen", return_value=process),
        patch.object(wheel_support, "wait_for_url") as wait_for_url,
        patch.object(wheel_support, "_stop_process") as stop_process,
    ):
        with wheel_support.running_gui(install) as base_url:
            assert base_url == "http://127.0.0.1:43123"

    wait_for_url.assert_called_once_with(
        "http://127.0.0.1:43123/api/starter", process, timeout=30
    )
    stop_process.assert_called_once_with(process)


def test_running_gui_preserves_body_failure_when_cleanup_also_fails(tmp_path):
    install = wheel_support.Install(
        python=tmp_path / "python",
        command=tmp_path / "rheplicant",
        gui_command=tmp_path / "rheplicant-gui",
        cwd=tmp_path,
        env={"PATH": "/bin"},
    )
    probe = MagicMock()
    probe.__enter__.return_value = probe
    probe.getsockname.return_value = ("127.0.0.1", 43124)
    process = MagicMock()

    with (
        patch.object(wheel_support.socket, "socket", return_value=probe),
        patch.object(wheel_support.subprocess, "Popen", return_value=process),
        patch.object(wheel_support, "wait_for_url"),
        patch.object(
            wheel_support,
            "_stop_process",
            side_effect=OSError("cleanup raced"),
        ),
    ):
        with pytest.raises(RuntimeError) as caught:
            with wheel_support.running_gui(install):
                (tmp_path / "rheplicant-gui.stdout.log").write_text("server out")
                (tmp_path / "rheplicant-gui.stderr.log").write_text("server err")
                raise ValueError("playwright failed")

    assert "playwright failed" in str(caught.value)
    assert "stdout:\nserver out" in str(caught.value)
    assert "stderr:\nserver err" in str(caught.value)
    assert isinstance(caught.value.__cause__, ValueError)


def test_running_gui_preserves_startup_failure_when_cleanup_also_fails(tmp_path):
    install = wheel_support.Install(
        python=tmp_path / "python",
        command=tmp_path / "rheplicant",
        gui_command=tmp_path / "rheplicant-gui",
        cwd=tmp_path,
        env={"PATH": "/bin"},
    )
    probe = MagicMock()
    probe.__enter__.return_value = probe
    probe.getsockname.return_value = ("127.0.0.1", 43126)

    def fail_startup(*_args, **_kwargs):
        (tmp_path / "rheplicant-gui.stdout.log").write_text("startup out")
        (tmp_path / "rheplicant-gui.stderr.log").write_text("startup err")
        raise TimeoutError("starter timed out")

    with (
        patch.object(wheel_support.socket, "socket", return_value=probe),
        patch.object(wheel_support.subprocess, "Popen", return_value=MagicMock()),
        patch.object(wheel_support, "wait_for_url", side_effect=fail_startup),
        patch.object(
            wheel_support,
            "_stop_process",
            side_effect=OSError("cleanup raced"),
        ),
    ):
        with pytest.raises(RuntimeError) as caught:
            with wheel_support.running_gui(install):
                pytest.fail("startup failure must prevent the body")

    assert "starter timed out" in str(caught.value)
    assert "stdout:\nstartup out" in str(caught.value)
    assert "stderr:\nstartup err" in str(caught.value)
    assert isinstance(caught.value.__cause__, TimeoutError)


def test_running_gui_surfaces_cleanup_failure_after_a_successful_body(tmp_path):
    install = wheel_support.Install(
        python=tmp_path / "python",
        command=tmp_path / "rheplicant",
        gui_command=tmp_path / "rheplicant-gui",
        cwd=tmp_path,
        env={"PATH": "/bin"},
    )
    probe = MagicMock()
    probe.__enter__.return_value = probe
    probe.getsockname.return_value = ("127.0.0.1", 43125)

    with (
        patch.object(wheel_support.socket, "socket", return_value=probe),
        patch.object(wheel_support.subprocess, "Popen", return_value=MagicMock()),
        patch.object(wheel_support, "wait_for_url"),
        patch.object(
            wheel_support,
            "_stop_process",
            side_effect=OSError("cleanup failed"),
        ),
    ):
        with pytest.raises(OSError, match="cleanup failed"):
            with wheel_support.running_gui(install):
                pass


@_POSIX_ONLY
def test_process_group_never_asks_the_kernel_about_a_doubled_pid():
    """The hazard the TYPE test exists for, rather than an ``except TypeError``.

    ``MagicMock`` satisfies ``__index__`` -- ``unittest.mock._return_values``
    maps it to ``1`` -- so ``os.getpgid(mock.pid)`` does not raise. It answers
    with the process group of PID 1, and the ``killpg`` that followed would be
    aimed at init's group: ``PermissionError`` as a user, and a signal that
    lands as root. An exception handler around the call is no defence at all
    because nothing is ever raised, which is why the guard is a type test and
    why ``getpgid`` must not even be reached.
    """
    process = MagicMock()
    assert process.pid.__index__() == 1  # the hazard, stated rather than implied

    with patch.object(wheel_support.os, "getpgid") as getpgid:
        assert wheel_support._process_group(process) is None

    getpgid.assert_not_called()


@_POSIX_ONLY
def test_a_child_that_shares_our_process_group_is_not_a_group_we_own():
    """``start_new_session`` is what makes the child a group LEADER, so
    ``getpgid(pid) == pid`` is the certificate that the group is ours to
    signal. A child started without it sits in pytest's own group, and
    signalling that group would take the suite down with the server it was
    cleaning up. Answering ``None`` degrades to signalling the child alone."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert os.getpgid(child.pid) == os.getpgrp()
        assert os.getpgid(child.pid) != child.pid
        assert wheel_support._process_group(child) is None
    finally:
        child.kill()
        child.wait(timeout=10)


@_POSIX_ONLY
def test_a_child_inside_somebody_elses_group_is_not_a_group_we_own():
    """The case the ``getpgrp()`` lock cannot see, and the reason ``group !=
    pid`` is a separate condition rather than a rephrasing of it.

    A child placed in a THIRD group -- neither its own nor pytest's -- is a
    MEMBER, not the leader, so that group was created by somebody else and its
    other members are somebody else's processes. Measured: with the ``group !=
    pid`` arm deleted, this answers the leader's group and the sweep would
    SIGKILL every sibling in it. The two children stay inside pytest's session
    because ``setpgid`` may only join a group in the caller's own session,
    which is also why the group here cannot be built out of a new session.
    """
    idle = [sys.executable, "-c", "import time; time.sleep(30)"]
    leader = subprocess.Popen(idle, process_group=0)
    member = subprocess.Popen(idle, process_group=leader.pid)
    try:
        assert os.getpgid(member.pid) == leader.pid
        assert os.getpgid(member.pid) != member.pid
        assert os.getpgid(member.pid) != os.getpgrp()
        assert wheel_support._process_group(member) is None
    finally:
        for child in (member, leader):
            child.kill()
            child.wait(timeout=10)


@_POSIX_ONLY
def test_a_child_in_its_own_session_is_the_group_we_signal():
    """The positive branch of the same certificate."""
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    try:
        assert wheel_support._process_group(child) == child.pid
    finally:
        child.kill()
        child.wait(timeout=10)


#: A stand-in for the installed GUI: it spawns a long-lived descendant exactly
#: as the real server does -- ``rheplicant.gui.jobs._run_isolated_job`` runs
#: ``python -m _rheplicant_bootstrap.gui_worker`` through ``_drained_run``, so a
#: job in flight IS a live grandchild of pytest -- and only then binds the port
#: ``wait_for_url`` polls.  The pid file is written and fsynced BEFORE the bind,
#: which is what lets the test read it the moment ``/api/starter`` answers,
#: instead of sleeping and hoping.
_FAKE_GUI = '''
import http.server
import json
import os
import subprocess
import sys

host = sys.argv[sys.argv.index("--host") + 1]
port = int(sys.argv[sys.argv.index("--port") + 1])
descendant = subprocess.Popen([sys.executable, "-c", sys.argv[-1]])

with open(os.environ["RHEPLICANT_FAKE_GUI_PIDS"], "w", encoding="utf-8") as handle:
    json.dump({"server": os.getpid(), "descendant": descendant.pid}, handle)
    handle.flush()
    os.fsync(handle.fileno())


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *_arguments):
        pass


http.server.HTTPServer((host, port), Handler).serve_forever()
'''

#: Every descendant below announces itself only once its SIGTERM disposition
#: is installed, and the tests wait for that announcement before they let the
#: sweep run.  Without it the race is real and was measured: the descendant
#: reaches its ``signal.signal`` call an interpreter start and three imports
#: after ``Popen`` returns, while the fake server writes the pid file and binds
#: the port immediately, so a TERM that arrives first is taken on the DEFAULT
#: disposition.  The honours-TERM case then dies without recording its marker
#: (two failures in roughly six runs under load) and -- worse, because it is
#: silent -- the ignores-TERM case dies on the TERM it was written to ignore,
#: so the escalation to a group KILL is never exercised at all and the test
#: passes having measured nothing.
_DESCENDANT_READY = '''
import os


def _ready():
    marker = os.environ["RHEPLICANT_FAKE_GUI_PIDS"] + ".ready"
    handle = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.write(handle, b"ready")
    os.fsync(handle)
    os.close(handle)
'''

#: Records that it was ASKED to stop before stopping, so the group TERM is
#: observable and not merely inferred from the descendant being gone.
_DESCENDANT_HONOURS_TERM = _DESCENDANT_READY + '''
import signal
import time


def _record(*_ignored):
    marker = os.environ["RHEPLICANT_FAKE_GUI_PIDS"] + ".term"
    handle = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.write(handle, b"term")
    os.fsync(handle)
    os.close(handle)
    os._exit(0)


signal.signal(signal.SIGTERM, _record)
_ready()
time.sleep(3600)
'''

#: The worst case a scientific worker really presents: TERM does not end it, so
#: only the escalation to a group KILL can.
_DESCENDANT_IGNORES_TERM = _DESCENDANT_READY + '''
import signal
import time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
_ready()
time.sleep(3600)
'''


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # it exists; it is simply not ours any more
    return True


def _await_ready(record: pathlib.Path, *, timeout: float = 30.0) -> None:
    """Wait for the descendant to say its SIGTERM disposition is installed.

    Synchronising on this rather than on the server's port is what makes the
    sweep deterministic: until the marker exists, a TERM delivered to the group
    is taken on the descendant's DEFAULT disposition, and neither what it
    honours nor what it ignores has been tested.  A release gate must not be
    intermittently red, and must not be quietly green either.
    """
    marker = record.parent / (record.name + ".ready")
    deadline = time.monotonic() + timeout
    while not marker.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"the descendant never became ready: {marker}")
        time.sleep(0.02)


def _gone(pid: int, *, timeout: float = 10.0) -> bool:
    """Poll to a deadline rather than sleeping a guess.

    Neither process can linger as an unreaped zombie that answers signal 0:
    the server is a direct child and ``_stop_process`` reaps it, and the
    descendant is a GRANDchild, so it is reparented to init and reaped there.
    """
    deadline = time.monotonic() + timeout
    while _alive(pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


@pytest.fixture
def fake_gui(tmp_path):
    """An installed-GUI stand-in that spawns a genuinely long-lived descendant."""
    records: list[pathlib.Path] = []

    def make(descendant_source: str) -> tuple[wheel_support.Install, pathlib.Path]:
        index = len(records)
        cwd = tmp_path / f"run-{index}"
        cwd.mkdir()
        server = tmp_path / f"server-{index}.py"
        server.write_text(_FAKE_GUI)
        record = tmp_path / f"pids-{index}.json"
        shim = tmp_path / f"rheplicant-gui-{index}"
        # ``exec`` rather than a child: the pid Popen created stays the pid of
        # the server, so it is the session leader ``start_new_session`` made and
        # there is no shell in between to confuse the group identity.
        interpreter = shlex.quote(sys.executable)
        program = shlex.quote(str(server))
        source = shlex.quote(descendant_source)
        shim.write_text(f'#!/bin/sh\nexec {interpreter} {program} "$@" {source}\n')
        shim.chmod(0o755)
        records.append(record)
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        environment["RHEPLICANT_FAKE_GUI_PIDS"] = str(record)
        return (
            wheel_support.Install(
                python=pathlib.Path(sys.executable),
                command=shim,
                gui_command=shim,
                cwd=cwd,
                env=environment,
            ),
            record,
        )

    yield make

    for record in records:  # the backstop: a failing test leaks nothing
        if not record.exists():
            continue
        for pid in json.loads(record.read_text()).values():
            with contextlib.suppress(OSError, ValueError):
                os.kill(int(pid), signal.SIGKILL)


@_POSIX_ONLY
@pytest.mark.parametrize(
    "descendant",
    [_DESCENDANT_HONOURS_TERM, _DESCENDANT_IGNORES_TERM],
    ids=["honours-term", "ignores-term"],
)
def test_running_gui_kills_the_descendants_the_server_spawned(
    fake_gui, monkeypatch, descendant
):
    """The defect this pins: the GUI server spawns a scientific worker for the
    life of every job, and a teardown that signals only the DIRECT child leaves
    that worker running -- holding its outputs, its memory and its CPU -- long
    after the test that started it has reported green."""
    monkeypatch.setattr(wheel_support, "_GROUP_GRACE_SECONDS", 1.0)
    install, record = fake_gui(descendant)

    with wheel_support.running_gui(install) as base_url:
        pids = json.loads(record.read_text())
        server, child = pids["server"], pids["descendant"]
        with urllib.request.urlopen(base_url + "/api/starter", timeout=5) as reply:
            assert reply.status == 200
        _await_ready(record)
        assert _alive(server) and _alive(child)
        assert os.getpgid(server) == server  # its own session, per start_new_session
        assert os.getpgid(child) == server  # and the descendant is inside it
        assert os.getpgid(server) != os.getpgrp()  # which is emphatically not ours

    assert _gone(server), "the server outlived teardown"
    assert _gone(child), "the descendant outlived teardown"
    if descendant is _DESCENDANT_HONOURS_TERM:
        assert (record.parent / (record.name + ".term")).exists(), (
            "the descendant was killed without ever being asked to stop"
        )


@_POSIX_ONLY
def test_the_group_is_swept_even_when_stopping_the_child_raises(
    fake_gui, monkeypatch
):
    """The sweep is in a ``finally`` rather than on the success path. A child
    stop that blew up would otherwise strand every descendant, which is the
    same leak wearing a different hat."""
    monkeypatch.setattr(wheel_support, "_GROUP_GRACE_SECONDS", 1.0)

    def explode(process):
        # Reap first: the group TERM has already reached the server, and a
        # child left unwaited would be a zombie for the rest of the session.
        process.wait(timeout=10)
        raise OSError("stopping the child failed")

    monkeypatch.setattr(wheel_support, "_stop_child", explode)
    install, record = fake_gui(_DESCENDANT_IGNORES_TERM)

    with pytest.raises(OSError, match="stopping the child failed"):
        with wheel_support.running_gui(install):
            # The descendant must be ignoring TERM before the sweep runs, or
            # this measures the default disposition and not the escalation.
            _await_ready(record)

    child = json.loads(record.read_text())["descendant"]
    assert _gone(child), "the descendant outlived a failed child stop"


def test_fresh_gui_wheel_passes_playwright(
    packaged_distributions, packaged_install
):
    install = packaged_install(
        packaged_distributions["direct-wheel"], extras=("gui",)
    )
    assert "PYTHONPATH" not in install.env
    assert "PYTHONHOME" not in install.env
    with wheel_support.running_gui(install) as base_url:
        env = dict(install.env)
        env["RHEPLICANT_E2E_BASE_URL"] = base_url
        completed = subprocess.run(
            ["npm", "--prefix", "tools/config_gui_spike/react", "run", "test:e2e"],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=600,
        )
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
