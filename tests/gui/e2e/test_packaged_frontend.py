from __future__ import annotations

import subprocess
import urllib.error
from unittest.mock import MagicMock, call, patch

import pytest

from tests.config import wheel_support

PROJECT_ROOT = wheel_support.PROJECT_ROOT


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
