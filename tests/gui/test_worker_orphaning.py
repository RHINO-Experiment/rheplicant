"""A job worker whose server is SIGKILLed, and what is left holding the machine.

Plan 6C gave the worker its own process group, anchored by a process the parent
opened, and the parent sweeps that group on every path it controls.  That was a
net improvement and it moved one case rather than closing it: because the
worker is no longer in the SERVER's group, a group TERM aimed at the server no
longer reaches it, and a server that is ``SIGKILL``ed mid-job never runs a
``finally`` at all.  Nothing then ends the worker, and nothing ends the
science process the worker spawned -- which is where a JAX or GPU context
actually lives.

So the worker is not allowed to depend on its parent surviving in order to be
reaped.  These tests do not mock a ppid: they start a real parent, run a real
worker through the real ``drained_run``, kill the real parent, and ask the
kernel what is left.

The synchronisation is the part that has been got wrong here before.  An
earlier descendant-cleanup test let the parent race ahead while the child was
still three imports away from being ready; it failed two runs in six under load
and, worse, passed the rest of the time having measured nothing.  The worker
below announces itself only once its watch is armed AND its science child
exists, by writing a temporary file and renaming it over the record, so a
reader never sees a half-written announcement.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

_POSIX_ONLY = pytest.mark.skipif(
    os.name != "posix",
    reason="process groups, killpg and orphan reparenting are POSIX",
)

_SRC = Path(__file__).resolve().parents[2] / "src"

#: Stands in for the GUI server: it runs one job through the real
#: ``drained_run``, so the worker gets the real anchor, the real new process
#: group and the real stream draining.  The only thing added is that the
#: anchor's pid is recorded, because a test cannot otherwise learn the name of
#: a group the production code opens for itself.
_SERVER = '''
import json
import os
import sys

from _rheplicant_bootstrap import gui_child

record = sys.argv[1]
worker = sys.argv[2]
state = {"server": os.getpid()}


def _announce(**fields):
    state.update(fields)
    handle = open(record + ".writing", "w", encoding="utf-8")
    json.dump(state, handle)
    handle.flush()
    os.fsync(handle.fileno())
    handle.close()
    os.replace(record + ".writing", record)


opened = gui_child._group_anchor


def _anchored():
    anchor = opened()
    _announce(anchor=None if anchor is None else anchor.pid)
    return anchor


gui_child._group_anchor = _anchored
_announce()
gui_child.drained_run(
    [sys.executable, worker, record + ".worker"],
    input_bytes=b"",
    stdout=gui_child.StreamTail(limit=1024),
    stderr=gui_child.StreamTail(limit=1024),
    timeout=3600,
)
'''

#: The real ``gui_worker.main``, with only the science replaced: a long-lived
#: child -- what a JAX or GPU context would be -- and then a wait that outlives
#: any deadline in this file.  Going through ``main`` is the point: whatever
#: arms the watch has to be armed by the entry point the worker really uses.
_WORKER = '''
import json
import os
import subprocess
import sys
import time

import _rheplicant_bootstrap.gui_worker as gui_worker

record = sys.argv[1]


def _science(_yaml_text):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"])
    # Announced last, and only here: by this line the watch main() arms is
    # already running and the child that has to be swept already exists, so a
    # reader that waits for this record is waiting for something true.
    handle = open(record + ".writing", "w", encoding="utf-8")
    json.dump(
        {"worker": os.getpid(), "science": child.pid, "group": os.getpgrp()},
        handle,
    )
    handle.flush()
    os.fsync(handle.fileno())
    handle.close()
    os.replace(record + ".writing", record)
    time.sleep(3600)
    return {}


gui_worker._run_validation = _science
raise SystemExit(gui_worker.main(["validate"]))
'''


#: The same worker, except that it says who it is BEFORE ``main`` runs and
#: then spends five seconds getting there.  That gap is a real one -- an
#: interpreter start and this module's imports -- widened here so a kill can be
#: aimed into it deterministically instead of raced for.  A worker orphaned in
#: that window captures init as its parent and the ppid it watches never
#: changes again, so nothing but the anchor's death can tell it what happened.
_WORKER_SLOW_TO_ARM = '''
import json
import os
import sys
import time

import _rheplicant_bootstrap.gui_worker as gui_worker

record = sys.argv[1]
handle = open(record + ".writing", "w", encoding="utf-8")
json.dump({"worker": os.getpid(), "group": os.getpgrp()}, handle)
handle.flush()
os.fsync(handle.fileno())
handle.close()
os.replace(record + ".writing", record)
time.sleep(5)


def _science(_yaml_text):
    time.sleep(3600)
    return {}


gui_worker._run_validation = _science
raise SystemExit(gui_worker.main(["validate"]))
'''


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # it exists; it is simply not ours any more
    return True


def _gone(pid: int, *, timeout: float) -> bool:
    """Poll to a deadline rather than sleeping a guess."""
    deadline = time.monotonic() + timeout
    while _alive(pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


def _group_empty(group: int) -> bool:
    """Whether anything is left in ``group``; signal 0 sends nothing.

    ``EPERM`` counts as empty for the reason ``gui_child.reap_group`` gives:
    on darwin a group whose last member is a zombie mid-reap answers with it
    rather than ``ESRCH``, and every member of this group is a descendant of
    this test, so it cannot mean somebody else's process.
    """
    try:
        os.killpg(group, 0)
    except (ProcessLookupError, PermissionError):
        return True
    return False


def _group_gone(group: int, *, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while not _group_empty(group):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


def _await(path: Path, *, timeout: float = 60.0) -> dict[str, int]:
    """Wait for one announcement, which is renamed into place or not there."""
    deadline = time.monotonic() + timeout
    while True:
        with contextlib.suppress(OSError, ValueError):
            return json.loads(path.read_text(encoding="utf-8"))
        if time.monotonic() >= deadline:
            raise AssertionError(f"nothing announced itself at {path}")
        time.sleep(0.02)


@pytest.fixture
def orphaned_job(tmp_path):
    """Start a server running one job, and hand back every pid involved.

    Teardown ``SIGKILL``s everything it recorded, so a failing assertion in the
    body of a test cannot leak a process for the rest of the session.
    """
    recorded: list[int] = []
    started: list[subprocess.Popen[bytes]] = []

    def start(worker_source: str = _WORKER):
        server_script = tmp_path / "server.py"
        server_script.write_text(_SERVER, encoding="utf-8")
        worker_script = tmp_path / "worker.py"
        worker_script.write_text(worker_source, encoding="utf-8")
        record = tmp_path / "pids.json"
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(_SRC)
        process = subprocess.Popen(
            [sys.executable, str(server_script), str(record), str(worker_script)],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        started.append(process)
        pids = _await(Path(str(record) + ".worker"))
        pids.update(_await(record))
        recorded.extend(int(value) for value in pids.values())
        return process, pids

    yield start

    for pid in recorded:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)
    for process in started:
        with contextlib.suppress(OSError):
            process.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=10)


@_POSIX_ONLY
def test_a_worker_whose_server_is_killed_takes_its_whole_group_with_it(
    orphaned_job,
):
    """The case Plan 6C moved rather than closed, measured end to end.

    ``SIGKILL`` cannot be caught, so the server runs no ``finally`` and sweeps
    nothing.  The worker is outside the server's group, so a group signal
    aimed at the server never reaches it either.  What has to hold is that the
    worker notices on its own -- and takes the anchor and its own science
    child with it, or the orphan has merely been renamed.
    """
    process, pids = orphaned_job()
    server, worker, science, anchor = (
        pids["server"], pids["worker"], pids["science"], pids["anchor"]
    )
    assert pids["group"] == anchor, "the worker did not run in the anchor's group"
    assert pids["group"] != os.getpgrp(), "the job ran in the test runner's group"
    assert _alive(worker) and _alive(science) and _alive(anchor)

    os.kill(server, signal.SIGKILL)
    process.wait(timeout=30)

    assert _gone(worker, timeout=30.0), "the worker outlived the server that made it"
    # The assertion that carries the group sweep.  Measured before the watch
    # existed: the anchor DOES follow its dead parent, because its stdin write
    # end went with it -- so the anchor alone would have been an assertion two
    # different implementations both satisfy.  Nothing reaches the science
    # child except a signal to the group it is in.
    assert _gone(science, timeout=30.0), "the science process outlived its worker"
    assert _gone(anchor, timeout=30.0), "the anchor outlived the group it held open"
    assert _group_gone(pids["group"], timeout=30.0), "the job group still has members"


@_POSIX_ONLY
def test_a_worker_orphaned_before_its_watch_is_armed_is_still_reclaimed(
    orphaned_job,
):
    """The blind spot in watching a ppid, aimed at rather than raced for.

    The value a worker captures when it arms is already init's if the parent
    died first, and it never changes again -- so the one check that looks
    portable and sufficient is inert for exactly the job it was meant to save.
    Here the kill lands squarely inside the worker's start-up, and what has to
    reclaim the job is the other fact: the anchor holding this group open goes
    when the parent that opened it goes.
    """
    process, pids = orphaned_job(_WORKER_SLOW_TO_ARM)
    worker, group = pids["worker"], pids["group"]
    assert _alive(worker), "the worker was not running when the server was killed"
    assert group != os.getpgrp(), "the job ran in the test runner's group"

    os.kill(pids["server"], signal.SIGKILL)
    process.wait(timeout=30)

    assert _gone(worker, timeout=30.0), "a worker orphaned mid-start-up ran on"
    assert _group_gone(group, timeout=30.0), "the job group still has members"


#: Answers with the group ``gui_worker`` would sweep, and with the group it is
#: actually in, so the two can be compared from outside the process.
_GROUP_PROBE = (
    "import os, sys;"
    "sys.path.insert(0, sys.argv[1]);"
    "import _rheplicant_bootstrap.gui_worker as worker;"
    "print(worker._job_group(os.getppid()), os.getpgrp())"
)


@_POSIX_ONLY
@pytest.mark.parametrize("opened", [True, False], ids=["opened-for-us", "inherited"])
def test_only_a_group_the_parent_opened_for_this_worker_is_ours_to_sweep(opened):
    """The guard that keeps an orphan sweep from becoming a massacre.

    A worker that inherited its parent's group shares it with everything else
    that parent lives among -- a shell's job, a supervisor's other children,
    a whole test session -- and none of that is the worker's to ``SIGKILL``.
    Real kernel state on both sides rather than a patched ``getpgid``: the
    child either is placed in a group of its own or is not.
    """
    answer = subprocess.run(
        [sys.executable, "-c", _GROUP_PROBE, str(_SRC)],
        capture_output=True,
        text=True,
        check=True,
        **({"process_group": 0} if opened else {}),
    )
    swept, group = answer.stdout.split()

    if opened:
        assert group != str(os.getpgrp()), "the child was not given its own group"
        assert swept == group, "a group opened for the worker was declined"
    else:
        assert group == str(os.getpgrp()), "the child did not inherit our group"
        assert swept == "None", "an inherited group was claimed as the worker's"
