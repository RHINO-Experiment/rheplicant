"""Running one bounded child process, and reclaiming everything it touched.

A GUI job is a child process the parent must survive.  Three things have to
hold for that, and all three are about ownership rather than about limits:

* Its **streams** are consumed as they arrive, by a thread that owns both the
  descriptor it reads and the sink it writes.  Nothing else reads that
  descriptor, nothing else mutates that sink, and the thread is the only thing
  that ever closes either.  ``gui_limits`` decides how much survives; this
  decides who is allowed to touch it while it is arriving.

* Its **descriptors and threads** are reclaimed on every path.  Memory and
  disk are bounded by ``gui_limits``; descriptors and threads are bounded only
  by whoever hands them back, and a long-lived server that leaks one per job
  dies at ``accept()`` with ``EMFILE`` rather than at any limit named there.

* Its **process group** is the parent's to sweep.  ``subprocess.run``'s timeout
  ends the direct child alone, so a worker that spawned anything leaves the
  descendants running -- holding its outputs, its CPU, and the stream the
  parent is still waiting on.

The process-group half is shared verbatim with ``tests/config/wheel_support``,
which sweeps the packaged GUI server the same way.  Two copies of this
reasoning would be two behaviours the first time one of them was corrected.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from threading import Event, Thread

from _rheplicant_bootstrap.gui_limits import (
    MAX_CHILD_STREAM_BYTES,
    MAX_FRAME_BYTES,
    MAX_FRAME_TAIL_BYTES,
)

FRAME_PREFIX = b"\x1eRHEPLICANT_GUI_JOB "

READ_CHUNK = 64 * 1024
"""Bytes taken from a child pipe per read: the parent's cost, not the child's."""

DRAIN_SECONDS = 30.0
"""Longest the parent waits for a drained stream to end after the child has."""

GROUP_GRACE_SECONDS = 5.0
"""How long a TERMed process group may take to drain before the rest is killed.

Bounded for the same reason every other wait here is: a descendant that will
not stop must not be able to stop the parent instead.
"""

_WAIT_SECONDS = 0.25
"""How long a reader blocks per look before it checks whether it may stop.

A compromise with only one bad direction: too long delays a stop that is
already an error path, too short spins a thread that lives as long as the job.
"""

_STOP_SECONDS = 2.0
"""Longest a reader asked to stop is given to notice; see :meth:`finish`."""

_POSIX = os.name == "posix"

if _POSIX:  # pragma: no branch - the non-POSIX floor is stated, not exercised
    import select


class ChildStreamAbandoned(RuntimeError):
    """One child stream that did not end when the child that owned it did.

    Something other than the worker still holds the write end, which means the
    bytes in that stream are not the worker's to be read as.  The job ends
    here, explicitly, rather than reporting whatever a descendant happened to
    have written as the worker's own answer.
    """


# --- the sinks: what survives one child stream ---------------------------


class ChildStream:
    """One child stream, consumed as it arrives and never held whole.

    The child chooses how much it writes; this chooses how much survives, and
    ``MAX_CHILD_STREAM_BYTES`` chooses when writing that much is itself the
    failure.  Past that point nothing more is stored, by the parent or by the
    filesystem, and the job ends as a bounded error.

    Exactly one thread -- the :class:`_StreamReader` that owns this -- calls
    :meth:`feed` and :meth:`close`, and it does so in that order and never
    concurrently.  The state below is therefore unsynchronised on purpose;
    what enforces it is the handover in :meth:`_StreamReader.finish`, not a
    lock, because a lock would let the parent read a half-fed sink politely
    instead of not at all.
    """

    def __init__(self) -> None:
        self.total = 0
        self.read_error: BaseException | None = None

    @property
    def overflowed(self) -> bool:
        """Whether the child wrote more than the parent agreed to accept.

        Strictly more: ``MAX_CHILD_STREAM_BYTES`` is the largest total the
        parent accepts, so a stream of exactly that many bytes is inside the
        agreement and one byte further is outside it.
        """
        return self.total > MAX_CHILD_STREAM_BYTES

    def feed(self, chunk: bytes) -> None:
        self.total += len(chunk)
        if not self.overflowed:
            self._consume(chunk)

    def close(self) -> None:
        """Finish a stream that ended wherever the child left it."""

    def _consume(self, chunk: bytes) -> None:
        raise NotImplementedError


class StreamTail(ChildStream):
    """Keep one stream's tail: the end is where a failure lands."""

    def __init__(self, *, limit: int) -> None:
        super().__init__()
        self._limit = limit
        self._kept = bytearray()

    def tail(self) -> bytes:
        return bytes(self._kept[-self._limit :])

    def _consume(self, chunk: bytes) -> None:
        self._kept += chunk
        if len(self._kept) > 2 * self._limit:
            del self._kept[: len(self._kept) - self._limit]


class FrameScan(ChildStream):
    """Keep the last complete result frame out of an unbounded stdout stream.

    The worker writes its frame as its last statement, but ``atexit`` handlers
    and native libraries write to the same descriptor after ``main`` returns.
    Searching a fixed tail afterwards therefore loses a legitimate result to
    whatever followed it -- four megabytes of shutdown chatter reports a
    finished job as a job with no result.  Recognising the frame as it streams
    past costs the same whether the noise comes before it or after it, and
    only the frame itself is ever buffered.
    """

    def __init__(self) -> None:
        super().__init__()
        self._carry = b""
        self._capturing = False
        self._buffer = bytearray()
        self._length = 0
        self.payload: bytes | None = None
        self.oversized = 0

    def close(self) -> None:
        carry, self._carry = self._carry, b""
        self._scan(carry, len(carry))
        if self._capturing:
            self._commit()

    def _consume(self, chunk: bytes) -> None:
        data = self._carry + chunk
        # A prefix split across two reads is still a prefix: hold back the
        # bytes one could have started in and read them again next time.
        edge = max(0, len(data) - (len(FRAME_PREFIX) - 1))
        self._carry = data[self._scan(data, edge) :]

    def _scan(self, data: bytes, edge: int) -> int:
        """Consume ``data`` up to ``edge``; return the first byte left over."""
        at = 0
        while True:
            if not self._capturing:
                start = data.find(FRAME_PREFIX, at)
                if start < 0:
                    return max(at, edge)
                self._start()
                at = start + len(FRAME_PREFIX)
                continue
            stop = data.find(b"\n", at)
            start = data.find(FRAME_PREFIX, at)
            if start >= 0 and (stop < 0 or start < stop):
                # A second prefix before the line ends: the last frame wins,
                # exactly as searching backwards through the stream did.
                self._keep(data[at:start])
                self._start()
                at = start + len(FRAME_PREFIX)
                continue
            if stop >= 0:
                self._keep(data[at:stop])
                self._commit()
                at = stop + 1
                continue
            left = max(at, edge)
            self._keep(data[at:left])
            return left

    def _start(self) -> None:
        self._capturing = True
        self._buffer = bytearray()
        self._length = 0

    def _keep(self, piece: bytes) -> None:
        self._length += len(piece)
        room = MAX_FRAME_TAIL_BYTES - len(self._buffer)
        if room > 0:
            self._buffer += piece[:room]

    def _commit(self) -> None:
        self._capturing = False
        if self._length > MAX_FRAME_BYTES:
            self.payload, self.oversized = None, self._length
        else:
            self.payload, self.oversized = bytes(self._buffer), 0
        self._buffer = bytearray()


# --- the reader: one thread that owns a descriptor and a sink ------------


class _StreamReader:
    """One child pipe read to exhaustion by a thread that always ends.

    Two things are owned here, and owning them is the whole point.

    The **descriptor** is read and closed by this thread alone.  The parent
    must never close a descriptor a live reader still holds: the number would
    be handed straight to the next file this process opens, and the reader
    would go on writing that file's contents into a job result.  Nor may the
    parent simply abandon it, which is the leak this replaces -- one
    descriptor and one thread per job, forever, until ``accept()`` fails.

    The **sink** is fed and closed by this thread alone.  The parent used to
    close a frame scan while this thread was still inside ``feed``: in one
    ordering the scan committed a frame a DESCENDANT had written and the job
    reported it as the worker's science, and in the other a bytearray was
    mutated under itself and the ``IndexError`` escaped a thread whose only
    handler was for ``OSError``, leaving a half-fed sink that read as an
    answer.  :meth:`finish` is the only handover, and it answers whether there
    is anything to hand over at all.
    """

    def __init__(self, handle: int, sink: ChildStream) -> None:
        self._handle = handle
        self._sink = sink
        self._stop = Event()
        self._thread = Thread(target=self._read, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def finish(self, timeout: float) -> bool:
        """Hand the sink back, and answer whether the stream ended by itself.

        ``True`` means the write end was closed by everything that held it, so
        what the sink holds is what the child wrote and nothing else is coming.
        ``False`` means something outlived the child and still holds the
        stream: the reader is told to stop so its thread and its descriptor
        come back, and the sink is NOT the caller's to read.
        """
        self._thread.join(timeout)
        if not self._thread.is_alive():
            return True
        self._stop.set()
        self._thread.join(_STOP_SECONDS)
        return False

    def _read(self) -> None:
        try:
            while not self._stop.is_set():
                if not self._readable():
                    continue
                chunk = os.read(self._handle, READ_CHUNK)
                if not chunk:
                    return
                self._sink.feed(chunk)
        except BaseException as error:  # noqa: BLE001 -- recorded, not raised
            # Deliberately wider than ``OSError``.  A sink mutates bytearrays,
            # so anything that disturbs one raises ``IndexError`` or
            # ``BufferError`` rather than ``OSError``; those used to escape
            # this thread entirely, and a sink left half-fed by an exception
            # nobody saw is read by the parent as the child's whole answer.
            self._sink.read_error = error
        finally:
            self._release()

    def _readable(self) -> bool:
        """Wait briefly for something to read, or for the stop to be worth
        checking again.

        Without ``select`` there is nothing to wait ON but the read itself, so
        the loop degrades to the blocking read it replaced: the stop is then
        only honoured once something arrives.  POSIX is where the GUI server,
        its process groups and this whole file are supported.
        """
        if not _POSIX:  # pragma: no cover - stated floor, not a tested path
            return True
        return bool(select.select((self._handle,), (), (), _WAIT_SECONDS)[0])

    def _release(self) -> None:
        try:
            self._sink.close()
        except BaseException as error:  # noqa: BLE001 -- see ``_read``
            self._sink.read_error = error
        finally:
            with suppress(OSError):
                os.close(self._handle)


# --- the process group: named by the parent, swept by the parent ---------


def process_group(process: subprocess.Popen[bytes] | subprocess.Popen[str]) -> int | None:
    """The process group ``process`` leads, or ``None`` if it leads none.

    A process started with ``start_new_session=True`` or ``process_group=0``
    is a process-group leader, so its group id EQUALS its pid.  Verifying that
    identity is the safety property, not a formality: the group is the unit we
    signal, and signalling the wrong group means signalling our own, which
    would take the caller down with the child it was cleaning up.

    ``None`` means "no group we own; signal the child alone, as before", and is
    the answer whenever the identity cannot be established:

    * ``process.pid`` is not exactly an ``int``.  A ``Popen`` always has one; a
      test double need not.  This is a TYPE test rather than ``except
      TypeError`` because a ``MagicMock`` satisfies ``__index__`` -- it answers
      ``1``, per ``unittest.mock._return_values`` -- so ``os.getpgid(mock.pid)``
      does not raise, it quietly reports the group of PID 1, and the kill that
      followed would be aimed at init.  Nothing is ever raised, so nothing
      could ever be caught.
    * ``os.getpgid`` raises ``ProcessLookupError``: the child has already been
      reaped and its group went with it.
    * the group is not the child's own pid -- the new group did not take
      effect, or the child is a MEMBER of a group somebody else leads.  Either
      way the group was not created here and its other members are not ours.
    * the group is our own.  This arm is deliberately unreachable while the one
      above stands, and is kept anyway: it is what makes deleting that arm a
      survivable mistake rather than a suicide.
    * ``os.getpgid``/``os.killpg`` do not exist.  POSIX only.
    """
    pid = getattr(process, "pid", None)
    if type(pid) is not int or pid <= 0:
        return None
    try:
        group = os.getpgid(pid)
    except (OSError, AttributeError):
        return None
    if group != pid or group == os.getpgrp():
        return None
    return group


def signal_group(group: int | None, number: int) -> None:
    """Signal every member of ``group``; a group with nothing left to signal is
    not an error.

    Two errnos mean exactly that, and both are swallowed:

    * ``ESRCH`` -- the group drained between two syscalls.
    * ``EPERM`` -- MEASURED, not anticipated.  On darwin a group whose last
      member is a zombie mid-reap answers ``killpg`` with ``EPERM`` rather than
      ``ESRCH``.  It is safe to swallow here BECAUSE :func:`process_group` has
      already certified that this group is led by our own child: ``EPERM``
      cannot mean "somebody else's group", only "what is left cannot be killed
      anyway".

    Any other ``OSError`` is a real fault and propagates, so the caller decides
    whether it should surface.
    """
    if group is None:
        return
    try:
        os.killpg(group, number)
    except (ProcessLookupError, PermissionError):
        return


def reap_group(group: int | None, *, grace_seconds: float) -> None:
    """Wait a bounded time for a TERMed group to drain, then kill the rest.

    The probe asks one question -- "is anything still in there that we could
    and should kill?" -- so both errnos that answer "no" end the wait. See
    :func:`signal_group` for why ``EPERM`` is one of them.
    """
    if group is None:
        return
    deadline = time.monotonic() + grace_seconds
    while True:
        try:
            os.killpg(group, 0)  # a membership probe; signal 0 sends nothing
        except (ProcessLookupError, PermissionError):
            return               # nothing left in it that we can kill
        if time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    signal_group(group, signal.SIGKILL)


def stop_child(process: subprocess.Popen[bytes] | subprocess.Popen[str]) -> None:
    """The direct child alone: TERM, a bounded wait, then KILL."""
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        if process.poll() is not None:
            return
    try:
        process.kill()
    except ProcessLookupError:
        return
    process.wait(timeout=5)


def stop_process_group(
    process: subprocess.Popen[bytes] | subprocess.Popen[str],
    *,
    stop: Callable[[subprocess.Popen[bytes] | subprocess.Popen[str]], None] = stop_child,
    grace_seconds: float = GROUP_GRACE_SECONDS,
) -> None:
    """Stop ``process`` AND everything left in the group it leads.

    The group is read BEFORE the first ``poll()`` because ``poll()`` reaps, and
    a reaped pid has no group left to ask about.  A process that died on its
    own but has not been waited on yet is a zombie, which still carries its
    group, so the realistic case -- the leader died while a descendant ran --
    is covered.  The case this does NOT cover is written down rather than
    papered over: if something else reaped the process before this ran, the
    group is unknowable and descendants are not swept.

    The sweep is in a ``finally`` rather than on the success path.  A stop that
    blew up would otherwise strand every descendant, which is the same leak
    wearing a different hat.
    """
    group = process_group(process)
    signal_group(group, signal.SIGTERM)
    try:
        stop(process)
    finally:
        reap_group(group, grace_seconds=grace_seconds)


# --- the anchor: the only unforgeable name a new group can have ----------

_ANCHOR_ARGV = (sys.executable, "-I", "-S", "-c", "import sys; sys.stdin.read()")


def _group_anchor() -> subprocess.Popen[bytes] | None:
    """Open one process group the parent can name, or ``None`` if it cannot.

    A group id is only trustworthy if it came from a ``Popen`` the parent
    created: with a new group the id EQUALS the leader's pid, and the pid is
    the one thing no descendant can rewrite.  ``subprocess.run`` never yields a
    pid, and every other way of learning one -- a line on stderr, a file named
    by an environment variable, a byte on an inherited descriptor -- is a
    channel a descendant inherits too.  ``_audit_path``'s lesson is exactly
    this one: a value parsed out of a stream that then becomes a directory is
    bad enough, and a value parsed out of a stream that then becomes a group
    we SIGKILL is worse.

    So the parent opens the group itself, with a process whose pid it was
    handed.  The anchor is born with ``process_group=0``, which makes it a
    group leader in the parent's OWN session -- a new *session* would be
    unjoinable, since ``setpgid`` may only join a group inside the caller's
    session, and joining by name is the entire purpose.  It then does nothing
    but hold the group open: it dies on TERM like anything else, and it dies on
    EOF as well, so a parent that is killed without sweeping does not leave it
    behind.

    ``None`` is the honest answer wherever this cannot be done -- no
    ``process_group`` support, no POSIX, a spawn that failed -- and it costs
    only the sweep.  The reader threads still hand their descriptors back.
    """
    if not _POSIX:  # pragma: no cover - stated floor, not a tested path
        return None
    try:
        anchor = subprocess.Popen(
            list(_ANCHOR_ARGV),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            process_group=0,
        )
    except (OSError, ValueError, TypeError):  # pragma: no cover - platform floor
        return None
    if process_group(anchor) is None:  # pragma: no cover - kernel refused
        _discard(anchor)
        return None
    return anchor


def _discard(anchor: subprocess.Popen[bytes]) -> None:
    """Close an anchor down without pretending it ever led a group."""
    with suppress(OSError):
        stop_child(anchor)
    if anchor.stdin is not None:
        with suppress(OSError):
            anchor.stdin.close()


def _sweep(anchor: subprocess.Popen[bytes] | None, *, grace_seconds: float) -> None:
    """Take back the whole group the job ran in, on every path out of it."""
    if anchor is None:
        return
    try:
        stop_process_group(anchor, grace_seconds=grace_seconds)
    finally:
        if anchor.stdin is not None:
            with suppress(OSError):
                anchor.stdin.close()


# --- the run itself ------------------------------------------------------


def drained_run(
    argv: Sequence[str],
    *,
    input_bytes: bytes,
    stdout: ChildStream,
    stderr: ChildStream,
    timeout: float,
    drain_seconds: float = DRAIN_SECONDS,
    grace_seconds: float = GROUP_GRACE_SECONDS,
) -> subprocess.CompletedProcess[bytes]:
    """Run one child, consuming both of its streams while it writes them.

    Pipes rather than temporary files: a file the child sizes is not a bound,
    and reading a bounded tail from it afterwards bounds only what the parent
    keeps, while the file grows to whatever a runaway ``print`` produced.
    Draining as the child writes means nothing it prints is stored anywhere
    beyond each sink's own limit, and the timeout means a child that never
    exits cannot hold its job -- and every identical resubmission -- forever.

    The order of the cleanup is the argument:

    1. the parent's copies of the write ends are closed, because a reader
       reaches the end of its stream only once every copy is gone;
    2. each reader is given until ``drain_seconds`` to reach that end.  Ending
       is what makes the stream the CHILD's: if it does not end, something the
       child left behind still holds it, and those bytes are not the child's to
       be read as.  That case raises rather than answering;
    3. the group is swept whatever happened, so the descendant that held the
       stream open does not go on holding the machine.

    The sweep comes after the wait rather than before it on purpose.  Killing
    the holder first would deliver the EOF, and the job would then report a
    frame a descendant wrote as though the worker had written it.
    """
    anchor = _group_anchor()
    # Two nested ``finally`` blocks rather than one, because the two reclaims
    # have to happen in this order and on different scopes: the readers are
    # waited for as soon as the child is gone, and the group is taken back
    # afterwards -- on EVERY path out of here, including one where opening the
    # pipes or starting a reader is itself what failed and there is no child.
    try:
        group = None if anchor is None else process_group(anchor)
        out_read, out_write = os.pipe()
        err_read, err_write = os.pipe()
        readers = (
            _StreamReader(out_read, stdout),
            _StreamReader(err_read, stderr),
        )
        for reader in readers:
            reader.start()
        try:
            completed = subprocess.run(
                list(argv),
                input=input_bytes,
                stdout=out_write,
                stderr=err_write,
                timeout=timeout,
                check=False,
                **({} if group is None else {"process_group": group}),
            )
        finally:
            os.close(out_write)
            os.close(err_write)
            deadline = time.monotonic() + drain_seconds
            ended = tuple(
                reader.finish(max(0.0, deadline - time.monotonic()))
                for reader in readers
            )
    finally:
        _sweep(anchor, grace_seconds=grace_seconds)
    abandoned = tuple(
        name
        for name, over in zip(("stdout", "stderr"), ended, strict=True)
        if not over
    )
    if abandoned:
        raise ChildStreamAbandoned(
            f"GUI scientific worker {' and '.join(abandoned)} did not end when "
            "the worker did: something it left behind still holds the stream, "
            "so what was written there cannot be read as the worker's result"
        )
    return completed


__all__ = [
    "DRAIN_SECONDS",
    "FRAME_PREFIX",
    "GROUP_GRACE_SECONDS",
    "READ_CHUNK",
    "ChildStream",
    "ChildStreamAbandoned",
    "FrameScan",
    "StreamTail",
    "drained_run",
    "process_group",
    "reap_group",
    "signal_group",
    "stop_child",
    "stop_process_group",
]
