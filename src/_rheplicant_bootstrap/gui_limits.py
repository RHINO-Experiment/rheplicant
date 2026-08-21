"""The bounded-diagnostic contract shared by the GUI worker and its parent.

A GUI job crosses a process boundary twice: the fresh-interpreter worker in
``_rheplicant_bootstrap.gui_worker`` frames a result, and the long-lived parent
in ``rheplicant.gui.jobs`` stores it and serves it over HTTP.  Neither side may
retain an amount of text or structure chosen by the job rather than by this
module -- an exception message, a child's stdout and a findings list are all
inputs, and an input is not a budget.

Every number below is a *limit*, not a suggestion, and every truncation leaves
``TRUNCATION_MARKER`` in the text it shortened, so a reader can always tell a
short answer from a shortened one.  Whole entries removed from a mapping or a
list are reported under ``TRUNCATION_KEY`` instead, which no payload can write
into or spell.  The limits live here rather than in either caller because the
worker imports only from ``_rheplicant_bootstrap`` (it must never import
``rheplicant.gui``), and because two copies of a limit are two limits.

Bounding late is not bounding: ``MAX_STREAM_BYTES`` and
``MAX_CHILD_STREAM_BYTES`` exist so the parent consumes a child's stream as it
arrives instead of storing the whole stream -- in memory or on disk -- and
slicing it afterwards.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

TRUNCATION_MARKER = "…[truncated]"
"""Left in place of whatever a limit removed from one string."""

_KEY_ESCAPE = "\x00"
"""Reserved first character of a mapping key; payload keys never keep one."""

TRUNCATION_KEY = f"{_KEY_ESCAPE}gui.truncated"
"""Names, out of band, what a limit removed from one mapping or list.

The marker inside a string says that string was shortened.  A mapping loses
whole entries instead, and a count written under a key the payload can also
author is not a channel but a collision: a result whose own key is the marker
has its value silently replaced by the count, and a resource innocently named
``probe …[truncated]`` announces a truncation that never happened.

This key is unforgeable because it is reserved by escaping rather than by
hope.  Every payload key that already begins with ``_KEY_ESCAPE`` gains one
more before it is stored, so a stored payload key that begins with the escape
always carries at least two of them, while this constant carries exactly one.
Its presence therefore means this module put it there, and its absence means
nothing was dropped -- whatever the payload contains.
"""

MAX_TEXT_CHARACTERS = 4_000
"""Longest diagnostic string kept: a message, a stream excerpt, a map key.

Bounded text is *exactly* this long at most, marker included, so bounding an
already-bounded string is a no-op rather than a second marker.
"""

MAX_STREAM_BYTES = 64 * 1024
"""Most bytes of a child's stderr the parent ever holds in memory.

The parent drains the child's streams as they arrive and keeps only this much
of the tail, so its peak memory is this constant rather than whatever the
child decided to print.
"""

MAX_CHILD_STREAM_BYTES = 64 * 1024 * 1024
"""Most bytes one child stream may total before the job is refused outright.

Keeping a bounded tail bounds the parent, not the child: redirecting the
stream to a temporary file trades a memory bound for a disk bound sized
entirely by the job, and a runaway print then fills the volume instead of the
heap.  Everything past this many bytes is discarded as it arrives and the job
ends as a bounded error, because a stream this large is a defect rather than
a diagnostic.
"""

MAX_WORKER_SECONDS = 60 * 60
"""Longest one worker subprocess may run before the parent ends it.

A child that never exits holds its job in ``running`` forever, and a job that
never reaches a terminal status refuses every identical resubmission for the
life of the process.  Generous enough for a formal run launched from the GUI,
finite because nothing else is.
"""

TRUNCATED_FINDING_CHECK = "gui.diagnostics.truncated"
"""Names, out of band, how many findings a limit removed from one result.

Reserved by escaping, exactly as ``TRUNCATION_KEY`` is, and for the same
reason: a findings list is the one place a payload authors both the shape and
the text of an entry, so a marker anybody may spell is a collision rather than
a channel.  Every stored finding whose own check ends in this name gains a
``_KEY_ESCAPE`` before it, so a stored check that IS this name was written by
:func:`bounded_findings` and by nothing else.

The escape hangs off ``endswith`` rather than ``startswith`` because this name
does not begin with the escape: escaping only the checks that already carry
one would leave the bare name itself forgeable.
"""

MAX_FINDING_COUNT = 100
"""Most validation findings carried in one result, marker finding included."""

MAX_COLLECTION_LENGTH = 256
"""Most entries kept from one list or mapping inside a result.

Deliberately far above 64: a forward preview publishes a 64x64 waterfall grid
(64 rows of 64 samples) and that is science, not diagnostics.
"""

MAX_NESTING_DEPTH = 12
"""Deepest nesting kept inside a result; deeper values become the marker."""

MAX_RESULT_BYTES = 1024 * 1024
"""Encoded-JSON budget for one whole job result.

Length, depth and string caps each bound one dimension; only a total budget
bounds their product, which is what a response body actually costs.
"""

MAX_FRAME_BYTES = 2 * 1024 * 1024
"""Largest worker result frame the parent will parse.

A backstop above ``MAX_RESULT_BYTES``: the worker has already bounded the
result, and JSON escaping is the only thing that can grow it further.
"""

MAX_FRAME_TAIL_BYTES = MAX_FRAME_BYTES + 8192
"""Most bytes of one candidate result frame the parent buffers while scanning.

Strictly larger than ``MAX_FRAME_BYTES`` so a legal frame is always wholly
inside the window, and constant so the buffer never follows the child: the
parent recognises a frame as it streams past rather than searching a window
of stdout afterwards, where output written after the frame -- by ``atexit``
handlers, or by a native library shutting down -- would push it out of reach.
"""

_SCALAR_BYTES = 25
"""Charged for one number, bool or None -- the widest float repr is 24 bytes."""

_SEPARATOR_BYTES = 2
"""Charged for the comma and colon around one collection entry."""


def bounded_text(value: object, *, limit: int = MAX_TEXT_CHARACTERS) -> str:
    """Keep the head of one diagnostic string, marking anything removed.

    The result is never longer than ``limit``.  A limit too small to hold the
    marker keeps as much of the marker as fits and none of the text: saying
    "something was removed" is the last thing worth the room.
    """
    text = value if isinstance(value, str) else str(value)
    if len(text) <= limit:
        return text
    if limit <= len(TRUNCATION_MARKER):
        return TRUNCATION_MARKER[:limit] if limit > 0 else ""
    return text[: limit - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER


def bounded_stream_text(value: object, *, limit: int = MAX_TEXT_CHARACTERS) -> str:
    """Keep the tail of one captured stream: the end is where failures land.

    Bounded to ``limit`` exactly as ``bounded_text`` is, from the other end.
    """
    text = value if isinstance(value, str) else str(value)
    if len(text) <= limit:
        return text
    if limit <= len(TRUNCATION_MARKER):
        return TRUNCATION_MARKER[:limit] if limit > 0 else ""
    return TRUNCATION_MARKER + text[len(TRUNCATION_MARKER) - limit :]


def bounded_stream_bytes(value: bytes, *, limit: int = MAX_STREAM_BYTES) -> str:
    """Decode at most ``limit`` trailing bytes of a stream that may be binary."""
    return bounded_stream_text(value[-limit:].decode("utf-8", "replace"))


def bounded_findings(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Cap how many findings a result carries and how long each one is."""
    listed = list(rows)
    if len(listed) <= MAX_FINDING_COUNT:
        return [_bounded_finding(row) for row in listed]
    kept = [_bounded_finding(row) for row in listed[: MAX_FINDING_COUNT - 1]]
    omitted = len(listed) - len(kept)
    kept.append(
        {
            "check": TRUNCATED_FINDING_CHECK,
            "severity": "report",
            "where": "document",
            "message": f"{omitted} further findings were dropped {TRUNCATION_MARKER}",
            "layer": "gui",
        }
    )
    return kept


class _BoundedResult(dict[str, object]):
    """One result this module has already bounded, marked so it is not twice.

    Bounding is NOT idempotent, and no key-rewriting rule can make it so.  The
    reserved key is reserved by escaping every payload key clear of it, which
    means ``_stored_key`` must rewrite ``TRUNCATION_KEY`` when a payload spells
    it -- and must leave it alone when this module wrote it.  Those are the
    same input, so one function cannot do both, and a second pass therefore
    either escapes a real notice out of the channel or counts the notice as one
    more entry and rewrites its number.

    Provenance is the only thing that can tell the two apart, so it is what is
    carried.  This is a ``dict``: it encodes, compares and serialises as the
    mapping it is, and the mark exists only between :func:`bounded_worker_result`
    and :func:`bounded_result` inside one process.  It does not survive the
    frame, which is correct -- what crosses the frame is bounded on both sides
    of it exactly once.
    """

    __slots__ = ()


def bounded_result(value: object) -> object:
    """Return a plain, JSON-ready copy of one job result inside every limit."""
    if isinstance(value, _BoundedResult):
        return value
    bounded, _ = _bounded_value(value, depth=0, budget=MAX_RESULT_BYTES)
    return bounded


def bounded_worker_result(value: object) -> object:
    """Take one result the worker already bounded, without bounding it twice.

    ``gui_worker`` calls :func:`bounded_result` before it frames its result and
    the parent's job store calls it again on what it receives, so every result
    that crosses the worker boundary is bounded twice and the second pass reads
    the first pass's notices as payload.  See :class:`_BoundedResult` for why
    the passes cannot simply be made to compose.

    Provenance is not taken on trust: the one dimension a frame can still
    exceed is the encoded size, because ``MAX_FRAME_BYTES`` deliberately sits
    above ``MAX_RESULT_BYTES``.  Anything larger than the parent's own budget
    was not produced by this module's bounding, so it is bounded for real --
    notices and all, because at that point there are no notices worth keeping.
    """
    if not isinstance(value, Mapping):
        return bounded_result(value)
    try:
        encoded = json.dumps(value, ensure_ascii=False).encode("utf-8", "strict")
    except (TypeError, ValueError, UnicodeError):
        return bounded_result(value)
    if len(encoded) > MAX_RESULT_BYTES:
        return bounded_result(value)
    return _BoundedResult(value)


def bounded_frame(frame: Mapping[str, object]) -> bytes:
    """Encode one worker result frame, degrading rather than failing to write.

    Both degradations are terminal frames rather than exceptions: a worker
    that raises here writes nothing at all, and a parent that receives nothing
    reports a finished job as a missing result.
    """
    try:
        encoded = json.dumps(frame, sort_keys=True).encode("utf-8", "strict")
    except (TypeError, ValueError) as error:
        return _degraded_frame(
            "GuiFrameUnencodable",
            f"The worker result cannot be encoded as JSON: {bounded_text(error)}",
        )
    if len(encoded) <= MAX_FRAME_BYTES:
        return encoded
    return _degraded_frame(
        "GuiFrameTooLarge",
        f"The worker result encoded to {len(encoded)} bytes, beyond the "
        f"{MAX_FRAME_BYTES}-byte frame limit {TRUNCATION_MARKER}",
    )


def _degraded_frame(exception_type: str, message: str) -> bytes:
    frame = {
        "status": "error",
        "exception_type": exception_type,
        "message": bounded_text(message),
    }
    return json.dumps(frame, sort_keys=True).encode("utf-8", "strict")


def _bounded_finding(row: Mapping[str, object]) -> dict[str, object]:
    """Store one finding with both reserved channels escaped clear of it.

    A finding is a mapping inside a result, so its keys reach the same
    reserved key the enclosing result's do -- and its ``check`` reaches the
    marker :func:`bounded_findings` writes.  Neither escape was applied here,
    which left the findings marker forgeable by any payload row willing to
    spell it.
    """
    return {
        bounded_text(_stored_key(key if isinstance(key, str) else str(key))): (
            _bounded_finding_value(key, item)
        )
        for key, item in row.items()
    }


def _bounded_finding_value(key: object, item: object) -> object:
    if not isinstance(item, str):
        return item
    return bounded_text(_stored_check(item) if key == "check" else item)


def _text_cost(text: str) -> int:
    return len(json.dumps(text))


def _bounded_value(
    value: object, *, depth: int, budget: int
) -> tuple[object, int]:
    """Bound one node, returning it with the budget its subtree left over."""
    if budget <= 0:
        return TRUNCATION_MARKER, 0
    if depth >= MAX_NESTING_DEPTH:
        return TRUNCATION_MARKER, max(0, budget - _text_cost(TRUNCATION_MARKER))
    if isinstance(value, str):
        text = bounded_text(value)
        return text, max(0, budget - _text_cost(text))
    if isinstance(value, Mapping):
        return _bounded_mapping(value, depth=depth, budget=budget)
    if isinstance(value, tuple | list):
        return _bounded_sequence(value, depth=depth, budget=budget)
    return _bounded_scalar(value, budget=budget)


def _bounded_scalar(value: object, *, budget: int) -> tuple[object, int]:
    """Keep one scalar only if JSON can carry it; degrade to its text if not.

    "Bounded" has to include "encodable": a value this returns unchanged is a
    value ``bounded_frame`` will later encode, and a raw ``bytes`` or a numpy
    scalar reaching that point kills the worker with no frame at all rather
    than costing it a few characters here.
    """
    if value is None or isinstance(value, bool | int | float):
        return value, max(0, budget - _SCALAR_BYTES)
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        text = bounded_text(value)
        return text, max(0, budget - _text_cost(text))
    return value, max(0, budget - _SCALAR_BYTES)


def _bounded_mapping(
    value: Mapping[object, object], *, depth: int, budget: int
) -> tuple[dict[str, object], int]:
    items = list(value.items())
    kept = items[:MAX_COLLECTION_LENGTH]
    bounded: dict[str, object] = {}
    left = budget
    for index, (key, item) in enumerate(kept):
        if left <= 0:
            return _marked_mapping(bounded, len(items) - index), 0
        name = bounded_text(_stored_key(key if isinstance(key, str) else str(key)))
        left = max(0, left - _text_cost(name) - _SEPARATOR_BYTES)
        bounded[name], left = _bounded_value(item, depth=depth + 1, budget=left)
    if len(items) > len(kept):
        return _marked_mapping(bounded, len(items) - len(kept)), left
    return bounded, left


def _bounded_sequence(
    value: Sequence[object], *, depth: int, budget: int
) -> tuple[list[object], int]:
    listed = list(value)
    kept = listed[:MAX_COLLECTION_LENGTH]
    bounded: list[object] = []
    left = budget
    for index, item in enumerate(kept):
        if left <= 0:
            bounded.append(_dropped(len(listed) - index))
            return bounded, 0
        left = max(0, left - _SEPARATOR_BYTES)
        item_value, left = _bounded_value(item, depth=depth + 1, budget=left)
        bounded.append(item_value)
    if len(listed) > len(kept):
        bounded.append(_dropped(len(listed) - len(kept)))
    return bounded, left


def _stored_key(name: str) -> str:
    """Escape one payload key clear of the reserved truncation channel.

    Injective, which is the property that matters and not merely a pleasant
    one: two payload keys stored under one name lose an entry silently, and a
    rule that escaped only the keys EQUAL to ``TRUNCATION_KEY`` would map that
    key onto the very name an already-escaped payload key is stored under.
    """
    return f"{_KEY_ESCAPE}{name}" if name.startswith(_KEY_ESCAPE) else name


def _stored_check(name: str) -> str:
    """Escape one payload finding's check clear of the reserved marker.

    Injective for the same reason and by the same argument: a name that ends
    in the reserved check always gains exactly one escape, and one that does
    not is stored unchanged, so no escaped name can equal an unescaped one.
    """
    return (
        f"{_KEY_ESCAPE}{name}"
        if name.endswith(TRUNCATED_FINDING_CHECK)
        else name
    )


def _marked_mapping(bounded: dict[str, object], omitted: int) -> dict[str, object]:
    """Report what a mapping lost without overwriting what it kept."""
    return {**bounded, TRUNCATION_KEY: _notice(omitted)}


def _dropped(omitted: int) -> dict[str, object]:
    """A list's notice, in the one shape a payload entry cannot forge."""
    return {TRUNCATION_KEY: _notice(omitted)}


def _notice(omitted: int) -> str:
    return f"{omitted} further entries were dropped {TRUNCATION_MARKER}"


__all__ = [
    "MAX_CHILD_STREAM_BYTES",
    "MAX_COLLECTION_LENGTH",
    "MAX_FINDING_COUNT",
    "MAX_FRAME_BYTES",
    "MAX_FRAME_TAIL_BYTES",
    "MAX_NESTING_DEPTH",
    "MAX_RESULT_BYTES",
    "MAX_STREAM_BYTES",
    "MAX_TEXT_CHARACTERS",
    "MAX_WORKER_SECONDS",
    "TRUNCATED_FINDING_CHECK",
    "TRUNCATION_KEY",
    "TRUNCATION_MARKER",
    "bounded_findings",
    "bounded_frame",
    "bounded_result",
    "bounded_stream_bytes",
    "bounded_stream_text",
    "bounded_text",
    "bounded_worker_result",
]
