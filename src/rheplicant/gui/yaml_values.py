"""One comparison for "is this YAML value the same value?".

Two answers to that question used to ship side by side —
``gui/document.py::_same_value`` and ``gui/validation.py::_same`` — and they
disagreed. This module is the one they were both spelling.

**What they disagreed about, measured.** Not key ordering: both walk mappings
by lookup and both are order-independent, so the symptom the review reported
is not the one that exists. The real divergence is **numeric key identity**,
because ``key in right`` is a dict lookup and Python collapses ``1``,
``True`` and ``1.0`` into one key there::

    {1: "x"} vs {True: "x"}     strict False   lookup True
    {1: "x"} vs {1.0: "x"}      strict False   lookup True
    {"o": {0: "x"}} vs {"o": {False: "x"}}
                                strict False   lookup True

Values were never affected — both compare a value's ``type`` before ``==`` —
so the disagreement lives entirely in how a mapping's KEYS are matched.

**Why the strict answer is the right one.** These comparisons decide whether
an edit changed the document and whether the document differs from its
preset. In YAML ``1:``, ``true:`` and ``1.0:`` are three different scalars
that round-trip differently and mean different things to the loader, so a
document whose key changed from one to another HAS changed. The lax answer
reports no change, which is a real difference rendered as none.

**A known limit, recorded rather than quietly fixed.**
``validation.py::_walk_diff`` matches a mapping's keys with its own
``key not in preset`` / ``.get(key, ...)`` before it ever reaches this
function, so at that one site the numeric-key collapse still applies.
Repairing it means changing how the preset diff pairs keys, which changes
the shape of what the GUI renders rather than only a boolean — a separate
piece of work from unifying the comparator, and one that wants its own
before/after. This function is not where that bug lives, and importing it
there does not fix it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def same_value(left: object, right: object) -> bool:
    """Compare YAML values without collapsing bool/int/float distinctions.

    Mapping keys are matched **by value through this same function**, not by
    dict lookup, which is what keeps ``1`` and ``True`` apart. A left key that
    matches anything other than exactly one right key makes the mappings
    different: zero matches is a key the other side lacks, and two would mean
    the right mapping held two keys this function calls equal, which a dict
    cannot do — so the check costs nothing and forecloses the case rather than
    assuming it away.
    """
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if len(left) != len(right):
            return False
        for left_key, left_value in left.items():
            matches = [
                right_value
                for right_key, right_value in right.items()
                if same_value(left_key, right_key)
            ]
            if len(matches) != 1 or not same_value(left_value, matches[0]):
                return False
        return True
    if (
        not isinstance(left, str | bytes)
        and not isinstance(right, str | bytes)
        and isinstance(left, Sequence)
        and isinstance(right, Sequence)
    ):
        return len(left) == len(right) and all(
            same_value(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if type(left) is not type(right):
        return False
    return left == right


def plain(value: object) -> object:
    """Frozen or tuple-bearing YAML back as plain mutable ``dict``/``list``.

    The bounded loader returns a FROZEN value, and every consumer that means
    to edit, diff or render one needs it thawed first. This was written out
    twice, byte for byte, in ``document.py`` and ``validation.py``; it is one
    idea and now has one home, next to the comparison it is usually paired
    with.

    Sequences come back as ``list`` whatever they went in as, which is not a
    loss: YAML has one sequence type and the tuple is Python's. That is the
    same reading :func:`same_value` takes, so a value thawed here and compared
    there cannot disagree with itself.
    """
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


__all__ = ["plain", "same_value"]
