"""The four capability levels, as a JAX-free vocabulary.

This module holds the enum and nothing that uses it. It lives in the bootstrap
package for one measured reason: importing ``rheplicant`` or
``rheplicant.config`` loads JAX (checked 2026-09-20), and the GUI server has to
be able to say that a capability is a placeholder without paying for that. The
bootstrap package is the only part of this checkout that
``tests/config/test_entry_order.py`` holds JAX-free, so a level that the GUI can
read is a level defined here.

What each level means:

``MAINTAINED``
    Documented contract, regression-tested within its declared domain.

``EXPERIMENTAL``
    Usable, with stated limits and no general validity guarantee. Anything
    built on an upstream ``EXPERIMENTAL`` surface inherits this, because a
    guarantee cannot be stronger than what it rests on.

``PLACEHOLDER``
    Correct plumbing and shapes, stand-in physics. The contract -- shapes,
    purity, PRNG consumption, coordinate updates -- is real and tested; the
    arithmetic is a stand-in that real physics will replace.

``UNAVAILABLE``
    Named by the schema or API and refused with a typed error. The name exists
    so a document mentioning it gets an answer rather than a ``KeyError``.

The levels are not ordered and this enum deliberately does not make them
comparable. ``EXPERIMENTAL`` and ``PLACEHOLDER`` are different KINDS of
incompleteness, not two depths of one: a placeholder's contract is trustworthy
and its numbers are not, while an experimental surface's numbers may be right
and its contract may still move. Ranking them would invite ``>= EXPERIMENTAL``
tests that mean nothing.

Where a capability is a class, the level is a ``ClassVar`` on that class rather
than a row in a table here, and the registry is derived by walking the classes.
A table beside the classes is the copy that goes stale the first time someone
adds a class without touching it, which is the failure this package keeps
paying for elsewhere.
"""

from __future__ import annotations

import enum

__all__ = ["Maturity"]


class Maturity(enum.Enum):
    """How far a capability's implementation has been taken.

    The value is the lower-case spelling used in documents, generated
    documentation and the GUI, so a level round-trips through YAML and JSON
    without a second mapping to keep in step.
    """

    MAINTAINED = "maintained"
    EXPERIMENTAL = "experimental"
    PLACEHOLDER = "placeholder"
    UNAVAILABLE = "unavailable"

    def __str__(self) -> str:
        return self.value
