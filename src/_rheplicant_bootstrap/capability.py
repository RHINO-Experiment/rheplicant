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

A level belongs to a capability ON A SURFACE, not to a capability alone. That
is measured rather than asserted: ``NeuralOperator`` is refused from a config
document with a typed ``ConfigError`` naming its capability, which is the
``UNAVAILABLE`` definition exactly, and from Python it is a working operator
that ``docs/operators.md`` documents. One class, two answers. So the
``maturity`` ClassVar on a class records how far the IMPLEMENTATION has been
taken, and ``UNAVAILABLE`` is a surface's answer, recorded in
:data:`REGISTRY` beside the surface it belongs to. On that reading no class is
ever ``UNAVAILABLE``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

__all__ = ["Capability", "Maturity", "REGISTRY", "Surface"]


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


class Surface(enum.Enum):
    """Where a capability is asked for.

    The same capability can answer differently on two of these, which is why
    a row in :data:`REGISTRY` carries one. ``DOCUMENT`` is a YAML
    configuration document; ``PYTHON`` is the importable API.
    """

    DOCUMENT = "document"
    PYTHON = "python"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Capability:
    """One capability that is NOT a class, and therefore has no ClassVar.

    Classes carry their level themselves and are walked, not listed. This
    table is for the rest: keys a schema names and refuses, and importable
    surfaces whose level is not a property of any single class.

    Attributes:
        name: how the surface names it -- a document key path, or a dotted
            import path.
        surface: which surface this row is about.
        maturity: the level on that surface.
        what: the capability, in the words the refusal already uses, so the
            table and the message cannot drift into two spellings.
        reference: the schema section or the ruling that decided it.
        inherits: the upstream module this level is taken FROM, when it is
            inherited rather than judged here. Checked: a row that names one
            must actually import it.
    """

    name: str
    surface: Surface
    maturity: Maturity
    what: str
    reference: str = ""
    inherits: str = ""


#: Capabilities with no class of their own.
#:
#: The eight ``DOCUMENT`` rows are schema §8's reserved keys, and they are the
#: source `rheplicant.config.preflight.document._CAPABILITY_KEYS` is now
#: derived from rather than a second copy of. Their ``what`` and ``reference``
#: are the exact strings the A39 refusal interpolates, so the table and the
#: message cannot say different things.
#:
#: The two ``PYTHON`` rows were labelled nowhere before. `npe` is Experimental
#: by INHERITANCE and the source is checked: it imports `bayesmith.amortize`,
#: which bayesmith's own `docs/stability.md` declares Experimental ("amortized
#: reference implementation"), and a guarantee cannot be stronger than what it
#: rests on. `compress_reduced_basis` is Experimental on its OWN merits, which
#: corrects the Stage 1 note (A7-5) that filed it as inheriting
#: `bayesmith.amortize`: measured 2026-09-20, `inference/compress.py` imports
#: nothing from bayesmith at all.
REGISTRY: tuple[Capability, ...] = (
    Capability(
        "campaign",
        Surface.DOCUMENT,
        Maturity.UNAVAILABLE,
        "capability 4 (streaming evidence)",
        "§8.2",
    ),
    Capability(
        "inference.transitions",
        Surface.DOCUMENT,
        Maturity.UNAVAILABLE,
        "capability 4 (streaming evidence)",
        "§8.2",
    ),
    Capability(
        "inference.parameters.<name>.scope",
        Surface.DOCUMENT,
        Maturity.UNAVAILABLE,
        "capability 4 (streaming evidence)",
        "§8.2",
    ),
    Capability(
        "inference.parameters.<name>.support",
        Surface.DOCUMENT,
        Maturity.UNAVAILABLE,
        "capability 4 (streaming evidence)",
        "§8.2",
    ),
    Capability(
        "inference.parameters.<name>.hyper",
        Surface.DOCUMENT,
        Maturity.UNAVAILABLE,
        "capability 4 (streaming evidence)",
        "§8.2",
    ),
    Capability(
        "model.<node>.type: NeuralOperator",
        Surface.DOCUMENT,
        Maturity.UNAVAILABLE,
        "capability 3 (neural surrogates)",
        "§8.1",
    ),
    Capability(
        "outputs.write.memory_archive",
        Surface.DOCUMENT,
        Maturity.UNAVAILABLE,
        "capability 4 (streaming evidence)",
        "§8.2",
    ),
    Capability(
        "outputs.write.posterior_net",
        Surface.DOCUMENT,
        Maturity.UNAVAILABLE,
        "capability 3 (neural surrogates)",
        "§8.1",
    ),
    Capability(
        "rheplicant.inference.npe",
        Surface.PYTHON,
        Maturity.EXPERIMENTAL,
        "neural posterior estimation: training and sampling an amortized posterior",
        "ledger A4-2, A7-5; bayesmith docs/stability.md",
        inherits="bayesmith.amortize",
    ),
    Capability(
        "rheplicant.inference.compress.compress_reduced_basis",
        Surface.PYTHON,
        Maturity.EXPERIMENTAL,
        "section 4.1's T1: compressing an epoch against a shared reduced "
        "basis, an approximation with no general validity guarantee",
        "ledger A7-5",
    ),
)
