"""Every shipped operator declares a capability level, and the prose agrees.

Before this file, "placeholder" was a word in fifteen class docstrings and
nothing read it. A reader could not get the list without grepping, the GUI
could not show it, and an operator whose physics arrived stayed labelled a
placeholder for as long as nobody reread the first line of its docstring.

The direction here is the one ``docs/`` and the plan both ask for: the
``maturity`` ClassVar is the registry, and the DOCSTRING is checked against it.
Reversing that would make the prose authoritative and leave the machine-readable
level free to drift, which is the arrangement that produced the problem.

Three things are asserted and they fail for different reasons, which is the
point of keeping them apart:

* every shipped operator has a level at all -- catches an operator added
  without one;
* the docstring's ``(placeholder)`` and the level agree, in BOTH directions --
  catches physics that arrived without the label being dropped, and a label
  dropped while the body is still a stand-in;
* the base class still has no default -- catches the change that would make
  the first test unable to fail.

That third one is not decoration. A default of ``MAINTAINED`` on
``AbstractOperator`` would make the first test pass for every operator anyone
ever adds, including one whose physics is a stand-in, and nothing else here
would notice.
"""

from __future__ import annotations

import inspect

import pytest

import rheplicant.radio as radio
from _rheplicant_bootstrap.capability import REGISTRY, Capability, Surface
from rheplicant.config.sections.model import operator_table
from rheplicant.core.capability import Maturity
from rheplicant.core.operator import AbstractOperator
from rheplicant.radio.sky.model import AbstractSkyModel
from rheplicant.radio.sky.projection import AbstractSkyProjector

#: What makes a public class a CAPABILITY rather than a record. Three bases,
#: and the answer is derived from the type relationship rather than from a list
#: of names -- a list would need editing every time a class is added, which is
#: the failure this whole registry exists to end.
CAPABILITY_BASES = (AbstractOperator, AbstractSkyModel, AbstractSkyProjector)


def _shipped() -> dict[str, type]:
    """Every concrete capability on the package's public radio surface.

    The first version of this walked ``operator_table()`` -- the classes a
    CONFIG DOCUMENT can address -- and that was too narrow by two. Measured
    2026-09-20: ``NeuralOperator`` is a public operator that no document may
    name (the config layer refuses it by design), and ``AbstractLinearFilter``
    is a public base. Neither was reached, so neither was required to declare
    a level, and ``NeuralOperator`` had none.

    ``rheplicant.radio.__all__`` is the real public surface, so that is what is
    walked. Two kinds of class are left out and both are derived, not listed:

    * abstract ones -- the three bases and anything with unimplemented
      abstract methods. A base must NOT carry a level; that is asserted
      separately, in the other direction.
    * classes that are not capabilities at all. ``Touchstone`` and
      ``RhinoObservation`` are parsed-data records; the capability is the
      reader that returns one, and readers are functions. They fall out of
      the ``issubclass`` test with no name written down.
    """
    found: dict[str, type] = {}
    for name in radio.__all__:
        obj = getattr(radio, name, None)
        if not inspect.isclass(obj) or not issubclass(obj, CAPABILITY_BASES):
            continue
        if obj in CAPABILITY_BASES or inspect.isabstract(obj):
            continue
        found[name] = obj
    return found


SHIPPED = sorted(_shipped().items())


@pytest.mark.parametrize(("name", "cls"), SHIPPED, ids=[n for n, _ in SHIPPED])
def test_every_shipped_operator_declares_a_level(name, cls):
    level = getattr(cls, "maturity", None)
    assert isinstance(level, Maturity), (
        f"{name} has no `maturity` ClassVar, so nothing can say how far its "
        "implementation has been taken. Declare one beside `requires` and "
        "`provides`; there is deliberately no default to inherit"
    )


@pytest.mark.parametrize(("name", "cls"), SHIPPED, ids=[n for n, _ in SHIPPED])
def test_the_docstring_and_the_level_agree(name, cls):
    """``(placeholder)`` in the summary line iff the level is PLACEHOLDER.

    Only the summary line is read. A docstring that discusses placeholders
    further down -- several explain what real physics would replace -- is
    prose about the future, not a claim about this class.
    """
    doc = (cls.__doc__ or "").strip().splitlines()
    summary = doc[0] if doc else ""
    says_placeholder = "placeholder" in summary.lower()
    is_placeholder = cls.maturity is Maturity.PLACEHOLDER

    if says_placeholder and not is_placeholder:
        pytest.fail(
            f"{name}'s summary line says placeholder but its level is "
            f"{cls.maturity}. If the physics arrived, drop the word from the "
            "docstring in the same change that raised the level"
        )
    if is_placeholder and not says_placeholder:
        pytest.fail(
            f"{name} is declared PLACEHOLDER but its summary line does not say "
            f"so: {summary!r}. A reader of the docstring would believe the "
            "numbers"
        )


def test_every_class_a_document_can_name_is_in_the_walk():
    """The narrower set must be inside the wider one.

    ``operator_table()`` is what a config document may address. If a class
    were registered there without being on ``radio.__all__``, this file would
    never see it and the level it failed to declare would be invisible -- the
    exact hole that hid ``NeuralOperator`` before the walk was widened, in the
    other direction.
    """
    addressable = {
        cls.__name__ for classes in operator_table().values() for cls in classes
    }
    walked = {name for name, _ in SHIPPED}
    missed = sorted(addressable - walked)
    assert not missed, (
        f"{missed} can be named in a config document but is not on "
        "rheplicant.radio.__all__, so the capability walk never reaches it"
    )


@pytest.mark.parametrize("base", CAPABILITY_BASES, ids=[b.__name__ for b in CAPABILITY_BASES])
def test_an_abstract_base_carries_no_level(base):
    """The bases annotate ``maturity``; they must not answer it.

    Asserted per base rather than once, so a default added to any one of the
    three names itself in the failure. Without this, a value on
    ``AbstractSkyModel`` would silently label both placeholder skies as
    whatever it said.
    """
    assert "maturity" not in vars(base), (
        f"{base.__name__} carries a DEFAULT maturity, so every subclass "
        "inherits a claim about its physics instead of declaring one"
    )


def test_the_base_class_has_no_default_level():
    """The annotation is there; a value is not.

    Without this, someone adding ``= Maturity.MAINTAINED`` to
    ``AbstractOperator`` would silently turn
    ``test_every_shipped_operator_declares_a_level`` into a test that cannot
    fail, and the failure mode is the worst available: unfinished physics
    labelled maintained.
    """
    assert "maturity" in AbstractOperator.__annotations__, (
        "AbstractOperator no longer annotates `maturity`, so subclasses have "
        "nothing to declare against"
    )
    assert "maturity" not in vars(AbstractOperator), (
        "AbstractOperator now carries a DEFAULT maturity. Remove it: a class "
        "that forgets to declare its level must raise, not inherit a claim "
        "that its physics is real"
    )


def test_an_operator_that_forgets_its_level_raises():
    """The mechanism the first test relies on, exercised directly.

    Asserting "there is no default" and asserting "a missing one is caught"
    are different claims, and only this one shows the AttributeError actually
    arrives.
    """

    class Forgetful(AbstractOperator):
        def __call__(self, state):  # pragma: no cover - never called
            return state

    with pytest.raises(AttributeError):
        _ = Forgetful.maturity


def test_the_level_is_not_a_pytree_leaf():
    """A ClassVar is excluded from the dataclass fields.

    ``PointSourceOperator`` already has a real, differentiable ``level``
    field, which is why this one is called ``maturity``. If the level ever
    became a field it would be traced, flattened and handed to `grad`.
    """
    import dataclasses

    for name, cls in SHIPPED:
        fields = {f.name for f in dataclasses.fields(cls)}
        assert "maturity" not in fields, (
            f"{name}.maturity is a dataclass field, so it is a pytree leaf "
            "and JAX will try to trace an enum"
        )


def test_the_levels_are_not_ordered():
    """EXPERIMENTAL and PLACEHOLDER are different kinds, not two depths.

    An ordering would invite ``>= EXPERIMENTAL`` filters, which would read as
    "at least this trustworthy" while meaning nothing: a placeholder's
    contract is dependable and its numbers are not, and an experimental
    surface is the other way round.
    """
    with pytest.raises(TypeError):
        _ = Maturity.PLACEHOLDER < Maturity.MAINTAINED


def test_a_level_spells_itself_the_way_a_document_does():
    """``str(level)`` is the document spelling, not ``Maturity.PLACEHOLDER``.

    Generated documentation, the GUI and any YAML or JSON that carries a level
    interpolate it directly. Without ``__str__`` they would each need their own
    mapping from enum member to word, which is four copies of one rule and the
    thing this registry exists to avoid.
    """
    assert str(Maturity.PLACEHOLDER) == "placeholder"
    assert f"{Maturity.EXPERIMENTAL}" == "experimental"
    assert {str(level) for level in Maturity} == {
        "maintained",
        "experimental",
        "placeholder",
        "unavailable",
    }


def test_the_vocabulary_is_jax_free():
    """The GUI must be able to read a level without loading JAX.

    Checked as a subprocess because this session has JAX imported already, so
    an in-process check would pass whatever the module did.
    """
    import subprocess
    import sys

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from _rheplicant_bootstrap.capability import Maturity; "
            "assert Maturity.PLACEHOLDER.value == 'placeholder'; "
            "assert 'jax' not in sys.modules, sorted(m for m in sys.modules if 'jax' in m); "
            "assert 'rheplicant' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr


def test_the_enum_is_documented_from_the_core_module():
    """`repr` and the API reference must not name the bootstrap package.

    The same arrangement `core.errors` uses for `AssemblyError`.
    """
    assert Maturity.__module__ == "rheplicant.core.capability"

    # NOT `inspect.getsourcefile`. It resolves through `__module__`, which this
    # arrangement deliberately rewrites, so it answers `core/capability.py` and
    # cannot see where the class was defined -- measured, it was the first
    # version of this assertion and it failed for that reason rather than
    # because anything was wrong. Identity is the property anyway: one enum
    # object, defined in the JAX-free module and re-exported, not two.
    from _rheplicant_bootstrap.capability import Maturity as Defined

    assert Maturity is Defined, (
        "rheplicant.core.capability no longer re-exports the bootstrap enum "
        "but defines its own. Two enums with equal members are not equal to "
        "each other, so the GUI's level and the operator's would stop matching"
    )


# ---------------------------------------------------------------------------
# The table: capabilities with no class, and the views derived from it.
# ---------------------------------------------------------------------------


def test_no_class_is_ever_unavailable():
    """The reading this registry is built on, pinned.

    ``UNAVAILABLE`` describes a SURFACE's answer, not an implementation.
    ``NeuralOperator`` is the case that settled it: refused from a config
    document with a typed error naming its capability, and a working operator
    from Python. If a class ever carried ``UNAVAILABLE`` the two readings
    would both be in the tree and nothing would say which was meant.
    """
    wrong = [name for name, cls in SHIPPED if cls.maturity is Maturity.UNAVAILABLE]
    assert not wrong, (
        f"{wrong} declare Maturity.UNAVAILABLE as a ClassVar. A class records "
        "how far its IMPLEMENTATION has been taken; that a surface refuses it "
        "belongs in REGISTRY beside the surface, not on the class"
    )


def test_every_registry_row_is_unique_on_its_surface():
    """Two rows for one (name, surface) is two answers to one question."""
    seen: dict[tuple[str, Surface], Capability] = {}
    for row in REGISTRY:
        key = (row.name, row.surface)
        assert key not in seen, (
            f"{row.name!r} appears twice for surface {row.surface}; the second "
            f"says {row.maturity} and the first says {seen[key].maturity}"
        )
        seen[key] = row


@pytest.mark.parametrize(
    "row",
    [r for r in REGISTRY if r.surface is Surface.PYTHON],
    ids=[r.name for r in REGISTRY if r.surface is Surface.PYTHON],
)
def test_a_python_row_names_something_that_exists(row):
    """A dotted path that no longer resolves is a row protecting nothing.

    The same failure as a comparison whose subject has left: the row keeps
    claiming a level for a surface that is gone, and nothing notices because
    the assertion never looks.
    """
    import importlib

    module_path, _, attribute = row.name.rpartition(".")
    try:
        module = importlib.import_module(row.name)
    except ImportError:
        module = importlib.import_module(module_path)
        assert hasattr(module, attribute), (
            f"REGISTRY names {row.name!r}, and neither the module nor "
            f"{attribute!r} inside {module_path!r} exists"
        )


@pytest.mark.parametrize(
    "row",
    [r for r in REGISTRY if r.inherits],
    ids=[r.name for r in REGISTRY if r.inherits],
)
def test_an_inherited_level_really_rests_on_what_it_names(row):
    """``inherits=`` is checked, not taken on trust.

    A level inherited from upstream is the one kind of claim this package
    cannot verify by running anything: bayesmith declares its own levels in
    its ``docs/stability.md`` and we read them. What CAN be checked here is
    the other half -- that the module claiming to inherit actually imports
    the module it names. Without this, a row could keep claiming Experimental
    long after the dependency it was inherited from had been dropped, and the
    capability would stay labelled for a reason that no longer applied.
    """
    import importlib
    import pathlib

    module = importlib.import_module(row.name)
    source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
    assert f"from {row.inherits} import" in source or f"import {row.inherits}" in source, (
        f"{row.name} is recorded as inheriting its level from {row.inherits}, "
        f"but its source does not import {row.inherits}. Either the dependency "
        "was dropped and the level should be judged on its own merits, or the "
        "row names the wrong module"
    )


def test_the_document_view_is_the_registry_and_not_a_second_table():
    """``_CAPABILITY_KEYS`` must carry exactly the registry's document rows.

    Not a tautology, although it is derived today: this is what turns a future
    edit that re-spells the dict as a literal into a red test rather than a
    silent second copy. The shape is asserted too, because
    ``rheplicant-agent``'s server unpacks ``(capability, section)``.
    """
    from rheplicant.config.preflight.document import _CAPABILITY_KEYS

    expected = {
        row.name: (row.what, row.reference)
        for row in REGISTRY
        if row.surface is Surface.DOCUMENT and row.maturity is Maturity.UNAVAILABLE
    }
    assert _CAPABILITY_KEYS == expected, (
        "_CAPABILITY_KEYS and REGISTRY's document rows disagree. If the dict "
        "was re-spelled as a literal, delete it and keep the comprehension: "
        "the rows live in REGISTRY so that the table, the A39 message and "
        f"rheplicant-agent cannot drift apart. View: {_CAPABILITY_KEYS!r}"
    )
    assert len(_CAPABILITY_KEYS) == 8, (
        f"schema §8 reserves eight keys; the view has {len(_CAPABILITY_KEYS)}"
    )
    for key, value in _CAPABILITY_KEYS.items():
        assert isinstance(value, tuple) and len(value) == 2, (
            f"{key} carries {value!r}; the consumer unpacks two strings"
        )


def test_the_refusal_message_quotes_the_registry():
    """Registry -> message, end to end.

    The table and the sentence a user reads are the pair most likely to drift,
    because nothing renders them side by side. This walks the actual refusal
    builder and requires the registry's own words in what comes out.
    """
    from rheplicant.config.preflight.document import _task3_capability

    row = next(
        r for r in REGISTRY
        if r.surface is Surface.DOCUMENT and r.name == "campaign"
    )
    finding = _task3_capability("campaign", "campaign")
    assert row.what in finding.message, (
        f"the A39 message does not contain {row.what!r}: the table and the "
        f"sentence have drifted. Message was: {finding.message!r}"
    )
    assert row.reference in finding.message
