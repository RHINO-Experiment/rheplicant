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

import pytest

from rheplicant.config.sections.model import operator_table
from rheplicant.core.capability import Maturity
from rheplicant.core.operator import AbstractOperator


def _shipped() -> dict[str, type]:
    """Every operator class addressable from a document.

    Derived from ``operator_table()``, which is itself derived from
    ``rheplicant.radio.__all__``, so a class added to the package surface
    arrives here with no list to update.
    """
    found: dict[str, type] = {}
    for classes in operator_table().values():
        for cls in classes:
            found[cls.__name__] = cls
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
