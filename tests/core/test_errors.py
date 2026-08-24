"""Tests for the error family, and for the two surfaces that re-export it."""

import inspect

import pytest

import rheplicant
import rheplicant.core
from rheplicant import DataIngestionError, DirtError
from rheplicant.core import errors as errors_module


def test_data_ingestion_error_is_catchable_as_the_family_and_as_value_error():
    assert issubclass(DataIngestionError, DirtError)
    assert issubclass(DataIngestionError, ValueError)
    with pytest.raises(DirtError):
        raise DataIngestionError("bad file")


def _declared_error_names() -> set[str]:
    """Every exception class ``rheplicant.core.errors`` defines itself."""
    return {
        name
        for name, obj in vars(errors_module).items()
        if inspect.isclass(obj)
        and issubclass(obj, Exception)
        and obj.__module__ == errors_module.__name__
    }


#: Which public surface owns each error, and therefore must export it.
#:
#: Not "every surface exports everything". ``rheplicant.core`` is the
#: domain-agnostic layer and its vocabulary is deliberately its own -- see
#: ``tests/core/test_basis.py::TestWhichErrorTheRefusalsRaise``, which pins
#: ``ParameterSpaceError`` OUT of ``rheplicant.core.__all__`` precisely so that
#: "a core module must not raise it" is a rule with teeth rather than a
#: preference. So the inference layer's error is exported by the inference
#: layer.
#:
#: (The first version of this test asserted the flat rule and pushed
#: ``ParameterSpaceError`` into core, which that basis test caught. The
#: layering argument was the better one; this table is what it looks like
#: written down.)
OWNERS = {
    "ParameterSpaceError": "rheplicant.inference",
    # Its subclass, and the same argument: a reader who wants the numbers a
    # refused ``check_linearity`` measured needs to name the class, and the
    # layer they already import ``check_linearity`` from is the one that
    # should hand it to them.
    "LinearityRefused": "rheplicant.inference",
}
DEFAULT_OWNER = "rheplicant.core"


def _surface(dotted: str):
    import importlib

    return importlib.import_module(dotted)


@pytest.mark.parametrize("name", sorted(_declared_error_names()))
def test_each_error_is_exported_by_the_layer_that_owns_it(name):
    """Every declared error has exactly one documented public home.

    Deriving the set from the module rather than listing it is what makes this
    worth having: a new error class has to be assigned an owner or this fails,
    so it cannot end up importable only from ``rheplicant.core.errors`` -- a
    path that reads as internal -- the way ``ParameterSpaceError`` did.
    """
    owner = _surface(OWNERS.get(name, DEFAULT_OWNER))
    assert name in owner.__all__, (
        f"{name} is not exported by {owner.__name__}. Export it there, or add "
        f"it to OWNERS with the layer that should own it."
    )
    # Not merely a name that resolves: a shadowing look-alike would satisfy the
    # line above and break every `except` clause written against it, silently,
    # because the raise site and the catch site would compare different objects.
    assert getattr(owner, name) is getattr(errors_module, name)


def test_core_keeps_its_own_error_vocabulary():
    """The negative half, stated here so both halves live together.

    Without this, someone reading only the table above would reasonably "fix"
    the asymmetry by exporting everything everywhere, and the layering claim
    would evaporate without a single test turning red.
    """
    for name, owner in OWNERS.items():
        if owner != DEFAULT_OWNER:
            assert name not in rheplicant.core.__all__, name
            assert name not in rheplicant.__all__, name
