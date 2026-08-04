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


@pytest.mark.parametrize("surface", [rheplicant, rheplicant.core], ids=["rheplicant", "core"])
def test_every_error_is_re_exported(surface):
    """The whole family reaches the top, or the odd one out gets noticed.

    ``ParameterSpaceError`` was the one class here that neither surface
    exported, and it is the error the entire inference layer raises -- the
    stochastic-twin refusal, the linearity refusal, the partition check, the
    identifiability verdict. A user writing ``except ParameterSpaceError`` had
    to import from ``rheplicant.core.errors`` while ``except PipelineError``
    worked from the top, which reads as "this one is internal" and is exactly
    backwards.

    Deriving the expected set from the module rather than listing it is what
    makes this worth having: a new error class is re-exported or this fails,
    so the gap cannot reopen by omission the way it opened the first time.
    """
    missing = _declared_error_names() - set(surface.__all__)
    assert not missing, f"{surface.__name__} does not export {sorted(missing)}"


@pytest.mark.parametrize("surface", [rheplicant, rheplicant.core], ids=["rheplicant", "core"])
def test_the_re_export_is_the_same_class(surface):
    """Not merely a name that resolves.

    A re-export that shadowed the class with a look-alike would satisfy the
    test above and break every ``except`` clause written against it, silently,
    because the raise site and the catch site would be comparing different
    objects.
    """
    for name in _declared_error_names():
        assert getattr(surface, name) is getattr(errors_module, name), name
