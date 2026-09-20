"""The neutral error boundary and the JAX-free shared records."""

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import Report, refuse
from rheplicant.core.errors import DirtError


def test_the_neutral_classes_are_the_public_classes():
    from _rheplicant_bootstrap.errors import ConfigError as NeutralConfigError
    from _rheplicant_bootstrap.errors import DirtError as NeutralDirtError

    assert ConfigError is NeutralConfigError
    assert DirtError is NeutralDirtError
    assert ConfigError.__module__ == "rheplicant.config.errors"
    assert DirtError.__module__ == "rheplicant.core.errors"
    assert issubclass(ConfigError, DirtError)
    assert issubclass(ConfigError, ValueError)


def test_report_is_additive_without_changing_exception_args():
    marker = object()
    error = ConfigError("first", "second", report=marker)
    assert error.args == ("first", "second")
    assert str(error) == "('first', 'second')"
    assert error.report is marker


def test_report_attaches_the_cumulative_report_when_supplied():
    first = refuse("A1", "one", "one is refused")
    current = Report(findings=(first,))
    cumulative = Report(findings=(first, refuse("A2", "two", "two is refused")))

    with pytest.raises(ConfigError) as caught:
        current.raise_if_refused(cumulative=cumulative)

    assert caught.value.report is cumulative
    assert str(caught.value) == first.message


def test_zero_argument_refusal_attaches_the_current_report():
    first = refuse("A1", "one", "one is refused")
    report = Report(findings=(first,))

    with pytest.raises(ConfigError) as caught:
        report.raise_if_refused()

    assert caught.value.report is report


def test_report_without_refusals_returns_without_raising():
    assert Report().raise_if_refused() is None


def test_shared_records_are_frozen_and_slotted():
    from _rheplicant_bootstrap.types import DestinationDescriptor, LayerIdentity, SourceInput

    assert not hasattr(LayerIdentity("base", None), "__dict__")
    assert not hasattr(DestinationDescriptor("model", "model_field", "noise"), "__dict__")
    assert dataclasses.fields(SourceInput)


def test_origin_render_is_stable_and_names_are_validated():
    from _rheplicant_bootstrap.types import Origin

    assert Origin("user").render() == "user"
    assert Origin("rheplicant-default").render() == "rheplicant-default"
    assert Origin("preset", "base preset").render() == "preset:n-6261736520707265736574"
    assert Origin("variant", "é").render() == "variant:n-c3a9"
    with pytest.raises(ValueError):
        Origin("user", "named")
    with pytest.raises(ValueError):
        Origin("preset", "")


class _HostileOriginText(str):
    def __str__(self):
        raise AssertionError("__str__ must not run")

    def __bool__(self):
        raise AssertionError("truth testing must not run")

    def __eq__(self, other):
        raise AssertionError("equality must not run")

    def __hash__(self):
        raise AssertionError("hash must not run")

    def encode(self, *args, **kwargs):
        raise AssertionError("encode must not run")

    def __repr__(self):
        raise AssertionError("repr must not run")


def test_origin_canonicalizes_text_before_validation_and_rendering():
    from _rheplicant_bootstrap.types import Origin

    origin = Origin(_HostileOriginText("preset"), _HostileOriginText("base preset"))

    assert type(origin.kind) is str
    assert type(origin.name) is str
    assert origin.render() == "preset:n-6261736520707265736574"


def test_invalid_hostile_origin_kind_raises_value_error_without_hooks():
    from _rheplicant_bootstrap.types import Origin

    with pytest.raises(ValueError, match="unknown origin kind"):
        Origin(_HostileOriginText("invalid"))


def test_destination_child_and_nested_preserve_the_parent_contract():
    from _rheplicant_bootstrap.types import DestinationDescriptor

    parent = DestinationDescriptor("model", "model_field", "noise")
    assert parent.child("sigma") == DestinationDescriptor(
        "model.sigma", "model_field", "noise.sigma"
    )
    assert parent.child(2, domain="resource_field", selector="[]") == DestinationDescriptor(
        "model[2]", "resource_field", "noise[]"
    )
    assert parent.nested("value") == DestinationDescriptor("model.value", "model_field", "noise")
    assert parent == DestinationDescriptor("model", "model_field", "noise")
    with pytest.raises(ValueError):
        DestinationDescriptor("", "model_field", "noise")
    with pytest.raises(ValueError):
        DestinationDescriptor("model", "model_field", "")


def test_the_assembly_refusal_is_a_neutral_class_too():
    """The command line classifies ``AssemblyError`` as a refusal, and the
    bootstrap may not import ``rheplicant`` to name it; so the class lives
    beside ``ConfigError`` and ``rheplicant.core.errors`` re-exports it."""
    from _rheplicant_bootstrap.errors import REFUSALS
    from _rheplicant_bootstrap.errors import AssemblyError as NeutralAssemblyError
    from rheplicant.core.errors import AmbiguousNodeError, AssemblyError

    assert AssemblyError is NeutralAssemblyError
    assert AssemblyError.__module__ == "rheplicant.core.errors"
    assert issubclass(AssemblyError, DirtError)
    assert issubclass(AssemblyError, ValueError)
    assert issubclass(AmbiguousNodeError, AssemblyError)
    assert REFUSALS == (ConfigError, AssemblyError)


_PICKLE_PROBE = """
import json, pickle, sys
import _rheplicant_bootstrap.errors as neutral
assert neutral.__file__.startswith({src!r}), neutral.__file__
if sys.argv[1] == "dumps":
    blob = pickle.dumps(getattr(neutral, sys.argv[2])("the message"))
    sys.stdout.buffer.write(blob)
else:
    loaded = pickle.loads(sys.stdin.buffer.read())
    print(json.dumps({{
        "module": type(loaded).__module__,
        "identical": type(loaded) is getattr(neutral, type(loaded).__name__),
        "args": list(loaded.args),
        "jax_imported": "jax" in sys.modules,
    }}))
"""


def _pickle_round_trip(name: str) -> dict:
    """Pickle ``name`` in one fresh process and unpickle it in another."""
    src = str(Path(__file__).parents[2] / "src")
    program = _PICKLE_PROBE.format(src=src)
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([src, *filter(None, [os.environ.get("PYTHONPATH")])]),
    }
    dumped = subprocess.run(
        [sys.executable, "-c", program, "dumps", name], capture_output=True, check=True, env=env
    )
    loaded = subprocess.run(
        [sys.executable, "-c", program, "loads"],
        input=dumped.stdout,
        capture_output=True,
        check=True,
        env=env,
    )
    return json.loads(loaded.stdout)


def test_the_moved_assembly_error_pickles_as_dirt_error_does():
    """Moving the class into the bootstrap must not change how it travels.

    Its ``__module__`` is ``rheplicant.core.errors``, so pickle records that
    path and unpickling imports ``rheplicant`` -- and with it JAX -- exactly
    as it does for ``DirtError``, whose ``__module__`` is rewritten the same
    way.  What must hold is that the class found there IS the bootstrap's
    object (a second class of the same name would fail ``pickle.dumps``
    outright) and that the message survives.
    """
    assembly = _pickle_round_trip("AssemblyError")
    dirt = _pickle_round_trip("DirtError")
    assert assembly == {
        "module": "rheplicant.core.errors",
        "identical": True,
        "args": ["the message"],
        "jax_imported": dirt["jax_imported"],
    }
    assert dirt["module"] == "rheplicant.core.errors"
    assert dirt["identical"] is True
