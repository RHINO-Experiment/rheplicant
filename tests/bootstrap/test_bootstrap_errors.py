"""The neutral error boundary and the JAX-free shared records."""

import dataclasses

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
    assert not hasattr(
        DestinationDescriptor("model", "model_field", "noise"), "__dict__"
    )
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


def test_destination_child_and_nested_preserve_the_parent_contract():
    from _rheplicant_bootstrap.types import DestinationDescriptor

    parent = DestinationDescriptor("model", "model_field", "noise")
    assert parent.child("sigma") == DestinationDescriptor(
        "model.sigma", "model_field", "noise.sigma"
    )
    assert parent.child(2, domain="resource_field", selector="[]") == DestinationDescriptor(
        "model[2]", "resource_field", "noise[]"
    )
    assert parent.nested("value") == DestinationDescriptor(
        "model.value", "model_field", "noise"
    )
    assert parent == DestinationDescriptor("model", "model_field", "noise")
    with pytest.raises(ValueError):
        DestinationDescriptor("", "model_field", "noise")
    with pytest.raises(ValueError):
        DestinationDescriptor("model", "model_field", "")
