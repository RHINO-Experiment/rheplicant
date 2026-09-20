"""The GUI's ``disabled``/``reason`` against the capability registry.

Section 2.4's last item. The catalog disabled eleven ``campaign.*`` widgets
and its ``campaign`` section with the sentence "Reserved for capability 4
(streaming evidence)." spelled twice -- and ``REGISTRY``'s own row for that
key already carried ``what="capability 4 (streaming evidence)"``. Three
spellings of one decision, and the registry is the one that decides it.

The property asserted here is the one that makes the catalog a VIEW rather
than a copy: for every document key the registry calls ``UNAVAILABLE``, the
catalog either offers no widget at all or disables every widget under it with
the registry's own words. Measured 2026-09-20: seven of the eight such keys
are simply absent from the form, which is the other correct answer -- a key
nobody can type needs no explanation -- and ``campaign`` is the one that is
offered and disabled.
"""

from __future__ import annotations

import pytest

from _rheplicant_bootstrap.capability import REGISTRY, Maturity, Surface
from rheplicant.gui.form_catalog import build_catalog
from rheplicant.gui.form_catalog_finalize import _reserved_reason

RESERVED = tuple(
    row
    for row in REGISTRY
    if row.surface is Surface.DOCUMENT and row.maturity is Maturity.UNAVAILABLE
)


@pytest.fixture(scope="module")
def catalog():
    return build_catalog()


def test_the_registry_has_reserved_document_keys_to_check():
    """Guard the guard: an empty ``RESERVED`` makes every test below vacuous."""
    assert RESERVED


@pytest.mark.parametrize("row", RESERVED, ids=[row.name for row in RESERVED])
def test_a_reserved_key_is_either_absent_or_disabled_with_the_registrys_reason(
    row, catalog
):
    under = [
        widget
        for widget in catalog.widgets
        if widget.path == row.name or widget.path.startswith(f"{row.name}.")
    ]
    if not under:
        pytest.skip(f"{row.name} is not offered by the form at all")
    enabled = [widget.path for widget in under if not widget.disabled]
    assert not enabled, (
        f"the registry calls {row.name!r} unavailable and the form offers "
        f"{enabled} as editable"
    )
    assert {widget.reason for widget in under} == {_reserved_reason(row.name)}


def test_at_least_one_reserved_key_is_actually_offered(catalog):
    """Otherwise every case above skips and the module proves nothing.

    If every reserved key stops being offered, this fails and someone decides
    whether the check still has a subject -- rather than a file of skips
    reading as a file of passes.
    """
    offered = [
        row.name
        for row in RESERVED
        if any(
            widget.path == row.name or widget.path.startswith(f"{row.name}.")
            for widget in catalog.widgets
        )
    ]
    assert offered == ["campaign"], offered


def test_the_reason_is_refused_when_the_registry_disagrees():
    """The direction that makes the derivation load-bearing.

    Asking for a reason for a key the registry does not call unavailable is a
    refusal rather than a default sentence, so a level RAISED upstream cannot
    leave the form quietly telling people the feature is reserved.
    """
    import dataclasses

    from _rheplicant_bootstrap.errors import ConfigError
    from rheplicant.gui import form_catalog_finalize

    with pytest.raises(ConfigError, match="no document row"):
        _reserved_reason("model")

    # The second branch needs a DOCUMENT row that is not unavailable, and the
    # registry has none today -- every document row it carries is reserved.
    # Monkeypatching is the only way to reach it, and reaching it is the
    # point: the day capability 4 lands, `campaign` becomes exactly this row
    # and the form must refuse to keep calling it reserved.
    row = next(row for row in REGISTRY if row.name == "campaign")
    raised = dataclasses.replace(row, maturity=Maturity.EXPERIMENTAL)
    original = form_catalog_finalize.REGISTRY
    form_catalog_finalize.REGISTRY = (raised,)
    try:
        with pytest.raises(ConfigError, match="rather than unavailable"):
            _reserved_reason("campaign")
    finally:
        form_catalog_finalize.REGISTRY = original


def test_no_disabled_widget_invents_a_capability_sentence(catalog):
    """A widget disabled for a reason the registry does not know.

    The three others -- `model.atmosphere_field`, `model.ground_field`,
    `model.beam` -- are disabled because the GRAPH reserves those nodes and no
    operator registers there, and they carry the node's own documentation as
    the reason. That is a different fact with a different source, and the test
    says so rather than letting the two blur.
    """
    from rheplicant.radio.graph import RADIO_GRAPH

    reserved_sentences = {_reserved_reason(row.name) for row in RESERVED}
    for widget in catalog.widgets:
        if not widget.disabled or widget.reason in reserved_sentences:
            continue
        node_id = widget.path.removeprefix("model.")
        assert node_id in RADIO_GRAPH.nodes, widget.path
        assert widget.reason == RADIO_GRAPH.nodes[node_id].doc
