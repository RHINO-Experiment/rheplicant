"""inference.twin: the fit twin is the model twin, repaired (schema §4.7).

``without:`` is the supported repair for the stochastic-stage refusal --
``refuse_stochastic_stages`` (``inference/parameters.py:125``) names
``Assembly.without`` itself.  ``replace:`` swaps one node's operator for a
declared one through ``Assembly.replace_node`` (``core/graph.py:473``), whose
named refusals -- aliased node, materialized junction, ``None`` -- speak for
themselves and are not restated here.  A replacement value is a §4.5.1 node
spec handed to the model section's own ``build_node_operator``, so a
replacement is spelled exactly like the node it replaces.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rheplicant.config.context import ResolutionContext
from rheplicant.config.errors import ConfigError
from rheplicant.config.resources import check_unknown_keys
from rheplicant.config.sections.model import build_node_operator

__all__ = ["build_fit_twin"]

_TWIN_KEYS = frozenset({"without", "replace"})


def build_fit_twin(
    section: Any, twin: Any, context: ResolutionContext
) -> tuple[Any, tuple[str, ...]]:
    """``inference.twin`` -> ``(repaired twin, replaced node ids)``.

    The replaced ids come back so the space builder can refuse a binding into
    a node this section just swapped out -- check B8's config-level form: the
    replacement would be overwritten at bind time and the document would be
    saying two contradictory things about one leaf.
    """
    if section is None:
        return twin, ()
    if not isinstance(section, Mapping):
        raise ConfigError(
            f"inference.twin: is a mapping with without: and/or replace:; "
            f"got {type(section).__name__} ({section!r})."
        )
    check_unknown_keys("inference.twin", dict(section), _TWIN_KEYS, label="twin:")
    if not hasattr(twin, "without"):
        raise ConfigError(
            "inference.twin: repairs a graph assembly, and this model is "
            f"kind: pipeline ({type(twin).__name__}). A pipeline is rebuilt, "
            "not repaired: declare the fit pipeline as its own variant."
        )
    repaired = twin
    without = (
        section["without"]
        if "without" in section
        else context.use_default("inference.twin.without", [])
    )
    if not isinstance(without, (list, tuple)) or not all(
        isinstance(node_id, str) for node_id in without
    ):
        raise ConfigError(f"inference.twin.without: is a list of node ids; got {without!r}.")
    for node_id in without:
        repaired = repaired.without(node_id)
    replace = section.get("replace") or {}
    if not isinstance(replace, Mapping):
        raise ConfigError(
            f"inference.twin.replace: is a mapping of node id -> node spec; got {replace!r}."
        )
    for node_id, spec in replace.items():
        repaired = repaired.replace_node(node_id, build_node_operator(node_id, spec, context))
    return repaired, tuple(replace)
