"""Private names cross a package boundary only through a pinned route.

``tests/config/test_config_surface.py`` already does this for ONE boundary,
``gui -> config``, by scanning text and allowing three files. This is the same
idea for every package pair, by AST rather than by text, and at the
granularity that turned out to matter: **the route is the FILE**, and the
names it may take are pinned within it.

That granularity is not cosmetic. Fifty-nine of the seventy-eight names on
this list come through ``gui/form_catalog.py``, which is the gateway
``CLAUDE.md`` sanctions -- every other GUI module takes config vocabulary from
its ``__all__`` rather than from ``config`` directly. Listing those
fifty-nine as fifty-nine exceptions would read as fifty-nine problems; listing
them as one route with a membership set says what is true, and still puts a
new name through that gateway in front of a reviewer.

**Asserted in both directions, which is the config boundary's own lesson.** An
entry that no longer imports anything fails here. An unused exemption is the
file that could start reaching across with nothing to say so, and that boundary
carried two such entries -- ``gui/jobs.py`` and ``gui/outputs.py`` -- holding a
permission neither used.

A package is ``rheplicant.<sub>`` or ``_rheplicant_bootstrap``. Private imports
WITHIN one package are that package's business and are not scanned: §1.2 scopes
this to boundaries, and a module reaching into its own sibling is a design
question for that package rather than a coupling between two.

Relative imports are skipped for the same reason -- ``from .thing import _x``
cannot leave the package it is written in.
"""

from __future__ import annotations

import ast
import collections
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _package_of(module: str) -> str:
    """The package a dotted module name belongs to, or ``""`` for neither."""
    parts = module.split(".")
    if parts[0] == "_rheplicant_bootstrap":
        return "_rheplicant_bootstrap"
    if parts[0] == "rheplicant":
        return f"rheplicant.{parts[1]}" if len(parts) > 1 else "rheplicant"
    return ""


def _routes() -> dict[tuple[str, str], frozenset[str]]:
    """``(file, provider package) -> the private names it imports``.

    By AST, so a docstring or a comment mentioning a name cannot register a
    route -- the opposite trade from the config boundary's text scan, which
    trips on a ``TYPE_CHECKING`` import and says so.
    """
    found: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for path in sorted(SRC.rglob("*.py")):
        relative = str(path.relative_to(SRC))
        here = _package_of(relative[:-3].replace("/", "."))
        if not here:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module is None or node.level:
                continue
            there = _package_of(node.module)
            if not there or there == here:
                continue
            for alias in node.names:
                if alias.name.startswith("_"):
                    found[(relative, there)].add(alias.name)
    return {key: frozenset(names) for key, names in found.items()}


#: Why each route exists. One sentence, and the burden is on the route.
REASONS = {
    (
        "rheplicant/config/inflight/axes.py",
        "rheplicant.core",
    ): "one refusal helper, shared so the axis pass and core answer a stored "
    "dtype the same way rather than twice",
    (
        "rheplicant/config/preflight/model.py",
        "rheplicant.core",
    ): "the graph descendant walk, shared so pre-flight and the build agree "
    "about what a node reaches",
    (
        "rheplicant/gui/form_catalog.py",
        "_rheplicant_bootstrap",
    ): "the runtime key vocabulary, which the bootstrap owns because it is "
    "read before the package is importable",
    (
        "rheplicant/gui/form_catalog.py",
        "rheplicant.config",
    ): "THE gateway: the config vocabulary every other GUI module takes from "
    "this file's __all__ instead of from config (CLAUDE.md)",
    (
        "rheplicant/gui/form_catalog_finalize.py",
        "_rheplicant_bootstrap",
    ): "the output and report key vocabulary, same owner and same reason as form_catalog.py's",
    (
        "rheplicant/gui/form_rules.py",
        "rheplicant.radio",
    ): "the filter mode names, so the form offers exactly the modes the operator accepts",
    (
        "rheplicant/gui/outputs.py",
        "_rheplicant_bootstrap",
    ): "the marker id and product formats the bootstrap writes, read back "
    "here so the GUI names what the runner produced",
    (
        "rheplicant/inference/parameters.py",
        "rheplicant.core",
    ): "the aliased-leaf walk, shared so a parameter path means the same "
    "thing to the parameter space and to the tree it indexes",
}

#: Route -> the private names it may take. Both the ROUTES and the NAMES are
#: pinned: a new name through an allowed file is still a change to what two
#: packages share, and still belongs in a review.
ALLOWED: dict[tuple[str, str], frozenset[str]] = {
    ("rheplicant/config/inflight/axes.py", "rheplicant.core"): frozenset(
        {
            "_refuse_a_time_axis_the_stored_dtype_cannot_carry",
        }
    ),
    ("rheplicant/config/preflight/model.py", "rheplicant.core"): frozenset(
        {
            "_descendants",
        }
    ),
    ("rheplicant/gui/form_catalog.py", "_rheplicant_bootstrap"): frozenset(
        {
            "_RUNTIME_KEYS",
        }
    ),
    ("rheplicant/gui/form_catalog.py", "rheplicant.config"): frozenset(
        {
            "_ADAM_DEFAULTS",
            "_ALLOWED_KEYS",
            "_AUX_KEYS",
            "_BANK_KEYS",
            "_BENCHMARK_DEFAULTS",
            "_BENCHMARK_KEYS",
            "_BINDING_KEYS",
            "_CABLE_KEYS",
            "_COMPARE_KEYS",
            "_CONDITION_KEYS",
            "_CREATE_DEFAULTS",
            "_CREATE_KEYS",
            "_ENGINE_KEYS",
            "_ENVIRONMENT_KEYS",
            "_ESTIMATE_DEFAULTS",
            "_ESTIMATE_KEYS",
            "_FAN_MODES",
            "_FORMAT_KEYS",
            "_FORM_KEYS",
            "_FREQ_KEYS",
            "_FROM_FILE_KEYS",
            "_GCR_KEYS",
            "_GLS_KEYS",
            "_GRADIENT_KEYS",
            "_IDENTIFIABILITY_KEYS",
            "_INFERENCE_KEYS",
            "_INITS",
            "_KEYS",
            "_KINDS",
            "_KIND_KEYS",
            "_KNOB_DEFAULTS",
            "_LATENT_KEYS",
            "_LST_KEYS",
            "_METRICS",
            "_MMODES_KEYS",
            "_MODES",
            "_NPE_KEYS",
            "_NUTS_DEFAULTS",
            "_NUTS_KEYS",
            "_OBSERVATION_KEYS",
            "_OPTIMIZE_KEYS",
            "_PREDICT_KEYS",
            "_REALISE_KINDS",
            "_RUN_KEYS",
            "_SAMPLE_DEFAULTS",
            "_SAMPLE_KEYS",
            "_SCORE_KEYS",
            "_SHORTHAND",
            "_SIM_KEYS",
            "_SITE_KEYS",
            "_TERMINATION_KEYS",
            "_TIME_KEYS",
            "_TOUCHSTONE_KEYS",
            "_TRAINABLE_KEYS",
            "_TRAIN_DEFAULTS",
            "_TRAIN_KEYS",
            "_TWIN_KEYS",
            "_WIENER_KEYS",
            "_object_fields",
        }
    ),
    ("rheplicant/gui/form_catalog_finalize.py", "_rheplicant_bootstrap"): frozenset(
        {
            "_OUTPUT_KEYS",
            "_PLAN4B_TOP",
            "_PLAN4B_WRITE",
            "_PRODUCT_FORMATS",
            "_REPORT_COLUMNS",
            "_REPORT_FORMATS",
            "_REPORT_RELATIVE",
            "_STDOUT",
            "_WRITE_KEYS",
        }
    ),
    ("rheplicant/gui/form_rules.py", "rheplicant.radio"): frozenset(
        {
            "_MODES",
        }
    ),
    ("rheplicant/gui/outputs.py", "_rheplicant_bootstrap"): frozenset(
        {
            "_MARKER_ID",
            "_PLAN4B_WRITE",
            "_PRODUCT_DEFAULT_FORMATS",
            "_PRODUCT_FORMATS",
            "_REPORT_COLUMNS",
        }
    ),
    ("rheplicant/inference/parameters.py", "rheplicant.core"): frozenset(
        {
            "_aliased_leaf_paths",
        }
    ),
}


def test_no_private_name_crosses_a_boundary_off_the_list():
    """Direction one: a route that is not allowed."""
    live = _routes()
    stray = sorted(set(live) - set(ALLOWED))
    assert not stray, (
        "these files import a private name from another package with no entry "
        f"on the list: {[(f, p, sorted(live[(f, p)])) for f, p in stray]}. "
        "Either take the name from the provider's public surface, or add the "
        "route here WITH a reason -- a private name is a promise to nobody, "
        "and crossing a package with it makes two packages one."
    )


def test_every_allowed_route_is_still_used():
    """Direction two, and the one that rots quietly.

    The config boundary carried two entries whose files imported nothing from
    config at all. An exemption nobody uses is not harmless: it is a standing
    permission for that file to start reaching across, with nothing in the
    review to say it began.
    """
    live = _routes()
    unused = sorted(set(ALLOWED) - set(live))
    assert not unused, (
        f"these routes are allowed and no longer used: {unused}. Delete the "
        "entries; an unused permission is the one that lets a file start "
        "reaching across unnoticed."
    )


@pytest.mark.parametrize("route", sorted(ALLOWED), ids=[f"{f}->{p}" for f, p in sorted(ALLOWED)])
def test_each_route_takes_exactly_the_names_it_is_allowed(route):
    """A new name through a sanctioned file is still a new coupling."""
    live = _routes().get(route, frozenset())
    allowed = ALLOWED[route]
    assert live == allowed, {
        "route": route,
        "newly imported": sorted(live - allowed),
        "no longer imported": sorted(allowed - live),
    }


def test_every_allowed_route_says_why():
    """A list of exceptions with no reasons becomes a list nobody questions."""
    missing = sorted(set(ALLOWED) - set(REASONS))
    assert not missing, f"no reason recorded for {missing}"
    extra = sorted(set(REASONS) - set(ALLOWED))
    assert not extra, f"a reason for a route that is not allowed: {extra}"
    for route, reason in REASONS.items():
        assert len(reason) > 30, f"{route} needs a reason, not a label: {reason!r}"
