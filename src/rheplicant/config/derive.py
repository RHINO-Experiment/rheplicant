"""Form 6: derivations -- a closed registry, one entry per package function.

Every entry names a real call. That is the difference between this and an
expression language: there is nothing here a user can compose, only quantities
the package already computes and that a config would otherwise ask a human to
recompute by hand.

``channel_spacing`` and ``sample_cadence`` are not conveniences. Schema check
A13 requires ``cw_tone.line_width`` to be written with no default, and it must
lie above ``MIN_WIDTH_IN_CHANNELS[lineshape] * median(|diff(freq)|)`` -- note
that ``MIN_WIDTH_IN_CHANNELS`` is a per-lineshape *dict*
(``radio/instrument/calibration.py:142``, ``{"sinc2": 1.0, "gaussian": 0.25}``),
not one number. The arithmetic v0 handed the user was
``25e6 / (N_FREQ - 1) = 806451.6129032258``; rounding that to 0.8 MHz is
refused at trace time and rounding the other way silently mis-sizes the
protection mask.

The measurement is a ``median`` and not a ``mean``, which on the uniform grids
the tests build is a distinction of one float32 ulp and on a real observing
grid is the whole point: a band with a flagged block dropped out of it has one
enormous gap, and the mean of the gaps is then a spacing no pair of adjacent
channels actually has.
"""

import contextlib
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np

from rheplicant.config.context import ResolutionContext
from rheplicant.config.delivery import record_resolved_delivery
from rheplicant.config.errors import ConfigError
from rheplicant.config.registry import LiveNames
from rheplicant.config.units import canonical_unit
from rheplicant.config.values import (
    VALUE_MODIFIERS,
    ResolutionTarget,
    ResolvedValue,
    register_form,
)

#: name -> (function, accepted argument keys, keys routed to the function so
#: it may refuse them by name). Paired deliberately: a derivation added to one
#: half and forgotten in the other is exactly the bug
#: inference/uncertainty.py:85-87 argues a single table prevents.
#:
#: ``refused`` is separate from ``arguments`` rather than folded into it
#: because the two say opposite things to a reader of a refusal. ``n`` on
#: ``basis_matrix`` must reach the function -- only the function can explain
#: why a written ``n`` is a mistake rather than an unknown key -- but listing
#: it among "its arguments are ..." would invite exactly the thing it refuses.
_DERIVATIONS: dict[str, tuple[Callable[..., ResolvedValue], frozenset[str], frozenset[str]]] = {}


def register_derivation(
    name: str, arguments: frozenset[str] = frozenset(), refused: frozenset[str] = frozenset()
):
    """Register one derivation. Returns the function."""

    def _register(fn):
        _DERIVATIONS[name] = (fn, arguments, refused)
        return fn

    return _register


#: Every registered derivation, live. Plans 1B and 2 add to it.
DERIVATIONS = LiveNames(_DERIVATIONS)


@contextlib.contextmanager
def _package_guard(name: str, controls: str):
    """Let no package refusal out of a derivation without the document's context.

    The wrapper :func:`rheplicant.config.files._read` applies to a reader, for
    the same reason. The function's own message is the authority on what it
    refused -- ``core/basis.py:206`` lists the live ``BASIS_KINDS`` and argues
    why the nearest-sounding name is not a safe guess, which is better than any
    restatement here and, being quoted rather than copied, cannot drift from
    the alphabet it quotes -- but it knows nothing about the document, so it
    cannot say which key the author should change. Quoting it verbatim keeps
    one copy of the argument and adds the half ``core`` cannot have.

    Wrapping rather than letting ``StateValidationError`` through is what keeps
    this layer's contract: a loader catches ``ConfigError`` to report "this
    document is wrong", and ``n_basis: 16`` against an 8-channel grid is a
    document error however deep in ``core`` it is noticed.

    ``ConfigError`` passes through untouched -- a refusal raised in this module
    already names the document, the key and the remedy. The catch is
    ``Exception`` rather than an enumeration for the reason ``_read`` gives:
    a guard that lists the types it expects reads every other shape as success.
    The exception's own type is named, so a defect in a package function is
    still reported as one rather than disguised as a bad document.
    """
    try:
        yield
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(
            f"from: {name} was refused by the function it names. "
            f"{type(exc).__name__}: {exc} -- that message is the package's, and it "
            f"knows nothing about the document. {controls}"
        ) from exc


def _median_gap(axis, *, name: str, axis_name: str):
    """``median(|diff(axis)|)``, or a refusal naming which axis was missing.

    ``abs`` before ``median``, not after: a descending grid is legal
    everywhere in this package, and its diffs are all negative, so the
    unsigned version returns a negative spacing -- which then compares below
    every floor it is checked against and reads as "comfortably fine".
    """
    if axis is None:
        raise ConfigError(
            f"{name}: this run declares no {axis_name} grid, so there is nothing to "
            "measure. observation.freq.grid and observation.time.grid are both "
            "required, so this is reachable only when a value node is resolved before "
            "the observation is."
        )
    # float64, via numpy, and NOT jnp. `diff` of a float32 axis whose values
    # are large and close cancels: measured on a 449-channel 50-200 MHz grid,
    # the gaps span 16 Hz around a true spacing of 334821.429 Hz, and the
    # median lands wherever that jitter puts it. That is how the same document
    # reported a "336 Hz median channel spacing" on one machine and "328 Hz"
    # on another -- a user-facing number moving 2.4 % with the CPU rather than
    # with the data. In float64 the same gaps span 6e-08 Hz.
    #
    # jnp cannot do this: `jax_enable_x64` is off here (float32 is the
    # package's production dtype) so a float64 request is silently truncated.
    # numpy can, and may: every caller of this consumes a concrete scalar at
    # config time -- `float(...)` in grids.py, a `ResolvedValue` in the two
    # derivations -- so nothing here is ever traced.
    gaps = np.abs(np.diff(np.asarray(axis, dtype=np.float64)))
    if gaps.size == 0:
        raise ConfigError(
            f"{name}: the {axis_name} axis has {axis.shape[0]} sample(s), so it has no "
            "spacing. A single-sample axis is legal elsewhere; it is not something to "
            "measure."
        )
    # Back to a jax scalar, and that is not cosmetic: `config/delivery.py`
    # refuses a value node for an `eqx.field(static=True)` float BY the value
    # being an array, and returning a Python float here silently turned that
    # refusal off -- caught by exactly one test in 6092, which is the kind of
    # margin this conversion exists to keep. The accuracy is what changed, not
    # the type: float32 holds this median to about seven digits, against the
    # 16 Hz of jitter the float32 `diff` was introducing.
    return jnp.asarray(np.median(gaps))


@register_derivation("channel_spacing", frozenset({"times"}))
def _channel_spacing(node, context, modifiers, target):
    gap = _median_gap(context.freq, name="channel_spacing", axis_name="frequency")
    times = (
        node["times"]
        if "times" in node
        else context.use_default("value.from.channel_spacing.times", 1.0)
    )
    return ResolvedValue(gap * float(times), canonical_unit("Hz"), "from", modifiers)


@register_derivation("sample_cadence", frozenset({"times"}))
def _sample_cadence(node, context, modifiers, target):
    gap = _median_gap(context.time, name="sample_cadence", axis_name="time")
    times = (
        node["times"]
        if "times" in node
        else context.use_default("value.from.sample_cadence.times", 1.0)
    )
    return ResolvedValue(gap * float(times), canonical_unit("s"), "from", modifiers)


@register_derivation("basis_matrix", frozenset({"kind", "n_basis", "axis"}), frozenset({"n"}))
def _basis_matrix(node, context, modifiers, target):
    from rheplicant.core.basis import basis_matrix

    if "n" in node:
        raise ConfigError(
            "basis_matrix: 'n' is never written -- it comes from the run's own grid. A "
            "design matrix built for a different number of samples returns a smooth, "
            "plausible, wrong temperature (radio/t_sys.py), and taking n from the grid "
            "is what makes that impossible rather than merely discouraged. Remove n:; "
            "axis: freq takes it from observation.freq.grid and axis: time from "
            "observation.time.grid."
        )
    axis = node.get("axis")
    grid = {"freq": context.freq, "time": context.time}.get(axis)
    if grid is None:
        raise ConfigError(
            f"basis_matrix: axis={axis!r} is not one of the run's axes ('freq', 'time'), "
            "or that axis is not declared. The axis is what supplies n."
        )
    for required in ("kind", "n_basis"):
        if required not in node:
            raise ConfigError(f"basis_matrix: {required!r} is required.")
    n = int(grid.shape[0])
    with _package_guard(
        "basis_matrix",
        f"The document controls kind: and n_basis:; n is {n}, taken from the run's "
        f"{axis} grid, and is not writable.",
    ):
        matrix = basis_matrix(node["kind"], n=n, n_basis=int(node["n_basis"]))
    return ResolvedValue(matrix, canonical_unit("dimensionless"), "from", modifiers)


@register_derivation("unit_mean_free", frozenset({"bandpass"}))
def _unit_mean_free(node, context, modifiers, target):
    """``receiver.py:103``: ``(bandpass / mean(bandpass))[:-1]``.

    The result is one element SHORTER than the bandpass it came from, and that
    is the function's whole point rather than an accident to be smoothed over
    here: it is the coordinate a ``Latent`` is declared in, and the dropped
    element is the constraint that pins the mean at one. A derivation that
    quietly restored the length would hand the gain and the bandpass back the
    exactly-null direction the coordinate exists to remove.
    """
    from rheplicant.config.values import resolve_operand
    from rheplicant.radio.instrument.receiver import unit_mean_free

    if "bandpass" not in node:
        raise ConfigError("unit_mean_free: 'bandpass' is required and is itself a value node.")
    resolved = resolve_operand(
        node["bandpass"],
        context,
        parent=target,
        segment="bandpass",
        formula="unit_mean_free",
        role="bandpass",
    )
    bandpass = jnp.asarray(resolved.value)
    if target is not None:
        record_resolved_delivery(context, target.destination.nested("bandpass"), resolved.unit)
    with _package_guard(
        "unit_mean_free",
        f"The document controls the bandpass: node, which resolved to shape "
        f"{tuple(bandpass.shape)}; unit_mean_free takes a 1-D (n_freq,) bandpass.",
    ):
        free = unit_mean_free(bandpass)
    return ResolvedValue(free, canonical_unit("dimensionless"), "from", modifiers)


@register_form("from", arguments=None)
def _from(
    node: dict,
    context: ResolutionContext,
    modifiers: dict,
    target: ResolutionTarget | None,
) -> ResolvedValue:
    name = node["from"]
    entry = _DERIVATIONS.get(name)
    if entry is None:
        raise ConfigError(
            f"Unknown derivation {name!r}; the registry holds {sorted(_DERIVATIONS)}. It "
            "is closed on purpose: every entry names one function this package already "
            "computes, so a derivation is a reference rather than arithmetic. A "
            "quantity that is not here is built as a resources.arrays entry through "
            "the python: hatch, which states its own cost."
        )
    fn, arguments, refused = entry
    unknown = sorted(set(node) - {"from"} - arguments - refused - set(VALUE_MODIFIERS))
    if unknown:
        raise ConfigError(
            f"Derivation {name!r} does not take {unknown}; its arguments are {sorted(arguments)}."
        )
    return fn(node, context, modifiers, target)
