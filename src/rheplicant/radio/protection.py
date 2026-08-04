"""Protected channels: keeping a known calibrator out of the RFI flags.

A continuous-wave calibration tone is a narrow, bright, persistent line —
which is, from a flagger's point of view, the definition of RFI. Both shipped
flaggers duly flag it at fraction 1.0, and flagging sits *downstream* of
``cw_tone`` on the same trunk, so the pipeline that is supposed to use the
calibrator destroys it on the first observation.

"Narrow" is not "one channel". A tone is observed through the spectrometer's
channel response, so it wets a set of channels, and if it drifts that set
moves during the run. Both mask shapes below therefore matter in practice:
``(n_freq,)`` for a line that stays put, ``(n_time, n_freq)`` for one that
does not. Which channels a given tone actually wets is
:class:`~rheplicant.radio.instrument.calibration.CWCalibrationOperator`'s to
decide — it is the only thing on the path that knows the lineshape.

The mechanism is an ``aux`` channel rather than a flagger setting, and that is
the whole design decision:

* the operator that INJECTS the tone knows which channel it went into, and
  writes the protection itself (:func:`protect`);
* the flaggers read it if it is there (:func:`unflag_protected`).

A flagger has no way to tell a calibration tone from RFI, and the operator has
no way *not* to know. Put the switch on the flagger instead and it is one the
user must remember to turn on for every run, with the failure showing up as a
slightly worse calibration rather than as an error — the kind of setting that
gets forgotten exactly once and then never noticed.

What this does NOT do: it removes the flag, it does not remove the tone's
influence on the flagger's own fit. ``MomentRFI`` fits a surface to the
waterfall, and a 5000 K spike biases that fit near the tone whether or not the
spike is flagged afterwards. Protecting a channel is not the same as excluding
it from the estimator, and only the first is claimed here.

Nor is protection free: a channel that is protected is a channel where genuine
RFI now survives into the data. That is the deliberate trade — the tone channel
is known-bright by construction, so a flagger's verdict there carries no
information anyway — but it is a trade, and the raw data still shows what
happened.
"""

from typing import Any

import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError

#: ``state.aux`` key carrying the protected-channel mask (``True`` = keep).
#:
#: A boolean ``(n_freq,)`` channel mask, or a full ``(n_time, n_freq)`` one for
#: a calibrator that is switched in and out.
PROTECTED_KEY = "protected"


def protect(aux: dict[str, Any], mask: jax.Array) -> dict[str, Any]:
    """Return a new ``aux`` with ``mask`` added to the protected channels.

    Composing rather than clobbering, for the same reason the flaggers compose
    their masks: two calibrators (a tone and a switched load, say) both need
    their channels kept, and whichever ran second would otherwise unprotect the
    first. A ``(n_freq,)`` channel mask composed with a ``(n_time, n_freq)``
    waterfall gives a waterfall, which is right: a channel protected at all
    times stays protected at all times.

    The shape checks live here, at the WRITE, and not only in
    :func:`unflag_protected`. Three reasons, all of them about where the error
    lands. The operator that built the mask is still on the stack here, so the
    traceback names it; a flagger three stages downstream can only say that
    *something* wrote a bad mask. A pipeline with no flagger in it never calls
    ``unflag_protected`` at all, so a malformed mask would ride in ``aux``
    unexamined to the end of the run. And two masks that cannot compose fail
    here as a sentence rather than as a raw broadcasting error from ``|``.

    Args:
        aux: the state's aux mapping.
        mask: ``(n_freq,)`` channel mask or ``(n_time, n_freq)`` waterfall
            mask, ``True`` = keep.

    Raises:
        StateValidationError: if ``mask`` is neither a channel mask nor a
            waterfall mask, or if it cannot compose with a mask already there.
    """
    mask = jnp.asarray(mask)
    if mask.ndim not in (1, 2):
        raise StateValidationError(
            f"a protected mask must be a (n_freq,) channel mask or a "
            f"(n_time, n_freq) waterfall mask, got ndim={mask.ndim}."
        )
    previous = aux.get(PROTECTED_KEY)
    if previous is None:
        return {**aux, PROTECTED_KEY: mask}
    previous = jnp.asarray(previous)
    if previous.shape[-1] != mask.shape[-1]:
        raise StateValidationError(
            f"the protection already declared covers {previous.shape[-1]} channels "
            f"but this mask covers {mask.shape[-1]}. Two calibrators on different "
            "frequency grids cannot both be right about the same data, and OR-ing "
            "them would either broadcast by accident or fail without saying whose "
            "band is wrong."
        )
    if previous.ndim == 2 and mask.ndim == 2 and previous.shape[0] != mask.shape[0]:
        raise StateValidationError(
            f"the protection already declared covers {previous.shape[0]} time "
            f"samples but this mask covers {mask.shape[0]}. Two drifting "
            "calibrators must be describing the same run."
        )
    return {**aux, PROTECTED_KEY: previous | mask}


def unflag_protected(flags: jax.Array, aux: dict[str, Any]) -> jax.Array:
    """Clear ``flags`` wherever ``aux`` says the channel is protected.

    A no-op when nothing declared protection, so a flagger keeps working
    unchanged in a pipeline with no calibrator in it.

    A waterfall mask is bound to the TIME AXIS it was written on: its row ``i``
    names the channels the calibrator wet at sample ``i`` of the axis that
    existed when the mask was built. Any stage that changes the number of
    samples — averaging into chunks, selecting a subset — leaves it stale, and
    a stale mask must be dropped or re-derived, not carried. That is why the
    time axis is checked here and not only the channel one.

    Args:
        flags: ``(n_time, n_freq)`` boolean flags, ``True`` = flagged.
        aux: the state's aux mapping.

    Raises:
        StateValidationError: if the mask is neither a channel mask nor a full
            waterfall mask, if its channel axis does not match the flags', or
            if it is a waterfall over a different number of samples. All three
            would otherwise broadcast — a ``(n_freq,)`` mask against a
            transposed waterfall, a mask built for a different band, or a
            single-row waterfall left over from a shape-changing stage — and
            silently protect the wrong channels, or every one of them.
    """
    protected = aux.get(PROTECTED_KEY)
    if protected is None:
        return flags
    protected = jnp.asarray(protected)
    if protected.ndim not in (1, 2):
        raise StateValidationError(
            f"aux[{PROTECTED_KEY!r}] must be a (n_freq,) channel mask or a "
            f"(n_time, n_freq) waterfall mask, got ndim={protected.ndim}."
        )
    if protected.shape[-1] != flags.shape[-1]:
        raise StateValidationError(
            f"aux[{PROTECTED_KEY!r}] has {protected.shape[-1]} channels but the "
            f"flags have {flags.shape[-1]}. A mask built for a different band "
            "would broadcast against these flags only by accident, and protect "
            "whichever channels happened to line up."
        )
    if protected.ndim == 2 and protected.shape[0] != flags.shape[0]:
        raise StateValidationError(
            f"aux[{PROTECTED_KEY!r}] is a waterfall mask over "
            f"{protected.shape[0]} time samples but the flags cover "
            f"{flags.shape[0]}. A waterfall mask names which channels the "
            "calibrator wet at each sample of the axis it was WRITTEN on, so a "
            "stage that changed the number of samples (averaging into chunks, "
            "selecting a subset) has left this one stale — row i no longer "
            "refers to sample i. A single-row mask is the dangerous case, "
            "because it would broadcast over every sample and protect the whole "
            "run rather than fail; it is refused here for that reason. Drop the "
            "mask or re-derive it after a stage that changes the time axis."
        )
    keep = protected if protected.ndim == 2 else protected[None, :]
    return flags & ~keep.astype(bool)


__all__ = ["PROTECTED_KEY", "protect", "unflag_protected"]
