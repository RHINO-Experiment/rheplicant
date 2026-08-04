"""Protected channels: keeping a known calibrator out of the RFI flags.

A continuous-wave calibration tone is a narrow, bright, persistent spike in
one channel — which is, from a flagger's point of view, the definition of RFI.
Both shipped flaggers duly flag it at fraction 1.0, and flagging sits
*downstream* of ``cw_tone`` on the same trunk, so the pipeline that is supposed
to use the calibrator destroys it on the first observation.

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
    first.
    """
    previous = aux.get(PROTECTED_KEY)
    combined = mask if previous is None else (previous | mask)
    return {**aux, PROTECTED_KEY: combined}


def unflag_protected(flags: jax.Array, aux: dict[str, Any]) -> jax.Array:
    """Clear ``flags`` wherever ``aux`` says the channel is protected.

    A no-op when nothing declared protection, so a flagger keeps working
    unchanged in a pipeline with no calibrator in it.

    Args:
        flags: ``(n_time, n_freq)`` boolean flags, ``True`` = flagged.
        aux: the state's aux mapping.

    Raises:
        StateValidationError: if the mask is neither a channel mask nor a full
            waterfall mask, or if its channel axis does not match the flags'.
            Both would otherwise broadcast — a ``(n_freq,)`` mask against a
            transposed waterfall, or a mask built for a different band — and
            silently protect the wrong channels.
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
    keep = protected if protected.ndim == 2 else protected[None, :]
    return flags & ~keep.astype(bool)


__all__ = ["PROTECTED_KEY", "protect", "unflag_protected"]
