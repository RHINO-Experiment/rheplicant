"""BackendOperator — PLACEHOLDER backend processing.

Real physics to come: correlator/spectrometer integration, RFI flagging
(MomentRFI integration point), frequency rebinning, waterfall product
generation. This placeholder averages time chunks — and, importantly,
demonstrates the contract that *shape-changing operators must update the
coordinates along with the data*, and refuse what they cannot update.
"""

from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State
from rheplicant.radio.protection import PROTECTED_KEY, reduce_protection

#: ``state.aux`` key carrying the RFI flag mask (``True`` = flagged), written by
#: both flaggers in :mod:`rheplicant.radio.backend.flagging` and read by
#: :class:`~rheplicant.inference.noise.FlaggedNoise` and
#: :class:`~rheplicant.radio.filters.skyspace.SkySpaceFilter`.
FLAGS_KEY = "flags"

#: Prefix of the keys :meth:`~rheplicant.core.state.State.checkpoint` writes.
SNAPSHOT_PREFIX = "snapshot/"

#: Sentinel: carry this ``aux`` entry across the average unchanged, on purpose.
_CARRY = object()


def _any_over_chunks(mask: Any, n_chunk: int) -> jax.Array:
    """``True`` wherever any sample in the chunk was ``True``.

    Reads truthiness, not order, so a non-boolean mask (an accumulated count,
    or a float mask carrying NaN) reduces to ``True`` for the chunk rather than
    to something a comparison would have to get right — ``nan > x`` is False,
    and a guard built on that would let the contaminated chunk through.
    """
    mask = jnp.asarray(mask)
    n_out = mask.shape[0] // n_chunk
    return mask.reshape(n_out, n_chunk, *mask.shape[1:]).any(axis=1)


def _stale_leaf_shapes(value: Any, n_time: int) -> tuple[tuple[int, ...], ...]:
    """Shapes of the array leaves of ``value`` whose LEADING axis is ``n_time``.

    Walks the pytree rather than looking at ``value.shape``, so an aux entry
    holding a tuple or dict of arrays is caught by whichever member is bound to
    the time axis. Leaves with no axes at all (a scalar, a label) are not per
    anything and never match.
    """
    return tuple(
        shape
        for shape in (jnp.shape(leaf) for leaf in jax.tree_util.tree_leaves(value))
        if len(shape) >= 1 and shape[0] == n_time
    )


class BackendOperator(AbstractOperator):
    """Average ``state.data`` over time chunks of ``n_chunk`` samples (placeholder).

    Updates ``coords.time`` to the per-chunk mean times, reduces the ``aux``
    entries whose semantics it knows onto the same chunk axis, and refuses by
    name an ``aux`` entry bound to the pre-average time axis that it does not
    know how to reduce.

    This is one of exactly two places in the package that does ARITHMETIC on
    ``coords.time`` values rather than reading its length (the other is
    :class:`~rheplicant.radio.instrument.calibration.CWCalibrationOperator`'s
    drift), and it is where a time axis the stored dtype cannot carry surfaces
    as a wrong number rather than as an exception. Measured on 8 samples 100 s
    apart on a unix-epoch axis, ``n_chunk=2``, before
    :class:`~rheplicant.core.coordinates.Coordinates` guarded its own store::

        chunk times  [1750000128, 1750000256, 1750000384, 1750000640]
        float64 truth[1750000050, 1750000250, 1750000450, 1750000650]
        error [s]    [       +78,         +6,        -66,        -10]

    Wrong by 78 s of a 100 s cadence, with two of the eight input samples
    already merged before the mean ran. Nothing here can detect that -- the
    merged values arrive indistinguishable from real ones -- so the check lives
    at the store, in ``Coordinates``, and this operator inherits it: the
    ``replace`` below re-runs it on the chunk-mean axis it produces.

    ``aux`` and the chunk axis
    --------------------------
    ``data`` and ``coords.time`` are not the only things bound to the time axis.
    Every per-time array in ``state.aux`` is stale the moment the axis changes
    length, and the three cases measured on one ``(6, 4)`` fixture with
    ``n_chunk=3``, when this operator left ``aux`` alone, failed with three
    different amounts of noise:

    ==========================  ===================================================
    ``aux`` entry               what happened before
    ==========================  ===================================================
    ``aux["flags"]``            carried at ``(6, 4)``; refused two stages later by
                                ``FlaggedNoise.std``
    ``aux["protected"]``, 2-D   carried at ``(6, 4)``; refused by the next
                                ``FlaggingOperator``, which names the staleness
    ``aux["switch"]``, 1-D      carried at ``(6,)`` with **no error anywhere**
    ==========================  ===================================================

    The third is the one worth the guard. The first two are loud because
    something downstream knows the shape those keys are supposed to have; a key
    this package has never heard of has no such consumer, so a wrong-length
    array rides to the end of the run misaligned with ``data`` and
    ``coords.time`` and nothing can tell. Worse, each output chunk now spans
    ``n_chunk`` of its entries, so for ``[0, 1, 0, 1, 0, 1]`` at ``n_chunk=3``
    there is no right value even in principle — every chunk covers both switch
    positions.

    So: ``flags`` and a 2-D ``protected`` are reduced with ``any`` over the
    chunk (see below); ``snapshot/...`` is carried unchanged, being deliberately
    a record of the pre-average axis; anything else whose leading axis is the
    pre-average ``n_time`` is refused, named. The refusal is on SHAPE alone, so
    it holds under ``jit`` and cannot be defeated by the values — including an
    integer switch index whose chunks happen to be constant, which is refused
    with the rest. Averaging one would silently return a float where an index
    was expected, and asking whether a chunk is constant is a value check this
    operator is not allowed to make.

    That rule is deliberately conservative: an array whose leading axis merely
    *coincides* with ``n_time`` is refused too, because nothing distinguishes it
    from a genuinely per-time one. The way out is in the message — reduce it
    before this stage, or pop it from ``aux`` and put the reduced version back
    after — and it is a sentence rather than a wrong number.

    ``any`` and not ``all``, for both reduced keys: this placeholder averages
    every sample in the chunk, flagged ones included, so the chunk mean is
    contaminated if ANY sample in it was. ``all`` would call a chunk clean with
    two of three samples RFI-blasted and carry that RFI forward as good data.
    When the mean learns to exclude flagged samples the two must change
    together — at that point a chunk is bad only when it has no good sample
    left, and ``all`` becomes the right reduction.

    Attributes:
        n_chunk: samples per integration chunk (static configuration).
    """

    requires: ClassVar[tuple[str, ...]] = (
        "data",
        "coords.time",
        f"aux.{FLAGS_KEY}",
        f"aux.{PROTECTED_KEY}",
    )
    provides: ClassVar[tuple[str, ...]] = (
        "data",
        "coords.time",
        f"aux.{FLAGS_KEY}",
        f"aux.{PROTECTED_KEY}",
    )
    graph_node: ClassVar[str] = "averaging"

    n_chunk: int = eqx.field(static=True)

    def __check_init__(self):
        if not isinstance(self.n_chunk, int) or self.n_chunk < 1:
            raise StateValidationError(f"n_chunk must be a positive int, got {self.n_chunk!r}.")

    def __call__(self, state: State) -> State:
        n_time = state.data.shape[0]
        if n_time % self.n_chunk != 0:
            raise StateValidationError(
                f"n_time={n_time} is not divisible by n_chunk={self.n_chunk}."
            )
        n_out = n_time // self.n_chunk
        data = state.data.reshape(n_out, self.n_chunk, *state.data.shape[1:]).mean(axis=1)

        aux = self._reduce_aux(state.aux, n_time, n_out)
        coords = state.coords
        if coords is not None and coords.time is not None:
            coords = coords.replace(time=coords.time.reshape(n_out, self.n_chunk).mean(axis=1))
        return state.replace(data=data, coords=coords, aux=aux)

    # -- aux ------------------------------------------------------------------

    def _reduce_aux(self, aux: dict[str, Any], n_time: int, n_out: int) -> dict[str, Any]:
        """A new ``aux`` on the chunk axis, or a refusal naming the key that blocks it."""
        if n_out == n_time:
            # n_chunk == 1: the time axis is unchanged, so nothing in aux went
            # stale and refusing a per-time array here would be a pure false
            # positive on a run this operator did not reshape.
            return aux
        reduced: dict[str, Any] = {}
        for key, value in aux.items():
            outcome = self._reduction(key, value, n_time)
            if outcome is None:
                self._refuse_unreducible(key, value, n_time, n_out)
                reduced[key] = value
            elif outcome is _CARRY:
                reduced[key] = value
            else:
                reduced[key] = outcome
        return reduced

    def _reduction(self, key: str, value: Any, n_time: int) -> Any:
        """The chunk-axis form of one ``aux`` entry.

        ``_CARRY`` means "correct across the average as it stands"; ``None``
        means this operator has no reduction for it and the caller must be told.
        """
        if key.startswith(SNAPSHOT_PREFIX):
            return _CARRY
        shape = jnp.shape(value) if hasattr(value, "shape") else ()
        on_time_axis = len(shape) >= 1 and shape[0] == n_time
        if key == FLAGS_KEY and on_time_axis:
            return _any_over_chunks(value, self.n_chunk)
        if key == PROTECTED_KEY:
            if len(shape) == 1:
                return _CARRY  # a channel mask names channels, not samples
            if len(shape) == 2 and on_time_axis:
                return reduce_protection(jnp.asarray(value), self.n_chunk)
        return None

    def _refuse_unreducible(self, key: str, value: Any, n_time: int, n_out: int) -> None:
        """Raise if ``value`` is bound to the pre-average time axis."""
        stale = _stale_leaf_shapes(value, n_time)
        if not stale:
            return
        shown = ", ".join(str(shape) for shape in stale)
        raise StateValidationError(
            f"BackendOperator(n_chunk={self.n_chunk}) averages {n_time} time "
            f"samples into {n_out} chunks, and has no reduction for "
            f"aux[{key!r}] ({shown} — leading axis {n_time}, the PRE-average "
            f"time axis). Carrying it would hand you an array misaligned with "
            f"data and coords.time, and unlike aux[{FLAGS_KEY!r}] and "
            f"aux[{PROTECTED_KEY!r}] — which a later stage refuses by shape — "
            f"nothing downstream knows what this key means, so nothing could "
            f"tell. Each output chunk merges {self.n_chunk} of its entries, so "
            f"for a categorical array (a switch index, a scan number) there is "
            f"no right value even in principle, and averaging one would return "
            f"a float where an index was expected. Reduce it onto the chunk "
            f"axis before this stage, or pop it from aux here and put the "
            f"reduced version back afterwards. Keys under "
            f"{SNAPSHOT_PREFIX!r} are exempt: a snapshot is deliberately a "
            f"record of the axis that existed before."
        )
