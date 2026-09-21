"""Block geometry, and refusing a stack that is not one.

Column spans, square blocks and quadratic forms, plus the two refusals that
catch a foreign stack or a foreign block before the arithmetic reaches it.
These are the checks a chain applies to its INPUT, which is why they are
neither the model nor the recursion.
"""

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.sqrtinfo import SqrtInfo

#: What a chain can do about an ``epoch_id`` it already holds, which is not what
#: a bag can do -- see :meth:`ChainMemory.remember` for the 0.1129 nats that
#: separate the two answers, and :func:`~rheplicant.inference.memory.reject_bad_term`
#: for why the rule is shared and this sentence is not.
_CHAIN_REPEAT_REMEDY = (
    "and it would land last, so the campaign would also say that night happened "
    "again after the ones that followed it. duplicate=True is the bag's answer "
    "and a chain refuses it: drop the repeat, or give the retried recording its "
    "own epoch_id."
)


def _column_spans(names: tuple[str, ...], shapes: tuple[tuple[int, ...], ...]) -> dict[str, range]:
    """``{name: which columns of a factor over these names it owns}``.

    One copy, read by :func:`_square_block` when it permutes an epoch into the
    memory's order and by :func:`_reject_a_foreign_stack` when it asks whether a
    stored block came from a given epoch. A second implementation of this
    arithmetic would let the check agree with a permutation the accumulator did
    not perform, which is the one way the check could pass on a block that is
    genuinely somebody else's.
    """
    spans: dict[str, range] = {}
    position = 0
    for name, shape in zip(names, shapes, strict=True):
        size = int(np.prod(shape, dtype=int))
        spans[name] = range(position, position + size)
        position += size
    return spans


def _square_block(info: SqrtInfo, order: tuple[str, ...]) -> SqrtInfo:
    """One epoch's joint form, permuted into ``order`` and padded to square.

    Square because ``lax.scan`` needs one shape per iteration, and by
    ``combine(null, info)`` rather than by ``jnp.pad`` because that is the QR the
    accumulator already uses: the offset it returns is the one the filter
    consumes, corner included. Padding with zeros would produce the same factor
    and a **different offset**, which is exactly the class of error section 6 is
    most exposed to -- the fold's corner is the largest of the six constants at
    +45.95 nats over this fixture's six epochs.
    """
    if tuple(info.names) != order:
        shapes = dict(zip(info.names, info.shapes, strict=True))
        columns = _column_spans(tuple(info.names), tuple(info.shapes))
        permutation = jnp.asarray([column for name in order for column in columns[name]], dtype=int)
        info = SqrtInfo(
            factor=info.factor[:, permutation],
            target=info.target,
            offset=info.offset,
            names=order,
            shapes=tuple(shapes[name] for name in order),
        )
    return SqrtInfo.combine(SqrtInfo.null(info.names, info.shapes), info)


def _quadratic_form(
    factor: Any, target: Any, offset: Any, columns: Sequence[int] | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """The three coefficients of ``offset - ||R x - z||^2 / 2`` as a form in x.

    Namely the Gram ``R^T R``, the cross term ``R^T z``, and the log-density's
    own CONSTANT ``offset - z.z/2``. They are what a stored block and the epoch
    it came from have in common, and all that they have in common.
    ``combine(null, info)`` re-triangularises, so the stored factor is **not**
    the epoch's factor even up to roundoff -- it has a different number of rows
    and a different sign convention -- while the form it stands for is preserved
    exactly. Comparing coefficients is therefore the check; comparing arrays
    would refuse every block the accumulator actually builds.

    **The third one was ``z.z + offset`` until 2026-08-28, and that is not a
    coefficient of anything.** ``combine`` moves the part of the residual no
    quadratic form in ``x`` can express into the corner ``rho``: ``z.z`` falls
    by ``rho^2`` and ``offset`` is paid ``-rho^2/2``, so their SUM falls by
    ``1.5 * rho^2`` while the density is unchanged. The old spelling was
    therefore preserved only when ``rho`` was empty -- which it is for every
    full-rank epoch, and was for every fixture that reached this code path.

    It surfaced as a platform difference, which is worth recording because the
    guard's own message denies it. An exactly collinear night's QR has a zero
    pivot, and *where LAPACK puts it* differs: on darwin/arm64 column 1 reduces
    to 1.36e-16, the null direction stays inside the kept block and ``rho`` is
    empty; on linux/x86-64 it reduces to exactly 0, the mass moves to the
    corner, and ``rho^2 = 14.62``. The stored block was right on both. The
    guard refused it on one, with a ``StateValidationError`` reporting a
    shuffled campaign -- a data-integrity accusation, raised on correct data,
    reproducible only on the machine nobody develops on.
    """
    factor = np.asarray(factor, dtype=float)
    if columns is not None:
        factor = factor[:, list(columns)]
    target = np.asarray(target, dtype=float)
    return (
        factor.T @ factor,
        factor.T @ target,
        float(offset) - 0.5 * float(target @ target),
    )


def _reject_a_foreign_stack(
    stacked: tuple[Any, Any, Any],
    terms: tuple[Any, ...],
    order: tuple[str, ...],
    width: int,
) -> None:
    """The stack and the archive are one campaign, so they are checked as one.

    ``__init__`` took ``(factorization, stacked, epochs)`` and asked nothing
    about how they related, which made the class's own promise -- ordered,
    refuses to be shuffled -- a property of :meth:`ChainMemory.remember` rather
    than of the type. Four pairs were accepted, and none of them raised
    anywhere downstream:

    * the archive reversed over an unchanged stack. The campaign returns the
      right number, -171.621919 nats at ``PROBES[0]`` on ``chain_bank``, under
      the labels ``('e5', ..., 'e0')``. Every per-epoch report is then attached
      to the wrong night.
    * the stack reversed under an unchanged archive: -171.614950 nats, a
      different model, the same ids.
    * six blocks and two terms. This is the sharp one, because the damage
      compounds: ``remember``'s duplicate guard reads ``_epochs.ids``, which now
      describes two nights out of six, so ``remember(e3)`` saw no clash and
      folded ``e3`` in a **second** time -- seven blocks under
      ``('e0', 'e1', 'e3')``.
    * six blocks and no terms at all, which reports the six-night likelihood
      from an archive that says the campaign is empty.

    **What is compared.** Not the arrays: ``_square_block`` re-triangularises,
    so a stored block never equals its epoch's factor. The quadratic form does
    survive, so ``(A, b, c)`` from :func:`_quadratic_form` is compared on each
    side, each coefficient against ``sqrt(eps)`` times its own scale -- relative,
    for the reason :func:`~rheplicant.inference.sqrtinfo.marginalise` gives
    about pivots, and the same ``sqrt(eps)`` band. Measured over
    ``chain_bank``'s six epochs the worst honest disagreement is 1.4e-16 of the
    block's scale against a band of 1.5e-8, and pairing ``e0``'s block with
    ``e1``'s term disagrees by 3.1e-01. Eight orders of headroom either way.
    The scale was once shared, ``max(|A|, |b|, |c|)``, which let a constant of
    7.2e11 (one RHINO night's time-bandwidth product) wave through two nights
    swapped in the stack; see :func:`_reject_a_foreign_block`.

    **What it costs.** O(N) small numpy products per construction, so O(N^2)
    over a campaign built one night at a time -- the same order section 6
    already spends per likelihood call. Measured on a warm 64-epoch chain, best
    of three: 0.065 s to build with no check, 0.120 s with this one, 0.335 s if
    the blocks are indexed on device instead of converted in bulk, and 0.800 s
    if the check rebuilds each block through :func:`_square_block` rather than
    comparing coefficients. The last is the version that reads as the obvious
    one and is seven times the cost of the whole build.
    """
    factors, targets, offsets = stacked
    n_blocks = int(np.shape(factors)[0])
    shapes = (np.shape(factors), np.shape(targets), np.shape(offsets))
    if shapes != ((n_blocks, width, width), (n_blocks, width), (n_blocks,)):
        raise StateValidationError(
            f"The stack is shaped {shapes}, but {list(order)} is a width-{width} "
            f"vector, so {n_blocks} epochs of it is "
            f"{((n_blocks, width, width), (n_blocks, width), (n_blocks,))}. A "
            "stored block is a quadratic form in one specific ordered vector; a "
            "block of another width is not that form reshaped, it is a different "
            "model, and the filter would slice theta's columns off the chain's."
        )
    if len(terms) != n_blocks:
        raise StateValidationError(
            f"This chain carries {n_blocks} blocks and {len(terms)} epochs. They "
            "are one campaign recorded twice -- the stack is what the recursion "
            "reads and the archive is what names it -- so a mismatch means "
            "either an unnamed night in the arithmetic or a named one missing "
            "from it. With a short archive the duplicate guard reads epoch ids "
            "that no longer describe the stack, and remember() folds in an epoch "
            "that is already there. Build the chain with remember(), which grows "
            "both together."
        )
    if not terms:
        return
    # Converted once, in bulk. Indexing the device arrays instead costs a gather
    # and a transfer per block, which is most of what this check would spend:
    # 0.335 s against 0.120 s on the 64-epoch chain above.
    found = (
        np.asarray(factors, dtype=float),
        np.asarray(targets, dtype=float),
        np.asarray(offsets, dtype=float),
    )
    root_eps = float(np.sqrt(np.finfo(np.asarray(factors).dtype).eps))
    for index, term in enumerate(terms):
        _reject_a_foreign_block(tuple(part[index] for part in found), term, order, index, root_eps)


def _reject_a_foreign_block(
    block: tuple[Any, Any, Any],
    term: Any,
    order: tuple[str, ...],
    index: int,
    root_eps: float,
) -> None:
    """One block against one epoch. See :func:`_reject_a_foreign_stack`."""
    from rheplicant.inference.memory import _stored_names

    stored = _stored_names(term)
    if set(stored) != set(order):
        raise StateValidationError(
            f"Block {index} does not come from epoch {term.epoch_id!r}: that "
            f"epoch is over {list(stored)} and this chain's blocks are over "
            f"{list(order)}. Build the chain with remember(), which refuses the "
            "term before it reaches the stack."
        )
    columns = _column_spans(tuple(term.info.names), tuple(term.info.shapes))
    found = _quadratic_form(*block)
    expected = _quadratic_form(
        term.info.factor,
        term.info.target,
        term.info.offset,
        [column for name in order for column in columns[name]],
    )
    # One band PER COEFFICIENT, each against the scale of what it is made of
    # and never looser than the shared `max(|A|, |b|, |c|)` it replaced. The
    # shared band let the largest coefficient set it for the other two: under
    # RadiometerNoise `c` carries the time-bandwidth product (~7.2e11 for one
    # RHINO night), the band became ~1e4 in every coefficient, and two nights
    # of one design swapped in the stack were accepted. `max|A|` bounds the
    # Gram's roundoff; `sqrt(max|A| z.z)` bounds the cross term's, by
    # Cauchy-Schwarz on `b = R^T z`; the constant keeps the shared scale.
    # Measured in tests/evidence/test_chain_foreign_block_band.py.
    gram = float(np.max(np.abs(expected[0])))
    shared = max(gram, float(np.max(np.abs(expected[1]))), abs(expected[2]))
    # `hypot.reduce`, not `sqrt(z @ z)`: the square underflows to 0 below
    # |z| ~ 1e-154 in float64, which set this band to 0 and refused an honest
    # block on its roundoff.
    target = np.asarray(term.info.target, dtype=float)
    cross = min(shared, float(np.sqrt(gram)) * float(np.hypot.reduce(target)))
    # `not (difference <= tolerance)`, and `isfinite(tolerance)` beside it: NaN
    # loses both comparisons, so the plain `>` form would wave a poisoned block
    # through, and an epoch whose own coefficients are inf makes the tolerance
    # inf, which admits every block there is. Both ends, because a guard written
    # NaN-safely can still be defeated from the other one.
    legs = (
        ("Gram", float(np.max(np.abs(found[0] - expected[0]))), root_eps * gram),
        ("cross term", float(np.max(np.abs(found[1] - expected[1]))), root_eps * cross),
        ("constant", abs(found[2] - expected[2]), root_eps * shared),
    )
    failed = [leg for leg in legs if not (np.isfinite(leg[2]) and leg[1] <= leg[2])]
    if failed:
        label, difference, tolerance = failed[0]
        raise StateValidationError(
            f"Block {index} does not come from epoch {term.epoch_id!r}: their "
            f"quadratic forms differ by {difference:.3e} in the {label}, against "
            f"a band of {tolerance:.3e} at this epoch's scale. The stack is what the "
            "recursion reads and the archive is what names it, so a block "
            "paired with the wrong epoch is either a different model reported "
            "under these ids or this model reported under the wrong ones, and "
            "the chain is ordered, so a reordering is one of those. Build the "
            "chain with remember(), which appends the block and the epoch "
            "together."
        )
