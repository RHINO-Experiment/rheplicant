"""Cross-block identifiability: the rank test a per-block guard cannot perform.

Every convergence guard in :mod:`rheplicant.inference.linear` is computed
**from one block**. That is not an oversight, it is arithmetic: a residual
``‖M x - b‖`` and a condition number ``κ(AᵀN⁻¹A + S⁻¹)`` are both properties of
the operator for the block being solved, so neither can see a degeneracy whose
two halves live in *different* blocks. :func:`~rheplicant.inference.linear.
check_linearity` cannot see it either, and it is right not to: each conditional
of a bilinear model genuinely is affine.

The failure that follows is silent and large. An alternating solve over
``gain × T_ant`` with a free antenna temperature per (time, frequency) cell
reports ``κ = 1`` and a CG residual of 1.7e-7 while sitting 2288 K from the
truth. Nothing in the sweep is wrong; the *partition* is, and no per-block
number is entitled to say so.

What can say so is the rank of the Jacobian of the prediction with respect to
**all** the parameters at once::

    free-per-cell T_ant,  tone ON  (5000 K)   n_par=72 rank=64 nullity=8
    free-per-cell T_ant,  tone OFF            n_par=72 rank=64 nullity=8
    (3,3)-basis T_ant,    tone ON  (5000 K)   n_par=17 rank=17 nullity=0
    (3,3)-basis T_ant,    tone OFF            n_par=17 rank=16 nullity=1

Read that as: a known calibration tone buys **exactly nothing** against a
free-per-cell antenna temperature — the free cell at the tone's channel absorbs
the gain sample by sample, so the nullity stays at ``n_time`` either way — and
**everything** against a frequency-smooth one, where a delta at one channel is
not in the span of three smooth basis functions and cannot be reabsorbed.

Three things about the method are not decoration.

**The Jacobian's columns are normalised.** A latent whose natural scale is 1e3
and one whose scale is 1e-3 produce columns differing by 1e6 in norm, and a
rank verdict taken on those reports the choice of units rather than the
identifiability of the model. Column normalisation measures each parameter in
units of its own effect on the prediction, which is the only scale-free
question there is to ask.

**It runs in float64 regardless of the caller's configuration.** The measured
separation between an identified and a non-identified basis model is
``s_min/s_max`` = 6.8e-2 versus 6.6e-17. Computed in float32 the *same* null
direction of the *same* model surfaces at 3.1e-8 — 3.1x above the default
tolerance — and the degenerate model is reported as fully identified.

That is fragility rather than impossibility, and the difference is worth
stating precisely rather than overselling. In float32 the window between this
model's weakest identified direction (4.8e-5) and its null one (3.1e-8) is
still 3.2 decades wide; it simply no longer contains the default. A
per-precision retune of ``rtol`` would therefore recover *this* model. It would
not recover one a few decades worse conditioned, which float64 still resolves
with eight decades to spare. Forcing float64 is what lets one default be right
for both. See :data:`DEFAULT_RANK_RTOL` for how that default is chosen against
the spectrum.

**The result is named.** An anonymous index into a flattened vector tells a
user they have a problem and nothing about which; :meth:`IdentifiabilityReport.
direction` hands back ``{"gain": ..., "t_ant": ...}``, shaped like the latents,
so "the degenerate direction is *this* combination of gain and antenna
temperature" is something you can read and act on.

One thing about it is a limit rather than a feature. **Cost:** a dense Jacobian
and a dense SVD: ``O(n_data · n_par · min(...))``
time and ``n_data · n_par`` float64 words of memory. This is a design-time
diagnostic for tens to a few thousand parameters — the size a Gibbs partition
is *chosen* at — not something to run inside a sweep over a 10⁶-coefficient sky
block. For that block the matrix-free relative is
:func:`~rheplicant.inference.linear.condition_estimate`, which reports the
conditioning of one block without forming anything.
"""

import contextlib
import dataclasses
from collections.abc import Iterator, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from rheplicant.core.errors import ParameterSpaceError, StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State
from rheplicant.inference.parameters import ParameterSpace

#: Singular values at or below ``rtol * s_max`` are called null.
#:
#: Justified against the measured float64 spectrum of the motivating model —
#: the degenerate one, which is the hard case, since it is the only one with a
#: number on both sides of the cut:
#:
#: * its null direction sits at **6.6e-17** of the largest singular value;
#: * its weakest genuinely IDENTIFIED direction at **4.8e-5** (the (3,3)
#:   polynomial basis is itself mildly ill-conditioned; give the same model a
#:   calibration tone and the weakest identified direction rises to 6.8e-2);
#: * the SVD's own noise floor is a few times ``n · eps`` ≈ **1e-14**, below
#:   which no tolerance is measuring anything but roundoff.
#:
#: Every tolerance in (1e-13, 4.8e-5) therefore returns the same verdict — an
#: 8.7-decade window whose geometric centre is 2e-9. 1e-8 sits within a decade
#: of that centre: 3.7 decades of headroom before an identified direction gets
#: called null, and 6 decades of floor before roundoff gets called a signal. It
#: is also ``sqrt(eps)`` in float64, which is the scale at which a direction
#: stops being constrained by the data and starts being constrained by the
#: arithmetic.
#:
#: The upper end is model-dependent and the lower end is not: a genuinely
#: weakly-constrained direction can sit at 1e-6, and then this default is only
#: two decades away from mis-classifying it. Read
#: :attr:`IdentifiabilityReport.weakest_identified` before trusting the verdict,
#: and pass ``rtol=`` when it comes back close to the cut.
#:
#: **The test suite pins this value into 2.5 decades, not 8.7.** That is
#: deliberate but not obvious, so know where a retune will fail before reaching
#: for the assertions. Two counterfactuals elsewhere are stated against this
#: default rather than against a literal, and each end of the window is a
#: measured number that a real claim turns on:
#:
#: * **lower, 1.0e-10** — ``test_the_same_model_without_normalisation_would_be
#:   _called_degenerate``. The mixed-scale fixture's RAW spectrum ratio is
#:   1.0e-10, and that counterfactual only demonstrates anything while the ratio
#:   sits *below* this default. Retune under 1e-10 and the right repair is to
#:   widen the fixture's 1e10 scale gap, not to relax its assertion.
#: * **upper, 3.1168e-8** — ``test_float32_would_have_got_the_verdict_wrong``.
#:   The basis model's null direction surfaces there in single precision, so a
#:   default *above* it would call that direction null, float32 would get the
#:   verdict right, and the argument for the x64 machinery would be gone. The
#:   margin here is 3.1x, not a decade.
#:
#: ``test_the_suite_pins_this_constant_more_tightly_than_the_physics`` states
#: that window in one place; read it first if a retune starts failing tests.
DEFAULT_RANK_RTOL: float = 1e-8


@contextlib.contextmanager
def _in_float64() -> Iterator[None]:
    """Run the enclosed block with ``jax_enable_x64`` on, then restore it.

    Built on ``jax.config.read``/``update`` rather than on any one JAX
    version's context manager, because those have moved between releases and
    this package floors at ``jax>=0.5``. The ``finally`` is load-bearing: a
    diagnostic that raised while leaving x64 on would change the dtype of every
    array the caller made afterwards.

    NOT thread-safe — ``jax.config`` is process-global. Concurrent JAX work in
    another thread will see float64 defaults for the duration.
    """
    was = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", was)


def _resolve_names(space: ParameterSpace, names: Sequence[str] | None) -> tuple[str, ...]:
    """The latents to differentiate with respect to, in the caller's own order.

    ``None`` means all of them, in declaration order. The order is preserved
    rather than sorted because it is the coordinate system every number in the
    report is expressed in, and a caller who asked for ``("t_ant", "gain")``
    should be able to read the answer back in those terms.

    A bare string is one name, the same "one or many" convention
    :class:`~rheplicant.inference.parameters.Bind` uses for its ``latents``.
    Without that, ``names="gain"`` iterates into ``('g', 'a', 'i', 'n')``.
    """
    if names is None:
        return space.names
    selected = (names,) if isinstance(names, str) else tuple(names)
    if not selected:
        raise ParameterSpaceError(
            "identifiability() needs at least one latent name; names=() would report "
            "rank 0 of 0 parameters, which reads as a clean bill of health for a block "
            "that contains nothing. Pass names=None for the whole space."
        )
    unknown = [name for name in selected if name not in space.names]
    if unknown:
        raise ParameterSpaceError(
            f"`names` contains {unknown}, which is not a latent of this space; declared: "
            f"{list(space.names)}."
        )
    repeated = sorted({name for name in selected if selected.count(name) > 1})
    if repeated:
        raise ParameterSpaceError(
            f"`names` lists {repeated} more than once. Two copies of one latent are "
            "exactly degenerate with each other, so the repeat would manufacture a null "
            "direction that says nothing whatever about the model."
        )
    return selected


def _check_at(space: ParameterSpace, at: dict[str, jax.Array] | None) -> None:
    """``at`` may only name declared latents.

    A stray key would be carried into the values dict, ignored by ``bind``, and
    the report would describe the model at a point the caller did not ask for —
    finite, correctly shaped, and about the wrong model.
    """
    if not at:
        return
    unknown = [key for key in at if key not in space.names]
    if unknown:
        raise ParameterSpaceError(
            f"`at` names {unknown}, which is not a latent of this space; declared: "
            f"{list(space.names)}."
        )


def _check_differentiable(space: ParameterSpace, names: Sequence[str]) -> None:
    """Every SELECTED latent must carry a real-valued derivative.

    Only the selected ones: a complex sky-``alm`` block held fixed in another
    Gibbs block is no obstacle to asking whether the gain block is identified.
    """
    complex_names = [
        name
        for name in names
        if jnp.issubdtype(space.latent(name).init.dtype, jnp.complexfloating)
    ]
    if complex_names:
        raise ParameterSpaceError(
            f"Latent(s) {complex_names} are complex. The prediction is real, so the map "
            "from complex coefficients to data is R-linear but not C-linear and its rank "
            "over C is not the number you want — a block with n complex coefficients has "
            "2n real degrees of freedom, and they can be identified separately. Declare "
            "the real and imaginary parts as separate latents, or ask about a different "
            "block with names=."
        )
    not_floating = [
        name
        for name in names
        if not jnp.issubdtype(space.latent(name).init.dtype, jnp.floating)
    ]
    if not_floating:
        kinds = {name: str(space.latent(name).init.dtype) for name in not_floating}
        raise ParameterSpaceError(
            f"Latent(s) {kinds} are not floating-point, so the prediction has no "
            "derivative with respect to them and identifiability — a statement about a "
            "Jacobian — is not defined. An integer latent is not a continuous parameter; "
            "exclude it with names=."
        )


@dataclasses.dataclass(frozen=True)
class IdentifiabilityReport:
    """What the joint Jacobian's rank says about a set of latents.

    Deliberately a plain frozen dataclass rather than an ``eqx.Module``, for
    the same reason :class:`~rheplicant.inference.linear.LinearBlock` is: this
    is a derived linear-algebra verdict, not a differentiable model. It holds
    **numpy** arrays, not JAX ones — a float64 JAX array that escapes the x64
    context truncates, with a warning, the moment a default-precision caller
    touches it, which would throw away exactly the precision the diagnostic
    went to trouble to obtain. And ``rank``/``nullity`` are Python ints, which
    is why this function cannot be jitted: a rank is a decision, and a traced
    decision is one you cannot branch on.

    Attributes:
        names: the latents analysed, in the order the caller asked for.
        shapes: their shapes, in the same order.
        spans: ``(start, stop)`` of each latent within the flat parameter
            vector, in the same order.
        n_par: total number of real parameters — ``sum`` of the latents' sizes.
        n_data: size of the flattened prediction. ``nullity`` can never be
            below ``n_par - n_data``: more parameters than data points is a
            null space by counting alone.
        rank: number of singular values of the COLUMN-NORMALISED Jacobian
            strictly above :attr:`threshold`.
        nullity: ``n_par - rank`` — the dimension of the space of parameter
            perturbations the prediction is blind to.
        singular_values: ``(n_par,)`` descending. When ``n_data < n_par`` the
            SVD returns only ``n_data`` values and the rest are exact zeros;
            they are included rather than dropped, so ``rank`` is always the
            count of entries above the threshold and never needs a caveat.
        null_space: ``(nullity, n_par)`` orthonormal rows, in the
            column-normalised coordinates the rank verdict is taken in. Use
            :meth:`direction` for raw latent coordinates.
        jacobian: ``(n_data, n_par)`` column-normalised, as analysed.
        rtol: the relative tolerance used.
        threshold: ``rtol * singular_values[0]``, the absolute cutoff.
    """

    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    spans: tuple[tuple[int, int], ...]
    n_par: int
    n_data: int
    rank: int
    nullity: int
    singular_values: np.ndarray
    null_space: np.ndarray
    jacobian: np.ndarray
    column_norms: np.ndarray
    rtol: float
    threshold: float

    @property
    def weakest_identified(self) -> float:
        """``s[rank-1] / s[0]`` — how well the worst identified direction is seen.

        The headline number: how much less the data says about the direction it
        constrains least than about the one it constrains most. ``0.0`` when
        nothing at all is identified, which is the only case where the ratio
        has no meaning.
        """
        if self.rank == 0 or float(self.singular_values[0]) == 0.0:
            return 0.0
        return float(self.singular_values[self.rank - 1] / self.singular_values[0])

    def _row(self, index: int) -> np.ndarray:
        # jnp/np arrays index out of range by CLAMPING rather than raising, so
        # direction(5) on a 1-dimensional null space would silently hand back
        # direction 0 again — a wrong answer with no symptom.
        if not 0 <= index < self.nullity:
            raise StateValidationError(
                f"There is no null direction {index}: this model has nullity "
                f"{self.nullity}, so the valid indices are "
                f"{list(range(self.nullity)) or 'none — it is fully identified'}."
            )
        # `nullity` and `null_space` are two records of one fact, and the bounds
        # check above trusts the first while the lookup below uses the second. An
        # SVD taken with full_matrices=False truncates the second whenever there
        # are fewer data points than parameters — the motivating case — and the
        # two then disagree silently: this lookup runs off the end of the array
        # and numpy raises a bare IndexError naming neither cause. Name it.
        if self.null_space.shape != (self.nullity, self.n_par):
            raise StateValidationError(
                f"Inconsistent report: nullity is {self.nullity} over {self.n_par} "
                f"parameters, so null_space should have shape "
                f"{(self.nullity, self.n_par)}, but it has {self.null_space.shape}. "
                "The SVD behind it must be taken with full_matrices=True, or the "
                "null space is truncated whenever n_data < n_par."
            )
        return self.null_space[index]

    def direction(self, index: int) -> dict[str, np.ndarray]:
        """One null direction in RAW latent coordinates, split by name.

        Add a small multiple of this to the latents and the prediction does not
        move, to first order — that is the whole content of the report, and it
        is the form a caller acts in. Note that it is NOT the raw SVD row: the
        SVD is taken of the column-normalised Jacobian, so a null vector there
        has to be divided by the column norms again to become a perturbation of
        the parameters themselves. Returned with unit 2-norm over the flat
        vector; the scale is arbitrary, only the direction means anything.

        For per-latent weights that can be compared across quantities in
        different units, use :meth:`participation` instead — in raw kelvin and
        dimensionless gain, a null direction's two halves are not comparable
        numbers.
        """
        raw = self._row(index) / self.column_norms
        norm = float(np.linalg.norm(raw))
        # `norm` cannot reach 0 while `column_norms` holds the SAFE norms: `raw`
        # is then an orthonormal row divided by finite positive numbers. The
        # fallback is a floor under that invariant, not live code — no mutation
        # of it can be killed, which is why no test below tries to. It becomes
        # reachable only if `column_norms` is ever changed to store the raw
        # norms, and then it makes things worse rather than better: `norm` is
        # NaN, `NaN > 0.0` is False, and dividing by 1.0 preserves the NaN.
        raw = raw / (norm if norm > 0.0 else 1.0)
        return {
            name: raw[start:stop].reshape(shape)
            for name, shape, (start, stop) in zip(
                self.names, self.shapes, self.spans, strict=True
            )
        }

    def participation(self, index: int) -> dict[str, float]:
        """Fraction of a null direction carried by each latent, summing to 1.

        Measured in the COLUMN-NORMALISED coordinates, not raw ones: a 3000 K
        antenna temperature and a gain near 1 cannot be compared in their own
        units, and a raw-unit share would report which quantity is numerically
        larger rather than which one the degeneracy involves. In normalised
        coordinates the bilinear ``gain × T_ant`` degeneracy comes out at
        0.50/0.50, which is the true statement about it.
        """
        row = self._row(index)
        return {
            name: float(np.sum(row[start:stop] ** 2))
            for name, (start, stop) in zip(self.names, self.spans, strict=True)
        }


def _flat_view(
    values: dict[str, jax.Array], names: Sequence[str]
) -> tuple[jax.Array, tuple[tuple[int, ...], ...], tuple[tuple[int, int], ...]]:
    """``(x0, shapes, spans)`` for the selected latents, in the GIVEN order.

    Built by hand rather than with ``ravel_pytree``, which flattens a dict in
    SORTED key order. Sorting is fine as long as nothing else disagrees with
    it, and catastrophic as soon as something does: a report that flattened one
    way and named the other would attribute a degeneracy to the wrong latent
    with every shape still checking out.
    """
    pieces = [jnp.ravel(jnp.asarray(values[name], dtype=jnp.float64)) for name in names]
    shapes = tuple(jnp.shape(values[name]) for name in names)
    spans: list[tuple[int, int]] = []
    offset = 0
    for piece in pieces:
        spans.append((offset, offset + piece.size))
        offset += piece.size
    return jnp.concatenate(pieces), shapes, tuple(spans)


def identifiability(
    space: ParameterSpace,
    pipeline: AbstractOperator,
    state_template: State,
    *,
    names: Sequence[str] | None = None,
    at: dict[str, jax.Array] | None = None,
    rtol: float = DEFAULT_RANK_RTOL,
) -> IdentifiabilityReport:
    """Rank of the joint Jacobian: what the data cannot tell apart.

    The diagnostic that sees ACROSS Gibbs blocks. See the module docstring for
    why no per-block guard can, and for the measured case that motivates it.

    Args:
        space: the parameter declaration.
        pipeline: the forward model.
        state_template: the state it is evaluated on.
        names: which latents to differentiate with respect to — a sequence, or
            a bare string for one. ``None`` (the
            default) means all of them, in declaration order. A subset asks the
            **conditional** question a Gibbs block faces — "is this block
            identified, with the others held fixed?" — and the answer is
            routinely *yes* for every block of a partition whose joint model is
            degenerate. That is the whole reason this function takes the joint
            by default.
        at: values for the latents NOT selected (and starting values for those
            that are). Identifiability is a LOCAL property of a nonlinear
            model, so a sweep has to ask it where the sampler currently is;
            defaults to the space's declared initial values, which is right
            exactly once. Same contract as
            :func:`~rheplicant.inference.linear.linear_operator`'s ``at``.
        rtol: singular values at or below ``rtol * s_max`` are called null. See
            :data:`DEFAULT_RANK_RTOL` for how the default is chosen and when to
            override it.

    Returns:
        An :class:`IdentifiabilityReport`. The three numbers to read first are
        ``n_par``, ``nullity`` and, when the nullity is non-zero,
        ``report.participation(0)`` — which names the latents the degenerate
        direction mixes::

            report = identifiability(space, pipeline, state)
            if report.nullity:
                print(report.nullity, "blind directions;", report.participation(0))

    Raises:
        ParameterSpaceError: if ``names`` is empty, repeats a latent, or names
            one that is not declared; if ``at`` names an undeclared latent; or
            if a selected latent is complex or non-floating.
        StateValidationError: if the model computes its prediction in single
            precision, where the rank verdict cannot be supported.

    Note:
        This runs with ``jax_enable_x64`` forced on for the duration and
        restores the caller's setting afterwards — including on the way out of
        an exception. The setting is process-global, so this is not thread-safe
        against other JAX work.
    """
    selected = _resolve_names(space, names)
    _check_at(space, at)
    _check_differentiable(space, selected)

    with _in_float64():
        forward, values0 = space.forward_fn(pipeline, state_template)
        values0 = {**values0, **(at or {})}
        x0, shapes, spans = _flat_view(values0, selected)

        def f_flat(x: jax.Array) -> jax.Array:
            block = {
                name: jnp.reshape(x[start:stop], shape)
                for name, shape, (start, stop) in zip(selected, shapes, spans, strict=True)
            }
            return jnp.ravel(forward({**values0, **block}))

        jacobian = jax.jacfwd(f_flat)(x0)  # (n_data, n_par)
        # This catches a model that pins its OUTPUT to single precision, which
        # is the way it actually happens (one astype in one operator). It cannot
        # catch a model that rounds an INTERMEDIATE to float32 and promotes back
        # — the dtype is then honest and the digits are gone anyway. Nothing
        # cheap distinguishes that from a genuinely float64 model, so the limit
        # is stated rather than papered over: if a model does that, its
        # `weakest_identified` will sit near 1e-7 and should not be believed.
        if jacobian.dtype != jnp.float64:
            raise StateValidationError(
                f"This model computes its prediction in {jacobian.dtype} even with x64 "
                "enabled, so its own roundoff (~1e-7 relative) is larger than the "
                f"rank tolerance ({rtol:g}) and an exact degeneracy would be reported as "
                "identified — the measured case surfaces at 3.1e-8 in single precision "
                "against 6.6e-17 in double. Remove the float32 cast from the operator or "
                "binding that pins it."
            )

        norms = jnp.linalg.norm(jacobian, axis=0)
        # A zero column is an exact null direction; dividing it by its own zero
        # norm makes the whole spectrum NaN, and a NaN spectrum reports rank 0
        # for every model. Leaving it at zero is both finite and correct.
        safe_norms = jnp.where(norms > 0, norms, 1.0)
        normalised = jacobian / safe_norms
        # The full_matrices flag is load-bearing, and precisely in the headline
        # case: the free-per-cell model has 64 data points against 72 parameters.
        # Turned off, `right` comes back (n_data, n_par) with no rows past index
        # n_data, so `right[rank:]` below would be EMPTY while `nullity` still
        # reported n_par - rank — a report whose two halves disagree, and whose
        # direction() passes its bounds check and then indexes off the end. It is
        # also the obvious performance simplification, so it is pinned by
        # test_the_null_space_is_whole_when_there_are_fewer_data_than_parameters.
        _, spectrum, right = jnp.linalg.svd(normalised, full_matrices=True)

        n_par = int(x0.size)
        n_data = int(jacobian.shape[0])
        spectrum = np.asarray(spectrum, dtype=np.float64)
        # The SVD returns min(n_data, n_par) values; the remaining directions of
        # parameter space are exactly null, so pad rather than drop them. Then
        # rank is simply "how many are above the cutoff", with no caveat.
        spectrum = np.concatenate([spectrum, np.zeros(n_par - spectrum.size)])
        threshold = float(rtol * spectrum[0])
        rank = int(np.sum(spectrum > threshold))
        return IdentifiabilityReport(
            names=selected,
            shapes=shapes,
            spans=spans,
            n_par=n_par,
            n_data=n_data,
            rank=rank,
            nullity=n_par - rank,
            singular_values=spectrum,
            null_space=np.asarray(right[rank:], dtype=np.float64),
            jacobian=np.asarray(normalised, dtype=np.float64),
            column_norms=np.asarray(safe_norms, dtype=np.float64),
            rtol=float(rtol),
            threshold=threshold,
        )


__all__ = ["DEFAULT_RANK_RTOL", "IdentifiabilityReport", "identifiability"]
