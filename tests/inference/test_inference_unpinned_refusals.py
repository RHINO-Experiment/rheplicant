"""The inference-side residue of the raise audit.

``tools/raise_audit.py`` cross-references every ``ast.Raise`` in ``src/``
against a coverage run. Its findings mostly fell into families, covered by
tests that derive their population from the source. These three did not: each
is a single guard about a different thing, and each is worth a test for the
same reason -- a refusal the suite has never executed is not merely untested,
it is a sentence nobody has read, and the two cheapest ways for it to be wrong
are to name the wrong thing and to fire on the wrong side of its condition.
"""

import jax
import jax.numpy as jnp
import pytest

from rheplicant.core.errors import ParameterSpaceError, StateValidationError
from rheplicant.inference import (
    Bind,
    Latent,
    ParameterSpace,
    linear_operator,
    wiener_solve,
)
from rheplicant.inference.linear import LinearBlock
from rheplicant.inference.uncertainty import (
    fisher_information,
    parameter_covariance,
    propagate_covariance,
)
from rheplicant.radio import GainOperator, SkyOperator, assemble

# Not square, and no two entries equal: a guard about which axis is which
# cannot be satisfied by a transposed read of this.
N_TIME, N_FREQ = 4, 6


def _state():
    from rheplicant import Coordinates, State

    return State(
        data=None,
        coords=Coordinates(
            time=jnp.arange(float(N_TIME)),
            freq=jnp.linspace(60e6, 80e6, N_FREQ),
        ),
    )


def _twin():
    return assemble(
        SkyOperator(amplitude=jnp.array(100.0)),
        GainOperator(gain=jnp.linspace(0.9, 1.3, N_TIME)),
    )


class TestALatentMustDeclareItselfLinear:
    """``linear_operator`` refuses a latent that has not claimed linearity.

    The claim is what earns the conjugate machinery. Without the declaration
    there is nothing to check against, and building a "linear operator" for a
    latent that may not be affine would produce a matrix that is simply the
    Jacobian at one point -- finite, correctly shaped, and a linearisation
    rather than the operator it is named after.

    **Reaching it needs an explicit** ``name=``, which is why it was the
    uncovered one. With ``name=None`` the resolver first collects the latents
    that DID declare themselves, and a space where none did fails earlier with
    a different sentence ("No latent in this space is declared linear"). So the
    only route here is a space that has a linear latent and a caller who asks
    for a different one by name -- and that is also the realistic mistake, since
    a space with nothing linear is a space nobody was about to solve.
    """

    def _mixed_space(self) -> ParameterSpace:
        """One declared-linear latent and one not, so ``name=`` is meaningful."""
        return ParameterSpace(
            latents=[
                Latent("gt", init=jnp.ones(N_TIME), linear=True),
                Latent("amp", init=jnp.array(100.0), linear=False),
            ],
            bindings=[
                Bind("gt", into=lambda p: p["gain"].gain),
                Bind("amp", into=lambda p: p["uniform_sky"].amplitude),
            ],
        )

    def test_asking_for_the_undeclared_latent_by_name_is_refused(self):
        with pytest.raises(ParameterSpaceError, match=r"'amp' is not declared linear=True"):
            linear_operator(self._mixed_space(), _twin(), _state(), name="amp")

    def test_the_message_says_what_to_do_about_it(self):
        """A refusal that names a defect without naming the remedy sends the
        reader to the source, which is the cost the sentence exists to avoid."""
        with pytest.raises(ParameterSpaceError, match="Declare it"):
            linear_operator(self._mixed_space(), _twin(), _state(), name="amp")

    def test_the_declared_latent_in_the_same_space_builds_its_operator(self):
        """The other branch, in the SAME space, so the refusal is demonstrably
        about which latent was asked for and not about the space."""
        block = linear_operator(self._mixed_space(), _twin(), _state(), name="gt")
        assert block.shape == (N_TIME,)
        assert jnp.shape(block.offset) == (N_TIME, N_FREQ)

    def test_a_space_with_nothing_linear_fails_earlier_and_differently(self):
        """Pinning the reason this guard was unreachable without ``name=``.

        If the two messages ever merge, this fails -- and merging them would
        quietly make the guard above dead code again.
        """
        space = ParameterSpace(
            latents=[Latent("gt", init=jnp.ones(N_TIME), linear=False)],
            bindings=[Bind("gt", into=lambda p: p["gain"].gain)],
        )
        with pytest.raises(ParameterSpaceError, match="No latent in this space is declared"):
            linear_operator(space, _twin(), _state())


class TestAComplexPredictionIsRefusedAtTheConjugateSeam:
    """The conjugate solves are real-valued; a complex offset is refused early.

    ``bayesmith.exact.solve._conjugate_solve`` splits a latent into real degrees
    of freedom and takes
    ``jax.grad`` of a real pairing. A complex *prediction* has no such pairing,
    so without this guard the failure surfaces much later as a dtype error
    inside CG, naming a helper rather than the argument.
    """

    def _complex_block(self) -> LinearBlock:
        # Offset entries all distinct in BOTH parts, so a guard that only
        # looked at the real part, or only at one entry, would be visible.
        offset = jnp.arange(N_TIME * N_FREQ, dtype=jnp.float32).reshape(
            N_TIME, N_FREQ
        ) * (1.0 + 2.0j)
        return LinearBlock(
            name="z",
            shape=(N_TIME,),
            dtype=jnp.complex64,
            offset=offset,
            forward=lambda x: jnp.broadcast_to(x[:, None], (N_TIME, N_FREQ)),
            adjoint=lambda y: jnp.sum(y, axis=1),
            prior=None,
        )

    def test_a_complex_offset_is_refused_and_names_the_exit(self):
        with pytest.raises(ParameterSpaceError, match=r"wiener_solve expects a real-valued"):
            wiener_solve(
                self._complex_block(),
                jnp.zeros((N_TIME, N_FREQ), dtype=jnp.complex64),
                noise_std=1.0,
                prior_std=1.0,
            )

    def test_a_real_block_of_the_same_shape_is_not_refused(self):
        """The other branch, at the same shape, so the refusal is demonstrably
        about the dtype and not about anything else in the block."""
        real = self._complex_block()
        block = LinearBlock(
            name=real.name,
            shape=real.shape,
            dtype=jnp.float32,
            offset=jnp.real(real.offset),
            forward=real.forward,
            adjoint=real.adjoint,
            prior=None,
        )
        value, _ = wiener_solve(
            block,
            jnp.zeros((N_TIME, N_FREQ)),
            noise_std=1.0,
            prior_std=1.0,
        )
        assert jnp.shape(value) == (N_TIME,)
        assert jnp.all(jnp.isfinite(value))


class TestCovarianceProvenanceBeyondTheTreeStructure:
    """Same latent NAMES, different per-latent shapes.

    ``propagate_covariance`` has two provenance guards and only the first was
    ever executed. The first compares pytree structures -- but for a dict-based
    space a treedef encodes the KEY NAMES only, so two spaces whose latents are
    named alike and shaped differently pass it and go on to produce finite,
    wrong error bars. That is what the second guard is for, and it is the one
    that had no test.

    The source says as much in a comment. A comment explaining why a guard
    exists, above a guard nothing exercises, is the exact configuration where
    the guard and the comment can disagree without anyone finding out.
    """

    @staticmethod
    def _forward_for(n: int):
        def forward(params):
            return params["amp"][:, None] * jnp.ones((n, N_FREQ))

        return forward

    def _covariance_for(self, n: int):
        params = {"amp": jnp.linspace(1.0, 2.0, n)}
        fisher = fisher_information(self._forward_for(n), params, noise_std=1.0)
        return parameter_covariance(fisher), params

    def test_same_names_different_shapes_is_refused(self):
        cov, _ = self._covariance_for(N_TIME)
        # Same key, so the tree STRUCTURE matches and the first guard passes.
        other = {"amp": jnp.linspace(1.0, 2.0, N_TIME + 1)}
        with pytest.raises(StateValidationError, match=r"was computed for \{'amp'"):
            propagate_covariance(self._forward_for(N_TIME + 1), other, cov)

    def test_it_is_the_SHAPES_guard_that_fires_and_not_the_structure_one(self):
        """The two guards share their opening and closing sentences.

        Both begin "param_cov was computed for" and both end "the flattened
        orderings differ and the numbers would be wrong"; they differ only in
        the middle, where one says "parameter structure" and quotes treedefs
        and the other quotes a name-to-shape mapping. A test matching either
        shared sentence passes when the WRONG guard fires -- and since the
        structure guard runs first, a regression that made it over-eager would
        be invisible to exactly the test written to cover the second one.

        This is the "three substring matches satisfied by one over-broad
        message" hazard, and the first version of the test above had it.
        """
        cov, _ = self._covariance_for(N_TIME)
        other = {"amp": jnp.linspace(1.0, 2.0, N_TIME + 1)}
        with pytest.raises(StateValidationError) as excinfo:
            propagate_covariance(self._forward_for(N_TIME + 1), other, cov)
        message = str(excinfo.value)
        assert "parameter structure" not in message, message
        assert "was computed for {'amp'" in message, message

    def test_the_message_quotes_both_shapes(self):
        """Naming only one of them leaves the reader to work out which is
        which, and the whole failure is that the two look alike."""
        cov, _ = self._covariance_for(N_TIME)
        other = {"amp": jnp.linspace(1.0, 2.0, N_TIME + 1)}
        with pytest.raises(StateValidationError) as excinfo:
            propagate_covariance(self._forward_for(N_TIME + 1), other, cov)
        message = str(excinfo.value)
        assert f"({N_TIME},)" in message and f"({N_TIME + 1},)" in message, message

    def test_the_matching_covariance_propagates(self):
        """The other branch, and the reason the guard cannot simply be stricter.

        A correct pairing has to keep working, so the check is on names and
        shapes together rather than on identity of the object.
        """
        cov, params = self._covariance_for(N_TIME)
        std = propagate_covariance(self._forward_for(N_TIME), params, cov)
        assert std.shape == (N_TIME, N_FREQ)
        assert jnp.all(jnp.isfinite(std))
        assert jax is not None
