"""G15's discharge: the prior curvature is the far side's, the admission is not.

``fisher_information(space=...)`` spelled ``diag(1/sigma^2)`` itself until
bayesmith 0.5 published ``local_block(..., priors=True)``. That spelling is
gone and the number now comes from the graph's own nodes; the two agreed to
``0.0e+00`` against an independent numpy oracle in
``bayesmith/docs/probes/probe_16_g15_discharge.py``.

What did NOT move is the admission, and this module is about why it could not.
:func:`~rheplicant.inference.graph_bridge.translate` sorts bayesmith's refusals
into three families, and ``NotGaussian`` is the blameless third: caught and not
re-raised, left on the yielded ``Seam``. That is right for a caller asking "is
there an exact route here?" in order to branch, and wrong for this one, where a
Uniform prior is an error that three tests pin by name. Left to the far side,
those refusals do not arrive as a different exception -- they do not arrive at
all, and the caller reads ``.values`` off a name that was never assigned.

``tests/inference/test_fisher_prior.py`` pins the refusals and the numbers, and
it passed unchanged across this switch, which is the point. What is pinned HERE
is the part of the arrangement that has no number: that the admission runs
BEFORE the graph exists, and that the seam behaviour making that necessary is
still what it is.
"""

import jax
import jax.numpy as jnp
import pytest

from rheplicant.core.errors import ParameterSpaceError
from rheplicant.inference import Bind, Latent, ParameterSpace
from rheplicant.inference.uncertainty import (
    _declared_gaussian_priors,
    fisher_information,
)

pytest.importorskip("numpyro")
import numpyro.distributions as dist  # noqa: E402

N_DATA = 12
NOISE = 0.4
VALUES = {"a_vec": jnp.array([2.0, 3.0]), "z_scalar": jnp.array(1.0)}


@pytest.fixture
def design():
    return jax.random.normal(jax.random.key(3), (N_DATA, 3))


def make_forward(design):
    def forward(values):
        return design @ jnp.concatenate(
            [values["a_vec"], jnp.atleast_1d(values["z_scalar"])]
        )

    return forward


def space_with(vec_prior, scalar_prior=None):
    if scalar_prior is None:
        scalar_prior = dist.Normal(0.0, 2.0)
    return ParameterSpace(
        latents=[
            Latent("a_vec", init=jnp.array([2.0, 3.0]), prior=vec_prior),
            Latent("z_scalar", init=jnp.array(1.0), prior=scalar_prior),
        ],
        bindings=[
            Bind("a_vec", into=lambda p: p["x"], fn=lambda v: v),
            Bind("z_scalar", into=lambda p: p["y"], fn=lambda v: v),
        ],
    )


def test_a_refused_prior_never_reaches_graph_construction(design, monkeypatch):
    """P1's general rule, at the one exit that needed it.

    "Any refusal whose evidence the graph seam would erase lives in a
    pre-validation, before the graph is built."

    Moving this check to the far side would leave every assertion in
    ``test_fisher_prior.py`` red -- but red with ``UnboundLocalError``, which
    names neither the latent nor the prior. This says the check runs while
    there is still nothing to erase it.

    **Both halves are in one test on purpose.** Written as two, deleting the
    ``monkeypatch`` line left a green suite and a guard that had quietly become
    a duplicate of ``test_a_prior_with_no_quadratic_form_is_refused_by_name``
    -- measured, as a surviving mutant. Asserting first that an ADMITTED prior
    DOES reach the patched builder makes the patch load-bearing for the pass,
    so it cannot be removed silently.
    """
    from rheplicant.inference import graph_bridge

    class GraphWasBuilt(AssertionError):
        pass

    def refuse(*args, **kwargs):
        raise GraphWasBuilt("graph construction was reached")

    monkeypatch.setattr(graph_bridge, "graph_for_information", refuse)
    forward = make_forward(design)
    good = dist.Normal(jnp.zeros(2), jnp.full(2, 0.5))

    # The patch is live, and an admitted prior really does get past the
    # admission to the graph. Without this the assertion below would pass for
    # a `fisher_information` that refused everything, or one this patch never
    # reached at all.
    with pytest.raises(GraphWasBuilt):
        fisher_information(
            forward, VALUES, noise_std=NOISE, space=space_with(good)
        )

    # And a refused one stops before it.
    with pytest.raises(ParameterSpaceError, match="z_scalar"):
        fisher_information(
            forward,
            VALUES,
            noise_std=NOISE,
            space=space_with(good, scalar_prior=dist.Uniform(0.0, 3.0)),
        )


def test_the_seam_still_files_not_gaussian_as_a_blameless_verdict():
    """The premise the pre-validation rests on, pinned rather than asserted.

    If ``translate`` is ever changed to re-raise ``NotGaussian``, this goes red
    and whoever changed it is pointed at ``_declared_gaussian_priors``, whose
    whole shape is an answer to this behaviour. It is a documented decision,
    not an accident -- but a documented decision is exactly the kind that gets
    revisited by someone who has not read the document.
    """
    import bayesmith

    from rheplicant.inference.graph_bridge import translate

    errors = pytest.importorskip("bayesmith.errors")
    reached_the_end = False
    with translate("probe") as seam:
        raise errors.NotGaussian(
            "node 'x' returns Uniform", reason="not_normal", node="x"
        )
        reached_the_end = True  # noqa: F841 - unreachable by construction

    assert not reached_the_end, "the block must NOT continue past the raise"
    assert isinstance(seam.blameless, errors.NotGaussian)
    assert bayesmith is not None


class TestTheDeclarationIsCanonicalised:
    """What crosses the seam is one spelling, whatever was declared.

    Measured in probe 16: bayesmith's ``check_gaussian`` accepts a ``Normal``
    and an ``Independent`` and refuses an ``ExpandedDistribution``;
    ``_gaussian_parameters`` unwraps all three. Handing over the declaration
    as written would admit ``.expand([2])`` here and have it refused -- with
    the refusal erased -- one call later.
    """

    def test_an_expanded_normal_crosses_as_a_plain_full_shaped_normal(self):
        space = space_with(dist.Normal(0.0, 0.5).expand([2]))
        declared = _declared_gaussian_priors(
            space, ("a_vec", "z_scalar"), ((0, 2), (2, 3)), ((2,), ())
        )
        assert type(declared["a_vec"]) is dist.Normal
        assert jnp.shape(declared["a_vec"].scale) == (2,)
        assert jnp.allclose(declared["a_vec"].scale, 0.5)

    def test_the_three_admitted_spellings_give_one_matrix(self, design):
        """Same prior, three ways to write it, one posterior precision."""
        forward = make_forward(design)
        scale = jnp.full(2, 0.5)
        spellings = [
            dist.Normal(jnp.zeros(2), scale),
            dist.Normal(0.0, 0.5).expand([2]),
            dist.Normal(jnp.zeros(2), scale).to_event(1),
        ]
        matrices = [
            fisher_information(
                forward, VALUES, noise_std=NOISE, space=space_with(prior)
            ).matrix
            for prior in spellings
        ]
        for other in matrices[1:]:
            assert jnp.array_equal(matrices[0], other)

    def test_the_canonical_form_is_not_merely_the_declaration_passed_through(
        self,
    ):
        """The sibling: a pass-through would satisfy the shape checks above for
        a full-shaped Normal, and only differ on the wrapped spellings."""
        wrapped = dist.Normal(0.0, 0.5).expand([2])
        space = space_with(wrapped)
        declared = _declared_gaussian_priors(
            space, ("a_vec", "z_scalar"), ((0, 2), (2, 3)), ((2,), ())
        )
        assert declared["a_vec"] is not wrapped
        assert not isinstance(declared["a_vec"], dist.ExpandedDistribution)


def test_the_kind_is_the_far_sides_own_tag_and_not_a_second_rule(
    design, monkeypatch
):
    """``kind`` is read off the returned matrix, not re-decided here.

    The far side tags the quantity from the same flag that decided it. A local
    ``if space is not None`` would be a second copy of that rule and the one
    that goes stale, so this pins the derivation by making the far side answer
    something no local rule would produce.
    """
    from rheplicant.inference import uncertainty

    real = uncertainty._bayesmith_fisher

    def tagged(*args, **kwargs):
        found = real(*args, **kwargs)
        return type(found)(
            values=found.values,
            names=found.names,
            spans=found.spans,
            kind="a_tag_no_local_rule_would_invent",
        )

    monkeypatch.setattr(uncertainty, "_bayesmith_fisher", tagged)
    space = space_with(dist.Normal(jnp.zeros(2), jnp.full(2, 0.5)))
    got = fisher_information(
        make_forward(design), VALUES, noise_std=NOISE, space=space
    )
    assert got.kind == "a_tag_no_local_rule_would_invent"
