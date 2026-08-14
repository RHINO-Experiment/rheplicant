"""posterior_support: the three helpers nuts and npe share, driven directly.

Nothing here calls ``run_document`` -- that is the test-module seam this
suite has kept since 2C.
"""

import jax
import jax.numpy as jnp
import pytest

from rheplicant.config.document import load_document
from rheplicant.config.errors import ConfigError
from rheplicant.config.sections.posterior_support import (
    _draw_key,
    _sampled_space,
    _unravel,
)
from tests.config.exit_helpers import (
    PRIOR_FREE,
    PRIOR_FREE_TWO,
    TWO_LATENTS,
    conjugate_built,
    conjugate_document,
    spec,
)
from tests.config.test_config_document import synthetic_document

JOINT = {**TWO_LATENTS, "parameters": PRIOR_FREE_TWO,
         "joint_prior": {"jeffreys": {"over": ["d", "a"]}}}


def vector_space():
    """A space whose two latents are 3 wide and 1 wide -- total width 4.

    Built rather than bound as a constant because ``ParameterSpace.raw``
    closes over a ``bind``; five unravel tests want the same shape, and a
    scalar-only space cannot tell a correct unravel from one that assumes
    every latent is a scalar.
    """
    from rheplicant.inference import Latent, ParameterSpace

    return ParameterSpace.raw(
        [Latent("b", init=jnp.arange(3.0)),
         Latent("s", init=jnp.asarray(1.0))], bind=lambda p, v: p)


class TestTheTwoRoutesDisagreeAboutAPrior:
    """The asymmetry, tested on BOTH routes from one document.

    ``to_numpyro_model`` -> ``_require_priors`` accepts a latent the space's
    ``joint_prior`` covers (``numpyro_bridge.py:72``); ``simulate_pairs``
    tests ``latent.prior is None`` alone (``npe.py:111-118``).  Testing only
    the route the task happens to be about is 2C shape 4 -- a hole closed on
    one route and left open on its twin -- so all four cells are here.
    """

    def test_a_joint_prior_only_space_is_a_space_for_nuts(self):
        built = conjugate_built(inference=JOINT)
        assert _sampled_space(spec(kind="nuts"), built,
                              route="nuts") is built.inference.space

    def test_a_joint_prior_only_space_is_refused_for_npe(self):
        """The refusal, and the SENTENCE that says why the sibling differs.

        The last assertion is not decoration.  Without it every other string
        here is satisfied from elsewhere in the message -- "['d', 'a']" from
        `missing`, "kind: nuts" from the `instead` advice, "SIMULATES a bank"
        from the fixed prefix -- so nothing read the `because` clause and
        deleting it outright left this test green (measured).  The phrase
        pinned below occurs in that clause and nowhere else.
        """
        built = conjugate_built(inference=JOINT)
        with pytest.raises(ConfigError) as caught:
            _sampled_space(spec(kind="npe"), built, route="npe")
        message = str(caught.value)
        assert "SIMULATES a bank" in message
        assert "kind: nuts" in message
        assert "['d', 'a']" in message
        assert "which is why kind: nuts accepts this space" in message

    def test_a_prior_free_space_gets_no_joint_prior_advice_from_npe(self):
        """The npe refusal's OTHER branch: what it must NOT say.

        Both conditional clauses of that message are advice, and on a
        document with no `joint_prior` at all the joint-prior advice is
        FALSE: the nuts leg refuses this very document for the same missing
        prior, so "or run kind: nuts, which takes joint-prior coverage"
        sends the reader to an exit that will refuse them again.  Asserting
        a string is PRESENT cannot catch that; asserting it is ABSENT can.
        Measured: making either clause unconditional fails this test.
        """
        built = conjugate_built(parameters=PRIOR_FREE)
        with pytest.raises(ConfigError) as caught:
            _sampled_space(spec(kind="npe"), built, route="npe")
        message = str(caught.value)
        assert "refuses this document too" in message
        assert "which takes joint-prior coverage" not in message
        assert "inference.joint_prior covers" not in message

    def test_a_prior_free_space_is_refused_on_BOTH_routes(self):
        built = conjugate_built(parameters=PRIOR_FREE)
        with pytest.raises(ConfigError, match="draws a POSTERIOR"):
            _sampled_space(spec(kind="nuts"), built, route="nuts")
        with pytest.raises(ConfigError, match="SIMULATES a bank"):
            _sampled_space(spec(kind="npe"), built, route="npe")

    def test_no_inference_parameters_is_refused_before_the_prior_gate(self):
        built = load_document(synthetic_document())
        with pytest.raises(ConfigError, match="declares no "
                           "inference.parameters"):
            _sampled_space(spec(kind="nuts"), built, route="nuts")


class TestTheUnravel:
    """Written here, called by npe at Task 8, and tested at both ends.

    Both failure modes are silent: sorting the names hands every latent
    another latent's column, and assuming a scalar hands the first latent a
    slice of the second.  In both cases the draws are finite and correctly
    shaped, so the tests are about ORDER and WIDTH rather than about shape.
    """

    def test_the_columns_go_by_declaration_order_not_sorted(self):
        space = conjugate_built(inference=TWO_LATENTS).inference.space
        assert list(space.names) == ["d", "a"]
        out = _unravel(space, jnp.asarray([[1.0, 10.0], [2.0, 20.0]]))
        assert [float(v) for v in out["d"]] == [1.0, 2.0]
        assert [float(v) for v in out["a"]] == [10.0, 20.0]

    def test_a_vector_latent_takes_its_own_width(self):
        space = vector_space()
        out = _unravel(space, jnp.arange(8.0).reshape(2, 4))
        assert out["b"].shape == (2, 3)
        assert out["s"].shape == (2,)
        assert out["b"].tolist() == [[0.0, 1.0, 2.0], [4.0, 5.0, 6.0]]
        assert out["s"].tolist() == [3.0, 7.0]

    def test_a_width_the_space_cannot_account_for_is_refused(self):
        with pytest.raises(ConfigError, match="5 wide"):
            _unravel(vector_space(), jnp.zeros((2, 5)))

    @pytest.mark.parametrize("narrow", [3, 2])
    def test_a_flat_NARROWER_than_the_space_is_refused_too(self, narrow):
        """Both directions of one check, and this is the escaping one.

        The width comparison used to sit AFTER the reshape loop, so a
        too-WIDE flat got this layer's ConfigError while a too-NARROW one
        reached jax first -- measured, `TypeError: cannot reshape array of
        shape (2, 0) into shape (2,)`, a message naming no run, no document
        and no latent.  The refusal was already worded for both directions;
        only one of them could be reached.  Narrow is the likelier bug at
        Task 8, because it is what an estimator sized from a stale space
        produces.
        """
        with pytest.raises(ConfigError, match="accounts for 4"):
            _unravel(vector_space(), jnp.zeros((2, narrow)))

    def test_the_refusal_wears_the_callers_where(self):
        """`where=` is this module's addition to the pinned signature.

        Without it this is the one refusal in either new module that names
        no run, which in a layer where every other one opens with
        `runs['<name>']:` reads as a package error rather than as this
        layer's.  Task 8 has the name to hand.
        """
        with pytest.raises(ConfigError) as caught:
            _unravel(vector_space(), jnp.zeros((2, 5)), where="runs['bank']")
        assert str(caught.value).startswith("runs['bank']: the draws are 5 ")


class TestTheDrawKey:
    def test_the_key_is_the_reported_seed(self):
        built = conjugate_built(seeds={"chain": 3})
        key = _draw_key(spec(kind="nuts",
                             seed={"from": "runtime.seeds.chain"}),
                        "runs['chain']", built)
        assert bool(jnp.all(jax.random.key_data(key)
                            == jax.random.key_data(jax.random.key(3))))

    def test_a_missing_seed_is_refused_under_the_callers_where(self):
        built = conjugate_built(seeds={"chain": 3})
        with pytest.raises(ConfigError) as caught:
            _draw_key(spec(kind="nuts"), "runs['chain']", built)
        assert str(caught.value).startswith("runs['chain']:")

    def test_a_literal_seed_is_refused(self):
        built = conjugate_built(seeds={"chain": 3})
        with pytest.raises(ConfigError, match="must NAME an entry"):
            _draw_key(spec(kind="nuts", seed=7), "runs['chain']", built)

    def test_spec_overrides_the_runs_own_options(self):
        """The npe form: four named seeds, one run.

        The run here declares `chain` and the spec declares `bank`, and the
        key must be `bank`'s.  A `_draw_key` that ignored `spec` would still
        return a valid key from a valid seed -- which is why the two seeds
        are different numbers rather than the same one twice.
        """
        built = conjugate_built(seeds={"chain": 3, "bank": 5})
        run = spec(kind="npe", seed={"from": "runtime.seeds.chain"})
        key = _draw_key(run, "inference.npe.bank", built,
                        {"seed": {"from": "runtime.seeds.bank"}})
        assert bool(jnp.all(jax.random.key_data(key)
                            == jax.random.key_data(jax.random.key(5))))

    def test_an_EMPTY_spec_is_refused_rather_than_falling_back(self):
        """`spec={}` must not quietly borrow the run's own seed.

        `if spec is None` and `if not spec` are indistinguishable on every
        other call this plan makes -- nothing passes an empty mapping -- and
        they differ exactly here: under the truthy form a declared but empty
        `inference.npe:` subsection draws from the RUN's seed instead, and a
        bank the document meant to name is a bank provenance.json records
        under the wrong name.  Measured: with `if not spec` this is the only
        test in the module that goes red.
        """
        built = conjugate_built(seeds={"chain": 3})
        run = spec(kind="npe", seed={"from": "runtime.seeds.chain"})
        with pytest.raises(ConfigError, match="'seed' is required"):
            _draw_key(run, "inference.npe.bank", built, {})


class TestTheLazyImportInvariant:
    """The invariant this plan is most able to break, as behaviour.

    A text guard (`"import numpyro" not in the module head`) was written
    first and is NOT what ships: the module docstrings talk about the very
    import they forbid, so the guard was satisfiable by prose and fired on
    it.  That is 2C shape 9 -- a verification method with the code's blind
    spot.  A subprocess that imports and looks at `sys.modules` has no blind
    spot.
    """

    def test_importing_the_config_layer_leaves_numpyro_out(self):
        import subprocess
        import sys

        out = subprocess.run(
            [sys.executable, "-c",
             "import sys; import rheplicant.config; "
             "print('numpyro' in sys.modules)"],
            capture_output=True, text=True, check=True)
        assert out.stdout.strip() == "False", out.stdout

    def test_importing_nuts_alone_leaves_numpyro_out(self):
        import subprocess
        import sys

        out = subprocess.run(
            [sys.executable, "-c",
             "import sys; import rheplicant.config.sections.nuts; "
             "print('numpyro' in sys.modules)"],
            capture_output=True, text=True, check=True)
        assert out.stdout.strip() == "False", out.stdout


def test_conjugate_document_still_builds():
    assert load_document(conjugate_document()) is not None
