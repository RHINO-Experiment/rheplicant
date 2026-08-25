"""The surfaces bayesmith imports, guarded here because the symptom lands there.

The migration spec (bayesmith's
``docs/superpowers/specs/2026-08-24-rheplicant-migration.md``, §六 step 5)
names three parts of this package as **published across repositories**:
changing any one of them is a breaking change for bayesmith. It also says why
a note is not enough -- 散文挡不住，守卫才行, prose cannot stop it, only a
guard can.

That is not rhetoric, and the reason is worth stating exactly: **every symptom
of breaking one of these lands in the other repository.** Rename a keyword
argument here, or rebuild an exception class instead of re-exporting it, and
this suite stays green, because nothing inside this package imports these
names the way an outside consumer does. §六 step 2 is about to hollow
``rheplicant.inference`` into a thin shell that re-exports bayesmith, and a
shell is precisely the edit that preserves behaviour while destroying
identity: ``except ParameterSpaceError`` raises nothing when the class it
names has been re-created rather than re-exported. The handler simply stops
firing, on the far side of a repository boundary.

The three, as the spec states them:

1. **The ``build_forward_fn`` seam** (``inference/forward.py``) -- the route
   by which a whole ``Pipeline`` attaches as one deterministic node, 零适配层,
   with no adaptation layer. Its published import path, parameter names, order
   and default are the contract. Its *behaviour* is
   ``tests/inference/test_forward.py``'s subject and is deliberately not
   re-tested here; this file is about the surface, which that file does not
   pin.
2. **The ``core`` exception classes' shared identity.** An ``except`` clause
   compares the class object, not its name, so identity is the whole contract.
3. **``AbstractOperator.__call__`` does structural validation only** -- the
   premise of function-tracing safety, without which a ``Pipeline`` cannot be
   used as a node at all.

**Which of the three is exercised today, measured rather than assumed.**
Contracts 2 and 3 are live: bayesmith imports ``ParameterSpaceError`` and
``StateValidationError`` from ``rheplicant.core.errors``, and
``src/bayesmith/graph/nodes.py`` attaches an ``eqx.Module`` -- "a whole
rheplicant ``Pipeline`` is the motivating case" -- as a node's ``fn``,
invoked through ``__call__``. Contract 1 is **designed against but not yet
imported**: ``build_forward_fn`` appears in bayesmith's docs and in no
``.py`` file there. It is pinned anyway, because "published" is a statement
about what an outside consumer may rely on, and the seam is the documented
route for the case bayesmith already builds by hand.

**What this file does NOT cover.** Contract 3 is checked at the seam, on one
representative pipeline, plus a positive control that proves the check can
fail. It is not a sweep of every shipped operator: constructing a valid
``State`` for each is ``tests/radio/``'s subject, and a sweep whose fixtures
were wrong would report on its fixtures rather than on the operators. So a
value check on the path bayesmith traces is caught here, and one in an
operator that path never reaches is not.
"""

import inspect

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import rheplicant
import rheplicant.core.errors as core_errors
from rheplicant import Pipeline
from rheplicant.core.operator import AbstractOperator
from rheplicant.radio import GainOperator, SkyOperator


@pytest.fixture
def published_pipeline():
    """The shape bayesmith attaches as a deterministic node.

    Deterministic on purpose: a stochastic stage is refused at the seam, and
    a frozen draw is the defect that refusal exists for. 3 and 5 rather than
    two equal numbers, so that a swapped stage is a different gradient and not
    merely a different order.
    """
    return Pipeline(
        SkyOperator(amplitude=jnp.array(3.0)),
        GainOperator(gain=jnp.array(5.0)),
        names=("sky", "gain"),
    )


# --------------------------------------------------------------- contract 1


class TestTheForwardSeamKeepsItsPublishedSurface:
    """Path, name, order, default. Not behaviour -- see the module docstring."""

    def test_the_published_path_and_the_defining_module_hold_one_object(self):
        """``from rheplicant.inference import build_forward_fn`` is the
        published spelling; ``inference.forward`` is where it lives.

        These are asserted to be the same object rather than merely both
        importable, because the thin shell of §六 step 2 is free to make them
        differ -- a deprecating wrapper at the package level over an untouched
        function in the module would satisfy both imports and hand a consumer
        two different callables depending on which line it wrote.
        """
        from rheplicant.inference import build_forward_fn
        from rheplicant.inference.forward import build_forward_fn as defined

        assert build_forward_fn is defined

    def test_the_seam_keeps_its_parameter_names_order_and_default(self):
        """Positional order and keyword names are both part of the contract.

        An outside caller may write either spelling, so neither may move.
        The default is pinned as the *object* ``eqx.is_inexact_array``: it is
        what makes "every floating-point leaf is trainable" true without the
        caller saying so, and swapping it for an equivalent-looking predicate
        would silently change which leaves a consumer gets back.
        """
        from rheplicant.inference import build_forward_fn

        parameters = inspect.signature(build_forward_fn).parameters
        assert list(parameters) == ["pipeline", "state_template", "filter_spec"]
        assert parameters["pipeline"].default is inspect.Parameter.empty
        assert parameters["state_template"].default is inspect.Parameter.empty
        assert parameters["filter_spec"].default is eqx.is_inexact_array

    def test_the_seam_returns_the_pair_a_consumer_unpacks(self, published_pipeline, template_state):
        """``(forward, params0)``, with ``forward`` a one-argument callable.

        The arity is the contract an engine writes its inner loop against;
        a seam that grew a second required argument would break every caller
        while every assertion about the returned value stayed true.
        """
        from rheplicant.inference import build_forward_fn

        built = build_forward_fn(published_pipeline, template_state)
        assert isinstance(built, tuple) and len(built) == 2
        forward, params0 = built
        assert callable(forward)
        assert len(inspect.signature(forward).parameters) == 1
        assert jax.tree.leaves(params0), "no trainable leaves came back"


# --------------------------------------------------------------- contract 2


def _published_exceptions() -> dict[str, type]:
    """Every exception class ``rheplicant.core.errors`` defines.

    Read off the module, never a list kept here. A hand-kept list is the
    failure this repository has already had once: a gate that could not see a
    page disappear was guarding a variable, not the repository. The same shape
    applies to a new exception class -- one added tomorrow is checked by these
    assertions without anyone remembering to add it.
    """
    return {
        name: obj
        for name, obj in vars(core_errors).items()
        if inspect.isclass(obj)
        and issubclass(obj, BaseException)
        and obj.__module__ == core_errors.__name__
    }


class TestTheExceptionClassesKeepTheirIdentity:
    def test_the_sweep_actually_sees_the_module(self):
        """A sweep that found nothing would pass every assertion below.

        The named three are the ones bayesmith imports today; the floor is
        there so that a module gutted down to one class is not read as a
        package that simply publishes fewer errors.
        """
        found = _published_exceptions()
        assert len(found) >= 8, sorted(found)
        assert {"ParameterSpaceError", "StateValidationError", "DirtError"} <= set(found)

    @pytest.mark.parametrize("name", sorted(_published_exceptions()))
    def test_each_stays_catchable_as_the_family_and_as_its_builtin(self, name):
        """Two ``except`` clauses an outside consumer is entitled to write.

        ``DirtError`` is the family catch the class docstring promises. The
        builtin base is the other half: a generic ``except ValueError`` in
        consumer code keeps working only while these keep deriving from one.
        ``DirtError`` itself is the root and derives from ``Exception``, which
        is why it is asked only for the first property.
        """
        klass = _published_exceptions()[name]
        assert issubclass(klass, core_errors.DirtError)
        if klass is not core_errors.DirtError:
            assert issubclass(klass, (ValueError, RuntimeError)), klass.__mro__

    @pytest.mark.parametrize("name", sorted(_published_exceptions()))
    def test_a_root_re_export_is_the_same_object_and_not_a_copy(self, name):
        """Where the root re-exports one, it must be the identical class.

        Not every one is re-exported -- ``ParameterSpaceError``, the class
        bayesmith imports most, is deliberately reachable only at
        ``rheplicant.core.errors``. So the assertion is conditional on the
        name being present, and the direction that matters is the one it
        does make: if ``rheplicant.X`` exists at all, it is not a second
        class with the same name. Two same-named classes are the shape that
        makes an ``except`` clause miss while every import succeeds.
        """
        klass = _published_exceptions()[name]
        at_root = getattr(rheplicant, name, None)
        if at_root is not None:
            assert at_root is klass
        assert (name in rheplicant.__all__) == (at_root is not None), (
            f"{name} is listed in rheplicant.__all__ but not reachable on the "
            f"package, or reachable but unlisted -- __all__ is what a reader "
            f"is told is public."
        )


# --------------------------------------------------------------- contract 3


class TestAPipelineStaysTraceableAsANode:
    """``__call__`` does structural validation only, checked where it is used.

    bayesmith stores a whole ``Pipeline`` in a node's ``fn`` field and invokes
    it through ``__call__`` under ``eqx.filter_jit`` and ``eqx.filter_grad``.
    Both halves are asserted, because they fail differently: a value check
    raises, while a leaf that stops being traceable returns a wrong number
    quietly.
    """

    def test_jit_reproduces_the_eager_result(self, published_pipeline, template_state):
        eager = published_pipeline(template_state).data
        traced = eqx.filter_jit(lambda model, state: model(state).data)(
            published_pipeline, template_state
        )
        assert jnp.array_equal(traced, eager)

    def test_the_gradient_reaches_the_parameters_inside_the_pipeline(
        self, published_pipeline, template_state
    ):
        """The silent half, and the reason this is asserted by value.

        bayesmith's ``graph/nodes.py`` records the trap in its own words: mark
        such a field static and ``filter_grad`` returns each parameter's
        *original value* in place of a gradient -- nothing raises, the answer
        is simply wrong. A test asserting only that gradients are finite and
        correctly shaped passes in exactly that case.

        So the expected values are derived from the fixture rather than
        written down: the pipeline computes ``amplitude * gain`` over every
        sample, so summing it gives ``n * amplitude * gain`` and the two
        gradients are ``n * gain`` and ``n * amplitude``. Both differ from the
        parameters themselves, which is what makes the pinned-value failure
        visible here.
        """
        amplitude, gain = 3.0, 5.0
        n = published_pipeline(template_state).data.size

        @eqx.filter_grad
        def total(model, state):
            return jnp.sum(model(state).data)

        gradient = total(published_pipeline, template_state)
        got = [float(leaf) for leaf in jax.tree.leaves(eqx.filter(gradient, eqx.is_inexact_array))]
        assert got == [n * gain, n * amplitude], got

    def test_a_value_check_inside_call_is_caught_here(self, template_state):
        """The positive control: proof this pair of tests can still fail.

        Without it the two above are a green light of unknown strength -- they
        pass on a compliant pipeline, and nothing shows they would notice a
        breach. So the breach is committed on purpose: a Python ``if`` on a
        traced array, which is exactly what the ``AbstractOperator`` docstring
        forbids and exactly what an author reaches for when adding a
        "harmless" sanity check.

        Declared inside the test rather than at module level on purpose.
        ``tests/config/test_config_delivery.py`` walks
        ``AbstractOperator.__subclasses__()`` with no module filter, so a
        contract-violating class living at module scope would enter a
        package-wide census as though the package shipped it. A local class is
        collected once the test returns.
        """

        class ValueCheckingOperator(AbstractOperator):
            """Breaks the contract deliberately; see the enclosing test."""

            limit: jax.Array

            def __call__(self, state):
                if jnp.any(state.data > self.limit):  # the forbidden branch
                    raise ValueError("data exceeds the limit")
                return state

        offending = Pipeline(
            SkyOperator(amplitude=jnp.array(3.0)),
            ValueCheckingOperator(limit=jnp.array(1.0)),
            names=("sky", "check"),
        )
        with pytest.raises(jax.errors.TracerBoolConversionError):
            eqx.filter_jit(lambda model, state: model(state).data)(offending, template_state)
