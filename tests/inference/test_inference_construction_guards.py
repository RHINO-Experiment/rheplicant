"""Construction-time refusals for malformed configuration, and what they miss.

A full-suite coverage run found eighteen ``raise`` statements in this family
that the suite had never executed: guards that read a constructor argument --
or a function's keyword -- decide it cannot mean anything, and refuse. Nothing
here is subtle physics. It is the layer that turns ``n_components=0`` into a
sentence instead of a shape error inside an MLP three frames later.

**Why this file is not built the way ``tests/radio/test_coords_guard_family.py``
is built.** That file derives its population from the source, and it should: its
family is one sentence copy-pasted nine times, so enumerating the carriers and
writing one case per carrier reaches every raise in the family. This family does
not have that shape, and the difference is measured rather than assumed.
Enumerating ``__check_init__`` with ``ast`` finds **31 classes carrying 70
raises between them, spread over 62 distinct messages**. One case per class
therefore reaches at most 31 of 70 -- a table claiming to be the family would be
honest about under half of it. The guards are not interchangeable either:
"malformed" means a non-int for ``n_pix``, a wrong base class for ``projector``,
a shape disagreement for ``beam_alms``, an empty tuple for ``latents``. There is
no one malformed argument to feed all 31.

A ``__check_init__``-derived population would also have been quietly incomplete
in a second way: **five of the eighteen raises covered here are not in a
``__check_init__`` at all.** ``NeuralPosterior.create`` validates in a
classmethod, ``train_posterior`` in a module-level function, and
``ParameterSpace._resolve_targets`` at validate time rather than at
construction. A table built by scanning for ``__check_init__`` would have
excluded all five while looking complete.

**What the derived check IS worth here.** Of those 62 messages, 7 appear in more
than one class, and four of the seven are in this batch: ``ref_freq must be >
0`` exists three times and had been tested twice; ``n_pix must be a positive
int``, ``learning_rate must be > 0`` and ``n_steps must be a positive int``
exist twice each and had each been tested once. That is the coords pathology at
a smaller scale, and :func:`test_the_copied_guards_have_not_grown_a_new_copy`
is scoped to exactly it. The census is a source scan, so it covers the radio
copies too and lives here rather than being cut in half across two files.

**One finding is recorded here rather than fixed**: six of these guards compare
against a threshold, and ``nan`` fails every comparison, so a NaN configuration
is accepted in silence. See :class:`TestNaNIsNotRefusedByComparisonGuards`.
"""

import ast
import pathlib
from typing import ClassVar

import jax
import jax.numpy as jnp
import pytest

import rheplicant
from rheplicant.core.errors import ParameterSpaceError, StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State
from rheplicant.inference.calibrate import AdamCalibrator, GradientCalibrator
from rheplicant.inference.noise import RadiometerNoise
from rheplicant.inference.npe import NeuralPosterior, train_posterior
from rheplicant.inference.parameters import Bind, Latent, ParameterSpace

# --------------------------------------------------------------------------
# The derived half: guards that exist in more than one class.
# --------------------------------------------------------------------------

_SRC = pathlib.Path(rheplicant.__file__).parent

#: Message template -> every class whose ``__check_init__`` raises it. Four
#: entries, because four is how many of this batch's messages are copy-pasted;
#: the other fourteen are singletons and get hand-written cases below. Each
#: value is checked against the source, so a fourth ``ref_freq`` copy fails here
#: instead of joining the copies that had no test.
COPIED_GUARDS: dict[str, set[str]] = {
    "ref_freq must be > 0, got {}.": {
        "IonosphereOperator",
        "ForegroundOperator",
        "PowerLawSkyModel",
    },
    "n_pix must be a positive int, got {}.": {
        "UniformSkyModel",
        "PowerLawSkyModel",
    },
    "learning_rate must be > 0, got {}.": {
        "GradientCalibrator",
        "AdamCalibrator",
    },
    "n_steps must be a positive int, got {}.": {
        "GradientCalibrator",
        "AdamCalibrator",
    },
}


def _message_template(node: ast.Raise) -> str:
    """The raise's message with every interpolation collapsed to ``{}``.

    Two copies of a guard differ only in which ``self.x`` they interpolate, so
    the literal parts are what identifies them as the same sentence.
    """
    exc = node.exc
    if not isinstance(exc, ast.Call) or not exc.args:
        return ""
    parts: list[str] = []
    stack = [exc.args[0]]
    while stack:
        item = stack.pop(0)
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            parts.append(item.value)
        elif isinstance(item, ast.JoinedStr):
            stack = list(item.values) + stack
        elif isinstance(item, ast.BinOp):  # implicit concatenation of f-strings
            stack = [item.left, item.right] + stack
        elif isinstance(item, ast.FormattedValue):
            parts.append("{}")
    return "".join(parts)


def _classes_whose_check_init_says(template: str) -> set[str]:
    """Every class in ``src/`` with that sentence in its ``__check_init__``."""
    found: set[str] = set()
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if not (isinstance(item, ast.FunctionDef) and item.name == "__check_init__"):
                    continue
                for raised in ast.walk(item):
                    if isinstance(raised, ast.Raise) and raised.exc is not None:
                        if _message_template(raised) == template:
                            found.add(node.name)
    return found


@pytest.mark.parametrize("template", sorted(COPIED_GUARDS))
def test_the_copied_guards_have_not_grown_a_new_copy(template):
    """Derived from the source, for the four guards that are genuinely copies.

    ``ref_freq must be > 0`` was written three times and tested twice; the
    untested copy was the one in a file nobody was working in. A fourth copy
    added to a new operator fails here, naming itself, rather than repeating
    that.
    """
    assert _classes_whose_check_init_says(template) == COPIED_GUARDS[template], {
        "message": template,
        "carry it but are not listed": sorted(
            _classes_whose_check_init_says(template) - COPIED_GUARDS[template]
        ),
        "listed but no longer carry it": sorted(
            COPIED_GUARDS[template] - _classes_whose_check_init_says(template)
        ),
    }


def test_the_two_calibrators_are_both_exercised_by_this_file():
    """The copied-guard census is only useful if the copies are then tripped.

    ``GradientCalibrator``'s copies were covered and ``AdamCalibrator``'s were
    not, which is the only reason this file exists for calibrate.py at all. Both
    are refused here so the pair cannot drift back apart.
    """
    for cls in (GradientCalibrator, AdamCalibrator):
        with pytest.raises(StateValidationError, match="learning_rate must be > 0"):
            cls(learning_rate=-1.0)
        with pytest.raises(StateValidationError, match="n_steps must be a positive int"):
            cls(n_steps=0)


# --------------------------------------------------------------------------
# AdamCalibrator: three refusals, one exception type between them.
# --------------------------------------------------------------------------

#: One case per distinct sentence ``AdamCalibrator.__check_init__`` can produce.
#: All three are ``StateValidationError``, so the substrings are checked against
#: each OTHER as well as against their own case -- three matches on one
#: over-broad message would otherwise all pass.
ADAM_REFUSALS: dict[str, tuple[dict, str]] = {
    "learning_rate <= 0": ({"learning_rate": -1e-3}, "learning_rate must be > 0"),
    "n_steps < 1": ({"n_steps": 0}, "n_steps must be a positive int"),
    "beta outside [0, 1)": ({"beta1": 1.0}, "beta1/beta2 must be in [0, 1)"),
}

#: Further inputs that must reach the SAME sentence as one of the above. These
#: are the branches of a guard rather than separate guards: ``n_steps`` refuses
#: both non-ints and non-positives, and one condition covers beta1 and beta2.
ADAM_SAME_SENTENCE: list[tuple[dict, str]] = [
    ({"learning_rate": 0.0}, "learning_rate must be > 0"),  # the excluded endpoint
    ({"n_steps": 100.0}, "n_steps must be a positive int"),  # a float, not a size
    ({"n_steps": -7}, "n_steps must be a positive int"),
    ({"beta2": 1.0}, "beta1/beta2 must be in [0, 1)"),  # the excluded endpoint
    ({"beta1": -0.5}, "beta1/beta2 must be in [0, 1)"),
]


@pytest.mark.parametrize("case", sorted(ADAM_REFUSALS))
def test_adam_refuses_the_malformed_setting(case):
    kwargs, expected = ADAM_REFUSALS[case]
    with pytest.raises(StateValidationError) as excinfo:
        AdamCalibrator(**kwargs)
    assert expected in str(excinfo.value), str(excinfo.value)


@pytest.mark.parametrize(("kwargs", "expected"), ADAM_SAME_SENTENCE)
def test_adam_reaches_the_same_sentence_by_the_other_branch(kwargs, expected):
    with pytest.raises(StateValidationError) as excinfo:
        AdamCalibrator(**kwargs)
    assert expected in str(excinfo.value), str(excinfo.value)


def test_adams_three_refusals_do_not_share_a_message():
    """The check that stops one over-broad sentence from satisfying all three.

    Every test above asserts a substring is present. If ``__check_init__`` were
    rewritten to raise a single "invalid Adam settings (learning_rate, n_steps,
    beta1, beta2)" for everything, all three would still pass and the user would
    be told nothing. So each case's substring is required to be ABSENT from the
    other two messages.
    """
    messages = {}
    for case, (kwargs, _) in ADAM_REFUSALS.items():
        with pytest.raises(StateValidationError) as excinfo:
            AdamCalibrator(**kwargs)
        messages[case] = str(excinfo.value)

    for case, (_, expected) in ADAM_REFUSALS.items():
        for other, message in messages.items():
            if other == case:
                continue
            assert expected not in message, (case, other, message)


def test_adam_accepts_the_boundary_settings_and_runs():
    """The other branch, at the boundaries the guards name.

    ``n_steps=1`` is the smallest accepted size and ``beta=0.0`` is the closed
    end of ``[0, 1)`` -- both one step from a refusal. Constructing is not
    enough: a guard collapsed to ``raise`` unconditionally would fail the tests
    above too, but a guard whose CONDITION was inverted would let this construct
    and then produce nothing usable, so the calibrator is actually run.
    """
    calibrator = AdamCalibrator(learning_rate=1e-2, n_steps=1, beta1=0.0, beta2=0.0)

    # Asymmetric weights: a transposed or partially-applied forward would give a
    # different loss, which a symmetric [1, 1, 1] would hide.
    weights = jnp.array([2.0, 5.0, 11.0])
    params0 = jnp.array([0.3, -0.7, 1.1])
    observed = weights * jnp.array([1.0, 2.0, 3.0])

    fitted, losses = calibrator.fit(lambda p: weights * p, params0, observed)
    assert losses.shape == (1,)
    assert jnp.all(jnp.isfinite(fitted))
    assert not jnp.allclose(fitted, params0)  # a step was actually taken


def test_a_positive_int_that_is_a_bool_is_accepted_as_a_size():
    """``isinstance(True, int)`` is True, so ``n_steps=True`` means one step.

    Recorded, not asserted as desirable: it is what the guard says, and pinning
    it means a future ``type(...) is int`` tightening shows up as a change here
    rather than as a mystery in somebody's notebook.
    """
    assert AdamCalibrator(n_steps=True).n_steps is True


# --------------------------------------------------------------------------
# RadiometerNoise: the floor.
# --------------------------------------------------------------------------


def test_a_negative_floor_is_refused():
    with pytest.raises(StateValidationError) as excinfo:
        RadiometerNoise(channel_width=1e5, integration_time=2.0, floor=-1e-6)
    assert "floor must be >= 0" in str(excinfo.value)


def test_the_floor_guard_and_the_bandwidth_guard_say_different_things():
    """Both are ``StateValidationError`` from the same ``__check_init__``."""
    with pytest.raises(StateValidationError) as negative_floor:
        RadiometerNoise(channel_width=1e5, integration_time=2.0, floor=-1e-6)
    with pytest.raises(StateValidationError) as bad_bandwidth:
        RadiometerNoise(channel_width=0.0, integration_time=2.0)

    assert "floor must be >= 0" in str(negative_floor.value)
    assert "floor must be >= 0" not in str(bad_bandwidth.value)
    assert "positive channel_width" in str(bad_bandwidth.value)
    assert "positive channel_width" not in str(negative_floor.value)


@pytest.mark.parametrize("floor", [0.0, 1e-9, 4.0])
def test_a_non_negative_floor_is_accepted_and_is_applied(floor):
    """The other branch, including the boundary ``floor=0.0`` the guard admits.

    The prediction is deliberately below the largest floor tested, so the two
    accepted regimes give numerically different answers: without the floor the
    std tracks the prediction, with it the std is pinned.
    """
    noise = RadiometerNoise(channel_width=1e5, integration_time=2.0, floor=floor)
    std = noise.std(jnp.array([0.5, 2.5]))
    expected = jnp.maximum(jnp.array([0.5, 2.5]), floor) * noise.fractional
    assert jnp.allclose(std, expected)


# --------------------------------------------------------------------------
# npe: entry validation in a classmethod and in a module-level function.
# --------------------------------------------------------------------------

#: A bank small enough to build an estimator on in milliseconds. Twelve pairs is
#: chosen so ``round(validation_fraction * 12)`` crosses 1 between 0.04 and
#: 0.05, which is what pins the "holds out zero" boundary below.
N_BANK = 12


@pytest.fixture
def bank():
    """Asymmetric thetas and data -- no row or column is a copy of another."""
    thetas = jnp.linspace(-1.0, 3.0, N_BANK * 2).reshape(N_BANK, 2)
    data = jnp.linspace(7.0, 19.0, N_BANK * 3).reshape(N_BANK, 3)
    return thetas, data


@pytest.fixture
def estimator(bank):
    thetas, data = bank
    return NeuralPosterior.create(
        thetas, data, key=jax.random.key(0), n_components=1, width=4, depth=1
    )


@pytest.mark.parametrize("n_components", [0, -3])
def test_a_non_positive_component_count_is_refused(bank, n_components):
    thetas, data = bank
    with pytest.raises(StateValidationError) as excinfo:
        NeuralPosterior.create(
            thetas, data, key=jax.random.key(0), n_components=n_components, width=4, depth=1
        )
    assert "n_components must be positive" in str(excinfo.value)


def test_one_component_is_accepted_and_is_a_working_estimator(bank):
    """The boundary the guard admits -- and the documented right answer.

    ``n_components=1`` is exact for a Gaussian posterior, so this is not an edge
    case being tolerated; it is the setting the module's own docstring
    recommends. It has to build and evaluate, not merely construct.
    """
    thetas, data = bank
    q = NeuralPosterior.create(
        thetas, data, key=jax.random.key(0), n_components=1, width=4, depth=1
    )
    assert q.n_components == 1
    assert jnp.isfinite(q.log_prob(thetas[0], data[0]))


@pytest.mark.parametrize("n_steps", [0, -1])
def test_a_non_positive_step_count_is_refused(estimator, bank, n_steps):
    thetas, data = bank
    with pytest.raises(StateValidationError) as excinfo:
        train_posterior(estimator, thetas, data, key=jax.random.key(1), n_steps=n_steps)
    assert "n_steps must be positive" in str(excinfo.value)


@pytest.mark.parametrize("validation_fraction", [1.0, 1.5, -0.1])
def test_a_validation_fraction_outside_the_unit_interval_is_refused(
    estimator, bank, validation_fraction
):
    """``1.0`` is the excluded endpoint: holding out everything trains on none."""
    thetas, data = bank
    with pytest.raises(StateValidationError) as excinfo:
        train_posterior(
            estimator,
            thetas,
            data,
            key=jax.random.key(1),
            n_steps=1,
            validation_fraction=validation_fraction,
        )
    assert "validation_fraction must be in [0, 1)" in str(excinfo.value)


def test_a_fraction_too_small_for_the_bank_is_a_different_refusal(estimator, bank):
    """The second refusal about the same argument, and it must not be the first.

    ``0.04`` is inside ``[0, 1)``, so it passes the range guard and is then
    refused for a different reason: ``round(0.04 * 12) == 0`` holds out no
    simulations at all, which would silently train without a validation split
    while the caller believed there was one. Asserting only "some
    StateValidationError mentioning validation_fraction" would not tell these
    two apart.
    """
    thetas, data = bank
    with pytest.raises(StateValidationError) as excinfo:
        train_posterior(
            estimator, thetas, data, key=jax.random.key(1), n_steps=1, validation_fraction=0.04
        )
    message = str(excinfo.value)
    assert "holds out zero of" in message
    assert str(N_BANK) in message
    assert "must be in [0, 1)" not in message


def test_the_range_refusal_is_not_the_holds_out_zero_refusal(estimator, bank):
    """The converse direction of the test above."""
    thetas, data = bank
    with pytest.raises(StateValidationError) as excinfo:
        train_posterior(
            estimator, thetas, data, key=jax.random.key(1), n_steps=1, validation_fraction=1.0
        )
    assert "holds out zero of" not in str(excinfo.value)


@pytest.mark.parametrize("validation_fraction", [0.0, 0.05, 0.5])
def test_the_accepted_fractions_either_side_of_the_boundary_train(
    estimator, bank, validation_fraction
):
    """The other branch, straddling both boundaries at once.

    ``0.0`` is the admitted end of ``[0, 1)`` AND the value that switches the
    held-out split off, so it is the one input that must pass both guards for
    opposite reasons. ``0.05`` is the smallest fraction that holds out a
    simulation from a bank of twelve -- one step above the ``0.04`` refused
    above, which is what makes that refusal a boundary rather than an assertion
    about a large number.
    """
    thetas, data = bank
    trained, history = train_posterior(
        estimator,
        thetas,
        data,
        key=jax.random.key(1),
        n_steps=1,
        validation_fraction=validation_fraction,
    )
    assert history.train.shape == (1,)
    assert jnp.isfinite(trained.log_prob(thetas[0], data[0]))


# --------------------------------------------------------------------------
# parameters: Latent, Bind, ParameterSpace -- and one validate-time refusal.
# --------------------------------------------------------------------------

class _ShapedPrior:
    """Duck-typed stand-in for a NumPyro distribution.

    ``Latent`` reads ``prior.shape()`` through ``getattr`` and ``callable`` so
    that declaring a space costs no numpyro import. A stub is therefore a
    faithful input, and keeps this file's import list honest about what the
    guard actually depends on.
    """

    def __init__(self, shape: tuple[int, ...]):
        self._shape = shape

    def shape(self) -> tuple[int, ...]:
        return self._shape


#: One case per distinct sentence, across three classes that all raise
#: ``ParameterSpaceError``. Grouped by class so the disjointness check below can
#: compare only the messages a single class can produce -- which is where an
#: over-broad message would actually be written. A class contributing only one
#: sentence would make that check vacuous, which is why ``Latent``'s second
#: guard is here even though it was already covered.
PARAMETER_REFUSALS: dict[str, list[tuple[str, object, str]]] = {
    "Latent": [
        ("empty name", lambda: Latent("", init=1.0), "Latent name must be a non-empty string"),
        (
            "prior disagrees with init",
            lambda: Latent("gain", init=1.0, prior=_ShapedPrior((3,))),
            "prior has shape (3,) but init has shape ()",
        ),
    ],
    "Bind": [
        ("no latents", lambda: Bind(latents=[], into=lambda p: p.gain),
         "Bind needs at least one latent name"),
        ("non-string latent", lambda: Bind(latents=[7], into=lambda p: p.gain),
         "Bind latent names must be strings"),
        ("non-callable selector", lambda: Bind("gain", into=[7]),
         "`into` must hold callables"),
    ],
    "ParameterSpace": [
        ("no latents", lambda: ParameterSpace(latents=[]),
         "needs at least one Latent"),
        ("no bindings", lambda: ParameterSpace(latents=[Latent("gain", init=2.0)]),
         "needs bindings"),
    ],
}

#: Inputs that must reach a sentence already in the table above. ``Latent``
#: refuses a name that is empty and a name that is not a string with one
#: condition and one message, so this is a second branch, not a second guard --
#: keeping it out of the table above is what lets the disjointness check there
#: be strict.
PARAMETER_SAME_SENTENCE: list[tuple[str, object, str]] = [
    (
        "Latent: a non-string name",
        lambda: Latent(7, init=1.0),
        "Latent name must be a non-empty string",
    ),
    (
        "Bind: a non-string among several latent names",
        lambda: Bind(latents=["gain", 7], into=lambda p: p.gain, fn=lambda a, b: a),
        "Bind latent names must be strings",
    ),
]

_ALL_PARAMETER_CASES = [
    (owner, case, build, expected)
    for owner, cases in PARAMETER_REFUSALS.items()
    for case, build, expected in cases
]


@pytest.mark.parametrize(
    ("owner", "case", "build", "expected"),
    _ALL_PARAMETER_CASES,
    ids=[f"{owner}:{case}" for owner, case, _, _ in _ALL_PARAMETER_CASES],
)
def test_a_malformed_declaration_is_refused(owner, case, build, expected):
    with pytest.raises(ParameterSpaceError) as excinfo:
        build()
    assert expected in str(excinfo.value), str(excinfo.value)


@pytest.mark.parametrize(
    ("case", "build", "expected"),
    PARAMETER_SAME_SENTENCE,
    ids=[case for case, _, _ in PARAMETER_SAME_SENTENCE],
)
def test_a_declaration_reaches_the_same_sentence_by_the_other_branch(case, build, expected):
    with pytest.raises(ParameterSpaceError) as excinfo:
        build()
    assert expected in str(excinfo.value), str(excinfo.value)


@pytest.mark.parametrize("owner", sorted(PARAMETER_REFUSALS))
def test_one_classes_refusals_do_not_share_a_message(owner):
    """Within a class, no case's substring may match another case's message."""
    messages = {}
    for case, build, _ in PARAMETER_REFUSALS[owner]:
        with pytest.raises(ParameterSpaceError) as excinfo:
            build()
        messages[case] = str(excinfo.value)

    for case, _, expected in PARAMETER_REFUSALS[owner]:
        for other, message in messages.items():
            if other == case:
                continue
            assert expected not in message, (owner, case, other, message)


def test_the_minimal_well_formed_declaration_is_accepted():
    """The other branch: one latent, one binding, exactly at each guard's edge.

    ``ParameterSpace`` refuses an empty ``latents`` and an empty ``bindings``,
    so one of each is the boundary. If either guard's condition were inverted
    this would be refused, which none of the tests above would notice.
    """
    space = ParameterSpace(
        latents=[Latent("gain", init=2.0)],
        bindings=[Bind("gain", into=lambda p: p.gain)],
    )
    assert space.latents[0].name == "gain"
    assert len(space.bindings) == 1


class _ScaleOperator(AbstractOperator):
    """A pipeline with one array leaf and one leaf that is not an array.

    ``knob`` is deliberately not ``eqx.field(static=True)``: it is a pytree leaf,
    so an ``into`` selector reaches it and the "landed on static configuration"
    refusal does NOT fire -- which is what makes it the input for the "not an
    array leaf" refusal specifically.
    """

    gain: jax.Array
    knob: float

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)

    def __call__(self, state: State) -> State:
        return state.with_data(state.data * self.gain)


def test_a_selector_landing_on_a_non_array_leaf_is_refused():
    """Not a construction guard: ``ParameterSpace`` cannot see the pipeline yet.

    This one fires at ``validate`` time, which is why a population derived from
    ``__check_init__`` would have missed it. The refusal has to name the type it
    found, or the user has no way to tell this apart from the neighbouring
    refusal about static configuration.
    """
    pipeline = _ScaleOperator(gain=jnp.array(2.0), knob=3.0)
    space = ParameterSpace(
        latents=[Latent("gain", init=1.0)],
        bindings=[Bind("gain", into=lambda p: p.knob)],
    )
    with pytest.raises(ParameterSpaceError) as excinfo:
        space.validate(pipeline)
    message = str(excinfo.value)
    assert "not an array leaf" in message
    assert "float" in message
    assert "static configuration" not in message


def test_the_same_selector_on_an_array_leaf_validates():
    """The other branch, one attribute away from the refusal above."""
    pipeline = _ScaleOperator(gain=jnp.array(2.0), knob=3.0)
    space = ParameterSpace(
        latents=[Latent("gain", init=1.0)],
        bindings=[Bind("gain", into=lambda p: p.gain)],
    )
    assert space.validate(pipeline) is None


# --------------------------------------------------------------------------
# The finding.
# --------------------------------------------------------------------------


class TestNaNIsRefusedByEveryComparisonGuard:
    """Every numeric configuration guard in this family refuses NaN.

    It did not always. ``nan <= 0`` is False, so ``if x <= 0: raise`` does not
    fire and the NaN quietly becomes configuration -- and this class was first
    written to pin that as a **gap**, with the one-line remedy named in each
    docstring. The remedy has since been applied at all seven sites
    (``if not x > 0``), and these tests now pin the contract instead. That
    transition is the reason they were written down rather than reported.

    The worst of the seven is worth keeping in view, because it is the reason
    "NaN poisons the output anyway" is not a defence. ``RadiometerNoise.std``
    applies its floor under ``if self.floor > 0.0``, which is ALSO False for
    NaN -- so ``floor=nan`` used to yield a noise model that was finite,
    correctly shaped, and simply un-floored, with nothing downstream to hint
    the argument had been dropped. Every other case at least produced a NaN
    somewhere.

    The package already knew the safe form and used it twice
    (``if not 0.0 <= self.beta1 < 1.0``, ``if not 0.0 <= validation_fraction <
    1.0``); those are pinned below too, so the contrast that motivated the fix
    stays visible.
    """

    NAN = float("nan")

    def test_a_nan_learning_rate_is_refused_by_both_calibrators(self):
        """The guard is copy-pasted, so the fix had to be applied twice.

        Asserting both is what distinguishes a real fix from one applied to
        whichever copy the author happened to open.
        """
        for calibrator in (AdamCalibrator, GradientCalibrator):
            with pytest.raises(StateValidationError, match="learning_rate must be > 0"):
                calibrator(learning_rate=self.NAN)

    def test_a_nan_floor_is_refused_rather_than_silently_ignored(self):
        with pytest.raises(StateValidationError, match="floor must be >= 0"):
            RadiometerNoise(channel_width=1e5, integration_time=2.0, floor=self.NAN)

    def test_a_legitimate_floor_still_applies(self):
        """The other branch: the inversion must not have made the guard total.

        ``if not self.floor >= 0.0`` has to keep accepting 0.0 -- the boundary
        the message itself quotes -- and any positive floor.
        """
        for floor in (0.0, 0.25):
            noise = RadiometerNoise(channel_width=1e5, integration_time=2.0, floor=floor)
            assert float(noise.floor) == floor
            assert jnp.all(jnp.isfinite(noise.std(jnp.array([0.5, 2.5]))))

    def test_a_nan_channel_width_is_refused_before_it_can_poison_a_weight(self):
        with pytest.raises(StateValidationError, match="channel_width"):
            RadiometerNoise(channel_width=self.NAN, integration_time=2.0)

    def test_a_nan_integration_time_is_refused_too(self):
        """The guard is an ``and`` over two fields; one inverted arm is not a fix."""
        with pytest.raises(StateValidationError, match="integration_time"):
            RadiometerNoise(channel_width=1e5, integration_time=self.NAN)

    def test_the_isinstance_guards_do_refuse_nan(self):
        """Not every guard here has the gap -- these refuse NaN as a side effect.

        ``nan`` is a float, so ``not isinstance(n_steps, int)`` fires before any
        comparison is reached. The refusal is correct by accident of the type
        check, not by design, which is why it is worth pinning separately from
        the inverted-comparison guards below.
        """
        with pytest.raises(StateValidationError, match="n_steps must be a positive int"):
            AdamCalibrator(n_steps=self.NAN)

    @pytest.mark.parametrize("field", ["beta1", "beta2"])
    def test_the_inverted_comparison_guard_refuses_nan(self, field):
        """``if not 0.0 <= x < 1.0`` is the one-line form the six sites need."""
        with pytest.raises(StateValidationError, match=r"beta1/beta2 must be in \[0, 1\)"):
            AdamCalibrator(**{field: self.NAN})

    def test_a_nan_validation_fraction_is_refused(self, estimator, bank):
        """The same inverted form, in ``train_posterior``. It works there too."""
        thetas, data = bank
        with pytest.raises(StateValidationError, match=r"must be in \[0, 1\)"):
            train_posterior(
                estimator,
                thetas,
                data,
                key=jax.random.key(1),
                n_steps=1,
                validation_fraction=self.NAN,
            )

    def test_a_nan_component_count_is_refused_by_the_guard_written_for_it(self, bank):
        """It used to be refused by ``eqx.nn.MLP`` three frames deeper.

        ``nan < 1`` is False, so ``n_components must be positive`` never fired
        and the caller got a ``TypeError`` about shape sequences instead of the
        sentence written for them. The inversion means the sentence arrives.
        """
        thetas, data = bank
        with pytest.raises(StateValidationError, match="n_components must be positive"):
            NeuralPosterior.create(
                thetas, data, key=jax.random.key(0), n_components=self.NAN, width=4, depth=1
            )
