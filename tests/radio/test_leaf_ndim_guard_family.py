"""Three shape guards, copy-pasted across six operators, tested on one.

An operator validates two things about the arrays it holds and receives: the
rank of its own differentiable leaves, and the rank of ``state.data``. Both
checks are written as a sentence, and both sentences have been pasted from
operator to operator. A full-suite coverage run found that the only member of
each family whose ``raise`` had ever executed was ``AntennaLossOperator`` --
the file someone happened to be working in. Its siblings carry byte-identical
guards that no test has ever reached.

The three families, by their sentence:

``A``  ``"{leaf} must be scalar or (n_freq,), got ndim=..."``
       BeamSpill and AntennaLoss. ``CalLoadOperator`` was a member and LEFT
       when ``t_load`` gained a per-sample column, which the source-derived
       membership check is what noticed -- a member leaving is as visible as
       one arriving. Its own forms are pinned in ``TestCalLoadRankForms``.
       Note the two remaining members do not raise at the same moment: the
       check sits in ``__check_init__``, but the family's third member used to
       raise in ``__call__``, so the helper below constructs *and* calls before
       deciding nothing was refused. Splitting that per-member is what let the
       difference hide.

``B``  ``"{Operator}: gain must be scalar or 1D, got ndim=..."``
       Gain, ApplyCalibration. These two sentences used to be
       **byte-identical** -- same leaf name, no operator name, nothing to tell
       apart a forward-model gain from its inverse at the far end of the
       pipeline. Both now interpolate ``type(self).__name__``;
       ``test_the_gain_refusals_name_which_operator_raised_them`` asserts they
       are pairwise DISTINCT, because an interpolation resolving to a constant
       would satisfy "names an operator" and restore the collision.

``C``  ``"{Operator} expects (n_time, n_freq) data; got ..."``
       NoiseWave, BeamSpill, AntennaLoss. The operator name used to be
       **hardcoded into the f-string** rather than read from
       ``type(self).__name__`` -- the copy-paste-that-forgot-to-rename hazard
       in its purest form, since a wrong name still raises, still raises the
       right exception type, and still matches ``got ndim=`` while sending the
       reader to a different file. Now interpolated, so the hazard is
       structurally gone; ``test_the_data_refusal_names_the_operator_that_
       raised_it`` remains the check that would catch a regression.

Each family's population is derived by parsing the shipped source, not by
trusting the tables below, so a seventh copy cannot be added untested.

Fixture shapes are ``n_time=5``, ``n_freq=3`` -- deliberately non-square and
non-equal. Family B's gain is indexed along TIME; under a square fixture a
gain of length ``n_freq`` would satisfy a length check that is meant to be
about which axis it is, and the guard would look correct while being blind.
"""

import ast
import pathlib
import re

import jax.numpy as jnp
import pytest

import rheplicant
import rheplicant.radio as radio
from rheplicant.core.coordinates import Coordinates
from rheplicant.core.errors import StateValidationError
from rheplicant.core.state import State

N_TIME, N_FREQ = 5, 3

_SRC = pathlib.Path(rheplicant.__file__).parent

# --------------------------------------------------------------------------
# Population derivation: parse the source, do not trust the tables.
# --------------------------------------------------------------------------

_LEAF_NDIM = re.compile(r"must be scalar or \(n_freq,\), got ndim=")
_GAIN_NDIM = re.compile(r"must be scalar or 1D, got ndim=")
_DATA_2D = re.compile(r"expects \(n_time, n_freq\) data; got ")


def _owners_raising(pattern: re.Pattern) -> set[str]:
    """Every class in the shipped source that raises with ``pattern``.

    Deliberately an AST walk over the files rather than a scan of
    ``radio.__all__``: a copy pasted into a class that is not exported would
    be invisible to the export list, and "not exported" is not the same as
    "not reachable" -- these operators are reachable through the pipeline
    combinators regardless of what the package re-exports.
    """
    owners: set[str | None] = set()

    def visit(node, owner):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, child.name)
                continue
            if isinstance(child, ast.Raise) and pattern.search(ast.unparse(child)):
                owners.add(owner)
            visit(child, owner)

    for path in sorted(_SRC.rglob("*.py")):
        visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)), None)
    return owners


# --------------------------------------------------------------------------
# The tables. Values are valid scalars; each test perturbs ONE leaf.
# --------------------------------------------------------------------------

#: family A -- operator -> its leaves carrying the (n_freq,) rank guard.
LEAF_RANK_GUARDED: dict[str, dict[str, float]] = {
    "AntennaLossOperator": {"efficiency": 0.9, "t_physical": 290.0},
    "BeamSpillOperator": {"sky_fraction": 0.8, "t_ground": 290.0},
}

#: ``CalLoadOperator`` LEFT this family, and the derived check above is what
#: noticed. Its ``t_load`` now also accepts a per-sample ``(n_time, 1)`` column
#: -- the form a recording's thermistor log takes -- so its refusal no longer
#: says "must be scalar or (n_freq,)" and it is no longer a member of the
#: sentence this file is about. Its own accepted forms are pinned in
#: ``TestCalLoadRankForms`` below rather than left to the family that dropped
#: it, which is the whole point of deriving membership from the source: a
#: member leaving is as visible as a member arriving.

#: family B -- operator -> the leaf, which is ``gain`` for both of them.
GAIN_RANK_GUARDED: dict[str, str] = {
    "ApplyCalibrationOperator": "gain",
    "GainOperator": "gain",
}

#: family C -- operators refusing data that is not a 2-D waterfall.
DATA_RANK_GUARDED: tuple[str, ...] = (
    "AntennaLossOperator",
    "BeamSpillOperator",
    "NoiseWaveOperator",
)

#: NoiseWaveOperator needs its optional backend before it reaches the guard:
#: ``__call__`` imports rhino_cal_jax on its first line, ahead of the check.
_NEEDS_RHINO_CAL = {"NoiseWaveOperator"}


#: Sentinel for "caller did not ask for particular data". ``None`` cannot serve
#: here: an explicit ``data=None`` is one of the cases under test, and a
#: ``None`` default silently substituted the valid waterfall for it -- the
#: refusal tests passed vacuously until this sentinel was introduced.
_DEFAULT = object()


def _state(data=_DEFAULT, *, n_time=N_TIME, n_freq=N_FREQ) -> State:
    """A state carrying both data and coords, so one fixture serves everybody.

    Family A spans operators that read ``n_freq`` from ``state.data`` and one
    that reads it from ``coords.freq``; giving every member both removes the
    per-member special-casing that is how the untested copies stayed untested.
    """
    if data is _DEFAULT:
        # Ramped, not constant: a constant waterfall cannot distinguish an
        # operator that mixed its axes from one that did not.
        data = jnp.arange(float(n_time * n_freq)).reshape(n_time, n_freq) + 100.0
    return State(
        data=data,
        coords=Coordinates(
            time=jnp.arange(float(n_time)),
            freq=jnp.linspace(60e6, 85e6, n_freq),
        ),
    )


def _noise_wave(**overrides):
    """A single-source NoiseWaveOperator -- no switch array required."""
    kwargs = dict(
        t_unc=jnp.array(250.0),
        t_cos=jnp.array(30.0),
        t_sin=jnp.array(-40.0),
        t_rx=jnp.array(290.0),
        # One source, and channel-varying so a per-channel slip is visible.
        gamma_src_re=jnp.array([[0.30, 0.28, 0.26]]),
        gamma_src_im=jnp.array([[0.10, 0.05, 0.00]]),
        gamma_rec_re=jnp.array([0.08, 0.07, 0.06]),
        gamma_rec_im=jnp.array([-0.03, -0.02, -0.01]),
    )
    kwargs.update(overrides)
    return radio.NoiseWaveOperator(**kwargs)


def _build(name: str, **leaves):
    if name == "NoiseWaveOperator":
        return _noise_wave(**leaves)
    return getattr(radio, name)(**{k: jnp.asarray(v) for k, v in leaves.items()})


def _construct_and_call(name: str, leaves: dict, state: State | None = None):
    """Run both guard sites, because the family raises from either one.

    ``BeamSpillOperator`` and ``AntennaLossOperator`` refuse in
    ``__check_init__``; ``CalLoadOperator`` refuses the same sentence in
    ``__call__``. A test that only constructed would silently pass on the
    third member without ever reaching its raise.
    """
    operator = _build(name, **leaves)
    return operator(_state() if state is None else state)


def _skip_if_backend_missing(name: str) -> None:
    if name in _NEEDS_RHINO_CAL:
        pytest.importorskip("rhino_cal_jax", reason="rhino-cal-jax not installed")


# --------------------------------------------------------------------------
# Family A: leaf rank, "{leaf} must be scalar or (n_freq,)"
# --------------------------------------------------------------------------


class TestLeafRankFamily:
    def test_the_table_is_the_family_and_the_family_is_the_table(self):
        derived = _owners_raising(_LEAF_NDIM)
        assert derived == set(LEAF_RANK_GUARDED), {
            "carry the guard but are untested": sorted(
                derived - set(LEAF_RANK_GUARDED)
            ),
            "listed but no longer carry it": sorted(
                set(LEAF_RANK_GUARDED) - derived
            ),
        }

    @pytest.mark.parametrize(
        ("name", "leaf"),
        [(n, leaf) for n, leaves in LEAF_RANK_GUARDED.items() for leaf in leaves],
    )
    def test_a_two_dimensional_leaf_is_refused(self, name, leaf):
        leaves = dict(LEAF_RANK_GUARDED[name])
        leaves[leaf] = jnp.ones((2, N_FREQ))
        with pytest.raises(StateValidationError, match="ndim=2"):
            _construct_and_call(name, leaves)

    @pytest.mark.parametrize(
        ("name", "leaf"),
        [(n, leaf) for n, leaves in LEAF_RANK_GUARDED.items() for leaf in leaves],
    )
    def test_the_refusal_names_the_leaf_it_came_from(self, name, leaf):
        """Five leaves across three operators share one sentence.

        The only thing distinguishing them is the interpolated leaf name, so
        a copy that hardcoded the wrong one -- ``t_ground`` where it meant
        ``sky_fraction`` -- would raise, would be a ``StateValidationError``,
        and would satisfy the test above.
        """
        leaves = dict(LEAF_RANK_GUARDED[name])
        leaves[leaf] = jnp.ones((2, N_FREQ))
        with pytest.raises(StateValidationError) as excinfo:
            _construct_and_call(name, leaves)
        assert leaf in str(excinfo.value), str(excinfo.value)

    @pytest.mark.parametrize(
        ("name", "leaf"),
        [(n, leaf) for n, leaves in LEAF_RANK_GUARDED.items() for leaf in leaves],
    )
    def test_a_scalar_leaf_is_accepted(self, name, leaf):
        """Arm two of three. Without it, a guard collapsed to an
        unconditional raise passes every refusal test in this class."""
        out = _construct_and_call(name, dict(LEAF_RANK_GUARDED[name]))
        assert out.data.shape == (N_TIME, N_FREQ)

    @pytest.mark.parametrize(
        ("name", "leaf"),
        [(n, leaf) for n, leaves in LEAF_RANK_GUARDED.items() for leaf in leaves],
    )
    def test_a_per_channel_leaf_is_accepted(self, name, leaf):
        """Arm three: ``ndim == 1`` is the shape the docstrings advertise."""
        leaves = dict(LEAF_RANK_GUARDED[name])
        # Distinct per channel, so a length check that passed by accident on a
        # constant spectrum would not also hide a per-channel misalignment.
        base = float(leaves[leaf])
        leaves[leaf] = jnp.linspace(base * 0.9, base, N_FREQ)
        out = _construct_and_call(name, leaves)
        assert out.data.shape == (N_TIME, N_FREQ)


# --------------------------------------------------------------------------
# Family B: "gain must be scalar or 1D"
# --------------------------------------------------------------------------


class TestGainRankFamily:
    def test_the_table_is_the_family_and_the_family_is_the_table(self):
        derived = _owners_raising(_GAIN_NDIM)
        assert derived == set(GAIN_RANK_GUARDED), {
            "carry the guard but are untested": sorted(
                derived - set(GAIN_RANK_GUARDED)
            ),
            "listed but no longer carry it": sorted(
                set(GAIN_RANK_GUARDED) - derived
            ),
        }

    @pytest.mark.parametrize("name", sorted(GAIN_RANK_GUARDED))
    def test_a_two_dimensional_gain_is_refused(self, name):
        with pytest.raises(StateValidationError, match="ndim=2"):
            _construct_and_call(name, {"gain": jnp.ones((N_TIME, N_FREQ))})

    @pytest.mark.parametrize("name", sorted(GAIN_RANK_GUARDED))
    def test_the_refusal_names_the_leaf_it_came_from(self, name):
        with pytest.raises(StateValidationError) as excinfo:
            _construct_and_call(name, {"gain": jnp.ones((N_TIME, N_FREQ))})
        assert GAIN_RANK_GUARDED[name] in str(excinfo.value), str(excinfo.value)

    def test_the_gain_refusals_name_which_operator_raised_them(self):
        """The finding this class first pinned, now closed.

        Both operators used to raise the SAME STRING for the same mistake --
        same leaf name, no operator name -- so a user seeing ``gain must be
        scalar or 1D, got ndim=2.`` in a traceback-less log could not tell
        whether it came from ``GainOperator`` (forward model) or
        ``ApplyCalibrationOperator`` (its inverse), which sit at opposite ends
        of the pipeline. The test asserted the collision and said that a source
        change naming the operator was the fix.

        The source now interpolates ``type(self).__name__``, so the two differ.
        Asserting they are pairwise distinct rather than merely non-empty is
        what keeps this closed: an interpolation that resolved to a constant --
        a shared base class, a copy-pasted literal -- would satisfy "names an
        operator" and reintroduce the collision.
        """
        messages = {}
        for name in sorted(GAIN_RANK_GUARDED):
            with pytest.raises(StateValidationError) as excinfo:
                _construct_and_call(name, {"gain": jnp.ones((N_TIME, N_FREQ))})
            messages[name] = str(excinfo.value)

        assert len(set(messages.values())) == len(messages), messages
        for name, message in messages.items():
            assert name in message, (name, message)
            others = [n for n in GAIN_RANK_GUARDED if n != name]
            assert not any(other in message for other in others), (name, message)

    @pytest.mark.parametrize("name", sorted(GAIN_RANK_GUARDED))
    def test_a_scalar_gain_is_accepted(self, name):
        out = _construct_and_call(name, {"gain": jnp.asarray(2.0)})
        assert out.data.shape == (N_TIME, N_FREQ)

    @pytest.mark.parametrize("name", sorted(GAIN_RANK_GUARDED))
    def test_a_per_sample_gain_is_accepted(self, name):
        """``ndim == 1`` of length ``n_time`` -- the advertised shape."""
        gain = jnp.linspace(1.0, 2.0, N_TIME)
        out = _construct_and_call(name, {"gain": gain})
        assert out.data.shape == (N_TIME, N_FREQ)

    @pytest.mark.parametrize("name", sorted(GAIN_RANK_GUARDED))
    def test_a_gain_along_the_frequency_axis_is_refused(self, name):
        """The reason the fixture is not square.

        A gain of length ``n_freq`` is the exact mistake this guard exists to
        catch -- the right rank, indexed along the wrong axis. With
        ``n_time == n_freq`` it would broadcast happily and produce a finite,
        correctly-shaped, wrong waterfall.
        """
        assert N_TIME != N_FREQ, "a square fixture cannot see this"
        with pytest.raises(StateValidationError, match="samples but data has"):
            _construct_and_call(name, {"gain": jnp.ones(N_FREQ)})


# --------------------------------------------------------------------------
# Family C: "{Operator} expects (n_time, n_freq) data"
# --------------------------------------------------------------------------


def _valid_leaves(name: str) -> dict:
    if name == "NoiseWaveOperator":
        return {}
    return dict(LEAF_RANK_GUARDED[name])


class TestDataRankFamily:
    def test_the_table_is_the_family_and_the_family_is_the_table(self):
        derived = _owners_raising(_DATA_2D)
        assert derived == set(DATA_RANK_GUARDED), {
            "carry the guard but are untested": sorted(
                derived - set(DATA_RANK_GUARDED)
            ),
            "listed but no longer carry it": sorted(
                set(DATA_RANK_GUARDED) - derived
            ),
        }

    @pytest.mark.parametrize("name", DATA_RANK_GUARDED)
    @pytest.mark.parametrize("bad", ["1d", "3d", "none"])
    def test_data_that_is_not_a_waterfall_is_refused(self, name, bad):
        _skip_if_backend_missing(name)
        data = {
            "1d": jnp.arange(float(N_FREQ)),
            "3d": jnp.ones((2, N_TIME, N_FREQ)),
            "none": None,
        }[bad]
        with pytest.raises(StateValidationError, match="n_time, n_freq"):
            _construct_and_call(name, _valid_leaves(name), _state(data=data))

    @pytest.mark.parametrize("name", DATA_RANK_GUARDED)
    def test_the_data_refusal_names_the_operator_that_raised_it(self, name):
        """The check that catches a rename that did not happen.

        These three sentences interpolate ``type(self).__name__``; they used to
        hardcode the class name, which is why this check exists at all. A
        pasted-in fourth operator now gets its own name automatically, so
        what this pins is that the interpolation is still there and still
        resolves per class -- one that resolved to a constant would satisfy
        "names an operator" and restore the collision. A fourth operator pasted in
        from one of them would raise the right exception with another
        operator's name on it, and every other assertion in this file would
        still pass.
        """
        _skip_if_backend_missing(name)
        with pytest.raises(StateValidationError) as excinfo:
            _construct_and_call(
                name, _valid_leaves(name), _state(data=jnp.arange(float(N_FREQ)))
            )
        assert str(excinfo.value).startswith(name), str(excinfo.value)

    @pytest.mark.parametrize("name", DATA_RANK_GUARDED)
    def test_a_waterfall_is_accepted(self, name):
        """The other branch: ``ndim != 2`` collapsed to ``True`` is one edit."""
        _skip_if_backend_missing(name)
        out = _construct_and_call(name, _valid_leaves(name))
        assert out.data.shape == (N_TIME, N_FREQ)

    @pytest.mark.parametrize("name", DATA_RANK_GUARDED)
    def test_the_reported_shape_is_the_shape_that_was_rejected(self, name):
        """``got {got}`` is the only diagnostic the message carries.

        Asserted on a non-square, non-transposable shape so that a guard
        reporting ``state.data.T.shape`` -- or the shape it expected rather
        than the shape it got -- fails here instead of reading plausibly.
        """
        _skip_if_backend_missing(name)
        with pytest.raises(StateValidationError) as excinfo:
            _construct_and_call(
                name, _valid_leaves(name), _state(data=jnp.ones((2, N_TIME, N_FREQ)))
            )
        assert f"(2, {N_TIME}, {N_FREQ})" in str(excinfo.value), str(excinfo.value)

    @pytest.mark.parametrize("name", DATA_RANK_GUARDED)
    def test_absent_data_is_reported_as_none_and_not_as_a_shape(self, name):
        """The ``got = None if state.data is None`` arm.

        Without this, the conditional could be dropped and the guard would
        raise ``AttributeError`` on ``None.shape`` while chasing the message.
        """
        _skip_if_backend_missing(name)
        with pytest.raises(StateValidationError, match="got None"):
            _construct_and_call(name, _valid_leaves(name), _state(data=None))


class TestCalLoadRankForms:
    """``t_load``'s three accepted forms and everything else refused.

    ``CalLoadOperator`` left family A when ``t_load`` gained a per-sample
    column, so its guard is no longer covered by the sentence-derived tests
    above. These take it over. The distinguishing assertion is not "it was
    accepted" but WHICH AXIS the value ended up varying along -- a per-sample
    temperature applied per-channel is finite, correctly shaped, and describes
    a different instrument.
    """

    #: n_time != n_freq, so a column and a row are never interchangeable here.
    #: With N_TIME == N_FREQ every assertion below would hold under a
    #: transposed reading, and the guard would look correct while being blind.
    def _state(self):
        return _state()

    def test_a_scalar_fills_the_whole_waterfall(self):
        out = radio.CalLoadOperator(t_load=jnp.asarray(300.0))(self._state())
        assert out.data.shape == (N_TIME, N_FREQ)
        assert jnp.allclose(out.data, 300.0)

    def test_a_1d_array_is_read_per_FREQUENCY(self):
        """The convention, asserted rather than assumed.

        ``NoiseWaveOperator``'s temperature leaves read a bare 1-D array the
        same way; the two must not disagree, because a model carries both.
        """
        spectrum = jnp.linspace(280.0, 320.0, N_FREQ)
        out = radio.CalLoadOperator(t_load=spectrum)(self._state())
        assert jnp.allclose(out.data[0], spectrum)          # varies along FREQ
        assert jnp.allclose(out.data[:, 0], spectrum[0])    # flat along TIME

    def test_a_column_is_read_per_SAMPLE(self):
        drift = jnp.linspace(290.0, 300.0, N_TIME)
        out = radio.CalLoadOperator(t_load=drift[:, None])(self._state())
        assert jnp.allclose(out.data[:, 0], drift)          # varies along TIME
        assert jnp.allclose(out.data[0], drift[0])          # flat along FREQ

    def test_a_bare_1d_of_length_n_time_is_refused_not_guessed(self):
        """The ambiguity this convention exists to remove.

        With ``N_TIME != N_FREQ`` this is a length mismatch and the refusal is
        easy. The reason it matters is the case this fixture cannot show: on a
        square grid the same array reads equally well as either axis, NumPy
        settles it by aligning trailing axes, and every downstream number is
        finite and wrong. The message therefore names the convention and the
        way out, not merely the mismatch.
        """
        with pytest.raises(StateValidationError) as excinfo:
            radio.CalLoadOperator(t_load=jnp.ones(N_TIME))(self._state())
        message = str(excinfo.value)
        assert "always read as per-FREQUENCY" in message, message
        assert f"({N_TIME}, 1)" in message, message

    @pytest.mark.parametrize(
        "shape",
        [(N_TIME, 2), (N_FREQ, 1), (1, N_FREQ), (N_TIME, N_FREQ)],
        ids=["two-columns", "wrong-rows", "row-vector", "full-waterfall"],
    )
    def test_every_other_2d_shape_is_refused(self, shape):
        """Including ``(n_time, n_freq)``, which is explicit but unneeded.

        A load whose spectrum also moved would be a different model than this
        placeholder has. Refusing it keeps the guard narrow, and a narrow guard
        is easier to widen when that model arrives than to narrow after someone
        has relied on it.
        """
        with pytest.raises(StateValidationError, match=r"must be exactly \(\d+, 1\)"):
            radio.CalLoadOperator(t_load=jnp.ones(shape))(self._state())

    def test_three_dimensions_is_refused_by_the_final_arm(self):
        with pytest.raises(StateValidationError, match="t_load must be scalar"):
            radio.CalLoadOperator(t_load=jnp.ones((N_TIME, 1, 1)))(self._state())

