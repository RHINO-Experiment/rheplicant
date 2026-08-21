"""C15 -- ``inflight/noise_waves.py``, the per-channel rank report.

**These tests do not live in ``test_preflight_gated.py``, and that is a
measurement rather than a preference.**  ``preflight_helpers``'
``findings``/``ids``/``refusals``/``only`` all call ``preflight(document)`` --
the TEXT pass alone -- so a C15 test written with ``only(...)`` would pass
against an empty implementation and every other one could never pass.  C15 is
registered with ``@register_axes`` and is reached through
``inflight_helpers.axis_only`` / ``axis_findings``.

**Why the axes pass and not the text pass.**  The only text reader for
``n_freq`` is ``preflight/values.py::_a41_scope``, which answers ``None`` --
*"the document does not say"* -- for a non-``linspace``/``arange``/``modulo``/
``list`` grid, for a symbolic ``num:``, and for every ingested run.  A C15 in
pre-flight would therefore be silent by construction on a whole class of
documents, which is the false negative this plan exists to remove.
``Axes.context.shape_scope`` gives ``n_freq`` and ``n_source``
unconditionally, and the axes pass still runs in front of ``build_resources``.

**Every message is pinned by a whole-string equality or by a substring pinned
in a partner test that pins the whole.**  A ``match=`` is a search: Plan 3A's
surviving mutants lived almost entirely inside refusal text.

**Set assertions are subset-shaped**, scoped to :data:`MINE`: six wave-1
branches land checks that run on these same documents, so an ``== ()`` would
go red on a correct merge.
"""

import pytest

from rheplicant.config.document import load_document
from rheplicant.config.findings import REPORT
from rheplicant.config.inflight import AXIS_CHECKS
from rheplicant.config.inflight.noise_waves import (
    _NOISE_WAVE_LEAVES,
    _T2C_BASIS_TYPE,
    _noise_wave_rank,
)
from tests.config.exit_helpers import ONE_LATENT
from tests.config.inflight_helpers import axis_findings, axis_only
from tests.config.preflight_helpers import (
    BASE_OBSERVATION,
    NOISE_WAVE_BASIS,
    NOISE_WAVE_LOADS,
    NOISE_WAVE_MODEL,
    NOISE_WAVE_SWITCHING,
    preflight_document,
)

#: The ids THIS module is about.  Every "and nothing else" assertion below is
#: scoped to it, for the reason the module docstring gives.
MINE = frozenset({"C15"})


def silent_here(document) -> bool:
    """Did this document earn nothing from THIS module's own check?"""
    return {one.check for one in axis_findings(document)}.isdisjoint(MINE)


def _document(**inference):
    """The base document with a noise-wave model and ``inference`` merged in.

    ``preflight_document`` merges one level deep, so ``parameters=`` REPLACES
    the base document's single ``g`` latent rather than adding to it -- which
    is what makes ``k`` and the freed set read exactly what a test wrote.
    """
    return preflight_document(model=NOISE_WAVE_MODEL, inference=inference)


# --- the registration -------------------------------------------------------


class TestTheSlot:
    def test_c15_is_bound_in_the_axes_registry(self):
        """Not the pre-flight one, and not the built one.  A C15 registered in
        the text pass cannot read ``shape_scope`` at all -- D-9's whole
        argument -- and one registered in the built slot has paid for the beam
        before saying anything."""
        assert AXIS_CHECKS["C15"] is _noise_wave_rank

    def test_the_four_temperatures_are_the_operators_own_fields(self):
        """Read off ``dataclasses.fields``, never restated.  A fifth leaf on
        the operator that this frozenset does not know is a family C15 would
        count as held when the document frees it."""
        import dataclasses

        from rheplicant.radio.instrument.noise_wave import NoiseWaveOperator

        fields = {one.name for one in dataclasses.fields(NoiseWaveOperator)}
        assert _NOISE_WAVE_LEAVES <= fields
        assert _NOISE_WAVE_LEAVES == frozenset(
            {"t_unc", "t_cos", "t_sin", "t_rx"})

    def test_the_basis_detector_names_the_class_whose_graph_node_is_t_sys_extra(
            self):
        """**The measured trap.**  ``BasisTemperatureOperator.graph_node`` is
        ``t_sys_extra``, NOT ``noise_wave`` -- a check that looked for the
        basis under ``model.noise_wave`` finds nothing and reports a number
        the package's own docstring contradicts in both directions (12
        identified against a predicted 6; rank 5 against a bound of 7)."""
        from rheplicant.radio.t_sys import BasisTemperatureOperator

        assert _T2C_BASIS_TYPE == BasisTemperatureOperator.__name__
        assert BasisTemperatureOperator.graph_node == "t_sys_extra"


# --- standing down ----------------------------------------------------------


class TestItStandsDown:
    def test_C15_stands_down_with_no_free_noise_wave_temperature(self):
        """``k == 0``: no temperature is free, so there is nothing to report.

        **Kills** a check that reports on every document -- which is what a
        report with no ``k == 0`` guard is, since ``min(n_source, 0) * n_freq``
        is a perfectly well-formed zero.
        """
        assert silent_here(preflight_document())
        assert silent_here(preflight_document(model=NOISE_WAVE_MODEL))

    def test_C15_reports_when_one_temperature_is_freed(self):
        """The anti-vacuity partner of the test above.  Without it, a check
        whose body is ``return ()`` passes every stand-down test in this
        class."""
        found = axis_only(_document(parameters={
            "t": {"init": 1.0, "into": "noise_wave.t_unc"}}), "C15")
        assert found.severity == REPORT

    def test_a_latent_into_another_node_is_not_a_noise_wave_temperature(self):
        """``gain.gain`` is a leaf and is not one of the four."""
        assert silent_here(_document(parameters={
            "g": {"init": 1.0, "into": "gain.gain"}}))

    @pytest.mark.parametrize("into", ["bandpass.t_unc", "gain.t_rx",
                                      "t_unc", "global_signal.depth.t_cos"])
    def test_a_temperature_NAME_under_another_node_is_not_one_of_the_four(
            self, into):
        """**Kills the path HEAD gate being dropped**, which no ``gain.gain``
        document can see: both halves must be gated, and a check that tested
        the LEAF alone counts a ``t_unc`` written under any node at all.  This
        is A33's own first-version mistake -- it gated the leaf and left the
        head open -- on the other side."""
        assert silent_here(_document(parameters={
            "x": {"init": 1.0, "into": into}}))

    @pytest.mark.parametrize("into", ["noise_wave.gamma_src_re",
                                      "noise_wave.switch_key"])
    def test_a_noise_wave_field_that_is_not_one_of_the_four_is_not_counted(
            self, into):
        """MAJOR 6: **kills the LEAF-membership gate being dropped**, which
        no document under another HEAD can see -- ``gamma_src_re`` and
        ``switch_key`` are ``NoiseWaveOperator``'s own fields, correctly
        headed at ``noise_wave``, but they are couplings and a selector
        rather than one of the four counted temperatures.  Without this
        gate a latent into either would count as a freed temperature and
        raise ``k``."""
        assert silent_here(_document(parameters={
            "x": {"init": 1.0, "into": into}}))


# --- the number -------------------------------------------------------------


class TestTheNumber:
    def test_it_is_min_n_source_times_k_times_n_freq(self):
        """Three loads, four freed families, eight channels: ``min(3, 4) * 8``.

        **Kills** ``min(n_source, k)`` collapsing to ``k`` -- which would say
        32 here, and would say ``4 * 8 = 32`` on the ONE-load document below
        where the truth is 8.
        """
        document = preflight_document(
            model=NOISE_WAVE_MODEL,
            observation={**BASE_OBSERVATION,
                         "switching": NOISE_WAVE_SWITCHING},
            inference=dict(parameters={
                "a": {"init": 1.0, "into": "noise_wave.t_unc"},
                "b": {"init": 1.0, "into": "noise_wave.t_cos"},
                "c": {"init": 1.0, "into": "noise_wave.t_sin"},
                "d": {"init": 1.0, "into": "noise_wave.t_rx"}}))
        found = axis_only(document, "C15")
        assert "min(3, 4) * 8 = 24" in found.message

    def test_C15_uses_n_source_or_one(self):
        """A ``pointing.mode: none`` document with no declared switching order
        still reports, with ``n_source = 1``.

        **Kills** a bare ``len(switch_order)``, which is 0 here and makes the
        product 0 -- a check that "reports" a rank of zero on the correct
        one-load document.  The ``or 1`` is ``context.shape_scope``'s and is
        INHERITED here, never re-derived: ``shape_scope.n_source`` is already
        ``n_source_override or len(switch_order) or 1``.
        """
        document = preflight_document(
            model=NOISE_WAVE_MODEL,
            observation={**BASE_OBSERVATION, "pointing": {"mode": "none"}},
            inference=dict(parameters={
                "a": {"init": 1.0, "into": "noise_wave.t_unc"},
                "b": {"init": 1.0, "into": "noise_wave.t_cos"}}))
        found = axis_only(document, "C15")
        assert "min(1, 2) * 8 = 8" in found.message

    def test_the_message_names_all_four_temperatures_and_which_are_free(self):
        """§2.6 item 9: §6's singular *"the declared noise-wave block"* is
        ambiguous, so the sentence names the whole family and the subset.

        **Kills** a message that says only ``k = 2``, which tells a reader
        deciding a switching cadence nothing about WHICH two.
        """
        found = axis_only(_document(parameters={
            "a": {"init": 1.0, "into": "noise_wave.t_unc"},
            "b": {"init": 1.0, "into": "noise_wave.t_rx"}}), "C15")
        for leaf in _NOISE_WAVE_LEAVES:
            assert leaf in found.message
        assert "['t_rx', 't_unc']" in found.message

    def test_it_reports_and_never_refuses(self):
        """§6: *"Reported, not refused"*, and the module docstring of
        ``radio/instrument/noise_wave.py`` says why -- the rule is a design
        aid, not a fault.

        Swept over every non-empty subset shape this check can reach, because
        a severity chosen inside one branch is exactly the shape that ships.
        """
        for freed in (["t_unc"], ["t_unc", "t_cos"],
                      ["t_unc", "t_cos", "t_sin", "t_rx"]):
            parameters = {name: {"init": 1.0, "into": f"noise_wave.{name}"}
                          for name in freed}
            found = axis_findings(_document(parameters=parameters))
            mine = [one for one in found if one.check == "C15"]
            assert [one.severity for one in mine] == [REPORT], freed

    def test_at_most_one_finding_however_many_latents(self):
        """``axis_only`` asserts it, and this names the property so the
        failure reads as "C15 fired per latent" rather than as an index
        error."""
        axis_only(_document(parameters={
            "a": {"init": 1.0, "into": "noise_wave.t_unc"},
            "b": {"init": 1.0, "into": "noise_wave.t_cos"},
            "c": {"init": 1.0, "into": "noise_wave.t_sin"}}), "C15")


# --- the twin: both routes into the leaf ------------------------------------


class TestTheTwin:
    """``inference.parameters.<n>.into`` and ``inference.bindings[].into`` are
    two spellings of one meaning, and ``build_space`` walks two loops over
    them.  3A's ``_bandpass_and_gain``/``_t11_bindings`` closed exactly this
    pair; a check that reads one is 2C's shape 4 in the one place this layer
    has an actual twin."""

    def test_C15_counts_a_binding_as_well_as_an_into(self):
        """**Kills** reading ``inference.parameters`` alone."""
        found = axis_only(_document(
            parameters={"a": {"init": 1.0}},
            bindings=[{"latents": ["a"], "into": "noise_wave.t_unc"}]), "C15")
        assert "min(1, 1) * 8 = 8" in found.message

    def test_the_two_routes_agree_on_one_document(self):
        """The same freed set through each spelling gives the same number.
        **Kills** a binding walk that read the head where the ``into:`` walk
        read the leaf."""
        through_into = axis_only(_document(parameters={
            "a": {"init": 1.0, "into": "noise_wave.t_unc"},
            "b": {"init": 1.0, "into": "noise_wave.t_cos"}}), "C15")
        through_bindings = axis_only(_document(
            parameters={"a": {"init": 1.0}, "b": {"init": 1.0}},
            bindings=[{"latents": ["a"], "into": "noise_wave.t_unc"},
                      {"latents": ["b"], "into": "noise_wave.t_cos"}]), "C15")
        assert "min(1, 2) * 8 = 8" in through_into.message
        assert "min(1, 2) * 8 = 8" in through_bindings.message

    def test_a_list_into_frees_every_leaf_it_names(self):
        """``into:`` is legally a string OR a list of strings
        (``sections/parameters.py::_names``).  **Kills** a normalisation that
        only handled the string."""
        found = axis_only(_document(parameters={
            "a": {"init": 1.0,
                  "into": ["noise_wave.t_unc", "noise_wave.t_cos"]}}), "C15")
        assert "min(1, 2) * 8 = 8" in found.message

    def test_an_index_after_the_leaf_still_counts_by_its_field(self):
        """MAJOR 6: **kills an index being read as the leaf name.**
        ``parse_path`` returns ``('noise_wave', 't_unc', 0)`` for
        ``noise_wave.t_unc[0]``, and the docstring's own promise is that the
        leaf is the LAST STRING segment -- the index is dropped before the
        membership test, not read as the leaf itself.  Without that,
        ``t_unc[0]`` silently drops out of ``freed`` and k reads one low."""
        found = axis_only(_document(parameters={
            "a": {"init": 1.0, "into": "noise_wave.t_unc[0]"}}), "C15")
        assert "min(1, 1) * 8 = 8" in found.message

    @pytest.mark.parametrize("bindings", [
        [{"into": "noise_wave.t_unc"}],
        [{"latents": 7, "into": "noise_wave.t_unc"}],
        [{"latents": ["ghost"], "into": "noise_wave.t_unc"}],
    ], ids=["missing", "non-string", "undeclared"])
    def test_a_binding_whose_latents_cannot_be_read_frees_nothing(
            self, bindings):
        """MINOR 1 (fix round): ``_t2c_routes`` used to count a ``bindings[]``
        entry's ``into:`` whatever ``latents:`` said, while
        ``preflight/model.py::_t11_bindings`` (A33's own walk over this same
        grammar) deliberately DROPS an entry whose ``latents:`` is missing,
        non-string, or names nothing ``inference.parameters`` declares -- the
        more specific refusal at build time is what names that fault, and a
        check that counted it anyway reported a rank the document cannot
        reach.  Measured before this fix: the ``undeclared`` cell alone
        (``latents: ['ghost']``, nothing named ``ghost`` in ``parameters:``)
        reported ``min(1, 1) * 8 = 8`` -- a temperature no declared latent
        actually reaches.  All three cells now stand C15 down entirely,
        exactly as ``_t11_bindings`` stands A33 down on the same three
        documents.
        """
        assert silent_here(_document(parameters={"a": {"init": 1.0}},
                                     bindings=bindings))


# --- declining --------------------------------------------------------------


class TestItDeclines:
    """§2.6 item 10: the basis regime is reachable TWO ways and a task closing
    one and leaving the other is 3A's recorded twin failure."""

    def test_C15_declines_under_a_basis_operator(self):
        """A lit ``t_sys_extra`` of type ``BasisTemperatureOperator``.

        ``**ONE_LATENT["parameters"]`` stays in the mix (MAJOR 4): the base
        document's ``inference.observed.at`` names ``g``, and a bare
        ``parameters={"a": ...}`` would leave that latent undeclared and
        ``load_document`` would refuse it for a reason this test is not
        about.
        """
        pytest.importorskip("rhino_cal_jax",
                            reason="rhino_cal_jax comes with rheplicant[cal]")
        document = preflight_document(
            model={**NOISE_WAVE_MODEL, "t_sys_extra": NOISE_WAVE_BASIS},
            inference=dict(parameters={
                **ONE_LATENT["parameters"],
                "a": {"init": 1.0, "into": "noise_wave.t_unc"}}))
        found = axis_only(document, "C15")
        assert found.severity == REPORT
        assert "does not apply" in found.message
        assert "min(1, 1) * 8" not in found.message
        load_document(document)

    def test_C15_declines_under_the_from_basis_route(self):
        """MAJOR 3: the OTHER spelling of the same operator --
        ``sections/model.py``'s ``t_sys_extra`` + ``from: basis`` route,
        which writes no ``type:`` at all.  A detector that only looked for
        ``type: BasisTemperatureOperator`` (or a ``python:`` naming it)
        missed this one: measured on this exact document under the original
        detector, ``declares_basis: False`` and C15 REPORTED A NUMBER the
        package's own docstring contradicts in both directions -- and the
        document LOADS, so the wrong number was not even confined to a
        document nobody could run.
        """
        pytest.importorskip("rhino_cal_jax",
                            reason="rhino_cal_jax comes with rheplicant[cal]")
        document = preflight_document(
            model={**NOISE_WAVE_MODEL,
                  "t_sys_extra": [{"from": "basis",
                                   "basis": {"ref": "resources.bases.b"},
                                   "coeff": {"zeros": [1, 3], "unit": "K"}}]},
            resources={"bases": {"b": {"time": {"kind": "legendre",
                                                "n_basis": 1},
                                       "freq": {"kind": "legendre",
                                               "n_basis": 3}}}},
            inference=dict(parameters={
                **ONE_LATENT["parameters"],
                "a": {"init": 1.0, "into": "noise_wave.t_unc"}}))
        found = axis_only(document, "C15")
        assert found.severity == REPORT
        assert "does not apply" in found.message
        assert "min(1, 1) * 8" not in found.message
        load_document(document)

    def test_C15_declines_under_a_python_relocated_basis_operator(self):
        """MAJOR 6: the THIRD spelling of the same operator -- a ``python:``
        relocation naming ``BasisTemperatureOperator`` by its module path
        rather than by ``type:``.  Untested before this commit: mutating the
        ``python:`` clause away left every test in this class green, because
        nothing drove that branch.
        """
        pytest.importorskip("rhino_cal_jax",
                            reason="rhino_cal_jax comes with rheplicant[cal]")
        document = preflight_document(
            model={**NOISE_WAVE_MODEL,
                  "t_sys_extra": [
                      {"python": "rheplicant.radio:BasisTemperatureOperator",
                       "coeff": {"zeros": [2, 3], "unit": "K"},
                       "time_basis": {"ones": ["n_time", 2]},
                       "freq_basis": {"ones": ["n_freq", 3]}}]},
            inference=dict(parameters={
                **ONE_LATENT["parameters"],
                "a": {"init": 1.0, "into": "noise_wave.t_unc"}}))
        found = axis_only(document, "C15")
        assert found.severity == REPORT
        assert "does not apply" in found.message
        assert "min(1, 1) * 8" not in found.message
        load_document(document)

    def test_C15_declines_under_a_transform(self):
        """A latent reaching the leaf through ``transform:``.  The SECOND
        route, and the one a task that only looked at ``model:`` would
        leave open."""
        found = axis_only(_document(parameters={
            "a": {"init": 1.0, "into": "noise_wave.t_unc",
                  "transform": "exp"}}), "C15")
        assert found.severity == REPORT
        assert "does not apply" in found.message
        assert "min(1, 1) * 8" not in found.message

    def test_C15_declines_under_a_transform_on_a_binding(self):
        """The transform route's own twin: ``inference.bindings[].transform``.
        **Kills** a transform test that read ``inference.parameters`` only."""
        found = axis_only(_document(
            parameters={"a": {"init": 1.0}},
            bindings=[{"latents": ["a"], "into": "noise_wave.t_unc",
                       "transform": "exp"}]), "C15")
        assert "does not apply" in found.message

    def test_identity_is_not_a_transform_that_breaks_the_counting(self):
        """``identity`` binds the leaf unchanged, so it ties no channels
        together and the per-channel rule still holds.  This layer already
        treats ``(None, "identity")`` as "no transform"
        (``sections/inference.py::_derive_truth``).

        **Kills** ``transform is not None`` -- which would decline on the one
        transform that changes nothing.
        """
        found = axis_only(_document(parameters={
            "a": {"init": 1.0, "into": "noise_wave.t_unc",
                  "transform": "identity"}}), "C15")
        assert "min(1, 1) * 8 = 8" in found.message

    def test_a_basis_at_t_sys_extra_of_another_type_does_not_decline(self):
        """The anti-vacuity partner: ``t_sys_extra`` lit by something that is
        NOT the basis leaves the counting rule in force.  **Kills** a detector
        that answered "basis" for a lit node whatever its type.

        ``GroundPickupOperator``, relocated onto ``t_sys_extra`` via
        ``python:`` (the package's own ``test_t_sys_extra_accepts_at_
        injection`` builds the identical relocation) -- ``ConstantTsysOperator``
        does not exist (MAJOR 4), and the old fixture's label-keyed mapping is
        A6-refused on this node besides (only ``cal_loads`` is FAN-shaped; a
        SUM node like ``t_sys_extra`` takes a LIST).
        """
        pytest.importorskip("rhino_cal_jax",
                            reason="rhino_cal_jax comes with rheplicant[cal]")
        document = preflight_document(
            model={**NOISE_WAVE_MODEL,
                  "t_sys_extra": [
                      {"python": "rheplicant.radio:GroundPickupOperator",
                       "coupling": {"value": 0.02, "unit": "dimensionless"},
                       "t_ground": {"value": 300.0, "unit": "K"}}]},
            inference=dict(parameters={
                **ONE_LATENT["parameters"],
                "a": {"init": 1.0, "into": "noise_wave.t_unc"}}))
        found = axis_only(document, "C15")
        assert "min(1, 1) * 8 = 8" in found.message
        load_document(document)


# --- the raise-guard --------------------------------------------------------


class TestItNeverRaises:
    """``passes.sweep`` turns any exception out of a check into a hard
    ``ConfigError`` that aborts the WHOLE pass and hides every later finding.
    ``paths.parse_path`` raises ``ConfigError`` on a non-``str`` and on a
    malformed path, and ``into:`` is user text."""

    @pytest.mark.parametrize("into", [7, None, ["noise_wave.t_unc", 7],
                                      "a..b", "", ["a..b"], {"x": 1},
                                      ["noise_wave.t_unc", "a..b"]])
    def test_an_unusable_into_does_not_abort_the_pass(self, into):
        """**Kills** a bare ``parse_path(path)``.  Its refusal is
        ``_selectors``'/``parse_path``'s own at build time, which names the
        value the user wrote; answering here would pre-empt it, and RAISING
        here would hide every finding after C15."""
        axis_findings(_document(parameters={"a": {"init": 1.0,
                                                  "into": into}}))

    @pytest.mark.parametrize("bindings", [7, "gain", [7], [{"into": 7}],
                                          [{"latents": "a", "into": None}]])
    def test_an_unusable_bindings_block_does_not_abort_the_pass(self,
                                                                bindings):
        axis_findings(_document(parameters={"a": {"init": 1.0}},
                                bindings=bindings))

    @pytest.mark.parametrize("model", [{"t_sys_extra": 7},
                                       {"t_sys_extra": [7]},
                                       {"t_sys_extra": None}])
    def test_an_unusable_t_sys_extra_does_not_abort_the_pass(self, model):
        axis_findings(preflight_document(
            model={**NOISE_WAVE_MODEL, **model},
            inference=dict(parameters={
                "a": {"init": 1.0, "into": "noise_wave.t_unc"}})))

    def test_a_document_with_no_inference_section_is_silent(self):
        assert silent_here(preflight_document(model=NOISE_WAVE_MODEL,
                                              inference=None))

    @pytest.mark.parametrize("name", ["a b", "7", "", "a..b", "(1, 2)"])
    def test_a_latent_NAME_that_is_not_a_path_segment_does_not_abort(self,
                                                                     name):
        """A latent's name is user text and reaches this pass BEFORE
        ``parse_latents`` has looked at it -- the axes hook runs at
        ``document.py``'s axes call and ``build_inference`` is two builders
        later.  Measured, ``parse_path('inference.parameters.a b')`` RAISES,
        and ``passes.check_where`` turns that into a ``ConfigError`` that
        aborts the whole axes pass and hides every finding after it.

        **Kills** the ``where`` being interpolated straight from the key.
        """
        found = axis_findings(_document(parameters={
            name: {"init": 1.0, "into": "noise_wave.t_unc"}}))
        assert [one.where for one in found if one.check == "C15"] == [
            "inference.parameters"]


# --- the whole message ------------------------------------------------------

#: The report, whole, on the one-latent document -- the shape 3A's surviving
#: mutants lived inside.
C15_ONE = (
    "inference.parameters.a frees ['t_unc'] of the four noise-wave "
    "temperatures ['t_cos', 't_rx', 't_sin', 't_unc']. Each switch position "
    "contributes one equation per frequency channel, so while every "
    "temperature is free PER CHANNEL the design matrix has rank "
    "min(n_source, k) * n_freq = min(1, 1) * 8 = 8. A four-family "
    "per-channel fit needs four distinct loads to be square; three loads "
    "leave it deficient by exactly n_freq, and sharing one Gamma across the "
    "cycle collapses every source onto the same row and drops the rank to "
    "n_freq whatever n_source is. Read a switching cadence off this number, "
    "and measure any other parameterization with "
    "rheplicant.inference.identifiability (check C15)."
)

#: The decline through a ``transform:``, whole.  It gives NO number for THIS
#: document in either direction -- the two arithmetic clauses inside it are
#: the package's own measured counter-examples and carry no ``n_freq`` of
#: this run's.
C15_DECLINED = (
    "inference.parameters.a frees ['t_unc'] of the four noise-wave "
    "temperatures ['t_cos', 't_rx', 't_sin', 't_unc'] through a transform:, "
    "so the per-channel counting rule does not apply and no counting rule "
    "replaces it. A basis ties the channels together and the rule fails "
    "in BOTH directions: per-channel counting understates (two loads and a "
    "3-coefficient basis identify all k * n_basis = 12 coefficients at k = 4, "
    "where min(n_source, k) * n_basis would say 6) and the bound "
    "rank <= min(n_source * n_freq, k * n_basis) overstates (one load whose "
    "Gamma is itself linear in frequency gives rank 5 against a bound of 7). "
    "Measure this parameterization with "
    "rheplicant.inference.identifiability instead (check C15)."
)

#: The same decline reached through the OTHER route -- a lit ``t_sys_extra``
#: of :data:`_T2C_BASIS_TYPE`.  Pinned separately because the two routes
#: differ in exactly one clause, and a task that closed one and left the other
#: is 3A's recorded twin failure.
C15_DECLINED_BASIS = C15_DECLINED.replace("through a transform:,",
                                          "through a frequency basis,")


class TestTheMessagesWhole:
    def test_the_report_is_pinned_whole(self):
        found = axis_only(_document(parameters={
            "a": {"init": 1.0, "into": "noise_wave.t_unc"}}), "C15")
        assert found.message == C15_ONE

    def test_the_decline_is_pinned_whole(self):
        found = axis_only(_document(parameters={
            "a": {"init": 1.0, "into": "noise_wave.t_unc",
                  "transform": "exp"}}), "C15")
        assert found.message == C15_DECLINED

    def test_the_basis_decline_is_pinned_whole(self):
        """The twin of the test above.  **Kills** the basis route being
        re-worded, or collapsing into the transform route's sentence, where
        no substring pin could see it."""
        pytest.importorskip("rhino_cal_jax",
                            reason="rhino_cal_jax comes with rheplicant[cal]")
        document = preflight_document(
            model={**NOISE_WAVE_MODEL, "t_sys_extra": NOISE_WAVE_BASIS},
            inference=dict(parameters={
                **ONE_LATENT["parameters"],
                "a": {"init": 1.0, "into": "noise_wave.t_unc"}}))
        found = axis_only(document, "C15")
        assert found.message == C15_DECLINED_BASIS
        load_document(document)

    def test_the_where_is_the_latent_the_reader_edits(self):
        """``Finding.where`` is a path into the USER'S document; ``sweep``
        validates the first segment against the section names."""
        assert axis_only(_document(parameters={
            "a": {"init": 1.0, "into": "noise_wave.t_unc"}},
        ), "C15").where == "inference.parameters.a"
        assert axis_only(_document(
            parameters={"a": {"init": 1.0}},
            bindings=[{"latents": ["a"], "into": "noise_wave.t_unc"}],
        ), "C15").where == "inference.bindings[0]"


# --- it does not stop the load ----------------------------------------------


class TestItDoesNotStopTheLoad:
    def test_a_reporting_document_still_loads(self):
        """A REPORT is not a refusal, and ``raise_if_refused`` must not turn
        one into a stopped load.  **Kills** the severity being chosen as
        REFUSE anywhere in the branch table.

        A document that really BUILDS, which is stronger than the axes-only
        documents above and is why it carries the loads its switching order
        names and keeps the base document's ``g`` -- ``observed.at`` names it,
        and a bare replacement of ``parameters:`` would be refused for that
        rather than for anything C15 decides.
        """
        pytest.importorskip("rhino_cal_jax",
                            reason="rhino_cal_jax comes with rheplicant[cal]")
        built = load_document(preflight_document(
            observation={**BASE_OBSERVATION,
                         "switching": NOISE_WAVE_SWITCHING},
            model={**NOISE_WAVE_MODEL, "cal_loads": NOISE_WAVE_LOADS},
            inference=dict(parameters={
                **ONE_LATENT["parameters"],
                "a": {"init": 1.0, "into": "noise_wave.t_unc"}})))
        assert "noise_wave" in built.twin.lit
