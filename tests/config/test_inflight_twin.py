"""B5 and C9 -- ``inflight/twin.py``.

Two rules, the same two ``test_inflight_grids.py`` carries: **every message is
pinned by equality on its whole text**, and **every registry and findings
assertion is subset shaped**, including "this document earns nothing", which
is scoped to :data:`MINE` through :func:`mine` rather than written ``== ()``.
Nine other tasks register checks that run on these same documents, so an exact
set is a merge hazard rather than a property; the property each test carries is
the whole-message pin beside it.

**The anti-property box, stated rather than implied.**  This slot runs when
``load_document`` is ready to return, so a document carrying a B5 violation
AND ``preflight_helpers.UNREADABLE_BEAM`` is refused by **the beam** --
``test_the_beam_wins_against_B5`` says so.  Schema §6's preamble ("all run
before any beam is analysed") is false about every row in this module, and a
task that wrote it would be repeating the mistake this plan exists to record.
"""

import dataclasses
import pathlib
import subprocess
import sys

import pytest

from rheplicant.config.document import load_document
from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import REFUSE
from rheplicant.config.inflight import BUILT_CHECKS, built
from rheplicant.config.inflight.twin import (
    NOISE_WAVE_TEMPERATURES,
    _square_grid_column,
    _switch_positions,
    _twins,
)
from tests.config.inflight_helpers import built_findings, built_only, built_run
from tests.config.preflight_helpers import (
    BASE_MODEL,
    BASE_OBSERVATION,
    UNREADABLE_BEAM,
    preflight_document,
)

_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: The ids this module is about.  Every "and nothing else" assertion is
#: intersected with it (§0.3 E.11): the shared base document already carries
#: other tasks' checks on other documents, and an exact set would be green in
#: this branch and red at the merge.
MINE = frozenset({"B5", "C9"})


def mine(document, **kwargs) -> frozenset[str]:
    """The ids from THIS module that fired on ``document``."""
    return frozenset(one.check
                     for one in built_findings(document, **kwargs)) & MINE


# --- the documents ---------------------------------------------------------

#: Three switch positions and two calibration loads -- A14's own arithmetic,
#: so the document is legal right up to the point B5 is about.
THREE = {"mode": "cycle", "order": ["antenna", "ambient", "hot"]}
TWO = {"mode": "cycle", "order": ["antenna", "ambient"]}
LOADS = {"ambient": {"t_load": {"value": 300.0, "unit": "K"}},
         "hot": {"t_load": {"value": 400.0, "unit": "K"}}}

#: The base model with every antenna-side source removed.  ``gain`` and
#: ``noise`` sit DOWNSTREAM of the switch, so the model still lights nodes --
#: which is what makes this different from an empty model and is why the
#: assembly still has a receiver_input to look at.
DARK = {key: value for key, value in BASE_MODEL.items()
        if key not in ("global_signal", "uniform_sky")}

#: A square grid: eight time samples against the base document's eight
#: channels.  C9's whole condition.
SQUARE_TIME = {"grid": {"arange": {"start": 0.0, "step": 2.0, "num": 8},
                        "unit": "s"}}


def dark_antenna(**patch):
    """Loads lit, antenna branch dark, three declared positions."""
    document = preflight_document(observation={"switching": THREE}, **patch)
    document["model"] = {**DARK, "cal_loads": LOADS}
    return document


def lit_antenna(**patch):
    """B5's own advice applied to :func:`dark_antenna`: a source on the branch."""
    return preflight_document(observation={"switching": THREE},
                              model={**BASE_MODEL, "cal_loads": LOADS},
                              **patch)


def square(cal_loads=None, model=None, switching=TWO, time=SQUARE_TIME):
    """A document whose time and frequency axes are the same length."""
    patch = dict(model or {})
    if cal_loads is not None:
        patch["cal_loads"] = cal_loads
    return preflight_document(
        observation={**BASE_OBSERVATION, "time": time, "switching": switching},
        model={**BASE_MODEL, **patch})


#: A ``noise_wave`` node whose four temperatures are written separately, so a
#: test can make exactly one of them the ambiguous shape.
def noise_wave(**over):
    node = {"type": "NoiseWaveOperator",
            "t_unc": {"value": 1.0, "unit": "K"},
            "t_cos": {"value": 1.0, "unit": "K"},
            "t_sin": {"value": 1.0, "unit": "K"},
            "t_rx": {"value": 1.0, "unit": "K"},
            "gamma_src_re": {"zeros": ["n_source", "n_freq"]},
            "gamma_src_im": {"zeros": ["n_source", "n_freq"]},
            "gamma_rec_re": {"zeros": ["n_freq"]},
            "gamma_rec_im": {"zeros": ["n_freq"]}}
    node.update(over)
    return node


PER_CHANNEL = {"ones": ["n_freq"], "unit": "K"}
AMBIGUOUS = {"ones": ["n_time"], "unit": "K"}
COLUMN = {"ones": ["n_time"], "unit": "K", "column": True}

#: Twelve values a document key may hold that are not what a check expects.
#: Driven against every document value these checks read: a membership test or
#: an index against one of them raises INSIDE the check, and ``sweep`` turns
#: that into "check RAISED", which aborts the pass and hides every finding
#: after it while every ``match=`` pin in the suite still passes.
HOSTILE = ([], {}, set(), None, 3, [[1, 2], [3]], "text", (1, 2),
           {"a": [1]}, [{"b": 2}], True, 0.5)


@pytest.fixture(scope="module")
def payload():
    """One real built payload, reused: ``built_run`` costs a few ms."""
    return built_run(square(cal_loads={"ambient": {"t_load": AMBIGUOUS}}))


# --- the messages, whole ---------------------------------------------------

_B5_HOW = (
    "twin['receiver_input'].names is ['cal_loads_1', 'cal_loads_2'] -- fold "
    "labels minted from graph node ids rather than the document's own, and "
    "names[0] moves with the model (it becomes 'astro_sum' the moment a "
    "second antenna-side node is lit), so a positional comparison cannot be "
    "written and a LENGTH comparison is the only one that exists"
)

_B5_REST = (
    " The antenna branch is dark: the calibration loads are lit and nothing "
    "upstream of antenna_loss is, so the loads slide down into a cycle whose "
    "position 0 is the antenna. Every switch label is then off by one and the "
    "last position selects nothing -- and nothing raises: measured, such a "
    "document builds, its twin runs, and the data comes back finite, "
    "correctly shaped and calibrated against the wrong load. {how}. Light a "
    "source on the antenna branch (global_signal, uniform_sky, foregrounds, "
    "point_sources, observed_astro_sky, ground_pickup, atmosphere or "
    "t_sys_extra): that is the fix, and it is the one that leaves a run which "
    "can still generate its own data. Dropping the cycle instead -- "
    "observation.switching: {{mode: none}}, and model.cal_loads with it -- "
    "silences this too, but on a model whose only sources ARE the loads it "
    "leaves a pure transform chain with nothing to transform, and the next "
    "simulation fails with 'This assembly is a pure transform chain'; it is a "
    "fix only for a run that has a source elsewhere. This check runs after "
    "build_resources, so it saves no beam: what it buys is the refusal "
    "instead of a plausible answer (check B5)."
)

B5_MESSAGE = (
    "observation.switching.order declares 3 switch positions ['antenna', "
    "'ambient', 'hot'] and the receiver_input switch this run's twin built "
    "has 2." + _B5_REST.format(how=_B5_HOW)
)

#: The FIT-twin branch, whole.  Its `where` is `inference.twin` and it carries
#: a ~40-word aside the model-twin message does not -- both of which were
#: unpinned in the first cut of this module and both of which a mutant could
#: rewrite arbitrarily while the suite stayed green.
B5_FIT_TWIN = (
    "observation.switching.order declares 3 switch positions ['antenna', "
    "'ambient', 'hot'] and the receiver_input switch the FIT twin built has 2."
    " That is the twin inference.twin's without:/replace: rebuilt, and it is "
    "the one the fit evaluates -- so the edit is there: "
    "inference.twin.without: is taking the last antenna-side source out of it "
    "while model: keeps one." + _B5_REST.format(how=_B5_HOW)
)

B5_TRAVERSED = (
    "observation.switching.order declares 2 switch positions ['antenna', "
    "'ambient'] and the receiver_input switch this run's twin built has 1."
    + _B5_REST.format(
        how="only one branch reaches receiver_input, so the fold traversed "
            "the selector as identity: there is no switch at all and every "
            "sample takes that one branch, whatever "
            "coords.extra['receiver_input'] holds")
)

_C9_TAIL = (
    " config/modifiers.py ships the mechanism -- column: true forces (n,) to "
    "(n, 1) -- and until this check nothing ever DEMANDED it: measured on a "
    "square 8x8 grid, a bare 8-element temperature loads clean and is read as "
    "per-frequency, silently, whichever axis its author meant. Off a square "
    "grid the 1-D form is legal and documented and a length mismatch is "
    "already caught by name, so this fires only where the two axes are "
    "genuinely indistinguishable. Write column: true beside the value if it "
    "is per-SAMPLE, or leave it as it stands if per-CHANNEL is what was meant "
    "and say so by writing the length as n_freq (check C9)."
)

_C9_HEAD = (
    "{where}: this run's time and frequency axes are both 8 long, and this "
    "value is a bare (8,) array -- so nothing says whether it is one "
    "temperature per SAMPLE or one per CHANNEL, and the package reads a bare "
    "1-D value as per-channel whatever was meant. The legal shapes here are "
)

C9_T_LOAD = (
    _C9_HEAD.format(where="model.cal_loads.ambient.t_load")
    + "scalar, (n_freq,) and (n_time, 1) -- and NOT (n_time, n_freq), which "
      "CalLoadOperator refuses by name because a load whose spectrum also "
      "moves is a different model than this one has." + _C9_TAIL
)

C9_T_UNC = (
    _C9_HEAD.format(where="model.noise_wave.t_unc")
    + "scalar, (n_freq,), (n_time, 1) and (n_time, n_freq)." + _C9_TAIL
)


class TestTheRegistry:
    def test_each_id_binds_to_its_own_function(self):
        assert BUILT_CHECKS["B5"] is _switch_positions
        assert BUILT_CHECKS["C9"] is _square_grid_column

    def test_both_slots_are_claimed(self):
        assert {"B5", "C9"} <= set(BUILT_CHECKS)

    def test_the_module_is_WIRED_and_not_merely_decorated(self):
        """The mutation ``BUILT_CHECKS["B5"] is _switch_positions`` cannot see.

        That assertion passes because THIS MODULE's own
        ``from ...inflight.twin import _switch_positions`` runs the decorator.
        Deleting the ``@register_built`` line and deleting the FOOT IMPORT in
        ``inflight/__init__.py`` are different mutations, and only the second
        decides whether B5 runs for a user -- every wave-1 task measured the
        same thing: with its foot import removed, its own module stayed
        entirely green while the id was absent from the registry in a fresh
        process.

        A SUBPROCESS, because in this process the import has already happened
        and cannot be un-happened.
        """
        done = subprocess.run(
            [sys.executable, "-c",
             "from rheplicant.config.inflight import BUILT_CHECKS\n"
             "assert {'B5', 'C9'} <= set(BUILT_CHECKS), sorted(BUILT_CHECKS)\n"],
            capture_output=True, text=True, cwd=str(_ROOT), check=False)
        assert done.returncode == 0, (
            "B5/C9 are decorated but not wired: importing the package does "
            "not import inflight/twin.py, so the built pass never runs "
            "them.\n" + done.stdout + done.stderr)

    def test_the_module_name_collides_with_no_entry_point(self):
        """R13, checked rather than assumed.

        ``inflight.axes()`` (the entry point) and ``inflight/axes.py`` (the
        module) collide, and BOTH foot-import spellings for that pair are
        silently wrong.  ``twin`` is not a name ``inflight/__init__.py``
        binds -- ``Built.twin`` is a dataclass FIELD, not a module-level name
        -- so the ordinary aliased form works here.  This is what says so, and
        what goes red the day someone adds ``def twin(...)`` beside ``def
        built``.
        """
        import rheplicant.config.inflight as package

        assert callable(package.axes) and callable(package.built)
        assert package.twin.__name__ == "rheplicant.config.inflight.twin"

    def test_the_base_document_earns_nothing_here(self):
        assert mine(preflight_document()) == frozenset()


class TestB5:
    """The dark antenna branch, on a document A14 and compose.py both accept."""

    def test_it_fires_and_names_the_document_key(self):
        found = built_only(dark_antenna(), "B5")
        assert found.severity == REFUSE
        assert found.where == "model.cal_loads"

    def test_the_whole_message(self):
        assert built_only(dark_antenna(), "B5").message == B5_MESSAGE

    def test_the_hook_stops_the_load(self):
        """A test about the FINDING uses ``built_run``; this one is about the
        HOOK, so it calls ``load_document`` under ``pytest.raises`` (§0.3
        C.3)."""
        with pytest.raises(ConfigError) as raised:
            load_document(dark_antenna())
        assert str(raised.value) == B5_MESSAGE

    def test_a_lit_antenna_branch_earns_nothing(self):
        """Three positions, two loads, a source on the antenna branch: the
        arithmetic B5 is about, correct.  This is also the mutant-killer for a
        POSITIONAL reading -- ``names`` here is ``('astro_sum', 'cal_loads_1',
        'cal_loads_2')`` and matches no declared label at all."""
        assert mine(lit_antenna()) == frozenset()

    def test_the_advice_applied_leaves_the_document_passing(self):
        """S4: take the check's own remedy and assert the document then loads."""
        assert load_document(lit_antenna()) is not None
        assert mine(lit_antenna()) == frozenset()

    def test_the_SECOND_remedy_is_conditional_and_the_message_says_so(self):
        """R4, and a measured advice loop closed rather than shipped.

        "Drop the cycle and model.cal_loads with it" reads as a second fix and
        is one only for a run with a source elsewhere.  Applied to THIS
        document -- whose only sources are the loads -- it leaves a pure
        transform chain, and the shipped fixture's ``inference.observed:
        {from: simulation}`` then fails at ``AssemblyError: This assembly is a
        pure transform chain``.  Measured, and the message names the
        condition rather than the bare alternative.
        """
        from rheplicant.core.graph import AssemblyError

        stripped = preflight_document(
            observation={"switching": {"mode": "none"}})
        stripped["model"] = dict(DARK)
        with pytest.raises(AssemblyError, match="pure transform chain"):
            load_document(stripped)
        assert "pure transform chain" in built_only(dark_antenna(),
                                                    "B5").message
        # ... and on a run that HAS a source elsewhere, the same edit works.
        with_source = preflight_document(
            observation={"switching": {"mode": "none"}})
        assert load_document(with_source) is not None
        assert mine(with_source) == frozenset()

    def test_a_traversed_selector_is_the_same_defect_and_is_caught(self):
        """One load and a dark antenna: the fold TRAVERSES receiver_input.

        There is no ``SelectOperator`` and no ``names`` at all, so an
        implementation that reads ``twin["receiver_input"].names`` and catches
        nothing else stands down on the very case where the switch has been
        deleted outright -- every sample takes the one load.
        """
        document = preflight_document(observation={"switching": TWO})
        document["model"] = {**DARK, "cal_loads": {"ambient": LOADS["ambient"]}}
        found = built_only(document, "B5")
        assert found.severity == REFUSE
        assert found.message == B5_TRAVERSED

    def test_the_FIT_TWIN_ALONE_going_dark_is_named_at_inference_twin(self):
        """R3 / §0.3 E.10 for B5: ``without:`` can darken the fit twin only.

        The raw twin here is CORRECT -- ``model:`` lights ``global_signal`` and
        ``uniform_sky``, so its selector has three positions -- and
        ``inference.twin.without:`` takes both back out of the twin the fit
        evaluates.  Measured: ``raw names ('astro_sum', 'cal_loads_1',
        'cal_loads_2')`` against ``fit names ('cal_loads_1', 'cal_loads_2')``.

        Two things are pinned that nothing pinned before: the ``where``
        (``inference.twin``, not ``model.cal_loads`` -- the model line is
        already right and sending a reader there would be sending them to the
        wrong file), and the whole ~40-word aside that names
        ``inference.twin.without:`` as the edit.  Both survived a mutation
        campaign otherwise.
        """
        # Built through ``preflight_document``'s own ``inference=`` keyword,
        # NOT by ``document["inference"] = {...}`` afterwards: the second is a
        # depth-1 replacement of a block ``exit_helpers._repaired`` has already
        # repaired, and ``test_config_fixture_contract.py``'s census refuses it
        # by name -- measured, it named this test.
        document = lit_antenna(
            inference={"twin": {"without": ["noise", "global_signal",
                                            "uniform_sky"]}})
        run = built_run(document)
        assert run.twin["receiver_input"].names == ("astro_sum",
                                                    "cal_loads_1",
                                                    "cal_loads_2")
        assert run.inference.fit_twin["receiver_input"].names == (
            "cal_loads_1", "cal_loads_2")
        found = built_only(document, "B5")
        assert found.severity == REFUSE
        assert found.where == "inference.twin"
        assert found.message == B5_FIT_TWIN

    def test_the_two_twins_are_named_DIFFERENTLY_and_not_by_one_sentence(self):
        """The half of the branch above a `where` assertion alone cannot see.

        A mutant that keeps the two ``where``s and collapses the two askings
        into one clause is invisible to any test that reads only the id and
        the path.  This is the equality pin's job, said out loud: the two
        whole messages differ, and they differ in the aside.
        """
        raw = built_only(dark_antenna(), "B5").message
        assert raw == B5_MESSAGE
        assert raw != B5_FIT_TWIN
        assert "inference.twin.without: is taking" in B5_FIT_TWIN
        assert "inference.twin.without: is taking" not in raw

    def test_no_switch_order_stands_it_down(self):
        """``switching: {mode: none}`` is a run with no cycle: there is
        nothing to compare and the base document must stay silent."""
        assert "B5" not in mine(preflight_document())
        assert "B5" not in mine(preflight_document(
            observation={"switching": {"mode": "none"}}))

    def test_the_INGESTED_route_is_a_RECORDED_FALSE_NEGATIVE(self, tmp_path):
        """S3, and §0.3 E.6 ruling 6's "guard it or record the false negative".

        ``preflight/observing.py`` stands A14 DOWN entirely on
        ``observation.from_file`` -- measured at its own ``if "from_file" in
        observation: return findings`` -- so the ingested route has no guard,
        and **this check is not it either**.  The reason is not an omission.

        On an ingested run the antenna's contribution IS the recorded
        waterfall rather than a model source, so the antenna branch is
        structurally dark: this document -- the one
        ``test_preflight_ingest.py`` calls its canonical WORKING document --
        gives ``names == ('cal_loads_1', 'cal_loads_2')`` against three
        declared positions, exactly B5's shape.  And there is no edit that
        would silence it, because an assembly carrying sources refuses a state
        that already has data.  Refusing here would therefore refuse every
        ingested switched run, the shipped canonical one included, with a
        sentence naming a remedy that cannot be applied (R4).

        What the route actually needs is a decision about what an ingested
        switched run assembles to, which is bigger than this check.  Recorded
        here by name, with both measurements, rather than left to be
        rediscovered.
        """
        pytest.importorskip("h5py", reason="h5py comes with rheplicant[rhino]")
        from rheplicant.core.graph import AssemblyError
        from tests.config.test_config_section_ingest import make_file

        make_file(tmp_path / "obs.hd5f")
        document = {
            "schema_version": 1, "runtime": {"seed": 1},
            "observation": {
                "from_file": {"format": "rhino_hdf5", "path": "obs.hd5f",
                              "freq_unit": "MHz", "settle_seconds": 0.0},
                "switching": {"order": ["antenna", "internal_load",
                                        "heated_load"]}},
            "model": {"cal_loads": {
                "internal_load": {"t_load": {"value": 300.0, "unit": "K"}},
                "heated_load": {"t_load": {"value": 400.0, "unit": "K"}}},
                "gain": {"gain": {"value": 2.0, "unit": "dimensionless"}}},
            "runs": [{"kind": "forward"}],
        }
        run = built_run(document, base_dir=str(tmp_path))
        # The shape IS B5's -- two positions against three declared labels.
        assert run.context.switch_order == ("antenna", "internal_load",
                                            "heated_load")
        assert run.twin["receiver_input"].names == ("cal_loads_1",
                                                    "cal_loads_2")
        # ... and the run cannot be evaluated at all, for a reason that has
        # nothing to do with the switch and that no B5 edit could remove.
        with pytest.raises(AssemblyError, match="contains source operators"):
            run.twin(run.state)
        # So the check stands down, and says so on the payload rather than
        # only in prose.
        assert mine(document, base_dir=str(tmp_path)) == frozenset()
        assert run.context.ingest is not None

    def test_a_switch_order_with_NO_selector_on_the_path_stands_it_down(self):
        """The other half of "loads absent is not this check's sentence".

        A payload can carry a declared order and a twin the fold never gave a
        ``receiver_input`` to -- no live branch reaches the selector, so there
        is no count to compare -- and "0 positions" would be a sentence about
        a missing ``model.cal_loads`` block, which is A14's.

        Assembled by hand because no synthetic document reaches that state
        (A14 refuses them at P-1) and the ingested ones stand down a line
        earlier: this drops the ``ingest`` from a real ingested payload, which
        is the narrowest change that exposes the branch.
        """
        pytest.importorskip("h5py", reason="h5py comes with rheplicant[rhino]")
        run = self._transform_only_ingested_run()
        assert "receiver_input" not in run.twin.materialized
        assert "receiver_input" not in run.twin.skipped
        synthetic = dataclasses.replace(
            run, context=dataclasses.replace(run.context, ingest=None))
        assert synthetic.context.switch_order == ("antenna", "internal_load",
                                                  "heated_load")
        assert list(_switch_positions(synthetic)) == []

    @staticmethod
    def _transform_only_ingested_run(tmp_path=None):
        import tempfile

        from tests.config.test_config_section_ingest import make_file

        directory = pathlib.Path(tmp_path or tempfile.mkdtemp())
        make_file(directory / "obs.hd5f")
        document = {
            "schema_version": 1, "runtime": {"seed": 1},
            "observation": {
                "from_file": {"format": "rhino_hdf5", "path": "obs.hd5f",
                              "freq_unit": "MHz", "settle_seconds": 0.0},
                "switching": {"order": ["antenna", "internal_load",
                                        "heated_load"]}},
            "model": {"gain": {"gain": {"value": 2.0,
                                        "unit": "dimensionless"}}},
            "runs": [{"kind": "forward"}],
        }
        return built_run(document, base_dir=str(directory))

    def test_the_mirror_document_hears_A14_and_not_this(self):
        """S4's stand-down: loads ABSENT is somebody else's sentence.

        ``preflight/observing.py``'s A14 refuses it at P-1, so the document
        never reaches the built pass at all -- and that is the right ordering,
        because A14 names the labels it wanted and this check could only say
        that a count did not match.  Fires on "loads present, antenna dark",
        never on "loads absent".

        (The plan's own TRAP names ``compose.py::cal_load_order_problem`` as
        the sibling here.  Measured, it is not: that function refuses the KEYS
        being wrong, and the mirror document -- no ``cal_loads`` key at all --
        is refused earlier and elsewhere, at ``preflight/observing.py``.
        §0.3 E.6 ruling 6 records the correction.)
        """
        mirror = preflight_document(observation={"switching": THREE})
        with pytest.raises(ConfigError) as raised:
            built_findings(mirror)
        assert "(check A14)." in str(raised.value)
        assert "check B5" not in str(raised.value)

    def test_the_beam_wins_against_B5(self):
        """§5's ANTI-PROPERTY for this task, as a named test.

        For every hoisting task the box reads "the violation wins against an
        unreadable beam".  For the BUILT slot it is the other way round and
        that is the point: this pass runs after ``build_resources``, so a
        document that is both wrong in B5's way and carrying an unreadable
        beam is refused by **the beam**.  A task claiming otherwise would be
        claiming schema §6's false preamble.
        """
        document = dark_antenna(resources=UNREADABLE_BEAM)
        with pytest.raises(ConfigError) as raised:
            load_document(document)
        assert "no_such_beam.npy" in str(raised.value)
        assert "check B5" not in str(raised.value)


class TestC9:
    """``column:`` demanded exactly where the two axes are indistinguishable."""

    def test_it_fires_on_a_cal_load_and_names_the_label(self):
        found = built_only(square(cal_loads={"ambient": {"t_load": AMBIGUOUS}}),
                           "C9")
        assert found.severity == REFUSE
        assert found.where == "model.cal_loads.ambient"

    def test_the_whole_cal_load_message(self):
        assert built_only(square(cal_loads={"ambient": {"t_load": AMBIGUOUS}}),
                          "C9").message == C9_T_LOAD

    def test_it_fires_on_a_noise_wave_temperature(self):
        found = built_only(
            square(cal_loads={"ambient": {"t_load": {"value": 300.0,
                                                     "unit": "K"}}},
                   model={"noise_wave": noise_wave(t_unc=AMBIGUOUS)}), "C9")
        assert found.where == "model.noise_wave"
        assert found.message == C9_T_UNC

    @pytest.mark.parametrize("field", NOISE_WAVE_TEMPERATURES)
    def test_all_four_temperatures_are_named_and_not_just_the_first(self, field):
        """Schema §6's C9 row says "a ``noise_wave`` temperature", singular.
        There are four, and a check keyed on ``t_unc`` alone would let the
        other three through with every shape correct."""
        document = square(
            cal_loads={"ambient": {"t_load": {"value": 300.0, "unit": "K"}}},
            model={"noise_wave": noise_wave(**{field: AMBIGUOUS})})
        assert built_only(document, "C9").message.startswith(
            f"model.noise_wave.{field}: this run's time and frequency axes")

    def test_TWO_cal_loads_are_reached_by_their_MINTED_ids_and_labelled_right(
            self):
        """The ``many`` fan, which every other C9 document here misses.

        Two calibration loads make ``cal_loads`` a multi-instance node, and
        ``twin['cal_loads']`` then raises ``AmbiguousNodeError`` **by design**
        -- the bare id addresses none of them.  A check that reached for it
        would raise INSIDE the pass, and ``sweep`` turns that into "in-flight
        check 'C9' RAISED", which **aborts the pass and hides every finding
        after it** while every ``match=`` pin in the suite still passes.  That
        is the §0.3 C.2 failure shape, and this is the document that would
        have triggered it.

        The second half is the label.  The twin's addresses are the minted
        ``cal_loads_1``/``cal_loads_2``; the document key comes back by
        POSITION off the FAN mapping, which ``cal_load_order_problem`` has
        already pinned equal to ``switching.order[1:]`` in that order.  The
        defect is in ``hot``, so the reader must be sent to
        ``model.cal_loads.hot`` -- taking ``labels[0]`` would name
        ``model.cal_loads.ambient`` and send them to a line that is fine.
        """
        from rheplicant.core.errors import AmbiguousNodeError

        document = square(
            switching=THREE,
            cal_loads={"ambient": {"t_load": {"value": 300.0, "unit": "K"}},
                       "hot": {"t_load": AMBIGUOUS}})
        run = built_run(document)
        assert run.twin.instances == (("cal_loads",
                                       ("cal_loads_1", "cal_loads_2")),)
        with pytest.raises(AmbiguousNodeError):
            run.twin["cal_loads"]
        found = built_only(document, "C9")
        assert found.where == "model.cal_loads.hot"
        assert found.message.startswith("model.cal_loads.hot.t_load:")

    def test_the_FIRST_of_two_cal_loads_is_labelled_right_too(self):
        """The mirror of the test above, and the reason it is not redundant.

        ``labels[0]`` names the right key for a defect in ``ambient`` and the
        wrong one for a defect in ``hot``; only the pair distinguishes a
        positional lookup from a constant.
        """
        document = square(
            switching=THREE,
            cal_loads={"ambient": {"t_load": AMBIGUOUS},
                       "hot": {"t_load": {"value": 400.0, "unit": "K"}}})
        assert built_only(document, "C9").where == "model.cal_loads.ambient"

    def test_TWO_ambiguous_temperatures_on_ONE_node_are_TWO_findings(self):
        """Why ``seen`` is keyed on the LEAF and not on the operator.

        ``test_all_four_temperatures_are_named_and_not_just_the_first`` is
        parametrized one temperature at a time, so it cannot see an
        operator-keyed de-duplication: with one ambiguous leaf per document
        the two implementations agree.  Two on one ``noise_wave`` separate
        them -- and the user-facing difference is real, because an
        operator-keyed one tells a reader about ``t_unc``, they fix it,
        reload, and are told about ``t_sin``.

        Subset-shaped rather than ``len(...) == 2``: the ``where``s are read
        off this module's own ids only.
        """
        document = square(
            cal_loads={"ambient": {"t_load": {"value": 300.0, "unit": "K"}}},
            model={"noise_wave": noise_wave(t_unc=AMBIGUOUS, t_sin=AMBIGUOUS)})
        fields = {one.message.split(":")[0]
                  for one in built_findings(document) if one.check == "C9"}
        assert {"model.noise_wave.t_unc", "model.noise_wave.t_sin"} <= fields
        assert "model.noise_wave.t_cos" not in fields
        assert "model.noise_wave.t_rx" not in fields

    def test_the_two_legal_shape_LISTS_differ(self):
        """§0.3 E.6 ruling 3: ``(n_time, n_freq)`` is REFUSED for ``t_load``
        and LEGAL for a noise-wave temperature, so one message quoting one
        list would be advice the package refuses on half its own subjects.

        Asserted as the two sentences being different rather than as two
        substrings, because the mutant this kills is "quote one list".
        """
        assert "(n_time, n_freq)" in C9_T_UNC
        assert "NOT (n_time, n_freq)" in C9_T_LOAD
        loads = built_only(square(cal_loads={"ambient": {"t_load": AMBIGUOUS}}),
                           "C9").message
        temps = built_only(
            square(cal_loads={"ambient": {"t_load": {"value": 300.0,
                                                     "unit": "K"}}},
                   model={"noise_wave": noise_wave(t_rx=AMBIGUOUS)}),
            "C9").message
        assert loads != temps

    def test_it_stands_down_OFF_a_square_grid(self):
        """The TRAP, and the whole reason the check is conditional.

        Off a square grid a 1-D length already says which axis it runs along,
        ``CalLoadOperator`` refuses a mismatch by name, and demanding
        ``column:`` there would refuse documents the package builds.  The base
        document is 16 time samples against 8 channels.
        """
        rectangular = preflight_document(
            observation={"switching": TWO},
            model={**BASE_MODEL,
                   "cal_loads": {"ambient": {"t_load": PER_CHANNEL}}})
        assert mine(rectangular) == frozenset()

    def test_the_advice_applied_leaves_the_document_passing(self):
        """S4: ``column: true`` beside the value, and the document loads."""
        fixed = square(cal_loads={"ambient": {"t_load": COLUMN}})
        assert load_document(fixed) is not None
        assert mine(fixed) == frozenset()

    def test_the_other_advice_applied_also_passes(self):
        """The message's second remedy: leave it per-CHANNEL and say so.

        ``{ones: [n_freq]}`` and ``{ones: [n_time]}`` are the same ARRAY on a
        square grid, so this is a documentation edit as far as the run is
        concerned -- and the check still fires on it, which is honest: the
        text is what disambiguates and the check reads the built leaf.  What
        is asserted here is that the remedy the message names first (the
        column) is the one that silences it.
        """
        assert "C9" in mine(square(cal_loads={"ambient": {"t_load": PER_CHANNEL}}))
        assert "C9" not in mine(square(cal_loads={"ambient": {"t_load": COLUMN}}))

    def test_a_scalar_and_a_2D_column_are_both_left_alone(self):
        for value in ({"value": 300.0, "unit": "K"}, COLUMN):
            assert mine(square(cal_loads={"ambient": {"t_load": value}})) \
                == frozenset()

    def test_the_beam_wins_against_C9(self):
        """§5's ANTI-PROPERTY, for C9 as well as for B5.

        The box is Task 7's alone and the task ships four rows, so each one
        gets it.  This slot runs after ``build_resources``: a document wrong
        in C9's way AND carrying an unreadable beam is refused by **the
        beam**, and ``check C9`` never appears.
        """
        document = square(cal_loads={"ambient": {"t_load": AMBIGUOUS}})
        document["resources"] = {**document.get("resources", {}),
                                 **UNREADABLE_BEAM}
        with pytest.raises(ConfigError) as raised:
            load_document(document)
        assert "no_such_beam.npy" in str(raised.value)
        assert "check C9" not in str(raised.value)

    def test_the_averaging_twin_does_not_move_the_grids_under_it(self):
        """S3: ``averaging`` against every other node that changes the time axis.

        ``BackendOperator`` reshapes the time axis INSIDE the twin; the grids
        this check reads are ``context.time`` and ``context.freq``, which are
        ``build_observation``'s and which no model node rewrites.  So a
        document that averages 8 samples into 2 is still square by this
        check's reckoning, and it is right to be: the LEAF was resolved
        against the declared grid, and it is the leaf whose axis is
        ambiguous.  Recorded as a test rather than as a sentence because the
        obvious alternative -- reading the post-averaging extent -- silently
        stops firing on exactly these documents.
        """
        averaged = square(cal_loads={"ambient": {"t_load": AMBIGUOUS}},
                          model={"averaging": {"n_chunk": 2}})
        assert "C9" in mine(averaged)

    def test_the_INFERENCE_TWIN_REPLACE_route_is_walked(self):
        """§0.3 E.10, and a correction to the plan's own premise.

        The plan assumed ``Built.twin`` already carries
        ``inference.twin.replace``.  Measured, it does NOT: ``Built.twin`` is
        the RAW twin and ``inference.fit_twin`` is the rebuilt one, with
        ``inference.replaced`` naming the nodes that moved.  Both objects are
        live -- ``run_forward`` evaluates the raw twin and the fit evaluates
        the other -- so both are walked and the finding names the document key
        that built the offending leaf.
        """
        document = square(
            cal_loads={"ambient": {"t_load": {"value": 300.0, "unit": "K"}}})
        document = preflight_document(
            observation={**BASE_OBSERVATION, "time": SQUARE_TIME,
                         "switching": TWO},
            model=document["model"],
            inference={**document["inference"],
                       "twin": {"without": ["noise"],
                                "replace": {"cal_loads": {"t_load": AMBIGUOUS}}}})
        run = built_run(document)
        assert run.inference.replaced == ("cal_loads",)
        found = built_only(document, "C9")
        assert found.where == "inference.twin.replace.cal_loads"
        assert found.message.startswith(
            "inference.twin.replace.cal_loads.t_load: this run's time and "
            "frequency axes are both 8 long")

    def test_a_leaf_the_fit_twin_did_not_rebuild_is_reported_ONCE(self):
        """The de-duplication, as a property rather than as a comment.

        The fit twin is a DIFFERENT object on almost every document -- the
        shipped fixture carries ``inference.twin.without: [noise]``, which
        re-assembles -- so a walk over both twins that keyed on the node id
        would report every shared leaf twice.  ``built_only`` asserts exactly
        one, so this test is the assertion; it is written out because the
        thing it pins is invisible in the passing case.
        """
        run = built_run(square(cal_loads={"ambient": {"t_load": AMBIGUOUS}}))
        assert run.inference.fit_twin is not run.twin
        assert run.inference.fit_twin["cal_loads"] is run.twin["cal_loads"]
        assert len(_twins(run)) == 2
        built_only(square(cal_loads={"ambient": {"t_load": AMBIGUOUS}}), "C9")


class TestTheChecksSurviveAValueThatIsTheWrongPythonType:
    """A document value a check reads may be a list, a dict, a set or ``None``.

    A membership test or an index against one of those raises inside the
    check, which ``sweep`` converts into "in-flight check 'C9' RAISED
    TypeError" -- **which aborts the pass and hides every finding after it,
    while every ``match=`` pin in the suite still passes**.  Wave 1 shipped
    exactly that bug once.  So every value these two checks read is driven
    with the whole hostile set, through the check functions directly, against
    a real payload.
    """

    @pytest.mark.parametrize("value", HOSTILE)
    @pytest.mark.parametrize("key", ["model", "resources", "inference"])
    def test_a_hostile_section(self, payload, key, value):
        run = dataclasses.replace(payload,
                                  document={**payload.document, key: value})
        assert isinstance(list(_switch_positions(run)), list)
        assert isinstance(list(_square_grid_column(run)), list)

    @pytest.mark.parametrize("value", HOSTILE)
    def test_a_hostile_cal_loads_block(self, payload, value):
        """The one document value C9 reads by name, to recover the label the
        twin's minted ``cal_loads_1`` id does not carry."""
        model = {**payload.document["model"], "cal_loads": value}
        run = dataclasses.replace(payload,
                                  document={**payload.document, "model": model})
        found = list(_square_grid_column(run))
        assert [one.check for one in found] == ["C9"]
        assert found[0].where.startswith("model.cal_loads")

    @pytest.mark.parametrize("field", ["twin", "inference", "context",
                                       "resources", "document"])
    def test_a_payload_field_that_is_not_what_it_should_be(self, payload, field):
        """A payload assembled by hand is a supported caller
        (``test_config_inflight.py`` assembles several), so neither check may
        assume the field is the class it usually is.

        ``document`` is on this list and was not: it is the field
        ``_cal_load_key`` calls ``.get()`` on, so excluding it left the one
        payload field two checks dereference unguarded outside the very list
        that admits hand-built payloads as supported.
        """
        run = dataclasses.replace(payload, **{field: object()})
        assert isinstance(list(_switch_positions(run)), list)
        assert isinstance(list(_square_grid_column(run)), list)

    @pytest.mark.parametrize("value", HOSTILE)
    @pytest.mark.parametrize("field", ["document", "context"])
    def test_a_payload_field_holding_a_hostile_VALUE(self, payload, field,
                                                     value):
        """The same sweep against the payload itself, not only its sections.

        Neither shape is reachable from a real document -- ``run.document`` is
        always ``_assemble``'s variant-applied Mapping and ``context.time`` is
        always an array -- but the sibling test above declares hand-built
        payloads supported, and a claim of "zero exceptions" has to cover what
        that sentence promises.
        """
        run = dataclasses.replace(payload, **{field: value})
        assert isinstance(list(_switch_positions(run)), list)
        assert isinstance(list(_square_grid_column(run)), list)

    @pytest.mark.parametrize("value", HOSTILE)
    def test_a_hostile_switch_order_or_grid_on_the_context(self, payload,
                                                           value):
        """``len(order)`` on an int and ``.shape`` on a list, guarded.

        B5 takes ``len()`` of ``context.switch_order`` and C9 reads
        ``context.time.shape``; both are always well formed on a real run and
        neither may raise on a payload that is not.
        """
        for field in ("switch_order", "time", "freq"):
            run = dataclasses.replace(
                payload,
                context=dataclasses.replace(payload.context,
                                            **{field: value}))
            assert isinstance(list(_switch_positions(run)), list)
            assert isinstance(list(_square_grid_column(run)), list)


class TestTheBuiltPassDoesNotEvaluateTheTwin:
    """A CALL COUNT rather than a wall clock, because the property allows it.

    "Under a bound" cannot distinguish a check that reads a shape from one
    that runs the model on a small document, and the small document is what
    every test here uses.  Zero calls can.  §0.1 puts a real forward pass in
    Plan 3C without exception; this is the assertion that says these two rows
    took none.
    """

    def test_neither_check_calls_the_assembled_twin(self, monkeypatch):
        """The payload is built FIRST and the counter installed after it.

        ``built_run`` runs a real forward pass of its own -- the shipped
        fixture's ``inference.observed: {from: simulation}`` makes
        ``build_observed`` do ``prediction = bound(state).data`` -- so a
        counter installed before it counts the LOAD's evaluation and the
        assertion could never be about the pass.
        """
        from rheplicant.core.graph import Assembly

        run = built_run(square(cal_loads={"ambient": {"t_load": AMBIGUOUS}}))
        calls = []
        original = Assembly.__call__
        monkeypatch.setattr(
            Assembly, "__call__",
            lambda self, state: (calls.append(1), original(self, state))[1])
        report = built(run)
        assert "C9" in report.checks()
        assert calls == []

    def test_neither_check_reads_a_file(self, monkeypatch):
        """The other half of the boundary, driven rather than parsed.

        ``test_config_inflight.py`` walks these modules statically for the
        NAME of a filesystem call; this drives them and asserts nothing was
        opened, which covers the indirection that walk records it cannot see.

        **The document must make BOTH rows do work, and the first cut of this
        test did not.**  It drove ``dark_antenna()``, on which C9 stands down
        in its first line (the grid is not square), so a planted ``open()`` in
        C9's live path would have survived: the assertion passed because
        nothing ran.  This document is square AND has a dark antenna branch,
        so both checks reach a finding -- asserted, so the vacuity cannot come
        back.
        """
        import builtins

        document = square(switching=THREE,
                          cal_loads={"ambient": {"t_load": AMBIGUOUS},
                                     "hot": {"t_load": AMBIGUOUS}},
                          model={key: value for key, value in BASE_MODEL.items()
                                 if key not in ("global_signal",
                                                "uniform_sky")})
        document["model"] = {
            key: value for key, value in document["model"].items()
            if key not in ("global_signal", "uniform_sky")}
        run = built_run(document)
        opened = []
        original = builtins.open
        monkeypatch.setattr(
            builtins, "open",
            lambda *a, **k: (opened.append(a[0]), original(*a, **k))[1])
        fired = built(run).checks()
        assert {"B5", "C9"} <= fired, fired
        assert opened == []
