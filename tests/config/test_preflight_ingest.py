"""A10, A45 and A46: the recording's unit, the switch's key, its columns.

Every test names the wrong implementation it kills, because a test that cannot
name one is decoration.  The three rows here are each late for a different
reason and the mutants that matter are different for each:

* **A10** already precedes the beam -- ``document.py`` builds the observation
  before the resources -- so the standard phase test is vacuous for it and
  would pass against the unhoisted code.  What is NOT vacuous is instrumenting
  the READ: measured at ``ea4839b`` with ``hashlib.sha256`` and
  ``Path.read_bytes`` spied on, a document whose ``from_file:`` omits
  ``freq_unit`` records ``['read_bytes', 'sha256']`` before the refusal
  arrives.  After the hoist the same probe records ``[]``.
* **A45 and A46 leg 2 DO lose to the beam today**, so the standard
  :data:`UNREADABLE_BEAM` phase test is meaningful for those two and is shipped
  for both.
* **A46 leg 3 is a WARN**, and ``preflight_helpers.ids()`` cannot express its
  stand-down: a warning is still in ``ids()`` on a document ``A14`` refuses.
  The stand-down is asserted through ``load_document`` inside
  ``warnings.catch_warnings(record=True)`` instead, which is what actually
  decides whether a user hears it.

**No check here walks the variant LAYERS**, and the three
``…selected_variant…`` tests below say why in one place:
``document.py::_assemble`` applies the selected variant BEFORE it runs the
pass, so the route a user actually takes is guarded and what layering would
add is the reporting of faults in variants **nobody selected** -- for every
check in this layer, not only these three.  Measured cold on the shipped
guard's own document, 22 samples each: **22.3 ms unlayered against 37.8 ms
layered, both under the 50 ms bound**.  The walk is dropped for the wave's
arithmetic (+15 ms of a budget with ~22 ms already spent and five siblings
landing), not for a breach.  That guard is load-flaky and reads **52.2 ms at
HEAD under `-n 16`, with no layering at all**; a number taken from it under
load is noise, which is how an earlier draft of this paragraph came to claim a
breach that did not happen.

**Report-level equality is banned here** (§0.3 E.11) and this module is one of
the reasons: a ``noise_wave`` written with literal extents already carries
**A41**, so every document below writes ``{zeros: ['n_source', 'n_freq']}``
instead, and every "and nothing else" assertion is a subset, a negative
membership or :func:`only`.
"""

import copy
import hashlib
import pathlib
import warnings

import pytest

from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import REFUSE, WARN
from rheplicant.config.layering import apply_variant
from rheplicant.config.preflight import CHECKS, preflight
from rheplicant.config.preflight.ingest import (
    _a45_carries_switch_key,
    _freq_unit,
    _switch_key,
    _thermistor_columns,
)
from rheplicant.config.sections.ingest import freq_unit_problem
from rheplicant.config.sections.model import operator_table
from rheplicant.config.sections.pointing import (
    compile_pointing,
    pointing_extra_keys,
)
from tests.config.message_binding import assert_bound_once
from tests.config.preflight_helpers import (
    BASE_MODEL,
    BASE_OBSERVATION,
    UNREADABLE_BEAM,
    ids,
    only,
    preflight_document,
    refusals,
)

h5py = pytest.importorskip("h5py", reason="h5py comes with rheplicant[rhino]")

from tests.config.test_config_section_ingest import make_file  # noqa: E402

#: The check ids this task decides.  Intersected with rather than compared to
#: (``test_preflight_model.py:110``'s idiom): the pass runs every registered
#: check, so an "and nothing else" assertion over the whole report would go red
#: on whichever sibling task first fires on one of these documents.
MINE = frozenset({"A10", "A45", "A46"})

#: The whole refusal A10 says, character for character.  **A HOISTED message**:
#: this literal is ``sections/ingest.py::freq_unit_problem``'s, with the
#: pass's own ``(check A10).`` tail, and
#: :meth:`TestTheOneBinding.test_every_literal_is_bound_once` asserts it is
#: written in exactly one module under ``src/``.
A10_MESSAGE = (
    "from_file: freq_unit is required and has no default -- the file does not "
    "record its frequency unit and its two producers disagree (rhino-cal "
    "writes Hz, the notebook writes MHz; radio/rhino.py's module docstring is "
    "the evidence). (check A10)."
)

#: The switch labels the shipped recording's log carries, in order.  The
#: ANTENNA is one of them, and that is the fact leg 3 turns on: the reader
#: demands a column for every label here, the file has two columns, so a
#: shared column is forced and is legal.
RECORDED_ORDER = ["antenna", "internal_load", "heated_load"]

#: The column map the canonical working document declares -- ``antenna`` and
#: ``internal_load`` share column 0, and the two LOADS do not.
WORKING_COLUMNS = {"antenna": 0, "internal_load": 0, "heated_load": 1}

#: A ``noise_wave`` spec's other six fields.  Symbolic extents throughout, so
#: no document in this module carries an unrelated A41 (§0.3 E.11).
_NOISE_WAVE_FIELDS = {
    "t_unc": {"zeros": ["n_freq"]}, "t_cos": {"zeros": ["n_freq"]},
    "t_sin": {"zeros": ["n_freq"]}, "t_rx": {"zeros": ["n_freq"]},
    "gamma_rec_re": {"zeros": ["n_freq"]},
    "gamma_rec_im": {"zeros": ["n_freq"]},
    "gamma_src_re": {"zeros": ["n_source", "n_freq"]},
    "gamma_src_im": {"zeros": ["n_source", "n_freq"]},
}

#: The class name the ``python:`` hatch relocates a noise wave with -- the
#: spelling a check keyed on ``type:`` cannot see.
NOISE_WAVE_CLASS = "rheplicant.radio:NoiseWaveOperator"

#: A three-label switch cycle behind the antenna, for the SYNTHETIC route.
SWITCHED = ["antenna", "ambient", "hot"]


def noise_wave(**extra):
    """A ``model.noise_wave`` spec, with ``extra`` written over its fields."""
    return {**_NOISE_WAVE_FIELDS, **extra}


def cal_loads(order=SWITCHED):
    """``model.cal_loads`` matching ``order[1:]``, which A14 wants present."""
    return {label: {"t_load": {"value": 300.0, "unit": "K"}}
            for label in order[1:]}


def switched_document(node=None, **patch):
    """The base document with a switch cycle and a ``noise_wave`` on it.

    The SYNTHETIC half of A45's ingested/synthetic twin: ``switching: {mode:
    cycle}`` is what makes ``receiver_input`` a key ``build_observation``
    itself writes, where an ingested run gets it from ``to_state`` afterwards.
    """
    document = preflight_document(
        model={**BASE_MODEL,
               "noise_wave": noise_wave() if node is None else node,
               "cal_loads": cal_loads()},
        observation={**BASE_OBSERVATION,
                     "switching": {"mode": "cycle", "order": list(SWITCHED),
                                   "dwell": 4}},
        **patch)
    return document


def recording(tmp_path):
    """Write the shipped test recording and hand back its directory."""
    make_file(tmp_path / "obs.hd5f")
    return str(tmp_path)


def ingested_document(*, columns="working", loads="both", freq_unit="MHz",
                      **patch):
    """A document whose observation IS a recording -- the INGESTED route.

    Hand-built rather than patched over :func:`preflight_document`, because an
    ingested run declares no ``freq:`` and no ``time:`` (the recording carries
    both) and ``build_observation`` refuses a document that declares either
    beside ``from_file:``.  This is
    ``test_config_section_model.py``'s canonical working document, which
    LOADS: measured, ``run.twin['cal_loads_1']`` is a ``CalLoadOperator`` at
    293.15 K.
    """
    from_file = {"format": "rhino_hdf5", "path": "obs.hd5f",
                 "settle_seconds": 0.0}
    if freq_unit is not None:
        from_file["freq_unit"] = freq_unit
    if columns == "working":
        columns = dict(WORKING_COLUMNS)
    if columns is not None:
        from_file["thermistor_columns"] = columns
    model = {"gain": {"gain": {"value": 2.0, "unit": "dimensionless"}}}
    if loads == "both":
        loads = {label: {"from": "thermistors", "label": label}
                 for label in RECORDED_ORDER[1:]}
    if loads is not None:
        model["cal_loads"] = loads
    document = {
        "schema_version": 1,
        "runtime": {"seed": 1},
        "observation": {"from_file": from_file,
                        "switching": {"order": list(RECORDED_ORDER)}},
        "model": model,
        "runs": [{"kind": "forward"}],
    }
    document.update(patch)
    return document


def load(document, tmp_path):
    """``load_document`` on a document whose recording lives in ``tmp_path``.

    Imported inside the helper: ``load_document`` is what every phase test
    here drives, and importing it at module scope would put the name in this
    module's namespace beside the pass it is meant to run BEFORE.
    """
    from rheplicant.config import load_document

    return load_document(document, base_dir=str(tmp_path))


class _ReadSpy:
    """Records whether the recording was slurped and hashed.

    Both verbs, because they are two halves of one line -- ``config/files.py``
    writes ``hashlib.sha256(path.read_bytes())``, so the argument is evaluated
    first and the order in the record is ``read_bytes`` then ``sha256``.  The
    ruling that mandates this probe wrote it the other way round; the
    repository wins (R6) and the assertion is on the SET, not the order.
    """

    def __init__(self, monkeypatch):
        self.seen = []
        real_sha, real_read = hashlib.sha256, pathlib.Path.read_bytes

        def sha256(*args, **kwargs):
            self.seen.append("sha256")
            return real_sha(*args, **kwargs)

        def read_bytes(target, *args, **kwargs):
            self.seen.append("read_bytes")
            return real_read(target, *args, **kwargs)

        monkeypatch.setattr(hashlib, "sha256", sha256)
        monkeypatch.setattr(pathlib.Path, "read_bytes", read_bytes)


class TestA10TheUnitIsAskedBeforeTheFileIsOpened:
    def test_the_missing_key_is_refused_by_the_pass(self):
        """Kills: not registering the check at all (R1).  Deleting
        ``@register("A10")`` leaves the whole shipped suite green -- the
        reader's own refusal still fires -- and turns this test red."""
        finding = only(ingested_document(freq_unit=None), "A10")
        assert finding.severity == REFUSE
        assert finding.where == "observation.from_file"

    def test_the_message_is_the_sections_own_sentence_whole(self):
        """Kills: any rewording of a HOISTED message.

        Equality on the whole text, because the pin that exists today is
        ``pytest.raises(ConfigError, match="freq_unit")`` -- eight characters
        of two hundred and thirty, which survives every rewrite of the clause
        a reader acts on."""
        assert only(ingested_document(freq_unit=None), "A10").message == \
            A10_MESSAGE

    def test_the_pass_and_the_section_say_one_sentence(self):
        """Kills: a hoist that COPIES the message rather than calling it.

        The pass's message must be the section function's plus the pass's own
        tail, and nothing else."""
        assert A10_MESSAGE == f"{freq_unit_problem({})} (check A10)."

    def test_the_sections_own_refusal_is_unchanged(self, tmp_path):
        """Kills: an extraction that changed what the reader raises.

        ``tests/config/test_config_section_ingest.py::
        test_freq_unit_is_required_with_no_default`` is the ``match=`` pin
        that must still pass; this is the equality half §0.3 F.3 requires
        beside it, because a ``match=`` pin does not prove a hoist."""
        from rheplicant.config import ResolutionContext, resolve_value

        make_file(tmp_path / "obs.hd5f")
        context = ResolutionContext(dtype="float32", base_dir=str(tmp_path))
        with pytest.raises(ConfigError) as caught:
            resolve_value(
                {"file": {"path": "obs.hd5f", "format": "rhino_hdf5"}},
                context)
        assert str(caught.value).endswith(freq_unit_problem({}))

    def test_a_declared_unit_earns_nothing_and_the_document_loads(
            self, tmp_path):
        """S4's second half: take the check's advice -- declare ``freq_unit``
        -- and the document it refused now passes.

        Kills: a check that fires on presence as well as absence."""
        document = ingested_document()
        assert ids(document) & MINE == frozenset()
        assert load(document, recording(tmp_path)) is not None

    def test_neither_the_digest_nor_the_read_is_reached(self, tmp_path,
                                                        monkeypatch):
        """The phase property, instrumented on the READ rather than on a beam.

        A10 already ran before ``build_resources`` at ``ea4839b`` -- so an
        ``UNREADABLE_BEAM`` phase test passes against the unhoisted code and
        proves nothing.  What the hoist actually buys is this: measured before
        it, the same probe records ``['read_bytes', 'sha256']``.

        Kills: leaving the question inside the reader."""
        base_dir = recording(tmp_path)
        spy = _ReadSpy(monkeypatch)
        with pytest.raises(ConfigError) as caught:
            load(ingested_document(freq_unit=None), base_dir)
        assert str(caught.value).startswith(A10_MESSAGE)
        assert spy.seen == []

    def test_that_probe_can_still_see_a_file_being_read(self, tmp_path,
                                                        monkeypatch):
        """ANTI-VACUITY for the probe above.  A spy that records nothing on a
        document that DOES read the recording is a green assertion measuring
        its own monkeypatch."""
        base_dir = recording(tmp_path)
        spy = _ReadSpy(monkeypatch)
        load(ingested_document(), base_dir)
        assert {"read_bytes", "sha256"} <= set(spy.seen)

    def test_the_value_node_twin_is_walked_too(self):
        """S3, and A10's real twin: ANY ``{file: {format: rhino_hdf5}}`` node
        reaches the same reader behind the same digest.

        Measured at ``resources.arrays.<n>``: the identical refusal, after the
        identical ``read_bytes``+``sha256``.

        Kills: a check that reads ``observation.from_file`` and nothing else
        -- the shape every 3A task shipped."""
        document = preflight_document(
            resources={"arrays": {"rec": {"file": {"path": "obs.hd5f",
                                                   "format": "rhino_hdf5"}}}})
        finding = only(document, "A10")
        assert finding.where == "resources.arrays.rec"
        assert finding.message == A10_MESSAGE

    def test_the_twin_replace_route_is_walked_too(self):
        """§0.3 E.10: ``inference.twin.replace.<node>`` reaches the same
        builders and is outside ``preflight/model.py::_nodes``.

        A10 walks it because it walks every section but ``variants:``.

        Kills: a section list that names ``observation`` and ``resources``."""
        document = preflight_document()
        document["inference"]["twin"]["replace"] = {"gain": {"gain": {
            "file": {"path": "obs.hd5f", "format": "rhino_hdf5"}}}}
        assert only(document, "A10").where == \
            "inference.twin.replace.gain.gain"

    def test_the_selected_variant_is_read_and_the_unselected_one_is_not(self):
        """S3: the ``variants:`` route, and exactly how much of it is guarded.

        ``document.py::_assemble`` applies the variant and THEN runs the pass,
        so the variant a user selects IS checked -- that is the half that
        matters and it is asserted here.  Layering over every DECLARED variant
        would add the un-selected ones and nothing else.  Measured cold on the
        shipped guard's document, 22 samples each: 22.3 ms unlayered against
        37.8 ms layered, both under the 50 ms bound.  The walk is dropped for
        the wave's arithmetic, not for a breach; the un-selected variant is a
        recorded false negative."""
        document = preflight_document(
            observation={**BASE_OBSERVATION},
            variants={"recorded": {"observation": {"from_file": {
                "format": "rhino_hdf5", "path": "obs.hd5f"}}}})
        assert "A10" not in ids(document)
        finding = only(apply_variant(document, "recorded"), "A10")
        assert finding.where == "observation.from_file"
        assert finding.message == A10_MESSAGE

    def test_a_selected_variants_file_node_is_not_reported_twice(self):
        """Kills: a walk that descends into ``variants:`` as well as into the
        document proper.

        ``apply_variant`` leaves ``variants:`` in the merged mapping, so
        without the skip the SELECTED variant's own patch is walked too and
        the same fault is reported at two paths.  Found by mutation: the test
        above cannot see it, because a patch writing ``observation.from_file``
        carries no ``{file:}`` wrapper for the recursive walk to find."""
        document = preflight_document(
            variants={"recorded": {"resources": {"arrays": {"rec": {
                "file": {"path": "obs.hd5f", "format": "rhino_hdf5"}}}}}})
        finding = only(apply_variant(document, "recorded"), "A10")
        assert finding.where == "resources.arrays.rec"

    def test_another_file_format_stands_down(self):
        """Kills: a check keyed on the presence of ``file:`` rather than on
        the format whose reader demands the key.  ``freq_unit`` is
        ``rhino_hdf5``'s, and ``files.py`` refuses it as an unknown key
        anywhere else."""
        document = preflight_document(
            resources={"arrays": {"rec": {"file": {"path": "d.npy",
                                                   "format": "npy"}}}})
        assert "A10" not in ids(document)

    def test_a_from_file_of_another_format_stands_down(self):
        """Kills: a check that reads ``observation.from_file`` without asking
        its format.  ``parse_from_file`` refuses a non-``rhino_hdf5`` format
        by name, which is the more specific sentence (§2.3)."""
        document = ingested_document(freq_unit=None)
        document["observation"]["from_file"]["format"] = "npz"
        assert "A10" not in ids(document)


class TestA45TheSwitchKeyNamesAKeyTheRunWrites:
    def test_a_key_the_run_never_writes_is_refused(self):
        """Kills: not shipping the check.  Measured, this document LOADS with
        an empty pre-flight report and the twin refuses only when it is
        evaluated -- as a ``StateValidationError`` naming
        ``coords.extra['my_switch']`` and no key of the document."""
        document = switched_document(noise_wave(switch_key="my_switch"))
        finding = only(document, "A45")
        assert finding.severity == REFUSE
        assert finding.where == "model.noise_wave.switch_key"

    def test_the_message_is_pinned_whole(self):
        """S1 for an INVENTED message: equality on the whole sentence, with
        the literal in the test."""
        document = switched_document(noise_wave(switch_key="my_switch"))
        assert only(document, "A45").message == (
            "model.noise_wave.switch_key: 'my_switch' is not a key this run "
            "writes into coords.extra, so the operator has no switch index to "
            "read -- with more than one source the twin refuses the moment it "
            "is evaluated, and with one it silently takes the first. This run "
            "writes ['receiver_input']; observation.extra and "
            "observation.pointing's materialise:/lst: are where another one "
            "would come from (check A45)."
        )

    def test_the_default_key_earns_nothing_and_the_document_loads(self):
        """S4's second half: the message says ``receiver_input`` is written;
        writing it makes the refusal go away and the document build."""
        document = switched_document(noise_wave(switch_key="receiver_input"))
        assert "A45" not in ids(document)
        from rheplicant.config import load_document

        assert load_document(document) is not None

    def test_the_field_omitted_entirely_earns_nothing(self):
        """Kills: a check that fires on absence.  ``switch_key`` is defaulted
        on the class, so omitting it is the ordinary spelling."""
        assert "A45" not in ids(switched_document())

    def test_receiver_input_is_accepted_on_an_ingested_run(self):
        """The measured trap: on an ingested run ``switching.receiver_input``
        is ``None`` at ``build_observation`` and the key arrives from
        ``to_state`` afterwards.

        Kills: ``receiver_input`` accepted only when ``switching.mode:
        cycle`` -- which refuses the ingested document this module is about.
        That naive gate is the one a reader writes from
        ``sections/switching.py``."""
        document = ingested_document()
        document["model"]["noise_wave"] = noise_wave(
            switch_key="receiver_input")
        assert "A45" not in ids(document)

    def test_the_message_lists_every_key_the_run_writes(self):
        """S1, on a document whose legal set is NOT a singleton.

        The interpolated set IS the advice: it is the list of keys the user
        may legally switch on.  The whole-message pin above runs on a document
        that writes only ``receiver_input``, and measured, replacing
        ``f"writes {sorted(written)}"`` with a hardcoded ``['receiver_input']``
        SURVIVED the whole of ``tests/config`` -- a user with a
        ``lst_deg``-writing pointing block would have been sent to the wrong
        fix and nothing would have noticed.

        Kills exactly that."""
        document = switched_document(noise_wave(switch_key="my_switch"))
        document["observation"] = {
            **document["observation"],
            "pointing": {"mode": "drift", "materialise": ["selfrot_deg"],
                         "lst": {"mode": "uniform_turn"}}}
        assert only(document, "A45").message == (
            "model.noise_wave.switch_key: 'my_switch' is not a key this run "
            "writes into coords.extra, so the operator has no switch index to "
            "read -- with more than one source the twin refuses the moment it "
            "is evaluated, and with one it silently takes the first. This run "
            "writes ['lst_deg', 'receiver_input', 'selfrot_deg']; "
            "observation.extra and observation.pointing's materialise:/lst: "
            "are where another one would come from (check A45)."
        )

    def test_a_key_observation_extra_writes_is_accepted(self):
        """Kills: a legal set of exactly ``{'receiver_input'}``."""
        document = switched_document(noise_wave(switch_key="my_switch"))
        document["observation"] = {
            **document["observation"],
            "extra": {"my_switch": {"zeros": ["n_time"]}}}
        assert "A45" not in ids(document)

    @pytest.mark.parametrize(("pointing", "key"), [
        ({"mode": "drift", "materialise": ["selfrot_deg"]}, "selfrot_deg"),
        ({"mode": "drift", "materialise": ["pointing"],
          "lst": {"mode": "uniform_turn"}}, "lst_deg"),
    ], ids=["materialise-selfrot", "lst"])
    def test_a_key_pointing_writes_is_accepted(self, pointing, key):
        """S3's named twin: ``pointing.extra`` against ``observation.extra``.

        ``compile_pointing`` writes ``lst_deg`` and ``selfrot_deg`` into
        ``pointing.extra`` and ``build_observation`` MERGES them in, so a
        check reading only ``observation.extra`` refuses documents that build.

        Kills exactly that reading -- which is what §6's one-liner invites."""
        document = switched_document(noise_wave(switch_key=key))
        document["observation"] = {**document["observation"],
                                   "pointing": pointing}
        assert "A45" not in ids(document)

    def test_the_python_spelling_is_read_too(self):
        """Kills the naive implementation: keying on the literal ``type:
        NoiseWaveOperator``.

        Measured, ``python: 'rheplicant.radio:NoiseWaveOperator'`` builds the
        same class at the same node, and Plan 3A's tests already exercise the
        spelling.  The class is resolved, never the token."""
        document = switched_document(
            noise_wave(python=NOISE_WAVE_CLASS, switch_key="my_switch"))
        assert only(document, "A45").where == "model.noise_wave.switch_key"

    def test_a_class_without_the_field_is_not_asked(self):
        """Kills: a check keyed on the KEY ``switch_key`` wherever it appears.

        A ``GainOperator`` given a ``switch_key`` is refused by ``_construct``
        -- *"does not take ['switch_key']"* -- which names the class.  Saying
        "is not a key coords.extra will carry" about it is the vaguer of the
        two sentences (§2.3)."""
        document = preflight_document(
            model={**BASE_MODEL, "gain": {
                "gain": {"value": 1.0, "unit": "dimensionless"},
                "switch_key": "my_switch"}})
        assert "A45" not in ids(document)

    def test_the_selector_field_is_not_a_document_key(self):
        """Kills: widening onto ``core/combinators.py::SelectOperator``, which
        carries a ``switch_key`` too -- defaulted, and set by the fold.

        It is not config-writable, and the proof is that no graph node
        registers it, so no document can name it and this check can never
        reach it."""
        from rheplicant.core.combinators import SelectOperator

        registered = {cls for classes in operator_table().values()
                      for cls in classes}
        assert SelectOperator not in registered
        assert _a45_carries_switch_key(SelectOperator)

    def test_a_non_mapping_observation_extra_stands_down(self):
        """S4: do not pre-empt.  ``_a45_written_keys`` answers ``None`` -- NOT
        ``frozenset()`` -- for an ``observation.extra`` this layer cannot read,
        because an empty legal set would refuse every ``switch_key`` in the
        document on the strength of a section nobody can parse.
        ``sections/observation.py`` refuses the SHAPE with the type it got.

        Kills: returning ``frozenset()`` there -- measured, that mutant makes
        A45 fire and survives the whole of ``tests/config``."""
        document = switched_document(noise_wave(switch_key="my_switch"))
        document["observation"] = {**document["observation"], "extra": ["a"]}
        assert "A45" not in ids(document)
        from rheplicant.config import load_document

        with pytest.raises(ConfigError) as caught:
            load_document(document)
        assert str(caught.value) == \
            "observation.extra: is a mapping; got list."

    def test_a_pipeline_models_twin_block_is_not_asked(self):
        """S4: do not pre-empt.  ``build_fit_twin`` refuses the WHOLE
        ``inference.twin:`` block on a ``kind: pipeline`` model, so a
        ``switch_key`` inside a block about to be rejected wholesale is not
        this check's sentence.

        Kills: dropping the ``model.kind != graph`` guard on the replace half
        -- measured, that mutant makes A45 fire inside a pipeline document's
        twin block and survives the suite.  (``_nodes`` already answers ``{}``
        for such a model, so the MODEL half needs no guard and this is the
        only place the question arises.)"""
        document = preflight_document()
        document["model"] = {"kind": "pipeline", "stages": [
            {"name": "gain", "type": "GainOperator",
             "gain": {"value": 1.1, "unit": "dimensionless"}}]}
        document["inference"]["twin"]["replace"] = {"noise_wave": {
            "type": "NoiseWaveOperator", **noise_wave(switch_key="nope")}}
        assert "A45" not in ids(document)
        from rheplicant.config import load_document

        with pytest.raises(ConfigError) as caught:
            load_document(document)
        assert "A pipeline is rebuilt, not repaired" in str(caught.value)

    def test_a_class_that_is_not_a_dataclass_answers_false(self):
        """The ``is_dataclass`` guard, defended rather than recorded.

        No DOCUMENT can reach it -- ``_a15_declared_class`` answers either
        ``None`` or a shipped operator class, and every one of those is an
        equinox ``Module`` and therefore a dataclass.  A direct call can,
        which is what makes this an ordinary test rather than §0.3 F.5 (10)'s
        undefendable-guard note: without the guard, ``field_specs`` raises
        ``TypeError`` here, and a raise inside a check costs the document
        every other finding (§2.3's TRAP)."""
        class NotADataclass:
            switch_key = "receiver_input"

        assert _a45_carries_switch_key(NotADataclass) is False
        assert _a45_carries_switch_key(None) is False

    def test_the_twin_replace_route_is_walked(self):
        """§0.3 E.10, and A45 is the row it matters most for: measured,
        ``replace.noise_wave.switch_key: nope`` LOADS CLEAN and detonates at
        fit time -- strictly worse than the ``model:`` route this check was
        written for.

        Kills: a walk over ``document['model']`` alone."""
        document = switched_document()
        document["inference"]["twin"]["replace"] = {"noise_wave": {
            "type": "NoiseWaveOperator", **noise_wave(switch_key="nope")}}
        finding = only(document, "A45")
        assert finding.where == "inference.twin.replace.noise_wave.switch_key"
        assert finding.message.startswith(
            "inference.twin.replace.noise_wave.switch_key: 'nope' is not a "
            "key this run writes into coords.extra")

    def test_a_compose_stage_is_walked(self):
        """A further twin, found rather than given: ``_t4_entries`` expands a
        ``compose:`` block's stages, so a ``switch_key`` one level down is
        read.  ``preflight/observing.py::_a15_sites`` -- the sibling walk this
        one is modelled on -- records that shape as a silent false negative.

        Kills: reading the node spec and never its stages."""
        document = switched_document({
            "compose": "cascade",
            "stages": [noise_wave(type="NoiseWaveOperator",
                                  switch_key="my_switch")]})
        assert only(document, "A45").where == \
            "model.noise_wave.stages[0].switch_key"

    def test_a_selected_variant_that_renames_the_key_is_read(self):
        """S3: the ``variants:`` route, guarded where it is taken -- the pass
        runs on the variant-applied mapping."""
        document = switched_document()
        document["variants"] = {
            **document.get("variants", {}),
            "other_switch": {"model": {"noise_wave": {
                "switch_key": "my_switch"}}}}
        assert "A45" not in ids(document)
        assert only(apply_variant(document, "other_switch"), "A45").where == \
            "model.noise_wave.switch_key"

    def test_a_non_string_key_stands_down(self):
        """S4: do not pre-empt.  ``deliver``'s ``static_str`` rule refuses a
        list with the type it got; "is not a key coords.extra will carry" is
        the vaguer sentence about the same line."""
        document = switched_document(noise_wave(switch_key=["a", "b"]))
        assert "A45" not in ids(document)

    def test_the_beam_does_not_win(self):
        """The standard phase test, which IS meaningful for this row:
        measured, A45 loses to an unreadable beam today.

        Kills: leaving the check anywhere after ``build_resources``."""
        document = switched_document(
            noise_wave(switch_key="my_switch"),
            resources=UNREADABLE_BEAM)
        assert "A45" in {one.check for one in refusals(document)}
        from rheplicant.config import load_document

        with pytest.raises(ConfigError) as caught:
            load_document(document)
        assert "no_such_beam.npy" not in str(caught.value)
        assert str(caught.value).startswith("model.noise_wave.switch_key: ")


class TestA46Leg2TheColumnALabelHas:
    def test_a_label_with_no_column_is_refused(self):
        """Kills: not shipping leg 2.  Measured, this document is refused from
        inside the reader -- ``DataIngestionError``, behind the whole
        recording, naming no key of the document."""
        document = ingested_document(
            loads={"internal_load": {"from": "thermistors",
                                     "label": "internal_load"},
                   "heated_load": {"from": "thermistors", "label": "ghost"}})
        finding = only(document, "A46")
        assert finding.severity == REFUSE
        assert finding.where == "model.cal_loads.heated_load"

    def test_the_message_is_pinned_whole(self):
        """S1 for an INVENTED message.

        ``radio/rhino.py`` is READ-ONLY for this task and its literal is not
        reused: that refusal reads ``obs.thermistor_k`` and quotes the file's
        own contents, neither of which exists at P-1."""
        document = ingested_document(
            loads={"internal_load": {"from": "thermistors",
                                     "label": "internal_load"},
                   "heated_load": {"from": "thermistors", "label": "ghost"}})
        assert only(document, "A46").message == (
            "model.cal_loads.heated_load: label: 'ghost' has no entry in "
            "observation.from_file.thermistor_columns, so this load's t_load "
            "is asked for a column the recording was never read with, and the "
            "refusal comes from inside the reader naming no key of this "
            "document. Declare thermistor_columns with a column for 'ghost' "
            "AND for every other switch label the recording visits, the "
            "antenna included -- the reader refuses a partial map (check A46)."
        )

    def test_thermistor_columns_absent_altogether_still_fires(self):
        """Kills the naive gate the task body proposed: *"emit only when the
        key is present"*.

        Measured, a document with ``from_file:`` and no ``thermistor_columns``
        beside a ``from: thermistors`` load is refused at
        ``rhino.py``'s ``cal_load_operators`` -- *"This observation carries no
        thermistor temperatures"* -- behind the whole recording, with a
        message naming no document key.  A key-presence gate says nothing
        about it."""
        document = ingested_document(columns=None)
        found = [one for one in preflight(document).findings
                 if one.check == "A46"]
        assert {one.where for one in found} == {
            "model.cal_loads.internal_load", "model.cal_loads.heated_load"}

    def test_it_stands_down_when_there_is_no_recording(self, tmp_path):
        """S4's stand-down: a document wrong in this check's way AND wrong in
        a way something else says better.

        With no ``observation.from_file`` at all, every thermistor label is
        equally uncolumned -- and ``sections/model.py`` already says the
        sentence that names the route to fix, which this check must not
        pre-empt."""
        document = preflight_document(
            model={**BASE_MODEL, "cal_loads": {
                "ambient": {"from": "thermistors", "label": "ambient"},
                "hot": {"from": "thermistors", "label": "hot"}}},
            observation={**BASE_OBSERVATION,
                         "switching": {"mode": "cycle",
                                       "order": list(SWITCHED), "dwell": 4}})
        assert "A46" not in ids(document)
        from rheplicant.config import load_document

        with pytest.raises(ConfigError) as caught:
            load_document(document)
        assert "declares no observation.from_file" in str(caught.value)

    def test_the_key_is_not_the_subject_the_label_is(self, tmp_path):
        """Kills the naive implementation §6 points at: constraining the
        ``cal_loads`` KEY.

        Measured, ``sections/model.py`` does ``cal_load_operators(
        context.ingest, labels=[label])[label]``, so the ``label:`` VALUE
        indexes ``thermistor_k``; the key is ``A14.cal_loads``' subject.  A
        CROSSED document -- keys and labels swapped -- LOADS today and must
        not be refused here.  (What it silently does is give each switch
        position the other one's physical temperature; that is a real defect
        and a different check's.)"""
        document = ingested_document(
            loads={"internal_load": {"from": "thermistors",
                                     "label": "heated_load"},
                   "heated_load": {"from": "thermistors",
                                   "label": "internal_load"}})
        assert "A46" not in ids(document)
        # BOTH halves of §0.3 E.9 ruling 3.  "earns no A46" alone would stay
        # green the day some other check started refusing a document that
        # works, and the ruling's whole point is that this one works.
        assert load(document, recording(tmp_path)) is not None

    def test_a_value_node_load_is_not_asked(self):
        """Kills: a check that reads every ``model.cal_loads`` entry.  A load
        whose ``t_load`` is a value node never touches the thermistor log."""
        document = ingested_document(
            loads={"internal_load": {"t_load": {"value": 300.0, "unit": "K"}},
                   "heated_load": {"t_load": {"value": 350.0, "unit": "K"}}})
        assert "A46" not in ids(document)

    def test_the_advice_applied_leaves_the_document_loading(self, tmp_path):
        """S4's second half, and the pre-identified advice LOOP (R4).

        The message names *every* switch label the recording visits, the
        antenna included -- not ``switching.order[1:]``.  Applied in full, the
        document loads."""
        document = ingested_document(columns=None)
        assert "A46" in ids(document)
        document["observation"]["from_file"]["thermistor_columns"] = \
            dict(WORKING_COLUMNS)
        assert "A46" not in ids(document)
        assert load(document, recording(tmp_path)) is not None

    def test_advice_that_names_only_the_loads_is_still_refused(self,
                                                              tmp_path):
        """Why the advice says "the antenna included": a column map covering
        ``switching.order[1:]`` and no more satisfies THIS check and is still
        refused by the reader, which demands a column for every label in the
        switch log.

        Kills: a message advising ``switching.order[1:]``, which would be
        advice that cannot be followed."""
        document = ingested_document(
            columns={label: index for index, label
                     in enumerate(RECORDED_ORDER[1:])})
        assert "A46" not in ids(document)
        with pytest.raises(ConfigError) as caught:
            load(document, recording(tmp_path))
        assert "thermistor_columns has no entry for ['antenna']" in \
            str(caught.value)

    def test_a_recording_of_another_format_stands_down(self, tmp_path):
        """S4: do not pre-empt.  ``thermistor_columns`` is a ``rhino_hdf5``
        key, so on any other format ``parse_from_file`` refuses the document
        with the sentence that names the real fault.

        Kills: dropping the ``format != rhino_hdf5`` guard -- measured, that
        mutant makes A46 fire on a ``format: npz`` recording and survives the
        whole of ``tests/config``."""
        document = ingested_document(columns=None)
        document["observation"]["from_file"]["format"] = "npz"
        assert "A46" not in ids(document)
        with pytest.raises(ConfigError) as caught:
            load(document, recording(tmp_path))
        assert str(caught.value) == (
            "observation.from_file: format is 'rhino_hdf5' (the one ingestion "
            "format this layer reads); got 'npz'."
        )

    def test_a_column_map_that_is_not_a_mapping_stands_down(self, tmp_path):
        """S4: do not pre-empt.  ``sections/ingest.py`` refuses the SHAPE with
        the value it got; "has no entry in thermistor_columns" about a list is
        the vaguer of the two sentences.

        Kills: dropping the non-``Mapping`` guard -- measured, that mutant
        makes leg 2 fire on a list-valued map and survives the suite."""
        document = ingested_document(columns=[0, 1])
        assert "A46" not in ids(document)
        with pytest.raises(ConfigError) as caught:
            load(document, recording(tmp_path))
        assert str(caught.value) == (
            "from_file: thermistor_columns is a mapping of switch label -> "
            "integer column of /temperatures; got [0, 1]."
        )

    def test_an_empty_label_stands_down(self, tmp_path):
        """S4: do not pre-empt.  ``sections/model.py`` refuses an empty
        ``label:`` by name; asking whether ``''`` has a thermistor column
        answers a question the user did not ask.

        Kills: dropping the empty-``label:`` skip -- measured, that mutant
        makes leg 2 fire on ``label: ""`` and survives the suite."""
        document = ingested_document(loads={
            "internal_load": {"from": "thermistors", "label": ""},
            "heated_load": {"from": "thermistors", "label": "heated_load"}})
        assert "A46" not in ids(document)
        with pytest.raises(ConfigError) as caught:
            load(document, recording(tmp_path))
        assert str(caught.value) == (
            "model.cal_loads: from: thermistors requires label: -- the switch "
            "label whose thermistor column becomes t_load."
        )

    def test_the_twin_replace_route_is_walked(self):
        """§0.3 E.10.  Measured, ``replace: {cal_loads: {from: thermistors,
        label: ghost}}`` reaches ``cal_load_operators`` and its
        ``DataIngestionError``."""
        document = ingested_document(inference={
            "twin": {"replace": {"cal_loads": {"from": "thermistors",
                                               "label": "ghost"}}},
            "parameters": {"g": {"init": 1.0, "into": "gain.gain"}},
            "observed": {"from": "simulation"},
            "noise": {"kind": "homoscedastic",
                      "sigma": {"value": 1.0, "unit": "K"}}})
        assert only(document, "A46").where == \
            "inference.twin.replace.cal_loads"

    def test_a_selected_variant_that_drops_a_column_is_read(self):
        """S3: the ``variants:`` route, guarded where it is taken."""
        document = ingested_document()
        document["variants"] = {"partial": {"observation": {"from_file": {
            "thermistor_columns": {"~heated_load": None}}}}}
        assert "A46" not in ids(document)
        assert only(apply_variant(document, "partial"), "A46").where == \
            "model.cal_loads.heated_load"

    def test_the_beam_does_not_win(self, tmp_path):
        """The standard phase test: measured, A46 leg 2 loses to an unreadable
        beam today, because ``build_resources`` runs before ``build_model``."""
        document = ingested_document(columns=None,
                                     resources=copy.deepcopy(UNREADABLE_BEAM))
        with pytest.raises(ConfigError) as caught:
            load(document, recording(tmp_path))
        assert "no_such_beam.npy" not in str(caught.value)
        assert str(caught.value).startswith("model.cal_loads.")


class TestA46Leg3TheColumnTwoLoadsShare:
    def _sharing(self, shared=0, **patch):
        """The two loads collide on ``shared``; the antenna takes the other.

        ``shared`` is a PARAMETER and not the constant 0 it started as: with
        one document the message's ``share column {column}`` clause is pinned
        only where the answer happens to be zero, and measured, hardcoding
        ``{0}`` there survived the whole of ``tests/config``.  The number is
        the column the reader must go and change."""
        other = 1 - shared
        return ingested_document(
            columns={"antenna": other, "internal_load": shared,
                     "heated_load": shared},
            **patch)

    def test_two_loads_on_one_column_earn_a_warning(self):
        """Kills: not shipping leg 3, and shipping it as a REFUSAL.

        Measured, this document LOADS -- both operators carry the same
        physical temperature and nothing says so."""
        finding = only(self._sharing(), "A46")
        assert finding.severity == WARN
        assert finding.where == "model.cal_loads"

    @pytest.mark.parametrize("shared", [0, 1], ids=["column-0", "column-1"])
    def test_the_message_is_pinned_whole(self, shared):
        """S1 for an INVENTED message, on BOTH columns the shipped recording
        has.  The remedy it names EXISTS: two columns, two loads.

        Kills: a message that hardcodes column 0 -- wrong advice on every
        recording whose loads collide anywhere else, and leg 3's whole purpose
        is to point at a column."""
        assert only(self._sharing(shared), "A46").message == (
            f"model.cal_loads: ['heated_load', 'internal_load'] share column "
            f"{shared} of observation.from_file.thermistor_columns, so their "
            "load operators carry one physical temperature between them and "
            "the calibration cannot tell those loads apart. Give each load "
            "the column its own thermistor was recorded in; a column shared "
            "with a label no load reads -- the antenna -- is legal and the "
            "file's own map often forces it (check A46)."
        )

    def test_the_canonical_working_document_earns_no_warning(self):
        """Kills the specified implementation: comparing the WHOLE
        ``thermistor_columns`` map.

        Measured, ``_thermistors_in_kelvin`` demands a column for every label
        the switch log carries -- ``antenna`` included -- and the shipped
        ``/temperatures/temperatures`` has 2 columns for 3 labels, so sharing
        is FORCED and ``radio/rhino.py`` documents it as legal.  A leg 3 over
        the whole map warns on the document the package ships tests for."""
        document = ingested_document()
        assert document["observation"]["from_file"]["thermistor_columns"][
            "antenna"] == document["observation"]["from_file"][
            "thermistor_columns"]["internal_load"]
        assert ids(document) & MINE == frozenset()

    def test_the_warning_is_said_out_loud_when_the_document_loads(
            self, tmp_path):
        """Kills: a WARN that never reaches a user.  ``ids()`` cannot express
        this -- it holds warnings too -- so the assertion is on what
        ``load_document`` actually emits."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load(self._sharing(), recording(tmp_path))
        assert any("share column 0" in str(one.message) for one in caught)

    def test_a_refused_document_never_says_it(self, tmp_path):
        """S4: leg 3 must not fire on a document ``A14`` already refuses, and
        it needs no rule of its own for that -- ``load_document`` calls
        ``raise_if_refused()`` before ``emit_warnings()``.

        Kills: adding a second ordering rule inside the check, and moving the
        raise-then-warn order in ``document.py``.  Note ``ids()`` still
        CONTAINS A46 here, which is exactly why this test drives
        ``load_document`` instead."""
        document = self._sharing(
            loads={"wrong_key": {"from": "thermistors",
                                 "label": "internal_load"},
                   "heated_load": {"from": "thermistors",
                                   "label": "heated_load"}})
        assert "A46" in ids(document)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ConfigError) as raised:
                load(document, recording(tmp_path))
        assert "check A14" in str(raised.value)
        assert not [one for one in caught if "share column" in str(one.message)]

    def test_a_replace_load_is_not_compared_against_the_models(self):
        """S4, and the restriction ``_a46_loads`` records rather than guesses.

        A ``replace:`` mapping carries at most ONE ``cal_loads`` operator, and
        it lives in a different twin from the model's own loads -- comparing
        the two would be a claim about two twins at once.  Here the model
        reads column 0 and the replacement reads column 0, and leg 3 is
        silent.

        Kills: widening leg 3 to ``model_loads + replace_loads`` -- measured,
        that mutant warns on this document and survives the suite."""
        document = ingested_document(
            columns={"antenna": 1, "internal_load": 0, "heated_load": 0},
            loads={"internal_load": {"from": "thermistors",
                                     "label": "internal_load"},
                   "heated_load": {"t_load": {"value": 350.0, "unit": "K"}}},
            inference={"twin": {"replace": {"cal_loads": {
                "from": "thermistors", "label": "heated_load"}}},
                "parameters": {"g": {"init": 1.0, "into": "gain.gain"}},
                "observed": {"from": "simulation"},
                "noise": {"kind": "homoscedastic",
                          "sigma": {"value": 1.0, "unit": "K"}}})
        assert "A46" not in ids(document)

    def test_a_boolean_column_is_not_a_column(self):
        """The ``isinstance(column, bool)`` clause, defended rather than
        recorded.  ``isinstance(True, int)`` is True, so without it ``True``
        and ``1`` group together and leg 3 warns that two loads share
        "column True".

        ``sections/ingest.py`` refuses a boolean column outright -- *"a
        mapping of switch label -> integer column"* -- so the honest answer
        here is to say nothing and let that sentence be the one the user
        reads.

        Kills: dropping the clause -- measured, that mutant warns on this
        document and survives the suite."""
        document = ingested_document(
            columns={"antenna": 0, "internal_load": True, "heated_load": 1})
        assert "A46" not in ids(document)

    def test_a_load_sharing_with_the_antenna_alone_is_silent(self):
        """The other half of "only the labels a load reads": ``antenna`` is in
        the map and no load reads it, so a column it shares is not a
        collision."""
        document = ingested_document(
            columns={"antenna": 0, "internal_load": 0, "heated_load": 1},
            loads={"internal_load": {"from": "thermistors",
                                     "label": "internal_load"},
                   "heated_load": {"t_load": {"value": 350.0, "unit": "K"}}})
        assert "A46" not in ids(document)


class TestThePointingKeySetHasOneBinding:
    @pytest.mark.parametrize(("spec", "expected"), [
        (None, frozenset()),
        ({"mode": "none"}, frozenset()),
        ({"mode": "baked", "provenance": {"who": "me"}}, frozenset()),
        ({"mode": "drift", "materialise": ["pointing"]}, frozenset()),
        ({"mode": "drift", "materialise": ["selfrot_deg"]},
         frozenset({"selfrot_deg"})),
        ({"mode": "drift", "materialise": ["pointing", "selfrot_deg"],
          "lst": {"mode": "uniform_turn"}},
         frozenset({"selfrot_deg", "lst_deg"})),
        ({"mode": "tracked", "table": {}, "lst": {"mode": "from_site"}},
         frozenset({"lst_deg"})),
        ({"mode": "tracked", "table": {}, "lst": {"mode": "from_site"},
          "selfrot": {"zeros": ["n_time"]}},
         frozenset({"lst_deg", "selfrot_deg"})),
        ("drift", frozenset()),
        ({"mode": "spiral"}, frozenset()),
        ({"mode": "drift", "materialise": "selfrot_deg"}, frozenset()),
    ], ids=["absent", "none", "baked", "drift-pointing-only", "drift-selfrot",
            "drift-both", "tracked", "tracked-selfrot", "not-a-mapping",
            "unknown-mode", "materialise-not-a-list"])
    def test_the_four_branches_and_the_stand_down(self, spec, expected):
        """§0.3 E.9's pinned signature.  ``frozenset()`` for a spec it cannot
        parse -- an unknown mode and a malformed ``materialise:`` are
        ``compile_pointing``'s own refusals, said with the shape they got."""
        assert pointing_extra_keys(spec) == expected

    def test_compile_pointing_reads_this_function_rather_than_its_own_copy(
            self, monkeypatch):
        """Kills the R1 mutant that matters here: re-deriving the key set
        inline in the pre-flight pass and leaving ``compile_pointing``'s four
        branches where they were.

        Two validators for one property is the ``_number``-vs-``_whole``
        divergence §2.2 exists to stop, and nothing else in the suite can tell
        one binding from two: patched here, the builder's answer MUST move."""
        import jax.numpy as jnp

        from rheplicant.config.context import ResolutionContext
        from rheplicant.config.sections import pointing as pointing_module
        from rheplicant.config.sections.observation import SiteFacts

        spec = {"mode": "drift", "materialise": ["selfrot_deg"]}
        time_s = jnp.arange(0.0, 4.0, 1.0)
        context = ResolutionContext(time=time_s, dtype="float32")
        site = SiteFacts(lat_deg=None, lon_deg=None, alt_m=None)
        built = compile_pointing(spec, context, time_s=time_s,
                                 epoch_unix_s=None, site=site)
        assert set(built.extra) == {"selfrot_deg"}

        monkeypatch.setattr(pointing_module, "pointing_extra_keys",
                            lambda spec: frozenset())
        starved = compile_pointing(spec, context, time_s=time_s,
                                   epoch_unix_s=None, site=site)
        assert starved.extra == {}


class TestTheRegistry:
    @pytest.mark.parametrize(("check", "function"), [
        ("A10", _freq_unit), ("A45", _switch_key),
        ("A46", _thermistor_columns),
    ])
    def test_each_id_binds_to_its_own_function(self, check, function):
        """Identity on this task's own ids -- one of the four assertion forms
        a six-way parallel wave leaves legal (§0.3 F.2)."""
        assert CHECKS[check] is function

    def test_the_three_ids_run_between_fitting_and_model(self):
        """``CHECKS`` insertion order IS run order, and the foot-import block
        is alphabetical, so ``ingest`` sits between ``fitting`` and ``model``.

        Kills: a head import of ``preflight.model`` or ``preflight.observing``
        in this task's module, which would register THEIR ids first and
        reorder the pass in silence.  It also kills a DELETED foot import --
        measured, removing this task's line from ``preflight/__init__.py``
        takes both the shared ``TestTheFootImportCannotRot`` guard and this
        test red, which is the blind spot Task 6 reported and this module does
        not have.

        **This is a RECORDED EXEMPTION, not an oversight.**  Standing-brief R8
        and §0.3 F.2 ban "any registration-*index* assertion", and read
        literally this is one.  The orchestrator has recorded the exemption in
        §0.3 F.2: **a RELATIVE index comparison between ids in named modules
        is legal; an ABSOLUTE index is not.**  The reason it is safe in a
        six-way parallel wave is that ``A28``, ``A10`` and ``A2`` live in
        ``fitting``, ``ingest`` and ``model``, and the four modules wave 1 adds
        to the foot block -- ``depends``, ``instrument``, ``noise``,
        ``resources`` -- change none of those three relative positions.  A
        length, a set, or an absolute index would still be a merge landmine
        and none is written here."""
        order = list(CHECKS)
        assert order.index("A28") < order.index("A10") < order.index("A2")
        assert order.index("A10") < order.index("A45") < order.index("A46")


class TestTheOneBinding:
    @pytest.mark.parametrize("literal", [
        # A10 -- the HOISTED row.  Bound in `sections/ingest.py` and called
        # from the pass; a copy would make this two.
        "freq_unit is required and has no default",
        # A45 and A46 are INVENTIONS and owe no one-binding row, but each of
        # their sentences must still be written once.
        "is not a key this run writes into coords.extra",
        "has no entry in observation.from_file.thermistor_columns",
        "of observation.from_file.thermistor_columns, so their load",
    ], ids=["A10", "A45", "A46-leg2", "A46-leg3"])
    def test_every_literal_is_bound_once(self, literal):
        """§2.2's one-binding rule as a command rather than a review step."""
        assert_bound_once(literal)


class TestTheCost:
    def test_the_three_checks_cost_a_seventh_of_a_millisecond(self):
        """3A's 0.05 s budget for the whole pass is not automatically
        re-earned by a task that adds to it, and this is what that budget
        bought here: measured 0.039 ms for the three together (best of 100,
        five repeats within 0.0003 ms of each other), against 1.27 ms for the
        same three wrapped in ``_task3_over_layers``.

        **This number is warm and in-process, and the shipped COLD guard is a
        different measurement that must not be conflated with it.**  Cold, on
        40 runs and 20 variants, 22 fresh-process samples each: 22.3 ms
        unlayered against 37.8 ms layered, both under §5's 50 ms -- so the
        layering is affordable in isolation, and the wave's arithmetic rather
        than a breach is why it is not used here.  That cold guard is
        load-flaky: it reads 52.2 ms at HEAD under ``-n 16`` with no layering
        at all.

        The bound is 0.15 ms, a 3.8x margin, deliberately not the 3008x Task
        1a shipped: VERIFIED to go red under a 10x slowdown of these three
        checks.  ``pytest-timeout`` is not installed in this worktree, so
        ``--timeout=`` is a usage error (exit 4) and is not used anywhere.

        The subject is this task's three checks and not ``preflight`` itself,
        because five sibling tasks land into the same registry in this wave: a
        bound on the whole pass would be a bound on their work too."""
        from tests.config.inflight_helpers import best_ms

        document = ingested_document()
        document["variants"] = {
            f"v{index}": {"observation": {"from_file": {"freq_unit": "Hz"}}}
            for index in range(8)}
        document["model"]["noise_wave"] = noise_wave()
        mine = (_freq_unit, _switch_key, _thermistor_columns)

        def run_them():
            return [list(check(document)) for check in mine]

        assert best_ms(run_them, repeats=100) < 0.15
