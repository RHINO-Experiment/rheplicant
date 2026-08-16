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

**All three checks now walk the variant LAYERS** (Plan 3C Task 0), through
``preflight/document.py::_task3_over_layers``.  ``document.py::_assemble``
applies the selected variant BEFORE it runs the pass, so the route a user
actually takes was already guarded; what the walk adds is the reporting of
faults in variants **nobody selected**.  The ``…unselected_variant…`` tests
below are the new false-negative closures, one per check plus A10's second
route; the ``test_the_selected_variant_is_still_read`` tests are the
surviving halves of the three that used to pin the false negative -- each
still keeps its ``apply_variant(...)`` call, which is the anti-vacuity read
proving the SELECTED route is still guarded and unprefixed.  Re-measured cold
on the shipped guard's own document at ``0030724``, min of five fresh
processes: **13.51 ms unlayered against 16.91 ms with these three walked**,
both well under the 50 ms bound -- ``preflight/document.py::_task3_layers``
memoises the layer walk per pass now, so a check added to it costs no
additional ``apply_variant`` calls.  Walking the other **18** non-layering
pre-flight functions the same way costs ~31 ms on the same guard, which is
why this task's scope is these three and no others (recorded in
``ingest.py``'s own module docstring, not reopened here).

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
    findings,
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

    def test_an_unselected_variant_that_omits_freq_unit_is_reported(self):
        """S3 INVERTED (§0.3 D-23 / Plan 3C Task 0): the fault a user wrote
        into a variant they did not select is now named before the run,
        prefixed with which layer said it.

        Kills: the check reading ``document[...]`` directly -- a walk wired
        to ``_task3_layers`` without ``_task3_over_layers`` would still find
        the fault, but report it with the BASE's own ``where`` and no
        ``variants.recorded: `` prefix, because it merges the variant's own
        section straight into the top level instead of walking a separate
        layer.  Both the ``where`` and the message prefix are asserted so
        that mutant is caught."""
        document = preflight_document(
            observation={**BASE_OBSERVATION},
            variants={"recorded": {"observation": {"from_file": {
                "format": "rhino_hdf5", "path": "obs.hd5f"}}}})
        finding = only(document, "A10")
        assert finding.where == "variants.recorded.observation.from_file"
        assert finding.message == f"variants.recorded: {A10_MESSAGE}"

    def test_the_selected_variant_is_still_read(self):
        """S3's surviving half: ``document.py::_assemble`` applies the
        variant and THEN runs the pass, so the variant a user actually
        SELECTS stays guarded on its own, unprefixed ``where`` -- this is
        the anti-vacuity read for the test above, on the same document."""
        document = preflight_document(
            observation={**BASE_OBSERVATION},
            variants={"recorded": {"observation": {"from_file": {
                "format": "rhino_hdf5", "path": "obs.hd5f"}}}})
        finding = only(apply_variant(document, "recorded"), "A10")
        assert finding.where == "observation.from_file"
        assert finding.message == A10_MESSAGE

    def test_an_unselected_variants_file_value_node_is_reported(self):
        """A10's SECOND route (§0.3 D-23), unselected.  ``_a10_sites`` walks
        two separate paths -- ``observation.from_file`` and any other
        ``{file:}`` value node -- and this is the second.

        Kills: wiring the walk to only the first route.  The test above
        alone passes against an implementation that only widens
        ``observation.from_file``'s own read to every layer and never
        touches ``_a10_file_nodes``'s recursion."""
        document = preflight_document(
            variants={"recorded": {"resources": {"arrays": {"rec": {
                "file": {"path": "obs.hd5f", "format": "rhino_hdf5"}}}}}})
        finding = only(document, "A10")
        assert finding.where == "variants.recorded.resources.arrays.rec"
        assert finding.message == f"variants.recorded: {A10_MESSAGE}"

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

    def test_an_unselected_variant_that_renames_the_key_is_reported(self):
        """A45's equivalent of A10's unselected-variant test (§0.3 D-23 /
        Plan 3C Task 0).

        Kills: wiring A10 and A46 to the walk and leaving A45 unwalked --
        the routes are three separate ``@register`` slots and a task that
        wires two of them passes every test in the other two classes."""
        document = switched_document()
        document["variants"] = {
            **document.get("variants", {}),
            "other_switch": {"model": {"noise_wave": {
                "switch_key": "my_switch"}}}}
        finding = only(document, "A45")
        assert finding.where == \
            "variants.other_switch.model.noise_wave.switch_key"
        assert finding.message == (
            "variants.other_switch: model.noise_wave.switch_key: "
            "'my_switch' is not a key this run writes into coords.extra, so "
            "the operator has no switch index to read -- with more than one "
            "source the twin refuses the moment it is evaluated, and with "
            "one it silently takes the first. This run writes "
            "['receiver_input']; observation.extra and "
            "observation.pointing's materialise:/lst: are where another one "
            "would come from (check A45)."
        )

    def test_the_selected_variant_is_still_read(self):
        """S3's surviving half: the pass runs on the variant-applied mapping
        for the route a user actually takes, unprefixed -- the
        anti-vacuity read for the test above."""
        document = switched_document()
        document["variants"] = {
            **document.get("variants", {}),
            "other_switch": {"model": {"noise_wave": {
                "switch_key": "my_switch"}}}}
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

    def test_an_unselected_variant_that_drops_a_column_is_reported(self):
        """A46's equivalent of A10's unselected-variant test (§0.3 D-23 /
        Plan 3C Task 0).

        Kills: wiring A10 and A45 to the walk and leaving A46 unwalked."""
        document = ingested_document()
        document["variants"] = {"partial": {"observation": {"from_file": {
            "thermistor_columns": {"~heated_load": None}}}}}
        finding = only(document, "A46")
        assert finding.where == "variants.partial.model.cal_loads.heated_load"
        assert finding.message == (
            "variants.partial: model.cal_loads.heated_load: label: "
            "'heated_load' has no entry in "
            "observation.from_file.thermistor_columns, so this load's "
            "t_load is asked for a column the recording was never read "
            "with, and the refusal comes from inside the reader naming no "
            "key of this document. Declare thermistor_columns with a "
            "column for 'heated_load' AND for every other switch label the "
            "recording visits, the antenna included -- the reader refuses a "
            "partial map (check A46)."
        )

    def test_the_selected_variant_is_still_read(self):
        """S3's surviving half: the pass runs on the variant-applied mapping
        for the route a user actually takes, unprefixed -- the
        anti-vacuity read for the test above."""
        document = ingested_document()
        document["variants"] = {"partial": {"observation": {"from_file": {
            "thermistor_columns": {"~heated_load": None}}}}}
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


class TestTheLayerWalkItself:
    """§0.3 D-23 / Plan 3C Task 0: the walk shared by all three checks, and
    the properties that are ``_task3_over_layers``'s and not any one check's
    own -- de-duplication, an un-spellable variant name, and a merged layer
    that ``_structural`` never validated.
    """

    def test_a_base_fault_is_reported_once_and_unprefixed(self):
        """``_task3_over_layers`` de-duplicates on the WHOLE ``Finding``: a
        base fault that survives unchanged into every declared variant's
        layer (because none of them touch the section it lives in) is said
        ONCE, at its own unprefixed ``where`` -- not once per layer.

        Kills: dropping the base-set de-duplication, under which one typo
        in the base document reads as several sentences, all but one of them
        blaming a variant that did not introduce it.  Two variants exercise
        this: the base fixture's own ``unity_gain`` (``preflight_document``'s
        default) and an ``unrelated`` one added here, neither of which
        touches ``observation.from_file``."""
        document = preflight_document(
            observation={**BASE_OBSERVATION, "from_file": {
                "format": "rhino_hdf5", "path": "obs.hd5f"}},
            variants={"unrelated": {"model": {"gain": {"gain": {
                "value": 2.0, "unit": "dimensionless"}}}}})
        assert len(document["variants"]) >= 2, (
            "this test's point is a base fault surviving into MULTIPLE "
            "layers unchanged; one variant would not distinguish "
            "de-duplication from a walk that only ever looked at the base")
        finding = only(document, "A10")
        assert finding.where == "observation.from_file"
        assert finding.message == A10_MESSAGE

    def test_a_variant_that_breaks_the_rule_DIFFERENTLY_is_still_reported(
            self):
        """The de-duplication key is the WHOLE ``Finding``, not its
        ``where``.  A variant that rebinds the SAME document path to a
        DIFFERENT bad value is a second violation, not a repeat of the
        base's.

        Kills: keying de-duplication on ``finding.where`` instead -- that
        mutant drops this variant's own finding because it shares a
        ``where`` with the base's, which is a lost check rather than a
        duplicated sentence."""
        document = switched_document(noise_wave(switch_key="bad_key_1"))
        document["variants"] = {
            **document.get("variants", {}),
            "other_key": {"model": {"noise_wave": {
                "switch_key": "bad_key_2"}}}}
        found = [one for one in findings(document) if one.check == "A45"]
        assert {one.where for one in found} == {
            "model.noise_wave.switch_key",
            "variants.other_key.model.noise_wave.switch_key"}
        messages = {one.message for one in found}
        assert any("'bad_key_1'" in message for message in messages)
        assert any("'bad_key_2'" in message for message in messages)

    def test_two_variants_that_break_the_same_rule_are_BOTH_reported(self):
        """MAJOR-2 (fix round).  The de-duplication key is the WHOLE
        UNPREFIXED ``Finding`` ``per_layer`` hands back -- taken BEFORE the
        wrapper applies the variant's own prefix -- so two variants that
        break the identical rule the identical way produce two
        structurally IDENTICAL raw findings, and only the BASE layer's own
        set may ever decide whether either is swallowed.  This is the
        sibling the test above cannot stand in for: that one breaks the
        rule DIFFERENTLY per variant (a different ``switch_key`` value),
        so its two raw findings were never equal to begin with.

        Kills: adding the prefixed branch's own finding to ``base`` inside
        ``_task3_over_layers`` (``elif finding not in base: base.add(
        finding)`` ahead of the ``yield``).  Measured, that mutant makes
        the SECOND variant to break the same rule the same way vanish
        silently -- ``v1``'s raw, unprefixed finding joins ``base`` on its
        own turn, and ``v2``'s, equal to it because both write the exact
        same ``observation.from_file`` fault, then reads as already seen.
        Neither variant touches the base's own ``observation``, so a
        correct pass reports two findings, one per variant, and the
        mutant reports one."""
        document = preflight_document(
            observation={**BASE_OBSERVATION},
            variants={
                "v1": {"observation": {"from_file": {
                    "format": "rhino_hdf5", "path": "obs.hd5f"}}},
                "v2": {"observation": {"from_file": {
                    "format": "rhino_hdf5", "path": "obs.hd5f"}}},
            })
        found = [one for one in findings(document) if one.check == "A10"]
        assert {one.where for one in found} == {
            "variants.v1.observation.from_file",
            "variants.v2.observation.from_file"}

    def test_a_non_identifier_variant_name_does_not_kill_the_pass(self):
        """A variant name need not be an identifier -- neither
        ``apply_variant`` nor ``parse_latents`` validates one, and
        ``variants: {bad-name: ...}`` loads today.  ``_task3_where`` cuts
        the un-spellable segment back to the longest prefix the path
        grammar CAN parse -- here, just ``variants`` -- rather than handing
        ``_check_where`` a path it must raise on.

        Kills: passing the raw, un-cut ``f'{prefix}.{finding.where}'``
        straight through as ``where``.  Measured, ``_check_where`` raises
        OUTSIDE the per-check ``try`` in ``passes.sweep``, so an un-cut
        segment here would not just fail this check -- it would abort the
        whole pass and hide every other finding."""
        document = preflight_document(
            observation={**BASE_OBSERVATION},
            variants={"bad-name": {"observation": {"from_file": {
                "format": "rhino_hdf5", "path": "obs.hd5f"}}}})
        finding = only(document, "A10")
        assert finding.where == "variants"
        assert finding.message == f"variants.bad-name: {A10_MESSAGE}"

    def test_a_non_identifier_document_path_does_not_kill_the_pass(self):
        """MAJOR-1 (fix round).  A DOCUMENT path can be non-identifier too,
        not just a variant name -- ``resources.arrays: {"my-array": ...}``
        loads today.  This is ``_a10_in``'s OWN call to ``_task3_where``
        (on the layer's own, unprefixed path) that is on trial here, not
        the wrapper's: the BASE layer's finding is yielded straight
        through ``_task3_over_layers`` -- ``if not prefix: base.add(
        finding); yield finding`` -- with no ``_task3_where`` of the
        wrapper's own to fall back on, so on the base layer the whole
        pass's survival rests on ``_a10_in`` doing this cut itself.

        Kills: dropping ``_task3_where(where)`` inside ``_a10_in`` and
        yielding the raw ``where``.  Measured, that mutant raises
        ``ConfigError`` from ``preflight/passes.py``'s ``_check_where`` --
        *"pre-flight check 'A10' emitted where='resources.arrays.my-array',
        which is not a document path"* -- OUTSIDE the per-check ``try``,
        aborting the whole pass rather than failing only this one."""
        document = preflight_document(
            resources={"arrays": {"my-array": {"file": {
                "format": "rhino_hdf5", "path": "obs.hd5f"}}}})
        finding = only(document, "A10")
        assert finding.where == "resources.arrays"

    def test_a_non_identifier_cal_load_key_does_not_kill_the_pass(self):
        """MAJOR-1 (fix round), A46's twin of the test above.  A
        ``model.cal_loads`` KEY need not be an identifier either --
        ``_t4_entries`` reads it off the mapping same as any other -- and
        ``_a46_in``'s own ``_task3_where`` call is what keeps leg 2's
        ``where`` spellable on the BASE layer, where the wrapper's own
        prefix-time call never runs.

        Kills: dropping ``_task3_where(where)`` inside ``_a46_in``'s leg 2
        and yielding the raw ``where``.  Measured, that mutant raises
        ``ConfigError`` from ``_check_where`` -- *"pre-flight check 'A46'
        emitted where='model.cal_loads.a-load', which is not a document
        path"*.  ``internal_load`` and ``heated_load`` keep their shipped,
        column-matched labels, so the only A46 finding on this document is
        the one ``a-load``/``'ghost'`` earns."""
        document = ingested_document()
        document["model"]["cal_loads"]["a-load"] = {
            "from": "thermistors", "label": "ghost"}
        finding = only(document, "A46")
        assert finding.where == "model.cal_loads"

    def test_a10s_path_free_message_lets_a_variants_own_fault_hide(self):
        """MAJOR-3 (fix round), PINNED rather than fixed.  ``A10_MESSAGE``
        is a CONSTANT -- it names no document key, unlike A45's and A46's
        sentences, which both interpolate the path they are about -- so
        once ``_task3_where`` cuts two DIFFERENT non-identifier array
        names back to the same ``where`` ("resources.arrays"), the two
        sites become the SAME unprefixed ``Finding``.
        ``_task3_over_layers``'s own docstring justifies its whole-
        ``Finding`` de-dup key on "every message this module emits opens
        with the label its ``where`` is derived from" -- true of A45 and
        A46, and NOT true of A10, which is the first path-free message on
        this walk (see ``ingest.py``'s module docstring).

        A variant that adds a SECOND broken array, while the base's own
        array is ALSO broken, has its own fault swallowed by the base's --
        not because it duplicates the base's finding in any interesting
        sense, but because both happen to collapse to a ``where`` and
        ``message`` neither can tell apart.

        RECORDED, not fixed here: a fix would change
        ``_task3_over_layers`` (outside this task's Files list -- MAJOR-2's
        own fix lives there instead) or ``A10_MESSAGE`` (a separate task,
        and one that would break ``assert_bound_once``).  This test exists
        so a future change to either is visible against a pinned baseline
        rather than silent."""
        broken_base = preflight_document(
            resources={"arrays": {"a-one": {"file": {
                "format": "rhino_hdf5", "path": "obs.hd5f"}}}},
            variants={"v": {"resources": {"arrays": {"a-two": {"file": {
                "format": "rhino_hdf5", "path": "obs.hd5f"}}}}}})
        found = [one for one in findings(broken_base) if one.check == "A10"]
        assert {one.where for one in found} == {"resources.arrays"}, (
            "the variant's own a-two fault is swallowed by the base's "
            "a-one finding -- both collapse to the identical unprefixed "
            "Finding because A10_MESSAGE carries no path")

        fixed_base = preflight_document(
            resources={"arrays": {"a-one": {"file": {
                "format": "rhino_hdf5", "path": "obs.hd5f",
                "freq_unit": "MHz"}}}},
            variants={"v": {"resources": {"arrays": {"a-two": {"file": {
                "format": "rhino_hdf5", "path": "obs.hd5f"}}}}}})
        finding = only(fixed_base, "A10")
        assert finding.where == "variants.v.resources.arrays"

    def test_two_broken_arrays_in_one_variant_are_reported_twice(self):
        """MINOR-4 (fix round), PINNED rather than fixed.  Neither branch
        of ``_task3_over_layers`` de-duplicates WITHIN one layer's own
        findings: the base branch (``if not prefix: base.add(finding);
        yield finding``) yields unconditionally, and the variant branch
        (``elif finding not in base``) only ever checks against the BASE
        layer's set, never against findings this same variant already
        produced this same walk.  A layer that carries two sites broken
        the identical way earns the identical ``Finding`` twice.

        Both directions are UNPINNED before this test: the current
        behaviour (two identical findings, asserted below) and the
        plausible "correct" fix (a per-layer ``seen`` set inside
        ``_task3_over_layers``) both leave the rest of ``tests/config``
        green.  RECORDED, not fixed here -- the fix lives in
        ``_task3_over_layers``, outside this task's Files list.

        Two hyphenated array names collapsing to the same
        ``_task3_where`` cut is the same mechanism
        :meth:`test_a10s_path_free_message_lets_a_variants_own_fault_hide`
        uses, but both broken arrays live in the ONE variant here rather
        than split across base and variant."""
        document = preflight_document(
            variants={"v": {"resources": {"arrays": {
                "a-one": {"file": {"format": "rhino_hdf5",
                                   "path": "obs.hd5f"}},
                "a-two": {"file": {"format": "rhino_hdf5",
                                   "path": "obs.hd5f"}},
            }}}})
        found = [one for one in findings(document) if one.check == "A10"]
        assert len(found) == 2
        assert {one.where for one in found} == {
            "variants.v.resources.arrays"}
        assert {one.message for one in found} == {f"variants.v: {A10_MESSAGE}"}

    #: One base document rich enough for all three checks to have something
    #: to walk: a valid ``rhino_hdf5`` ingestion (A10, A46 leg 2), a
    #: ``switch_key`` on both the ``model:`` and ``twin.replace:`` routes
    #: (A45), and a ``cal_loads`` entry on both routes too (A46).  Built once
    #: per call rather than shared, because every hostile patch below is a
    #: VARIANT layered over it and the base itself must stay untouched.
    @staticmethod
    def _hostile_document():
        return ingested_document(
            loads={"internal_load": {"from": "thermistors",
                                     "label": "internal_load"},
                   "heated_load": {"from": "thermistors",
                                   "label": "heated_load"}},
            inference={
                "twin": {"replace": {"noise_wave": {
                    "type": "NoiseWaveOperator",
                    **noise_wave(switch_key="receiver_input")}}},
                "parameters": {"g": {"init": 1.0, "into": "gain.gain"}},
                "observed": {"from": "simulation"},
                "noise": {"kind": "homoscedastic",
                          "sigma": {"value": 1.0, "unit": "K"}}})

    @pytest.mark.parametrize(("patch",), [
        ({"~observation": None},),
        ({"~model": None},),
        ({"~inference": None},),
        ({"model": ["oops"]},),
        ({"inference": "oops"},),
        ({"observation": 3},),
        ({"observation": {"extra": ["a"]}},),
        ({"model": {"kind": "pipeline"}},),
        ({"inference": {"twin": {"replace": "oops"}}},),
        ({"observation": {"from_file": "oops"}},),
        ({"observation": {"from_file": {
            "thermistor_columns": [0, 1]}}},),
        ({"model": {"cal_loads": {"internal_load": {"label": 3}}}},),
        ({"inference": {"twin": {"replace": {
            "cal_loads": {"from": "thermistors", "label": 3}}}}},),
        ({"resources": ["oops"]},),
    ], ids=[
        "delete-observation", "delete-model", "delete-inference",
        "model-not-a-mapping", "inference-not-a-mapping",
        "observation-not-a-mapping", "observation-extra-a-list",
        "model-kind-pipeline", "scalar-twin-replace", "scalar-from-file",
        "thermistor-columns-a-list", "non-string-label-model-route",
        "non-string-label-replace-route", "resources-a-list",
    ])
    def test_a_hostile_variant_layer_earns_no_raise(self, patch):
        """A merged variant layer is not ``_structural``-validated -- that
        guard runs once, on the top-level document, before layering -- so
        the layer these three checks walk is HOSTILE input in a way the
        un-layered document never was.  Fourteen shapes, one test each
        (Plan 3C Task 0's own measured battery, re-run here): none may
        raise, whatever it finds or does not find.

        Kills: any ``isinstance`` guard inside ``_a10_in``, ``_a45_in`` or
        ``_a46_in`` that assumed a section present in the base document is
        present -- or still a mapping -- in every layer."""
        document = self._hostile_document()
        document["variants"] = {"hostile": patch}
        # `findings()` raises `ConfigError` if any check RAISES (`passes.sweep`
        # turns that into "check 'X' RAISED ...", which is exactly the
        # failure this test exists to catch) or if `_check_where` refuses an
        # illegal `where` -- both would surface here rather than as a silent
        # False.
        findings(document)

    def test_the_variants_own_advice_applied_clears_a10_and_the_variant_loads(
            self):
        """R4 for A10: apply the refusal's own advice -- declare
        ``freq_unit:`` -- INSIDE the variant patch that broke it.  The id
        disappears (both unselected, which is Task 0's own new read, and
        selected), and ``load_document`` on that variant then reaches a
        later phase."""
        from rheplicant.config import load_document

        broken = preflight_document(
            observation={**BASE_OBSERVATION},
            variants={"recorded": {"observation": {"from_file": {
                "format": "rhino_hdf5", "path": "obs.hd5f"}}}})
        assert "A10" in ids(broken)  # Task 0: reported unselected, prefixed

        fixed = preflight_document(
            observation={**BASE_OBSERVATION},
            variants={"recorded": {"observation": {"from_file": {
                "format": "rhino_hdf5", "path": "obs.hd5f",
                "freq_unit": "MHz"}}}})
        assert "A10" not in ids(fixed)
        # A LATER reason -- no file at 'obs.hd5f' -- not A10's.
        with pytest.raises(ConfigError) as caught:
            load_document(fixed, variant="recorded")
        assert "freq_unit" not in str(caught.value)

    def test_the_variants_own_advice_applied_clears_a45_and_the_variant_loads(
            self):
        """R4 for A45: the message says ``receiver_input`` is written;
        writing it into the variant's own patch that broke it clears the id
        both unselected and selected, and the selected variant fully
        builds -- there is nothing else wrong with this document."""
        broken = switched_document()
        broken["variants"] = {
            **broken.get("variants", {}),
            "renamed": {"model": {"noise_wave": {
                "switch_key": "my_switch"}}}}
        assert "A45" in ids(broken)  # Task 0: reported unselected, prefixed

        fixed = switched_document()
        fixed["variants"] = {
            **fixed.get("variants", {}),
            "renamed": {"model": {"noise_wave": {
                "switch_key": "receiver_input"}}}}
        assert "A45" not in ids(fixed)
        from rheplicant.config import load_document

        assert load_document(fixed, variant="renamed") is not None

    def test_the_variants_own_advice_applied_clears_a46_and_the_variant_loads(
            self, tmp_path):
        """R4 for A46: the message names EVERY switch label the recording
        visits, the antenna included -- not ``switching.order[1:]``.
        Applied in full inside the variant patch that broke it, the id
        disappears both unselected and selected, and the selected variant
        loads."""
        broken = ingested_document()
        broken["variants"] = {"trimmed": {"observation": {"from_file": {
            "thermistor_columns": {"~heated_load": None}}}}}
        assert "A46" in ids(broken)  # Task 0: reported unselected, prefixed

        fixed = ingested_document()
        fixed["variants"] = {"trimmed": {"observation": {"from_file": {
            "thermistor_columns": dict(WORKING_COLUMNS)}}}}
        assert "A46" not in ids(fixed)
        assert load(apply_variant(fixed, "trimmed"), recording(tmp_path)) \
            is not None


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

    def test_this_modules_own_three_ids_run_in_the_order_it_writes_them(self):
        """The three ids this module registers keep their own relative order.

        This is the surviving half of a test that also asserted
        ``A28 < A10 < A2`` -- ``fitting`` before ``ingest`` before ``model``.
        **That half was refuted at the wave-1 merge and is deleted**, not
        repaired: measured on the merged branch, ``A2`` sits at index 17 and
        ``A10`` at 31, because ``preflight/beam_spill.py`` sorts FIRST in the
        alphabetical foot block and *head*-imports ``preflight.document`` and
        ``preflight.model``, so ``model``'s ids register before ``ingest`` is
        foot-imported at all.

        The rule that replaces the one this test was written against, and
        which §0.3 C.5's "alphabetical foot order gives run order" got wrong:
        **a foot-imported module's checks register after everything its own
        head imports transitively register.**  Position in the foot block
        decides nothing on its own.

        What remains here is intra-module and therefore immune to every
        sibling: A10, A45 and A46 are all registered by *this* file, in this
        order, so no other task's imports can reorder them.  The two
        properties the deleted half was buying are asserted directly instead
        -- the wiring by :meth:`test_the_foot_import_is_what_registers_them`,
        and the absence of a sibling head import by
        :meth:`test_this_module_head_imports_no_preflight_sibling`."""
        order = list(CHECKS)
        assert order.index("A10") < order.index("A45") < order.index("A46")

    def test_model_and_observing_are_imported_inside_the_functions(self):
        """The property the deleted ordering assertion was really buying.

        A **head** import of a ``preflight/`` sibling makes that sibling's
        ``@register`` decorators run before this module's own, so its ids land
        earlier in ``CHECKS`` -- which is run order.  This module's own source
        says it imports ``preflight.model`` and ``preflight.observing``
        *inside* its functions for exactly that reason; this asserts it rather
        than trusting the comment.

        ``preflight.document`` IS head-imported, for ``_task3_where``, and
        that is deliberate and allowed -- it is named here so the day someone
        adds a second head import the failure says which one and why.

        Only module-level statements count: ``ast.walk`` would descend into
        the function bodies and report the very imports whose placement is the
        point.  Measured at the wave-1 merge -- a walk over the whole tree
        reports ``document``, ``model`` and ``observing`` and the test is then
        asserting the opposite of what it means.
        """
        import ast
        import importlib

        module = importlib.import_module("rheplicant.config.preflight.ingest")
        source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
        package = "rheplicant.config.preflight"
        siblings = set()
        for node in ast.parse(source).body:          # module level ONLY
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(f"{package}."):
                    siblings.add(node.module)        # from ...preflight.X import y
                elif node.module == package:
                    # `from ...preflight import X` -- the sibling is the
                    # ALIAS, not the module path.  This is the spelling the
                    # foot-import block itself uses, and a walk that only
                    # inspects `node.module` misses it entirely: measured, a
                    # `from rheplicant.config.preflight import model` mutant
                    # SURVIVED the first version of this test.
                    siblings.update(f"{package}.{a.name}" for a in node.names
                                    if a.name != "register")
            elif isinstance(node, ast.Import):
                siblings.update(a.name for a in node.names
                                if a.name.startswith(f"{package}."))
        assert siblings == {"rheplicant.config.preflight.document"}, (
            f"{module.__name__}'s module-level preflight imports are "
            f"{sorted(siblings)}; only `document` (for `_task3_where`) is "
            "allowed, because a head import registers that sibling's ids "
            "before this module's own")

    def test_the_foot_import_is_what_registers_them(self):
        """R1: deleting the foot import leaves this file's own tests green.

        The file's ``from rheplicant.config.preflight.ingest import ...`` runs
        the ``@register`` decorators itself, so nothing in-process can tell
        "registered for a user" from "registered by this test module".  A
        subprocess that imports only the package is the honest form.
        """
        import subprocess
        import sys

        done = subprocess.run(
            [sys.executable, "-c",
             "from rheplicant.config.preflight import CHECKS\n"
             "print(sorted(k for k in CHECKS if k in ('A10', 'A45', 'A46')))"],
            capture_output=True, text=True)
        assert done.returncode == 0, done.stdout + done.stderr
        assert done.stdout.strip() == "['A10', 'A45', 'A46']"


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
    def test_the_three_checks_layered_cost_just_over_a_millisecond(self):
        """3A's 0.05 s budget for the whole pass is not automatically
        re-earned by a task that adds to it -- but this docstring's PRIOR
        number was itself stale (§0.3 D-23 / Plan 3C Task 0): it read 0.039 ms
        unlayered against 1.27 ms layered, and the 1.27 ms was measured
        BEFORE ``preflight/document.py::_task3_layers`` was memoised.  These
        three checks now walk every layer (Task 0's own change), so
        "unlayered" is no longer a state this test can measure at all --
        there is only the layered number, and it must be re-derived rather
        than carried over.

        **RE-DERIVED, not assumed**, by
        ``PYTHONPATH=. .venv/bin/python`` on this test's own document
        (``ingested_document()`` plus 8 declared variants and a lit
        ``noise_wave``): best of 100, five repeats agreeing to within 1 %:
        **0.290-0.301 ms**.  That is the number
        :data:`inflight_helpers.best_ms` returns for ``run_them`` below, run
        five times in five fresh interpreters.

        **This number is warm, memo-warm and in-process**, and the shipped
        COLD guard is a different measurement that must not be conflated with
        it: re-measured on the shipped guard's own document (40 runs, 21
        declared variants) at ``0030724``, min of five fresh processes,
        **13.51 ms unlayered against 16.91 ms with these three walked**, both
        far under §5's 50 ms.  ``preflight/document.py``'s own docstring and
        ``ingest.py``'s carry that measurement; this one is the WARM,
        three-checks-alone number and is not it.

        The bound is **1.2 ms**, a 4x margin over the measured 0.30 ms --
        the same shape of margin the retired 0.15 ms bound held over its own
        0.039 ms (3.8x) -- and RE-VERIFIED to go red under a 10x slowdown of
        these three checks: calling each check ten times per ``run_them()``
        call measures **2.99 ms** best-of-20, comfortably over 1.2 ms, while
        the un-slowed number stays comfortably under it.  Not the 3008x
        margin Task 1a shipped, and not the 1.27 ms stale number either --
        this is a re-derivation against a measured 4.2x improvement
        (1.27 ms / 0.30 ms) from the memo, not a threshold tuned to make a
        test pass.  ``pytest-timeout`` is not installed in this worktree, so
        ``--timeout=`` is a usage error (exit 4) and is not used anywhere.

        The subject is this task's three checks and not ``preflight`` itself,
        because five sibling tasks land into the same registry in this wave: a
        bound on the whole pass would be a bound on their work too.

        **What this bound does and does not catch (fix-round MINOR-2),
        re-derived rather than assumed.**  A 4x slow-down of these three
        checks (each called four times per ``run_them()`` instead of once)
        is NOT a regression this bound's 4x margin can promise to catch:
        min of 100, across three fresh interpreters, **1.188-1.203 ms** --
        straddling the 1.2 ms line rather than sitting comfortably on
        either side of it.  A full loss of the layer memo --
        ``_task3_build_layers`` (``document.py``) never storing into
        ``_TASK3_LAYER_MEMO``, so each of the three checks rebuilds this
        document's nine layers for itself instead of sharing the one build
        -- IS caught by this bound today, with room to spare: the same
        mutation, min of 100 across three fresh interpreters, measures
        **1.22-1.24 ms**.  That margin is not a promise, though, and must
        not be read as one: this test asserts wall-clock time on ONE
        document of one fixed shape, not the memo's own invariant, and a
        quieter machine or a smaller declared-variant count could let a
        real memo regression read under 1.2 ms undetected.  The actual
        guard for "built once per declared variant" is
        ``test_the_layers_are_built_once_per_declared_variant`` in
        ``test_preflight_document.py``, which counts ``apply_variant``
        calls directly rather than inferring them from a clock that moves
        with machine load -- this cost test is not a substitute for it and
        was never meant to be.  Last: this test's own document earns ZERO
        findings from all three checks (confirmed: ``sum(len(one) for one
        in run_them()) == 0``), so what it measures is the WALK and each
        check's own per-layer READ alone -- the ``dataclasses.replace`` /
        prefix / set-membership path inside ``_task3_over_layers`` that a
        document WITH findings exercises is never timed here at all."""
        from tests.config.inflight_helpers import best_ms

        document = ingested_document()
        document["variants"] = {
            f"v{index}": {"observation": {"from_file": {"freq_unit": "Hz"}}}
            for index in range(8)}
        document["model"]["noise_wave"] = noise_wave()
        mine = (_freq_unit, _switch_key, _thermistor_columns)

        def run_them():
            return [list(check(document)) for check in mine]

        assert best_ms(run_them, repeats=100) < 1.2
