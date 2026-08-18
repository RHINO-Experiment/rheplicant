"""A1's three measured holes, A38's presence half, and A39 by capability name.

Every test below names the wrong implementation it kills, because a test that
cannot say so is decoration -- 2C shipped twenty-seven surviving mutants and
every one of them was in a test.
"""

import ast
import inspect
import sys
import textwrap

import jax.numpy as jnp
import pytest

from _rheplicant_bootstrap.layering import initial_merge
from _rheplicant_bootstrap.types import Origin
from _rheplicant_bootstrap.variants import enumerate_layers_once
from rheplicant.config.context import ResolutionContext
from rheplicant.config.errors import ConfigError
from rheplicant.config.preflight import CHECKS, preflight
from rheplicant.config.preflight import _structural as _document_sweep
from rheplicant.config.preflight.document import (
    _CAPABILITY_KEYS,
    _TASK3_SPOKEN_FOR,
    _task3_allowed_run_options,
)
from rheplicant.config.sections import exits  # noqa: F401  -- fills PARSERS
from rheplicant.config.sections.exit_support import PARSERS
from rheplicant.config.sections.model import _pick_class
from rheplicant.config.sections.parameters import parse_latents
from tests.config.preflight_helpers import UNREADABLE_BEAM, preflight_document

_CTX = ResolutionContext(freq=jnp.linspace(60e6, 85e6, 8),
                         time=jnp.arange(16.0), dtype="float32")


#: The five ids this module's checks are registered under.  ``A1`` is three
#: functions and ``register`` binds ONE function per id, so the three carry a
#: suffix while every Finding they emit carries the schema's own ``"A1"``.
_IDS = ("A1.runs", "A1.variants", "A1.horizon", "A38", "A39")


def _findings(document, check):
    """Every finding ``preflight`` produces for one schema check id."""
    return [f for f in preflight(document).findings if f.check == check]


def _layer_prefixes(document) -> list[str]:
    merged = initial_merge(document, origin=Origin("user"))
    enumeration = enumerate_layers_once(
        merged.document, merged.origins, merged.deletions
    )
    return [layer.prefix for layer in enumeration.layers]


def _beam_entry(name, **horizon):
    """One named ``resources.beams`` entry.  Named, so a test can declare two."""
    return {name: {"format": "npy", "path": "beam.npy", "nside": 4,
                   "normalize": "pixel_sum", "frame": "beam_local",
                   "horizon": {"mode": "truncate_map", **horizon}}}


def _a_beam(**horizon):
    """A ``resources.beams`` entry.  No file is ever opened: P-1 reads text."""
    return {"beams": _beam_entry("horn", **horizon)}


def _latent(**extra):
    return {"twin": {"without": ["noise"]},
            "parameters": {"g": {"init": 1.0, "into": "gain.gain", **extra}}}


def _allowed_set(node, scope):
    """The frozenset a ``_sweep(run, X)`` argument denotes, from ``X``'s AST.

    Three shapes and no fourth, which is what the sweep census measured: a
    module-level name, a ternary over a ``run.kind == "<literal>"`` flag
    already resolved into ``scope``, and a ``frozenset(...)`` literal written
    at the call site.  Resolved rather than ``eval``-ed so that a fourth shape
    is a loud ``KeyError``/``AttributeError`` here instead of a silently
    different set.
    """
    if isinstance(node, ast.Name):
        return frozenset(scope[node.id])
    if isinstance(node, ast.IfExp):
        return _allowed_set(
            node.body if scope[node.test.id] else node.orelse, scope)
    assert isinstance(node, ast.Call) and node.func.id == "frozenset", (
        f"unhandled _sweep argument shape {ast.unparse(node)!r}")
    return frozenset(ast.literal_eval(node.args[0]) if node.args else ())


def _swept_by(kind: str) -> frozenset[str]:
    """The allowed-key set the registered PARSER for ``kind`` sweeps with.

    The grammar's owner is the handler parser (Tasks 7-9), and that is what
    this derives from: ``PARSERS[kind]``'s source, not the executor's.  Three
    shapes and no fourth, which is what the sweep census measured: a
    module-level name, a ternary over a ``spec.kind == "<literal>"`` flag
    already resolved into ``scope``, and a ``frozenset(...)`` literal written
    at the call site.  Resolved rather than ``eval``-ed so that a fourth shape
    is a loud ``KeyError``/``AttributeError`` here instead of a silently
    different set.
    """
    fn = PARSERS[kind]
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    scope = dict(vars(sys.modules[fn.__module__]))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Compare)
                and len(node.value.ops) == 1
                and isinstance(node.value.ops[0], ast.Eq)
                and ast.unparse(node.value.left) in ("run.kind", "spec.kind")
                and isinstance(node.value.comparators[0], ast.Constant)):
            scope[node.targets[0].id] = kind == node.value.comparators[0].value
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "_sweep"]
    assert len(calls) == 1, f"{kind}: {len(calls)} _sweep calls, expected one"
    return _allowed_set(calls[0].args[1], scope)


class TestRegistration:
    def test_the_five_ids_are_registered(self):
        """Kills the mutation that writes a check and never decorates it.

        Every other test here calls ``preflight``, so an unregistered check
        would already show as a missing finding -- but only for the checks
        whose findings some test reads.  This one names all five."""
        assert set(_IDS) <= set(CHECKS)

    def test_the_a1_ids_all_emit_the_schema_s_own_id(self):
        """Kills a Finding.check of "A1.runs": the user is told to read
        schema check A1, which is what §6 calls it."""
        document = preflight_document(
            runs=[{"kind": "forward", "tpyo": 1}],
            resources=_a_beam(apod_deg={"value": 1.0, "unit": "deg"}),
            variants={"bad": {"campaign": {}}})
        emitted = {f.check for f in preflight(document).findings}
        assert "A1" in emitted
        assert not any(f.check.startswith("A1.")
                       for f in preflight(document).findings)

    def test_every_finding_this_module_emits_ends_with_its_check_tag(self):
        """`Finding`'s own docstring: a message "ends with `(check A30).` when
        `check` is set", and §3.2(c) says the tail is APPENDED so a moved
        message survives verbatim.  Until this test the convention was stated
        in two docstrings and enforced nowhere for this module's five checks.

        Driven on a document that lights all three schema ids at once, so a
        tag dropped from any one of them is red -- including on a
        VARIANT-layer finding, whose sentence carries a prefix in front and
        must still carry the tag at the end."""
        document = preflight_document(
            runs=[{"kind": "forward", "tpyo": 1}],
            resources=_a_beam(apod_deg={"value": 1.0, "unit": "deg"}),
            inference={"twin": {"without": ["noise"]},
                       "transitions": {},
                       "parameters": {"d": {"init": 0.5,
                                            "into": ["global_signal.depth",
                                                     "gain.gain"]}}},
            variants={"bad": {"campaign": {}}})
        found = [f for f in preflight(document).findings
                 if f.check in ("A1", "A38", "A39")]
        assert {f.check for f in found} == {"A1", "A38", "A39"}
        assert len(found) == 5
        for one in found:
            assert one.message.endswith(f"(check {one.check}).")


class TestRunOptionKeys:
    def test_an_unknown_run_option_is_refused(self):
        """Kills the whole check: today load_document PASSES this document
        and only execute_run refuses it, after every earlier run has run."""
        found = _findings(preflight_document(
            runs=[{"kind": "forward", "tpyo_key": 3}]), "A1")
        assert [f.where for f in found] == ["runs[0]"]
        assert "tpyo_key" in found[0].message

    def test_the_typo_is_attributed_to_the_run_that_carries_it(self):
        """Shape 1 -- attribution, not presence.  A check that reported the
        FIRST run would satisfy `"tpyo_key" in message`, and would send the
        reader to edit a run that is correct.  Both the document path and the
        run's own name are pinned, and the innocent run's name is pinned
        ABSENT."""
        found = _findings(preflight_document(runs=[
            {"name": "innocent", "kind": "forward"},
            {"name": "guilty", "kind": "nuts", "num_smaples": 4}]), "A1")
        assert [f.where for f in found] == ["runs[1]"]
        assert "runs['guilty']" in found[0].message
        assert "innocent" not in found[0].message

    def test_a_legal_option_is_silent(self):
        """Kills `return [refuse(...)]` -- a check that refuses every run."""
        assert _findings(preflight_document(
            runs=[{"kind": "nuts", "num_samples": 4}]), "A1") == []

    def test_a_single_mapping_runs_section_is_swept_too(self):
        """`runs:` may be one mapping (parse_runs wraps it, runs.py:122-123).
        Kills a check written `for entry in document["runs"]`, which iterates
        that mapping's KEYS and finds nothing."""
        found = _findings(preflight_document(
            runs={"kind": "forward", "tpyo": 1}), "A1")
        assert [f.where for f in found] == ["runs[0]"]

    def test_an_unparseable_runs_section_yields_nothing_here(self):
        """`kind: not_an_exit` is parse_runs' refusal and run_document's to
        report.  Kills a check that re-implements the kind enum and hands the
        user two refusals, in two voices, for one typo."""
        assert _findings(preflight_document(
            runs=[{"kind": "not_an_exit", "tpyo": 1}]), "A1") == []

    def test_every_registered_handler_has_a_table_entry(self):
        """Kills a kind dropped from the table -- which would make its
        options unswept at P-1 with nothing to say so."""
        assert set(_task3_allowed_run_options()) == set(PARSERS)

    def test_the_table_is_the_handlers_own_allowed_sets(self):
        """The anti-drift guard, and the reason this task did not restate
        sixteen key lists.  It reads each registered parser's ONE
        ``_sweep(spec, X)`` call out of its source, resolves X in that
        module's globals -- binding the ``drawing``/``estimate`` flags, which
        are ``spec.kind == "<literal>"`` and so are decidable from the kind
        alone -- and compares.

        Kills: a key added to a parser's own frozenset and not to this
        table; a table entry copied and then edited; and the two conditional
        sweeps resolved to the wrong branch."""
        table = _task3_allowed_run_options()
        for kind in sorted(PARSERS):
            assert table[kind] == _swept_by(kind), kind

    def test_thirteen_of_the_sixteen_entries_are_the_handlers_own_object(self):
        """Not equality -- identity.  Thirteen kinds bind their allowed set to
        a module-level name this table imports, so the two CANNOT drift.  The
        other three (``forward``, ``fisher``, ``npe``) write theirs as a
        literal at the parser's ``_sweep`` call site and have no name to
        import; they are restated, and the test above is what holds them.  The
        count is asserted so that turning a fourteenth into a literal is a red
        test rather than a silent loss of the identity guarantee."""
        table = _task3_allowed_run_options()
        by_name = 0
        for kind, fn in PARSERS.items():
            module = sys.modules[fn.__module__]
            if any(value is table[kind] for value in vars(module).values()):
                by_name += 1
        assert by_name == 13

    def test_a_key_its_executor_refuses_by_name_is_left_to_that_executor(self):
        """`condition` + `prior_mean:` is the shape `conjugate.py:580-582`
        argues about by name: the bespoke refusal runs "BEFORE the sweep on
        purpose: the sweep would fire first with the generic 'does not take
        [...]' and the reader would fix the symptom by deleting a key they had
        good reason to write."  Hoisting the generic sweep to P-1 inverts
        exactly that, in a phase the executor cannot reach.

        Kills `_TASK3_SPOKEN_FOR` emptied.  Measured on this tree with the
        table replaced by `{}` and this module's derivation test deselected:
        SIX tests outside this module go red -- two in
        `test_config_exits_diagnostics.py`, two in
        `test_config_exits_predict.py`, one in `test_config_exits_npe.py` and
        one in `test_config_exits_plan.py` -- and every one of them is a
        document carrying one of the five keys."""
        assert _findings(preflight_document(
            runs=[{"kind": "condition", "prior_mean": {"g": 1.0}}]),
            "A1") == []
        # ...and the stand-down is for a run that CARRIES such a key, not for
        # the kind: an ordinary typo on a run with no spoken-for key is ours.
        found = _findings(preflight_document(
            runs=[{"kind": "condition", "tpyo": 1}]), "A1")
        assert [f.where for f in found] == ["runs[0]"]

    def test_the_stand_down_is_for_the_whole_run_and_not_for_the_key_alone(self):
        """Kills the refinement that looks strictly better and is not:
        dropping only the spoken-for key from the options P-1 sweeps and
        sweeping the rest.

        Measured -- that form turns `test_config_exits_diagnostics.py:324`
        red.  That test declares `prior_mean:` AND `tol:` on one `condition`
        run and pins the CENTRE refusal as the one heard, because "a message
        with no `tol` in it is proof of which check ran first"; a P-1 that
        swept the rest of the run answers "does not take ['tol']" one phase
        earlier, which is the exact inversion `_TASK3_SPOKEN_FOR` exists to
        prevent.

        The test above cannot see it: its spoken-for document carries no
        second key, so both implementations are silent on it."""
        assert _findings(preflight_document(runs=[
            {"kind": "condition", "prior_mean": {"g": 1.0}, "tol": 1.0e-9}]),
            "A1") == []

    def test_the_spoken_for_table_is_every_handler_that_raises_first(self):
        """DERIVED, not restated.  Each registered PARSER is read with `ast`:
        a string literal tested for membership in the options BEFORE that
        parser's one `_sweep(...)` call is a key the handler speaks about
        itself, and those are exactly the keys P-1 must stand down on.

        Kills a sixth such key shipping in a parser with
        `_TASK3_SPOKEN_FOR` untouched -- under which the generic sweep
        silently displaces a bespoke message again, in a phase no existing
        test covers.  A test that restated the five keys could not see that;
        this one goes red on the commit that adds the sixth.

        **The derivation follows one level of CALL**, into a helper defined in
        the parser's own module, and that is not generality for its own sake:
        plan 3A's Task 8 lifted three of these refusals to module level so
        that `preflight/fitting.py` could call the same object from the raw
        document (§2.2, one name, one binding, two call sites), and Tasks 7-9
        moved the call sites from the executors into the parsers without
        touching the behaviour.  A derivation that read the parser's body
        alone would have gone red on that refactor, which changed nothing the
        table is about, and "update `_TASK3_SPOKEN_FOR`" is the wrong repair
        for it: dropping `plan.estimate: {seed}` would let A1's generic sweep
        displace the A29 message again, which is the very thing the table
        exists to stop.  Restricted to the SAME module, so an `exit_support`
        helper called before the sweep does not drag its own membership tests
        in.

        **Measured against `d3ab22e`, re-measured after that refactor, and
        re-measured against the parsers at Task 10**: this derivation returns
        exactly `{condition: {prior_mean}, npe: {seed}, plan.estimate:
        {seed}, plan.sample: {seed}, predict: {from}}` -- five kinds, five
        keys, no extras -- which is the table in `preflight/document.py`
        character for character."""

        def options_membership(tree, cut):
            keys = set()
            for node in ast.walk(tree):
                if not isinstance(node, ast.Compare):
                    continue
                if cut is not None and node.lineno >= cut:
                    continue
                if not any(isinstance(op, ast.In) for op in node.ops):
                    continue
                if not isinstance(node.left, ast.Constant):
                    continue
                if not any(ast.unparse(one).endswith("options")
                           for one in node.comparators):
                    continue
                keys.add(node.left.value)
            return keys

        derived: dict[str, frozenset[str]] = {}
        for kind, fn in PARSERS.items():
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
            sweeps = [node.lineno for node in ast.walk(tree)
                      if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name)
                      and node.func.id == "_sweep"]
            assert sweeps, f"{kind}: no _sweep call"
            cut = min(sweeps)
            keys = options_membership(tree, cut)
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.lineno < cut):
                    continue
                helper = fn.__globals__.get(node.func.id)
                if (not inspect.isfunction(helper)
                        or helper.__module__ != fn.__module__):
                    continue
                keys |= options_membership(
                    ast.parse(textwrap.dedent(inspect.getsource(helper))),
                    None)
            if keys:
                derived[kind] = frozenset(keys)
        assert derived == _TASK3_SPOKEN_FOR, (
            "a handler parser's pre-sweep refusal changed; update "
            "_TASK3_SPOKEN_FOR in preflight/document.py")


class TestUnselectedVariants:
    def test_a_variant_that_would_be_refused_is_reported_now(self):
        """Measured: this document loads clean today and the refusal waits
        for the run that selects the variant.  Kills the whole check."""
        found = _findings(preflight_document(
            variants={"sneaky": {"campaign": {}}}), "A1")
        assert [f.where for f in found] == ["variants.sneaky.campaign"]
        assert "capability 4" in found[0].message

    def test_the_finding_names_the_offending_variant(self):
        """Shape 1.  Two variants, one bad: a check reporting `variants.good`
        or bare `variants` passes any membership test on the message and sends
        the reader to the wrong block."""
        document = preflight_document(variants={
            "good": {"model": {"gain": {"gain": {"value": 1.0,
                                                 "unit": "dimensionless"}}}},
            "bad": {"schema_version": 2}})
        with pytest.raises(ConfigError) as caught:
            preflight(document)
        assert str(caught.value) == (
            "variant 'bad' touches 'schema_version'. The version belongs to "
            "the document; a patch that changes -- or deletes -- how the "
            "document is read is not a patch."
        )
        assert "good" not in str(caught.value)

    def test_a_variant_deleting_a_required_section_is_caught(self):
        """`~model: null` is the delete form (layering.py:26-38).  Kills a
        check that only looks at the patch's own keys and never merges."""
        found = _findings(preflight_document(
            variants={"nomodel": {"~model": None}}), "A1")
        assert [f.where for f in found] == ["variants.nomodel.model"]
        assert "missing ['model']" in found[0].message

    @pytest.mark.parametrize(("variants", "kind"), [
        (["a"], "tuple"),
        ([], "tuple"),
        (0, "int"),
        ("", "str"),
        (None, "NoneType"),
    ], ids=["a-list", "an-empty-list", "a-zero", "an-empty-string",
            "an-explicit-null"])
    def test_a_non_mapping_variants_section_is_rejected(self, variants, kind):
        """Presence is significant: null and omission are not equivalent."""
        document = preflight_document()
        document["variants"] = variants
        with pytest.raises(ConfigError) as caught:
            preflight(document)
        assert str(caught.value) == (
            "variants: is a mapping of name -> patch; got " f"{kind}."
        )

    def test_a_legal_variant_is_silent(self):
        """Kills a check that reports every variant it merges."""
        assert _findings(preflight_document(variants={
            "unity": {"model": {"gain": {"gain": {"value": 1.0,
                                                  "unit": "dimensionless"}}}}}),
            "A1") == []

    def test_the_interior_of_an_unselected_variant_is_out_of_scope_here(self):
        """A1 owns grammar only; the pass driver sends A2 over this layer."""
        assert _findings(preflight_document(
            variants={"unused": {"model": {"ghost": {}}}}), "A1") == []


class TestHorizonNumbers:
    def test_a_value_node_apod_deg_is_refused_rather_than_a_bare_typeerror(self):
        """Measured on this tree: `apod_deg: {value: 0.1, unit: rad}` reaches
        the user as `TypeError: float() argument must be a string or a real
        number, not 'dict'` from inside build_beam, after the beam file has
        been read.  Kills the whole check."""
        found = _findings(preflight_document(
            resources=_a_beam(apod_deg={"value": 0.1, "unit": "rad"})), "A1")
        assert [f.where for f in found] == [
            "resources.beams.horn.horizon.apod_deg"]

    def test_el_deg_is_the_twin_and_is_refused_the_same_way(self):
        """Shape 4.  `beams.py:482` is `float(horizon.get("el_deg", 90.0))` --
        the same line, one key over -- and the survey named only `apod_deg`.
        A check written for one key alone passes every apod_deg test."""
        found = _findings(preflight_document(
            resources=_a_beam(el_deg={"value": 90.0, "unit": "deg"})), "A1")
        assert [f.where for f in found] == [
            "resources.beams.horn.horizon.el_deg"]

    def test_the_angle_is_attributed_to_the_beam_that_carries_it(self):
        """Shape 1 -- attribution, not presence, and the twin of
        `test_the_typo_is_attributed_to_the_run_that_carries_it`, which
        `runs[i]` had and this loop did not.

        Kills `where = f"resources.beams.{list(beams)[0]}..."` and any other
        form that names A beam rather than THE beam: both satisfy every
        assertion in this class, and both send a reader to edit the innocent
        entry.  The innocent name is pinned absent from the message too."""
        found = _findings(preflight_document(resources={"beams": {
            **_beam_entry("good", apod_deg=0.0),
            **_beam_entry("bad", apod_deg={"value": 0.1, "unit": "rad"})}}),
            "A1")
        assert [f.where for f in found] == [
            "resources.beams.bad.horizon.apod_deg"]
        assert "good" not in found[0].message

    def test_the_key_is_checked_even_where_the_mode_never_reads_it(self):
        """Measured: under `mode: none` the value is never touched, so the
        document builds and the declared apodisation silently does nothing.
        Kills a check gated on `mode == "truncate_map"`."""
        section = _a_beam()
        section["beams"]["horn"]["horizon"] = {
            "mode": "none", "apod_deg": {"value": 0.1, "unit": "rad"}}
        assert len(_findings(preflight_document(resources=section), "A1")) == 1

    @pytest.mark.parametrize("value", [0.0, 20, 90.0])
    def test_a_plain_number_is_accepted(self, value):
        """Kills a check that refuses the form the code actually takes."""
        assert _findings(preflight_document(
            resources=_a_beam(apod_deg=value)), "A1") == []

    def test_a_bool_is_not_a_number_here(self):
        """`float(True)` is 1.0, so a bool builds a 1-degree taper out of a
        typo.  Kills `isinstance(value, (int, float))` written without the
        bool exclusion -- which passes every other test in this class."""
        assert len(_findings(preflight_document(
            resources=_a_beam(apod_deg=True)), "A1")) == 1


class TestFanPresence:
    def test_two_targets_with_no_fan_are_refused_in_the_sugar_spelling(self):
        """Measured: this builds today with `Bind.fan = None`.  Kills the
        whole presence half; the registry-consistency half already exists at
        transforms.py:269-276 and is untouched."""
        found = _findings(preflight_document(inference={
            "twin": {"without": ["noise"]},
            "parameters": {"d": {"init": 0.5, "into": ["global_signal.depth",
                                                       "gain.gain"]}}}), "A38")
        assert [f.where for f in found] == ["inference.parameters.d"]

    def test_two_targets_with_no_fan_are_refused_in_the_bindings_spelling(self):
        """Shape 4.  transforms.py calls `_merged_fan` from TWO loops (:356
        and :395); a check written over `inference.parameters` alone leaves
        the longhand open, and measured, the longhand builds too."""
        found = _findings(preflight_document(inference={
            "twin": {"without": ["noise"]},
            "parameters": {"g": {"init": 1.0}},
            "bindings": [{"latents": ["g"], "into": ["gain.gain",
                                                     "global_signal.depth"]}]}),
            "A38")
        assert [f.where for f in found] == ["inference.bindings[0]"]

    def test_the_binding_index_is_the_one_that_carries_it(self):
        """Shape 1 again, on the third loop -- `runs[i]` had this test and
        neither of its twins did.

        Kills a hard-coded `inference.bindings[0]`, and any form that reports
        the first binding rather than the offending one: a document with two
        bindings sends the reader to edit the correct one, and every other
        assertion in this class -- all of which declare ONE binding -- stays
        green."""
        found = _findings(preflight_document(inference={
            "twin": {"without": ["noise"]},
            "parameters": {"g": {"init": 1.0}, "d": {"init": 0.5}},
            "bindings": [{"latents": ["g"], "into": ["gain.gain"]},
                         {"latents": ["d"], "into": ["global_signal.depth",
                                                     "gain.gain"]}]}), "A38")
        assert [f.where for f in found] == ["inference.bindings[1]"]
        assert "inference.bindings[1]" in found[0].message

    def test_one_target_written_as_a_list_is_not_refused(self):
        """§4.7.2 words the trigger as "req iff `into` is a list"; §2.2 words
        it as "more than one entry".  `parameters.py:114 _names` collapses
        `"a"` and `["a"]` to the same `("a",)`, so the §4.7.2 reading cannot
        be implemented without changing `_names` -- §2.6 item 5 decides for
        §2.2's.  This test is that decision, and it goes red if someone
        implements the other reading."""
        assert _findings(preflight_document(inference={
            "twin": {"without": ["noise"]},
            "parameters": {"g": {"init": 1.0, "into": ["gain.gain"]}}}),
            "A38") == []

    @pytest.mark.parametrize("transform", ["exp", "split_rows",
                                           {"affine": {"scale": 2.0}}])
    def test_a_transform_that_carries_its_own_fan_is_not_refused(self,
                                                                 transform):
        """Measured: every transform form except None and "identity" returns
        a non-None canonical fan, and `_merged_fan(None, canonical)` returns
        it -- so there is no guess left to refuse.  Kills a literal
        `len(into) > 1` with no transform clause, which would refuse
        `split_rows` (the one transform whose whole purpose is two targets)
        and turn tests/config/test_config_transforms.py:591 red."""
        assert _findings(preflight_document(inference={
            "twin": {"without": ["noise"]},
            "parameters": {"d": {"init": 0.5, "transform": transform,
                                 "into": ["global_signal.depth",
                                          "gain.gain"]}}}), "A38") == []

    def test_transform_identity_is_not_a_transform_for_this_purpose(self):
        """`parse_transform("identity")` returns `(None, None)` -- measured --
        so the ambiguity is exactly the same as writing no transform at all.
        Kills `if "transform" in spec: return`."""
        found = _findings(preflight_document(inference={
            "twin": {"without": ["noise"]},
            "parameters": {"d": {"init": 0.5, "transform": "identity",
                                 "into": ["global_signal.depth",
                                          "gain.gain"]}}}), "A38")
        assert [f.where for f in found] == ["inference.parameters.d"]

    @pytest.mark.parametrize("fan", ["broadcast", "distribute"])
    def test_a_declared_fan_silences_it(self, fan):
        """Kills a check that ignores the key it exists to require."""
        assert _findings(preflight_document(inference={
            "twin": {"without": ["noise"]},
            "parameters": {"d": {"init": 0.5, "fan": fan,
                                 "into": ["global_signal.depth",
                                          "gain.gain"]}}}), "A38") == []

    def test_fan_written_as_an_explicit_null_is_no_fan_at_all(self):
        """`fan: ~` is YAML for None, and `parse_transform`/`_merged_fan` see
        exactly what an absent key gives them: `_merged_fan(None, None)`
        returns None, `Bind.fan is None`, and the broadcast-versus-distribute
        ambiguity A38 exists to prevent ships.

        Kills `if "fan" in spec: return` -- membership rather than value --
        which passes every other test in this class, because every one of them
        either omits the key or gives it a real mode."""
        found = _findings(preflight_document(inference={
            "twin": {"without": ["noise"]},
            "parameters": {"d": {"init": 0.5, "fan": None,
                                 "into": ["global_signal.depth",
                                          "gain.gain"]}}}), "A38")
        assert [f.where for f in found] == ["inference.parameters.d"]

    def test_the_message_names_the_targets_in_the_written_order(self):
        """Shape 1.  `distribute` writes the k-th value into the k-th target,
        so a message that lists them in the wrong order tells the reader the
        opposite of what their document does."""
        found = _findings(preflight_document(inference={
            "twin": {"without": ["noise"]},
            "parameters": {"d": {"init": 0.5, "into": ["global_signal.depth",
                                                       "gain.gain"]}}}), "A38")
        assert "['global_signal.depth', 'gain.gain']" in found[0].message


class TestCapabilityKeys:
    #: The eight keys schema §7's summary table reserves at capability 3 or 4,
    #: with the capability and section §8.1 and §8.2 give each -- enumerated
    #: from the schema rather than from the survey.
    SCHEMA_8 = {
        "campaign": ("capability 4 (streaming evidence)", "§8.2"),
        "inference.transitions": ("capability 4 (streaming evidence)", "§8.2"),
        "inference.parameters.<name>.scope":
            ("capability 4 (streaming evidence)", "§8.2"),
        "inference.parameters.<name>.support":
            ("capability 4 (streaming evidence)", "§8.2"),
        "inference.parameters.<name>.hyper":
            ("capability 4 (streaming evidence)", "§8.2"),
        "model.<node>.type: NeuralOperator":
            ("capability 3 (neural surrogates)", "§8.1"),
        "outputs.write.memory_archive":
            ("capability 4 (streaming evidence)", "§8.2"),
        "outputs.write.posterior_net":
            ("capability 3 (neural surrogates)", "§8.1"),
    }

    def test_the_table_is_schema_8s_eight_keys_and_their_capabilities(self):
        """Kills a table that grew a key §8 does not reserve, one that lost
        the two `outputs.write` keys, AND one whose values are wrong.

        Value equality, not a key-set comparison, and the difference is not
        pedantry: no check can reach the two `outputs.write` rows today
        (`_structural` refuses `outputs:` first), so they carry no message and
        no test of their own, and they exist for exactly one stated reason --
        "so that Plan 4 inherits the capability and the section rather than
        re-deriving them".  For those two rows the VALUES are the entire
        payload.  Measured: swapping their two capabilities and two sections
        survives a key-set comparison at exit 0."""
        assert _CAPABILITY_KEYS == self.SCHEMA_8
        assert len(self.SCHEMA_8) == 8

    def test_transitions_names_its_capability_and_section(self):
        """The measured hole: today this falls to `inference:`'s generic
        unknown-key sweep -- "inference: does not take ['transitions']" --
        which reads as a typo rather than as a reserved key."""
        found = _findings(preflight_document(inference={
            "twin": {"without": ["noise"]}, "transitions": {"g": {"ou": {}}}}),
            "A39")
        assert [f.where for f in found] == ["inference.transitions"]
        assert "capability 4" in found[0].message
        assert "§8.2" in found[0].message

    @pytest.mark.parametrize("section,where", [
        ({"twin": {"without": ["noise"]}, "transitions": {}},
         "inference.transitions"),
        (_latent(support=[0.0, 1.0]), "inference.parameters.g.support"),
        (_latent(hyper={"of": ["x"]}), "inference.parameters.g.hyper"),
        (_latent(scope="per_epoch"), "inference.parameters.g.scope"),
        (_latent(scope="linked"), "inference.parameters.g.scope"),
    ])
    def test_each_inference_key_is_refused_by_capability_name(self, section,
                                                              where):
        """Shape 1 again: `where` is pinned per key, so a check that reported
        every one of them against `inference` -- which satisfies any
        "capability 4 in message" test -- goes red."""
        found = _findings(preflight_document(inference=section), "A39")
        assert [f.where for f in found] == [where]
        assert "capability 4" in found[0].message
        assert "§8.2" in found[0].message

    def test_a_neural_operator_type_is_refused_at_capability_3(self):
        """The one capability-3 key a v1 document can write.  Today it is
        refused correctly but at model-build time, i.e. after every beam in
        the document has been read."""
        found = _findings(preflight_document(model={
            "gain": {"gain": {"value": 1.0, "unit": "dimensionless"}},
            "bandpass": {"type": "NeuralOperator"}}), "A39")
        assert [f.where for f in found] == ["model.bandpass.type"]
        assert "capability 3" in found[0].message
        assert "§8.1" in found[0].message

    @pytest.mark.parametrize("patch,expected", [
        ({"inference": _latent(support=[0.0, 1.0])},
         "inference.parameters.g.support: is reserved with capability 4 "
         "(streaming evidence), schema §8.2, and refused in v1 (check A39)."),
        ({"inference": _latent(hyper={"of": ["x"]})},
         "inference.parameters.g.hyper: is reserved with capability 4 "
         "(streaming evidence), schema §8.2, and refused in v1 (check A39)."),
        ({"inference": _latent(scope="per_epoch")},
         "inference.parameters.g.scope: 'per_epoch' is reserved with "
         "capability 4 (streaming evidence), schema §8.2, and refused in v1 "
         "(check A39)."),
        ({"model": {"bandpass": {"type": "NeuralOperator"}}},
         "model.bandpass.type: NeuralOperator is reserved with capability 3 "
         "(neural surrogates), schema §8.1, and refused in v1 (check A39)."),
        ({"inference": {"twin": {"without": ["noise"]}, "transitions": {}}},
         "inference.transitions: is reserved with capability 4 (streaming "
         "evidence), schema §8.2, and refused in v1 (check A39)."),
    ], ids=["support", "hyper", "scope", "neural-operator", "transitions"])
    def test_each_corrected_message_is_pinned_verbatim(self, patch, expected):
        """§2.3 grants A39's correction exception "each with its own test
        pinning the new text", and this is that test.

        EQUALITY, and it has to be.  Every other assertion here reads
        `"capability 4" in message` and `"§8.x" in message` -- which is no
        stronger than what `test_config_section_parameters.py` and
        `test_config_section_model.py` already assert on the OLD route with
        `match="capability 4"`.  Membership therefore cannot tell the
        corrected sentence from the one it replaced, which is the entire
        content of the exception §2.3 grants.  Measured: rewording the whole
        sentence to `"...reserved, capability 4, schema §8.2"` survives every
        membership assertion in this class at exit 0.

        Equality also pins two things nothing else does: the `(check A39).`
        tail `Finding`'s own docstring requires of every id-carrying message,
        and `got=` -- without which the scope row stops naming WHICH scope was
        written, and the reader is told a key they cannot see is reserved."""
        found = _findings(preflight_document(**patch), "A39")
        assert [f.message for f in found] == [expected]

    def test_a_scope_typo_is_not_claimed_to_be_a_capability_key(self):
        """`scope: glboal` is a typo, not a reservation, and A39 refuses the
        two names §8.2 lists and nothing else.  Kills `if scope != "global"`,
        which would make this pass tell a user with a typo to wait for
        capability 4.

        **What this test does NOT claim.** The typo still reaches the user as
        a capability-4 sentence -- `sections/parameters.py:151` refuses ANY
        unknown `scope:` by naming capability 4, measured.  All this assertion
        buys is that P-1 does not add a second voice saying the same wrong
        thing one phase earlier; the wrong thing itself is unchanged and is
        outside every task's Files list here.
        """
        assert _findings(preflight_document(
            inference=_latent(scope="glboal")), "A39") == []

    @pytest.mark.parametrize("key,provoke", [
        ("campaign",
         lambda: _document_sweep({"schema_version": 1, "campaign": {}})),
        ("inference.parameters.<name>.support",
         lambda: parse_latents({"g": {"init": 1.0, "into": "gain.gain",
                                      "support": [0.0, 1.0]}}, _CTX)),
        ("inference.parameters.<name>.hyper",
         lambda: parse_latents({"g": {"init": 1.0, "into": "gain.gain",
                                      "hyper": {}}}, _CTX)),
        ("inference.parameters.<name>.scope",
         lambda: parse_latents({"g": {"init": 1.0, "into": "gain.gain",
                                      "scope": "per_epoch"}}, _CTX)),
        ("model.<node>.type: NeuralOperator",
         lambda: _pick_class("bandpass", (), {"type": "NeuralOperator"})),
    ])
    def test_the_sections_own_refusal_agrees_with_the_table(self, key, provoke):
        """§2.2 says one property gets one binding.  These five refusals stay
        where they are (this task edits no section module), so the two places
        are held together by THIS: the section's own live message must carry
        the same capability number and the same schema section the table
        gives the pre-flight refusal.  Kills the `_number`-vs-`_whole`
        divergence -- two validators for one property, disagreeing."""
        capability, section = _CAPABILITY_KEYS[key]
        with pytest.raises(ConfigError) as excinfo:
            provoke()
        message = str(excinfo.value)
        assert capability.split(" (")[0] in message
        assert section in message

    def test_the_two_outputs_keys_still_arrive_by_the_plan_4_route(self):
        """The recorded gap, pinned rather than described.  `_structural`
        refuses `outputs:` wholesale and raises before any check runs, so
        neither `memory_archive` nor `posterior_net` can be reached from
        here; §2.6 item 6's "every capability key names its capability" is
        unexecutable for these two without editing `_structural`, which is
        Task 2's file.  §6 carries it."""
        for key in ("memory_archive", "posterior_net"):
            with pytest.raises(ConfigError, match="Plan 4"):
                _document_sweep({"schema_version": 1,
                                 "outputs": {"write": {key: {}}}})


class TestTheVariantRoute:
    def test_a_capability_key_inside_an_unselected_variant_is_refused(self):
        """Shape 4, and the brief's own warning: these keys arrive through
        `variants:` as well as through the base.  Kills a check that reads
        only `document[...]`."""
        found = _findings(preflight_document(variants={"v": {
            "inference": {"twin": {"without": ["noise"]},
                          "transitions": {"g": {"ou": {}}}}}}), "A39")
        assert [f.where for f in found] == ["variants.v.inference.transitions"]

    def test_a_run_option_typo_inside_an_unselected_variant_is_refused(self):
        """The same twin for A1.  A variant patching `runs:` never changes
        which runs execute (runs.py:10-12), but it does change what
        `load_document(variant=...)` accepts -- so the typo is real and today
        nothing sees it at all."""
        found = _findings(preflight_document(
            variants={"v": {"runs": [{"kind": "forward", "tpyo": 1}]}}), "A1")
        assert [f.where for f in found] == ["variants.v.runs[0]"]

    def test_a_variant_layer_finding_says_so_in_its_own_sentence(self):
        """`raise_if_refused` quotes the MESSAGE, not the `where`
        (findings.py:170) -- so a finding whose sentence says only
        `inference.transitions:` sends a reader to grep a base document that
        does not contain the key at all.

        Kills the layer walk that rewrites `where` and leaves the sentence
        alone, which every other test in this class passes: they all read
        `where`."""
        found = _findings(preflight_document(variants={"v": {
            "inference": {"twin": {"without": ["noise"]},
                          "transitions": {}}}}), "A39")
        assert found[0].message.startswith("variants.v: ")

    @pytest.mark.parametrize("patch,where,named", [
        ({"variants": {"unity-gain": {"campaign": {}}}},
         "variants", "variants.unity-gain"),
        ({"variants": {"unity-gain": {
            "inference": {"twin": {"without": ["noise"]},
                          "transitions": {}}}}},
         "variants", "variants.unity-gain"),
        ({"inference": {"twin": {"without": ["noise"]},
                        "parameters": {"d-1": {"init": 0.5,
                                               "into": ["global_signal.depth",
                                                        "gain.gain"]}}}},
         "inference.parameters", "inference.parameters.d-1"),
        ({"resources": {"beams": _beam_entry(
            "horn-a", apod_deg={"value": 0.1, "unit": "rad"})}},
         "resources.beams", "resources.beams.horn-a.horizon.apod_deg"),
        ({"inference": {"twin": {"without": ["noise"]},
                        "parameters": {"d-1": {"init": 0.5,
                                               "into": "gain.gain",
                                               "support": [0.0, 1.0]}}}},
         "inference.parameters", "inference.parameters.d-1.support"),
        ({"inference": {"twin": {"without": ["noise"]},
                        "parameters": {"d-1": {"init": 0.5,
                                               "into": "gain.gain",
                                               "scope": "per_epoch"}}}},
         "inference.parameters", "inference.parameters.d-1.scope"),
        ({"model": {"band-pass": {"type": "NeuralOperator"}}},
         "model", "model.band-pass.type"),
    ], ids=["a-variant-name", "a-variant-name-carrying-a-key", "a-latent-name",
            "a-beam-name",
            "a-latent-name-under-a-capability-key", "a-latent-name-under-scope",
            "a-model-node-name"])
    def test_a_name_the_path_grammar_cannot_spell_does_not_crash_the_pass(
            self, patch, where, named):
        """A user's own names are not identifiers.  `variants: {unity-gain:
        ...}` and `parameters: {d-1: ...}` both load today -- apply_variant
        and parse_latents validate no name -- while `Finding.where` must
        parse: `preflight._check_where` calls `parse_path`, whose segment
        grammar (`paths.py:35`) admits no hyphen, and it raises OUTSIDE the
        per-check try, so the whole pass dies naming the check rather than
        reporting the violation.

        Kills the obvious `where=f"variants.{name}"` and
        `where=f"inference.parameters.{name}"`: `where` is cut back to the
        longest prefix the grammar can spell and the full path stays in the
        sentence, which is what the user reads.

        **All four call sites of `_task3_where` are driven here, and that is
        the point of the last four rows.** Measured: with only the
        `_variant_text` and `_task3_fan_one` rows, dropping the cutback from
        `_task3_horizon_in` and from `_task3_capability` survived eight
        modules at exit 0 -- and each of those two reaches an ordinary
        document (a hyphenated beam name, a hyphenated latent under
        `support:`/`scope:`, a hyphenated model node under `type:`) whose
        user then gets `pre-flight check 'A39' emitted where=... which is not
        a document path` and loses every other finding."""
        found = [f for f in preflight(preflight_document(**patch)).findings
                 if f.check in ("A1", "A38", "A39")]
        assert [f.where for f in found] == [where]
        assert named in found[0].message

    @pytest.mark.parametrize("unknown", ["bad-name", ".", "雪！"])
    def test_a_variant_unknown_top_key_is_cut_back_only_after_attribution(
            self, unknown):
        """Keep the raw key until the variant prefix makes a legal path."""
        found = [
            finding
            for finding in preflight(
                preflight_document(variants={"x": {unknown: {}}})
            ).findings
            if finding.check == "A1"
        ]

        assert [finding.where for finding in found] == ["variants.x"]
        assert unknown in found[0].message

    def test_a_non_string_variant_name_is_rejected_in_source_order(self):
        with pytest.raises(ConfigError) as caught:
            preflight(preflight_document(variants={1: {"campaign": {}}}))
        assert str(caught.value) == (
            "initial_merge document: unsupported evidence mapping key type int."
        )

    def test_a_base_finding_is_not_repeated_once_per_variant(self):
        """Kills the obvious implementation of the layer walk.  Three layers
        with the same A38 violation would hand the user the same sentence
        three times, in three `where`s, two of which name a variant that did
        not introduce it."""
        found = _findings(preflight_document(
            inference={"twin": {"without": ["noise"]},
                       "parameters": {"d": {"init": 0.5,
                                            "into": ["global_signal.depth",
                                                     "gain.gain"]}}},
            variants={"a": {"runtime": {"seed": 1}},
                      "b": {"runtime": {"seed": 2}}}), "A38")
        assert [f.where for f in found] == ["inference.parameters.d"]

    def test_a_variant_that_breaks_the_rule_DIFFERENTLY_is_still_reported(self):
        """The other side of the de-duplication, and the side nothing pinned.

        The test above says a base finding is not repeated per layer; it says
        nothing about WHICH KEY decides "the same finding".  Kills
        de-duplicating on `finding.where` (and on `finding.message`): the base
        binds two targets and the variant rebinds the same latent to three, so
        the `where` is identical and the violation is not -- and under the
        `where` key the variant's own refusal is swallowed, which is a lost
        check rather than a duplicated sentence.

        Measured: shipped code reports both, in layer order."""
        found = _findings(preflight_document(
            inference={"twin": {"without": ["noise"]},
                       "parameters": {"d": {"init": 0.5,
                                            "into": ["global_signal.depth",
                                                     "gain.gain"]}}},
            variants={"v": {"inference": {"parameters": {"d": {"into": [
                "gain.gain", "global_signal.depth", "noise.sigma"]}}}}}),
            "A38")
        assert [f.where for f in found] == ["inference.parameters.d",
                                            "variants.v.inference.parameters.d"]
        assert "3 targets" in found[1].message

    def test_the_walk_is_one_layer_per_declared_variant(self):
        """The cost contract, without a clock.  Kills a walk that merges
        variants pairwise (layering is one level deep by design) and a walk
        that forgets the base document.

        Asserted on SHAPE rather than against a fixed list, because
        `preflight_document` MERGES one level deep (§3.2(b)) and the base
        document already declares `variants: {unity_gain: ...}` -- so the walk
        is the base plus EVERY declared variant, this patch's two included."""
        document = preflight_document(variants={
            "a": {"runtime": {"seed": 1}}, "b": {"runtime": {"seed": 2}}})
        prefixes = _layer_prefixes(document)
        assert prefixes[0] == ""
        assert prefixes[1:] == [f"variants.{name}"
                                for name in document["variants"]]
        assert {"variants.a", "variants.b"} <= set(prefixes)

    def test_a_variant_apply_variant_refuses_is_a_controlled_pass_refusal(self):
        """The canonical enumerator never drops an invalid declared variant."""
        document = preflight_document(variants={"chained": {"variants": {}}})
        with pytest.raises(ConfigError) as caught:
            preflight(document)
        assert str(caught.value) == (
            "variant 'chained' declares 'variants'. Layering is one level "
            "deep by design: there is no ordering between variants and no "
            "variant builds on another, so a comparison's halves cannot "
            "drift apart through a chain."
        )


class TestASectionThisPassCannotRead:
    @pytest.mark.parametrize("patch", [
        {"inference": None},
        {"inference": "nope"},
        {"inference": {"twin": {"without": ["noise"]},
                       "parameters": {"g": "nope"}}},
        {"inference": {"twin": {"without": ["noise"]}, "bindings": "nope"}},
        {"inference": {"twin": {"without": ["noise"]}, "bindings": ["nope"]}},
        {"resources": {"beams": "nope"}},
        {"resources": {"beams": {"horn": "nope"}}},
        {"resources": {"beams": {"horn": {"horizon": "el_deg: 90"}}}},
        {"runs": "nope"},
    ], ids=["no-inference", "inference-not-a-mapping", "a-latent-that-is-not",
            "bindings-not-a-list", "a-binding-that-is-not",
            "beams-not-a-mapping", "a-beam-that-is-not",
            "horizon-not-a-mapping", "runs-not-a-list"])
    def test_a_section_this_pass_cannot_read_yields_nothing_here(self, patch):
        """Every check in this module is a TEXT check on a WELL-FORMED
        section.  The refusal for a malformed one belongs to the section that
        parses it, and a second voice for one mistake is what §2.2 forbids.

        Kills each isinstance guard individually -- a check written
        `document["inference"]["parameters"].items()` raises on three of these
        rows, and §2.3's TRAP is that a check which RAISES aborts the pass and
        hides every finding after it.  The failure is therefore not a wrong
        finding but a `ConfigError: pre-flight check ... RAISED`, which no
        assertion about findings would see: this one calls `preflight` and so
        does.

        `horizon: "el_deg: 90"` is a STRING carrying the key name, not a bare
        `"nope"`, and the difference is what makes the row discriminate:
        measured, `horizon = spec.get("horizon") or {}` survives every
        non-mapping that does not contain "el_deg" as a substring (`"el_deg"
        not in "nope"` is True and the loop simply skips), and raises
        `TypeError: string indices must be integers` on this one."""
        found = [f for f in preflight(preflight_document(**patch)).findings
                 if f.check in ("A1", "A38", "A39")]
        assert found == []


class TestThePhase:
    def test_the_typo_wins_against_a_beam_that_cannot_be_read(self):
        """The whole point of Plan 3A in one assertion.  Measured on this
        tree BEFORE the pass existed: with a missing beam file and a run-key
        typo in the same document, `load_document` reports the beam --
        "No file at 'no_such_beam.npy'" -- because `build_resources`
        (`document.py:75` since Task 2 moved the head of that function;
        `:104` before it) runs before anything reads `runs:`.  Kills a hook
        placed after `build_resources`, which every other test in this module
        would pass."""
        from rheplicant.config.document import load_document

        document = preflight_document(
            runs=[{"kind": "forward", "tpyo_key": 3}],
            resources=UNREADABLE_BEAM)
        with pytest.raises(ConfigError) as excinfo:
            load_document(document)
        assert "tpyo_key" in str(excinfo.value)
        assert "no_such_beam" not in str(excinfo.value)
