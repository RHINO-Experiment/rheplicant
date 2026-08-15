"""Tasks 7-10: the checks that read ``runs[]`` and ``inference:`` together.

Every document here comes from ``tests/config/preflight_helpers.py`` (plan §3:
the ONE place a pre-flight test document is built).  Every test names, in a
comment, the wrong implementation it kills -- a test that cannot name one is
decoration, and 2C shipped twenty-seven surviving mutants every one of which
lived in a test rather than in the code it guarded.

**What the package really does, measured at ``9ee99af`` and not all of it what
this task's brief said.**  Fifteen documents were driven directly through
``Block`` and ``SamplingPlan`` over the three-latent space below:

* the six the brief quotes -- uncovered, doubled, undeclared, A17, A18, A19 --
  come back word for word as it quotes them, and the all-linear
  ``engine: gradient`` block carrying ``steps: 5`` is ACCEPTED, which is §2.6
  item 3's decision as a measurement;
* **``steps: 0`` does not reach A17 at all.**  ``Block._check``
  (``plan.py:360-368``) refuses it first, as *"asks for steps=0; inner steps
  must be a positive int"*, and so it does for ``steps: True`` and
  ``steps: '5'``.  A17's *"would be silently ignored"* is therefore false of
  every value the package would not have accepted, and the refusal here says
  so rather than sending the reader to ``engine: gradient``, which would
  leave the run refused for the second reason;
* a repeated name inside ONE block and an unknown ``engine:`` are also
  ``Block``'s (``plan.py:344-359``), not ``SamplingPlan``'s, which is why the
  differential below builds the ``Block`` inside its own ``try``.

**And ``warm_start.blocks`` is the same partition, one call along.**
``exits.py:268`` builds ``_blocks(f"{where}: warm_start", ...)`` and hands the
result to ``SamplingPlan(space, *warm_blocks)`` -- the same constructor, the
same space, the same four refusals, at P3 behind the same beam.  A check
written on ``runs[].blocks`` alone would guard one route and leave its
identical sibling open.
"""

import pytest

from rheplicant.config.findings import REFUSE
from rheplicant.config.preflight import CHECKS, preflight
from rheplicant.config.preflight.fitting import (
    _ENGINES,
    _a18_linear,
    _blocks,
    _kinds,
    _latents,
    _runs,
)
from tests.config.preflight_helpers import UNREADABLE_BEAM, preflight_document

#: A linear and a non-linear latent, and the third that makes "uncovered"
#: distinguishable from "declared".  The two linear ones differ in NAME only
#: so that a message asserting the wrong one is caught: a fixture whose
#: latents were interchangeable could not tell `blocks[0]` from `blocks[1]`.
LINEAR_D = {"init": 0.5, "linear": True, "into": "global_signal.depth"}
LINEAR_A = {"init": 10.0, "linear": True, "into": "uniform_sky.amplitude"}
NONLINEAR_W = {"init": 5.0, "into": "global_signal.width"}
THREE = {"d": LINEAR_D, "a": LINEAR_A, "w": NONLINEAR_W}


def _plan(blocks, kind="plan.estimate", **options):
    return {"name": "fit", "kind": kind, "blocks": blocks, **options}


def _doc(blocks, parameters=None, **run):
    return preflight_document(
        inference={"parameters": parameters or THREE},
        runs=[_plan(blocks, **run)])


def _warm(blocks, warm_blocks, **warm):
    """A ``plan.sample`` whose ``warm_start`` declares its own partition."""
    return _doc(blocks, kind="plan.sample", n_sweeps=4,
                seed={"from": "runtime.seeds.sample"},
                warm_start={"kind": "plan.estimate", "blocks": warm_blocks,
                            "move": ["d"], **warm})


def _found(document):
    return list(_blocks(document))


def _hostile_document(patch):
    """The base document with ``patch`` over this module's own ``inference:``.

    Built by ``update``-ing a plain dict rather than by writing
    ``preflight_document(**{"inference": ..., **patch})``, and the difference
    is not style: ``test_config_fixture_contract._writes`` reads a ``{**base,
    "inference": ...}`` LITERAL as a module rolling its own document, and
    flags this one -- measured, that spelling turns
    ``test_no_module_beyond_the_recorded_four_rolls_its_own`` red.  The
    document still comes from ``preflight_document`` and nowhere else, which
    is what that guard is actually for.
    """
    sections = {"inference": {"parameters": THREE}}
    sections.update(patch)
    return preflight_document(**sections)


class TestBlocks:
    def test_an_uncovered_latent_is_refused_naming_it(self):
        # Kills an implementation that checks only that every BLOCK name is
        # declared: that direction passes on this document, and the latent
        # `w` is silently frozen at 5.0 for the whole run.  The name is
        # asserted, not just "A16": a message naming `d` would send the user
        # to a latent that IS covered.
        found = _found(_doc([{"names": ["d", "a"]}]))
        assert [f.check for f in found] == ["A16"]
        assert found[0].severity == REFUSE
        assert "['w']" in found[0].message
        assert found[0].where == "runs[0].blocks"

    def test_a_latent_in_two_blocks_names_BOTH_block_indices(self):
        # Kills the presence-not-attribution shape (2C's shape 1, 7
        # instances): `assert "d" in message` passes on a message that names
        # one block, or the wrong two.  The user has to know WHICH two blocks
        # to edit, so both indices are pinned and they are different.
        found = _found(_doc([{"names": ["d", "a", "w"]}, {"names": ["d"]}]))
        assert [f.check for f in found] == ["A16"]
        assert "blocks[0] and in blocks[1]" in found[0].message
        assert found[0].where == "runs[0].blocks[1]"

    def test_a_block_naming_an_undeclared_latent_is_refused(self):
        # A16's third leg, which schema line 1193's wording does not describe.
        # Kills an implementation that only walks `latents` and asks "is it
        # owned" -- `zzz` is in no `latents` iteration, so that direction
        # never sees it and the run is refused at P3 by `plan.py:551-558`.
        found = _found(_doc([{"names": ["d", "a", "w", "zzz"]}]))
        assert [f.check for f in found] == ["A16"]
        assert "'zzz'" in found[0].message
        assert "['d', 'a', 'w']" in found[0].message

    def test_one_name_twice_in_ONE_block_is_refused_as_a_repeat(self):
        # Kills `if name in owner` alone: with one block, `owner[name]` is
        # this same position, so that implementation emits "is in blocks[0]
        # and in blocks[0]" -- a message that reads as a bug in the checker.
        found = _found(_doc([{"names": ["d", "d", "a", "w"]}]))
        assert [f.check for f in found] == ["A16"]
        assert "lists 'd' twice" in found[0].message

    def test_the_partition_is_settled_before_the_engine_is_derived(self):
        # THE ordering test.  This block would trip A18 as well ('zzz' reads
        # as non-linear beside the linear 'd'), and an implementation that
        # derives the engine first tells the user their block "mixes
        # declared-linear latents ['d'] with non-linear ones ['zzz']" -- about
        # a latent that does not exist.  `plan.py:539-541` argues the order in
        # as many words; this is that argument as an assertion.
        found = _found(_doc([{"names": ["d", "zzz"]}, {"names": ["a", "w"]}]))
        assert {f.check for f in found} == {"A16"}
        assert not any("mixes" in f.message for f in found)

    def test_a_bad_engine_is_reported_even_when_the_partition_is_wrong(self):
        # Kills `if partition: yield from partition; continue`, which is the
        # brief's own shape.  `engine:` is `Block._check`'s (plan.py:353-359)
        # and reads no latent at all, so the package refuses it on a broken
        # partition too -- measured, `Block('d','a',engine='banana')` raises
        # before `SamplingPlan` is constructed at all.  Suppressing it behind
        # the partition costs a user with both faults a second round trip,
        # which is the whole content of collect-rather-than-raise.
        found = _found(_doc([{"names": ["d", "a"], "engine": "banana"}]))
        assert [f.check for f in found] == ["A16", ""]
        assert "does not cover ['w']" in found[0].message
        assert "'banana'" in found[1].message
        # ...and the DERIVED clauses stay suppressed on the same document:
        # A17/A18/A19 all read what a name resolves to, and this partition
        # says the names are wrong.
        assert not any("mixes" in f.message or "check A17" in f.message
                       for f in found)

    def test_steps_on_a_conjugate_engine_block_is_refused(self):
        # The derived-conjugate case: no `engine:` at all, both members
        # linear.  Kills an implementation that only refuses `steps:` when
        # `engine: conjugate` is WRITTEN -- measured, the package refuses this
        # document at P3 with "Block('d', 'a') is solved by the conjugate
        # engine, which has no inner steps".
        found = _found(_doc([{"names": ["d", "a"], "steps": 5},
                             {"names": ["w"]}]))
        assert [f.check for f in found] == ["A17"]
        assert "steps: 5" in found[0].message
        assert found[0].where == "runs[0].blocks[0]"
        assert "engine: gradient" in found[0].message

    @pytest.mark.parametrize("steps", [0, True, "5", 1.5],
                             ids=["zero", "bool", "string", "float"])
    def test_a_steps_the_package_refuses_outright_still_fires_but_says_so(
            self, steps):
        """The guard is ``steps is not None`` -- ``plan.py:673``'s own -- and
        the message is not the same message.

        Two mutations, and the second is the one the brief shipped.  Written
        ``if entry.get("steps"):`` a zero passes every assertion in the test
        above and reaches P3, which is what the first half here kills.  But
        A17's *"would be silently ignored"* is a claim about what the package
        would do, and for these four values it is **false**: measured,
        ``Block('d','a', steps=0)`` never reaches ``SamplingPlan`` at all --
        ``plan.py:360-368`` refuses it as *"inner steps must be a positive
        int"*, and so it does for ``True``, ``'5'`` and ``1.5``.  A refusal
        that told this reader to *declare engine: gradient* would name a fix
        leaving the run refused for a second reason, which is the defect
        class Task 6 shipped eight of.
        """
        found = _found(_doc([{"names": ["d", "a"], "steps": steps},
                             {"names": ["w"]}]))
        assert [f.check for f in found] == ["A17"]
        assert f"steps: {steps!r}" in found[0].message
        assert "silently ignored" not in found[0].message
        assert "positive int" in found[0].message
        # ...and the fix clause does NOT send them to the engine that would
        # refuse this value too.
        assert "declare engine: gradient" not in found[0].message

    def test_steps_on_an_all_linear_block_declared_gradient_is_ALLOWED(self):
        # §2.6 item 3, as the assertion that makes it a decision rather than
        # a sentence.  Kills the schema's literal "all-linear block" reading:
        # that implementation refuses this document, and the package
        # ACCEPTS it -- measured, `SamplingPlan(('d','a'):gradient,
        # ('w'):gradient)`.  A refusal here would refuse a document that runs.
        assert _found(_doc([{"names": ["d", "a"], "engine": "gradient",
                             "steps": 5}, {"names": ["w"]}])) == []

    def test_a_mixed_block_with_no_engine_is_refused_naming_both_sides(self):
        # Kills a message that says "cannot be derived" without saying which
        # members are which: the fix is to split them or to declare an
        # engine, and neither is actionable without both lists.
        found = _found(_doc([{"names": ["d", "w"]}, {"names": ["a"]}]))
        assert [f.check for f in found] == ["A18"]
        assert "['d']" in found[0].message and "['w']" in found[0].message

    def test_engine_conjugate_over_a_non_linear_member_is_refused(self):
        found = _found(_doc([{"names": ["w"], "engine": "conjugate"},
                             {"names": ["d", "a"]}]))
        assert [f.check for f in found] == ["A19"]
        assert "['w']" in found[0].message

    def test_a_mixed_block_asking_for_conjugate_fires_A19_and_not_A17(self):
        # The package raises A19 at `plan.py:663` and never reaches A17 at
        # `:674` -- measured on exactly this block.  Kills an implementation
        # that collects both: two findings about one block is one error
        # described twice, and the second one names a fix (drop steps:) that
        # would leave the document still refused.
        found = _found(_doc([{"names": ["d", "w"], "engine": "conjugate",
                              "steps": 3}, {"names": ["a"]}]))
        assert [f.check for f in found] == ["A19"]

    def test_an_unknown_engine_is_refused_here_and_names_the_two(self):
        # `_BLOCK_KEYS` (exits.py:164) accepts any string, so today
        # `engine: banana` reaches the user as a ParameterSpaceError from
        # `plan.py:353-358` -- measured.  Kills deleting the enum clause:
        # without it `_engine_of` returns "banana", `engine == CONJUGATE` is
        # False, and this pass accepts a document the run refuses.
        found = _found(_doc([{"names": ["d", "a"], "engine": "banana"},
                             {"names": ["w"]}]))
        assert [f.check for f in found] == [""]
        assert "'banana'" in found[0].message
        assert "['conjugate', 'gradient']" in found[0].message

    def test_an_engine_that_is_not_even_a_string_is_refused_by_the_same_clause(
            self):
        # Kills `declared not in _ENGINES` narrowed to `isinstance(declared,
        # str) and declared not in _ENGINES`: `engine: 5` then falls through
        # to `_engine_of`, which returns "" for a non-string override, and the
        # reader is told their block "mixes declared-linear latents [...] with
        # non-linear ones []" -- an A18 refusal about a block that is not
        # mixed at all.  `Block._check` calls this one by name, measured:
        # "asks for engine=5; the engines are ['conjugate', 'gradient']".
        found = _found(_doc([{"names": ["d", "a"], "engine": 5},
                             {"names": ["w"]}]))
        assert [f.check for f in found] == [""]
        assert "5" in found[0].message
        assert "mixes" not in found[0].message

    def test_the_engine_enum_is_the_packages_own(self):
        """The enum is a COPY, and this test is the whole reason that is safe.

        The brief said to import ``ENGINES`` at the head of
        ``preflight/fitting.py`` and pin the BINDING LINE, on the measurement
        that ``numpyro`` stays out of ``sys.modules``.  It does, and it is the
        wrong invariant: ``rheplicant/inference/__init__.py`` re-exports the
        layer eagerly, so reaching ``...engines`` imports
        ``rheplicant.inference`` -- which ``test_config_exits_predict.py:1046``
        forbids the config layer to do, in a fresh interpreter, by name.
        Measured at ``9ee99af``: that probe prints ``[]``, and with the head
        import it prints ``['rheplicant.inference']``.

        So the names are written out, and the closed set is held closed HERE
        instead.  All three, not just the frozenset: ``_T7_CONJUGATE`` is what
        A17's and A19's clauses compare against, so a drift in ``CONJUGATE``
        alone would leave ``_ENGINES`` equal and both clauses dead.  A third
        engine, or a rename, turns this red -- which is exactly the failure
        the import was for.
        """
        from rheplicant.config.preflight.fitting import (
            _T7_CONJUGATE,
            _T7_GRADIENT,
        )
        from rheplicant.inference.engines import CONJUGATE, ENGINES, GRADIENT

        assert _T7_CONJUGATE == CONJUGATE
        assert _T7_GRADIENT == GRADIENT
        assert _ENGINES == frozenset(ENGINES)

    def test_this_module_does_not_import_the_inference_layer(self):
        """The guard that forced the copy, as this module's own assertion.

        ``test_config_exits_predict.py:1046`` imports ``rheplicant.config``
        and ``sections.exits``; this one imports the pre-flight package
        itself, which is the module that would break it, and says so where
        the copy is.  Kills "put the import back": it is a two-character edit
        that every other test here stays green under.
        """
        import os
        import subprocess
        import sys

        script = ("import sys, rheplicant.config.preflight;"
                  "print('rheplicant.inference' in sys.modules)")
        env = {"PATH": "/usr/bin:/bin"}
        if os.environ.get("PYTHONPATH"):
            env["PYTHONPATH"] = os.environ["PYTHONPATH"]
        done = subprocess.run([sys.executable, "-c", script], env=env,
                              capture_output=True, text=True, check=True)
        assert done.stdout.strip() == "False", done.stdout

    @pytest.mark.parametrize(
        "label, blocks",
        [("all-linear, no engine", [("d", "a"), ("w",)]),
         ("conjugate + steps", [("d", "a", "steps=5"), ("w",)]),
         ("gradient + steps, all linear",
          [("d", "a", "engine=gradient", "steps=5"), ("w",)]),
         ("mixed, no engine", [("d", "w"), ("a",)]),
         ("conjugate over non-linear", [("w", "engine=conjugate"), ("d", "a")]),
         ("mixed + conjugate + steps",
          [("d", "w", "engine=conjugate", "steps=3"), ("a",)]),
         ("undeclared name", [("d", "a", "zzz"), ("w",)]),
         ("doubled", [("d", "a"), ("d",), ("w",)]),
         ("uncovered", [("d", "a")]),
         ("unknown engine", [("d", "a", "engine=banana"), ("w",)]),
         ("one name twice in one block", [("d", "d", "a"), ("w",)]),
         ("steps zero", [("d", "a", "steps=0"), ("w",)]),
         ("gradient over the non-linear one alone",
          [("w", "engine=gradient", "steps=3"), ("d", "a")])])
    def test_the_pass_agrees_with_the_package_on_every_case(self, label,
                                                            blocks):
        """The differential that catches mirroring drift.

        `_engine_of` mirrors `plan.py:643-682` and `_a16_partition` mirrors
        `plan.py:544-586`.  Mirrored logic drifts, and every test above would
        stay green while the pass and the run disagreed about which engine a
        block takes -- a document refused at P-1 for a reason the run would
        not have given.  So this drives BOTH: the package's own `Block` and
        `SamplingPlan(space, *blocks)`, which settle the partition and derive
        the engines in their constructors, and our `_blocks`; and asserts they
        agree on refuse-or-not for all thirteen cases.

        **`Block(...)` is inside the `try`, not before it**, and that is three
        of the thirteen rows: the unknown engine, the repeated name and
        `steps: 0` are refused by `Block._check` (`plan.py:344-368`) and never
        reach `SamplingPlan` at all.  A differential that built the blocks
        outside the `try` would ERROR on those three rather than compare them,
        so the three clauses that decide them would have no differential at
        all -- which is exactly where a hand-written engine enum or a dropped
        repeat clause would hide.

        **What this differential deliberately does not cover**, stated rather
        than left as a gap: `learning_rate:` on a conjugate block, which
        `Block._check` refuses (`plan.py:369-381`, measured) and nothing here
        does.  It is A17's shape and it has no schema §6 row, so it is not one
        of the four ids this function claims; recorded for §6's ledger.

        Measured: `ParameterSpace.raw` builds in 0.019 s and each
        `SamplingPlan` settles in 0.0001 s, so the whole parametrization is
        milliseconds.  Constructing a ParameterSpace is a TEST doing it, not
        the check -- `preflight/fitting.py` constructs nothing (§2.4).
        """
        from rheplicant.core.errors import ParameterSpaceError
        from rheplicant.inference import (
            Block,
            Latent,
            ParameterSpace,
            SamplingPlan,
        )

        space = ParameterSpace.raw(
            [Latent("d", init=0.5, linear=True),
             Latent("a", init=10.0, linear=True),
             Latent("w", init=5.0)],
            lambda pipeline, values: pipeline)
        document = []
        for spec in blocks:
            names = [part for part in spec if "=" not in part]
            knobs = dict(part.split("=", 1) for part in spec if "=" in part)
            document.append({"names": names, **{
                key: (int(value) if key == "steps" else value)
                for key, value in knobs.items()}})
        try:
            package = [
                Block(*[part for part in spec if "=" not in part], **{
                    key: (int(value) if key == "steps" else value)
                    for key, value in (part.split("=", 1) for part in spec
                                       if "=" in part)})
                for spec in blocks]
            SamplingPlan(space, *package)
            package_refuses = False
        except ParameterSpaceError:
            package_refuses = True
        assert bool(_found(_doc(document))) is package_refuses, label

    def test_the_space_names_are_the_declared_parameter_keys(self):
        """The assumption A16 rests on, pinned against a BUILT document.

        A16 compares `runs[].blocks[].names` to the keys of
        `inference.parameters`.  That is right only while one declaration is
        one latent.  Measured today on three shipped documents including one
        declaring `fan:`; the day a key splits a declaration into two space
        latents, this goes red and A16's comparison is wrong rather than
        merely untested.

        This is the ONE test in Tasks 7-9 that builds a document.  It costs
        0.487 s cold and 0.004 s warm (measured), and it is a test doing it,
        not a check.
        """
        from rheplicant.config.document import load_document
        from tests.config.exit_helpers import TWO_LATENTS, conjugate_document

        document = conjugate_document({"kind": "forward"},
                                      inference=dict(TWO_LATENTS))
        built = load_document(document)
        assert (tuple(document["inference"]["parameters"])
                == tuple(built.inference.space.names))

    def test_only_plan_runs_are_read(self):
        # `blocks:` is a `plan.*` key.  Kills a loop over every run: a
        # `kind: fisher` run carrying a stray `blocks:` is Task 3's
        # unknown-key hole, not A16's, and refusing it here would give the
        # user two refusals for one typo.
        document = preflight_document(
            inference={"parameters": THREE},
            runs=[{"name": "f", "kind": "fisher", "blocks": [{"names": ["d"]}]}])
        assert _found(document) == []
        assert _kinds(document) == frozenset({"fisher"})

    def test_the_four_ids_are_registered_and_the_pass_runs_them_once(self):
        # 2C's shape 3: a correct decision shipped with no test, so reverting
        # it stays green.  Deleting any one `@register` line leaves every test
        # above passing, because they call `_blocks` directly.  And a
        # `preflight()` that iterates `CHECKS.values()` without de-duplicating
        # by identity would emit this document's one refusal four times.
        for check in ("A16", "A17", "A18", "A19"):
            assert CHECKS[check] is _blocks
        report = preflight(_doc([{"names": ["d", "a"]}]))
        assert [f.check for f in report.refusals()] == ["A16"]

    def test_an_uncovered_latent_wins_against_a_beam_that_cannot_be_read(self):
        # §5's PHASE PROPERTY, this task's one real assertion of it.  Task
        # 2's phase guard registers four synthetic lambdas: it proves the
        # HOOK's position and says nothing about any shipped check.  Nine
        # tasks each own one document that carries a real violation AND an
        # unreadable beam, and the assertion is symmetric -- the violation's
        # own words come back, and `no_such_beam` does NOT.
        #
        # `load_document`, never `run_document`: §2.1 measured that
        # `parse_runs` (runs.py:149) speaks BEFORE P-1 on the run_document
        # path, so a `runs`-shaped violation driven that way proves nothing.
        from rheplicant.config.document import load_document
        from rheplicant.config.errors import ConfigError

        document = preflight_document(
            inference={"parameters": THREE},
            runs=[_plan([{"names": ["d", "a"]}])],
            resources=UNREADABLE_BEAM)
        with pytest.raises(ConfigError) as caught:
            load_document(document)
        assert "check A16" in str(caught.value)
        assert "does not cover ['w']" in str(caught.value)
        assert "no_such_beam" not in str(caught.value)

    def test_the_base_document_earns_none_of_the_four(self):
        # The fixture's own contract: a check that finds nothing on the base
        # has actually looked.  Kills every inverted trigger at once -- an
        # implementation that refuses a COVERED partition passes each "is
        # refused" test above and fails here.
        assert not ({"A16", "A17", "A18", "A19"}
                    & preflight(preflight_document()).checks())

    def test_the_three_readers_survive_a_document_that_declares_neither(self):
        # `_latents`/`_runs`/`_kinds` are imported by Tasks 8-11 and are the
        # first thing every fitting check calls.  A KeyError here would abort
        # the whole pass (§2.3's TRAP), taking every later finding with it.
        assert _latents({}) == {}
        assert _runs({}) == ()
        assert _kinds({}) == frozenset()
        assert _runs({"runs": {"kind": "forward"}}) == (
            {"kind": "forward", "name": "forward"},)
        assert _runs({"runs": [7, {"kind": "fisher", "name": "f"}]}) == (
            {}, {"kind": "fisher", "name": "f"})

    def test_a_malformed_latent_KEEPS_ITS_NAME(self):
        # `_latents` normalises a non-Mapping spec to `{}` and KEEPS the key.
        # Kills `if isinstance(spec, Mapping)` written as a filter that DROPS
        # it -- under which `'a'` stops being a declared latent, A16's
        # coverage set loses a name, and a block that covers it correctly is
        # refused for naming something `inference.parameters` "does not
        # declare".  Nothing else here declares a malformed latent, so the
        # mutation survives the whole module.
        assert _latents({"inference": {"parameters": {"d": LINEAR_D,
                                                      "a": "oops"}}}) == {
            "d": LINEAR_D, "a": {}}
        assert _found(_doc([{"names": ["d"]}, {"names": ["a", "w"]}],
                           parameters={"d": LINEAR_D, "a": "oops",
                                       "w": NONLINEAR_W})) == []

    def test_runs_does_not_mutate_the_callers_document(self):
        # `_runs` FILLS `name` the way runs.py:115 does, and it must do it on
        # a COPY.  Kills `run["name"] = run.get("name") or run["kind"]`
        # written against the caller's own dict -- under which P-1 edits the
        # user's document in place, every later check reads a `runs:` the
        # user did not write, and `load_document` hands the mutated mapping
        # on to `build_*`.  Every other test here reads the return value and
        # never looks at the input again.
        document = {"runs": [{"kind": "forward"}, {"kind": "fisher"}]}
        assert [run["name"] for run in _runs(document)] == ["forward",
                                                            "fisher"]
        assert document["runs"] == [{"kind": "forward"}, {"kind": "fisher"}]

    def test_linear_is_the_BOOL_True_and_not_a_truthy_value(self):
        # `parameters.py:178-180` refuses a non-bool `linear:` at P2, and
        # `Latent.linear` defaults to False -- so to the package only the
        # BOOL True is linear.  Kills `_a18_linear` written
        # `bool(body.get("linear"))` -- under which `linear: 1` derives
        # CONJUGATE here and GRADIENT there, and this pass refuses a `steps:`
        # the package would have accepted.  Every other cell in this module
        # writes a literal `True`, so none of them can see it.
        assert _a18_linear({"d": {"linear": True}}, "d") is True
        for truthy in (1, "yes", [0], 1.0):
            assert _a18_linear({"d": {"linear": truthy}}, "d") is False, truthy
        assert _a18_linear({"d": {}}, "d") is False
        # ...and an undeclared name is non-linear rather than a KeyError,
        # which is what keeps A16's "uncovered" refusal reachable.
        assert _a18_linear({}, "d") is False


class TestTheMessagesNameWhoMustEdit:
    """Attribution, per index and per run, on every loop this module writes.

    Task 3 shipped `runs[i]` guarded and its two twins unguarded, and the
    carry-forward says so: a suite that only ever puts ONE offender in a
    document cannot tell `runs[0]` from `runs[i]`.
    """

    def test_the_second_run_is_blamed_and_not_the_first(self):
        # Kills a hard-coded `runs[0]` and a `list(...)[0]`: the first run
        # here is a clean forward exit, so an implementation that reported
        # index 0 would send the reader to a run with no blocks at all.
        found = _found(preflight_document(
            inference={"parameters": THREE},
            runs=[{"name": "fwd", "kind": "forward"},
                  _plan([{"names": ["d", "a"]}], name="second")]))
        assert [f.where for f in found] == ["runs[1].blocks"]
        assert "runs['second']" in found[0].message

    def test_an_unnamed_run_is_named_for_its_kind(self):
        # §3.1's whole reason for putting `name`-filling in `_runs`: an
        # unnamed run is `runs['plan.estimate']` to `parse_runs`
        # (runs.py:115) and to every refusal this layer writes.  Kills
        # `run.get("name", "")` and `run["name"] or "?"` -- both give a
        # message prefix no other refusal in this package uses.
        found = _found(preflight_document(
            inference={"parameters": THREE},
            runs=[{"kind": "plan.estimate", "blocks": [{"names": ["d", "a"]}]}]))
        assert "runs['plan.estimate']:" in found[0].message

    def test_a_single_run_mapping_is_still_run_zero(self):
        # `runs: {kind: ...}` is one run (runs.py:122-123).  Kills `_runs`
        # written `document.get("runs") or []` -- under which a mapping is
        # iterated as its KEYS and every entry reads as malformed, so this
        # document earns nothing at all.
        found = _found(preflight_document(
            inference={"parameters": THREE},
            runs={"kind": "plan.estimate", "blocks": [{"names": ["d", "a"]}]}))
        assert [f.where for f in found] == ["runs[0].blocks"]

    def test_both_offending_blocks_are_reported_and_not_only_the_first(self):
        # Emit per offender, not per site (Task 5's rule).  Two independent
        # engine faults in one run: an implementation that `break`s or
        # `return`s on the first leaves the second to be discovered on the
        # next round trip, which is what the collect-rather-than-raise design
        # exists to prevent.
        found = _found(_doc([{"names": ["d", "a"], "steps": 5},
                             {"names": ["w"], "engine": "conjugate"}]))
        assert [(f.check, f.where) for f in found] == [
            ("A17", "runs[0].blocks[0]"), ("A19", "runs[0].blocks[1]")]

    def test_every_message_ends_with_its_own_check_tag(self):
        # Enforced from Task 3 on.  Kills a clause that forgets the tail, or
        # carries another check's -- `Report.raise_if_refused` quotes the
        # message and the tag is what a reader greps for.  The enum clause
        # carries NO id (schema §6 gives it no row), so it is the one that
        # must NOT end in a tag, and asserting that is what stops a later
        # edit giving it A17's.
        documents = [
            _doc([{"names": ["d", "a"]}]),                       # A16
            _doc([{"names": ["d", "a", "w", "zzz"]}]),           # A16
            _doc([{"names": ["d", "d", "a", "w"]}]),             # A16
            _doc([{"names": ["d", "a", "w"]}, {"names": ["d"]}]),  # A16
            _doc([{"names": ["d", "a"], "steps": 5}, {"names": ["w"]}]),
            _doc([{"names": ["d", "a"], "steps": 0}, {"names": ["w"]}]),
            _doc([{"names": ["d", "w"]}, {"names": ["a"]}]),     # A18
            _doc([{"names": ["w"], "engine": "conjugate"},
                  {"names": ["d", "a"]}]),                       # A19
        ]
        seen = set()
        for document in documents:
            for finding in _found(document):
                seen.add(finding.check)
                assert finding.message.endswith(
                    f"(check {finding.check})."), finding.message
        assert seen == {"A16", "A17", "A18", "A19"}
        [enum] = _found(_doc([{"names": ["d", "a"], "engine": "banana"},
                              {"names": ["w"]}]))
        assert enum.check == ""
        assert not enum.message.endswith(").")


class TestTheWarmStartIsTheSamePartition:
    """`warm_start.blocks` reaches the same `SamplingPlan`, one call along.

    `exits.py:268` -- `SamplingPlan(space, *_blocks(f"{where}: warm_start",
    warm.get("blocks")))`.  Same constructor, same space, same four refusals,
    same P3.  Every test here kills a `_blocks` that reads `run["blocks"]`
    alone, which is the shape every task in this plan so far has shipped and
    a reviewer has found.
    """

    def test_a_warm_starts_uncovered_latent_is_refused_too(self):
        found = _found(_warm([{"names": ["d", "a"], "engine": "gradient"},
                              {"names": ["w"]}],
                             [{"names": ["d", "a"]}]))
        assert [f.check for f in found] == ["A16"]
        assert found[0].where == "runs[0].warm_start.blocks"
        assert "['w']" in found[0].message

    def test_the_warm_start_refusal_does_not_claim_the_whole_run_is_frozen(
            self):
        # The message TEXT, which is where Task 6's mutants lived.  A16's
        # main-route sentence is "silently frozen at its declared init for
        # the whole run"; on a warm start that is false -- the main blocks
        # still update it, and what the warm estimate freezes is its own
        # phase.  Kills reusing one message for both sites, which is what
        # sharing the formatter without sharing the reasoning gives you.
        [found] = _found(_warm([{"names": ["d", "a"], "engine": "gradient"},
                                {"names": ["w"]}],
                               [{"names": ["d", "a"]}]))
        assert "for the whole run" not in found.message
        assert "warm_start" in found.message

    def test_a_warm_starts_engine_rules_are_read_too(self):
        found = _found(_warm([{"names": ["d", "a"], "engine": "gradient"},
                              {"names": ["w"]}],
                             [{"names": ["d", "a"], "steps": 5},
                              {"names": ["w"]}]))
        assert [f.check for f in found] == ["A17"]
        assert found[0].where == "runs[0].warm_start.blocks[0]"
        assert "warm_start.blocks[0]" in found[0].message

    def test_a_warm_start_on_plan_estimate_is_left_to_A1(self):
        # `_ESTIMATE_KEYS` (exits.py:165) has no `warm_start`, and Task 3's
        # A1 already refuses it at P-1 -- measured: "kind: plan.estimate does
        # not take ['warm_start']".  Kills reading `warm_start` on every
        # `plan.*` run: the reader would get a partition refusal about a
        # block list that is not read on this kind at all, in front of the
        # refusal that names the real fault.
        document = preflight_document(
            inference={"parameters": THREE},
            runs=[_plan([{"names": ["d", "a"], "engine": "gradient"},
                         {"names": ["w"]}],
                        warm_start={"kind": "plan.estimate", "move": ["d"],
                                    "blocks": [{"names": ["d"]}]})])
        assert _found(document) == []
        assert "A1" in preflight(document).checks()

    def test_a_clean_warm_start_earns_nothing(self):
        assert _found(_warm([{"names": ["d", "a"], "engine": "gradient"},
                             {"names": ["w"]}],
                            [{"names": ["d", "a"], "engine": "gradient"},
                             {"names": ["w"]}])) == []


class TestWhatThisCheckStandsDownOn:
    """Three shapes where a refusal here would be the wrong sentence.

    Task 5 paid for this rule and Task 6 paid for it again: if the package
    has a better sentence for a document, or the document's own text says a
    refusal is expected, stand down.
    """

    def test_a_run_that_EXPECTS_a_refusal_is_left_alone(self):
        # `expect: refuse` is an assertion ABOUT the refusal: `execute_run`
        # (exits.py:292-302) runs the executor and captures its error as the
        # run's product.  A P-1 refusal makes the document unloadable, so the
        # assertion can never be made at all -- measured,
        # `test_config_exits_plan.py:107` is exactly such a document
        # (`blocks: [{names: [g, ghost]}]`, `expect: refuse`) and it asserts
        # the captured error names `ghost`.  Kills a `_blocks` that reads
        # every run: that implementation turns that test red and takes
        # `expect: refuse` away from every plan document in the repo.
        assert _found(_doc([{"names": ["d", "a"]}], expect="refuse")) == []
        # ...and `expect: ok` is not a licence, which is the other half.
        assert len(_found(_doc([{"names": ["d", "a"]}], expect="ok"))) == 1

    @pytest.mark.parametrize("blocks", [
        "nope", {"names": ["d"]}, [], [5], [{"names": "d"}], [{"names": []}],
        [{"names": ["d", 7]}], [{"names": None}], [{}],
        [{"names": ["d", "a", "w"]}, "nope"],
    ], ids=["a-string", "a-mapping", "empty", "not-a-mapping", "names-a-string",
            "names-empty", "names-not-all-strings", "names-none", "no-names",
            "a-second-entry-that-is-not-a-mapping"])
    def test_a_blocks_list_the_grammar_refuses_by_shape_is_left_alone(
            self, blocks):
        """`exits._blocks` (`exits.py:180-202`) refuses each of these in its
        own words -- *"blocks: is a non-empty list of block mappings"*,
        *"blocks[0] is a mapping"*, *"blocks[0].names is a non-empty list of
        latent names"*.

        Kills the brief's literal guard, `isinstance(entries, list)`, on
        **six** of these ten rows: an empty list, a non-mapping entry and
        every malformed `names:` all pass it, after which `owner` stays empty
        and the reader is told their blocks *"do not cover ['d', 'a', 'w']"*
        -- a coverage sentence whose fix ("add it to a block") is not the
        fault, in front of the grammar's own.  Two rows kill something worse:
        `names: "d"` iterates a STRING into the characters `['d']`, and
        `names: 5` (not shown -- see the raising battery) iterates an int and
        aborts the whole pass.
        """
        document = preflight_document(inference={"parameters": THREE},
                                      runs=[_plan(blocks)])
        assert _found(document) == []

    def test_a_names_that_cannot_be_iterated_does_not_abort_the_pass(self):
        # §2.3's TRAP, in the shape Task 4 measured one level in: a value the
        # code iterates or hashes.  `(entry.get("names") or ())` is the
        # brief's own spelling and it raises `TypeError: 'int' object is not
        # iterable` on this document -- inside the pass that becomes "check
        # A16 RAISED" and DISCARDS every other finding in the report.
        document = preflight_document(inference={"parameters": THREE},
                                      runs=[_plan([{"names": 5}])])
        assert _found(document) == []
        assert preflight(document).checks() == frozenset()


class TestNoHostileDocumentCanAbortThePass:
    """228 hostile documents is Task 6's standard; this is the shape of them.

    A check that RAISES aborts the pass and hides every finding after it
    (§2.3's TRAP), and `preflight` turns that into "check 'A16' RAISED",
    which is a report with one entry where the user had four problems.
    """

    #: Each is merged OVER ``inference={"parameters": THREE}`` by
    #: :func:`_hostile_document` rather than passed beside it --
    #: ``preflight_document(inference=..., **patch)`` is a duplicate keyword
    #: when the patch names the same section, and the TypeError would read as
    #: a defect in the check.
    HOSTILE = [
        {},
        {"inference": None},
        {"inference": "nope"},
        {"inference": {"parameters": "nope"}},
        {"inference": {"parameters": ["d"]}},
        {"inference": {"parameters": {7: {"linear": True}}}},
        {"inference": {"parameters": {"d": None}}},
        {"runs": "nope"},
        {"runs": 7},
        {"runs": []},
        {"runs": [None]},
        {"runs": [{"kind": None}]},
        {"runs": [{"kind": 7, "blocks": [{"names": ["d"]}]}]},
        {"runs": [{"kind": "plan.estimate"}]},
        {"runs": [{"kind": "plan.estimate", "blocks": None}]},
        {"runs": [{"kind": "plan.estimate", "blocks": {"names": ["d"]}}]},
        {"runs": [{"kind": "plan.estimate", "blocks": [{"names": 5}]}]},
        {"runs": [{"kind": "plan.estimate", "blocks": [{"names": {"d": 1}}]}]},
        {"runs": [{"kind": "plan.estimate",
                   "blocks": [{"names": ["d"], "engine": ["conjugate"]}]}]},
        {"runs": [{"kind": "plan.estimate",
                   "blocks": [{"names": ["d"], "steps": {"a": 1}}]}]},
        {"runs": [{"kind": "plan.estimate", "name": 7,
                   "blocks": [{"names": ["d", "a", "w"]}]}]},
        {"runs": [{"kind": "plan.sample", "warm_start": "nope",
                   "blocks": [{"names": ["d", "a", "w"]}]}]},
        {"runs": [{"kind": "plan.sample", "warm_start": {"blocks": 7},
                   "blocks": [{"names": ["d", "a", "w"]}]}]},
        {"runs": [{"kind": "plan.sample",
                   "warm_start": {"blocks": [{"names": ["zzz"]}]},
                   "blocks": [{"names": ["d", "a", "w"]}]}]},
        {"runs": [{"kind": "plan.estimate", "expect": ["refuse"],
                   "blocks": [{"names": ["d", "a"]}]}]},
    ]

    @pytest.mark.parametrize("patch", HOSTILE,
                             ids=[str(index) for index in
                                  range(len(HOSTILE))])
    def test_the_check_returns_findings_and_raises_nothing(self, patch):
        document = _hostile_document(patch)
        for finding in _found(document):
            assert finding.check in ("A16", "A17", "A18", "A19", "")

    @pytest.mark.parametrize("patch", HOSTILE,
                             ids=[str(index) for index in
                                  range(len(HOSTILE))])
    def test_the_whole_pass_survives_each_of_them(self, patch):
        # The pass, not the function: `_check_where` runs OUTSIDE the
        # per-check `try`, so a `where` this module computes from a hostile
        # document could kill the pass even when the check itself returns
        # cleanly.  Kills a `where` built from a user-supplied string.
        preflight(_hostile_document(patch))
