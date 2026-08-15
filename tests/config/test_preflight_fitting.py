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


# --- Task 8: the prior gates and the seed asymmetry -------------------------
#
# `LINEAR_D`/`LINEAR_A`/`NONLINEAR_W`/`THREE` above are Task 7's and are reused
# rather than re-declared: a second three-latent constant in one module is the
# shape §3.1 pins names against, one file in.

#: `inference.joint_prior:`'s only grammar -- `transforms._joint_prior`
#: (`:298-321`) refuses anything else, and `{kind: jeffreys, names: [...]}` by
#: name, so coverage is a list of strings in the document and nothing has to
#: be built to read it.
JOINT = {"jeffreys": {"over": ["d", "a"]}}
PRIOR = {"normal": {"loc": 1.0, "scale": 1.0}}
COVERED = {"d": LINEAR_D, "a": LINEAR_A}
PRIORED = {"d": {**LINEAR_D, "prior": PRIOR}, "a": {**LINEAR_A, "prior": PRIOR}}


def _gates(document):
    from rheplicant.config.preflight.fitting import _prior_gates

    return list(_prior_gates(document))


class TestPriorGates:
    def test_a_joint_prior_beside_a_plan_run_is_refused(self):
        # Measured at P3 today: ParameterSpaceError, "This space declares
        # JeffreysPrior(over=['d', 'a']) ..., and no block would step it at
        # all", from `plan.py:631` -- after every beam.  Kills an
        # implementation gated on the partition SPLITTING the joint prior:
        # `_refuse_split_joint_prior` is unconditional (`plan.py:624-629`
        # chooses only the wording) and the un-split case is the silent one.
        found = _gates(preflight_document(
            inference={"parameters": COVERED, "joint_prior": JOINT},
            runs=[{"name": "fit", "kind": "plan.estimate",
                   "blocks": [{"names": ["d", "a"]}]}]))
        assert [f.check for f in found] == ["A20"]
        assert "['d', 'a']" in found[0].message

    def test_both_plan_kinds_trip_A20(self):
        # 2C's shape 4, a hole closed on one route and left open on its twin.
        # Measured: the same ParameterSpaceError on `plan.estimate` and on a
        # seeded `plan.sample`.
        for kind in ("plan.estimate", "plan.sample"):
            found = _gates(preflight_document(
                inference={"parameters": COVERED, "joint_prior": JOINT},
                runs=[{"name": "fit", "kind": kind,
                       "blocks": [{"names": ["d", "a"]}]}]))
            assert [f.check for f in found] == ["A20"], kind

    def test_a_joint_prior_beside_fisher_space_true_is_refused(self):
        found = _gates(preflight_document(
            inference={"parameters": COVERED, "joint_prior": JOINT},
            runs=[{"name": "f", "kind": "fisher", "space": True}]))
        assert [f.check for f in found] == ["A21"]

    def test_fisher_WITHOUT_space_true_is_not_refused(self):
        # `_run_fisher` (`exits.py:58-59`) passes `space=space if use_space
        # else None`, and `fisher_information` refuses a joint prior on the
        # `space=` leg alone (`uncertainty.py:313`).  Kills a gate on
        # `kind == "fisher"` by itself, which would refuse a document that
        # runs.
        for run in ({"name": "f", "kind": "fisher"},
                    {"name": "f", "kind": "fisher", "space": False}):
            assert _gates(preflight_document(
                inference={"parameters": COVERED, "joint_prior": JOINT},
                runs=[run])) == []

    def test_the_joint_prior_reader_takes_only_a_LIST_OF_STRINGS(self):
        """Both halves of ``_a20_joint_over``'s filter, which no document
        test above can see.

        Kills ``return tuple(over)``: ``over: [d, 7]`` then makes A20 announce
        that ``inference.joint_prior covers ['d', 7]`` -- a message naming
        something that is not a latent name, in front of
        ``JeffreysPrior.validate_against``'s, which says which name is wrong.
        And kills ``if over is None`` in place of the list test: ``over: 7``
        raises ``TypeError`` inside ``tuple()``, which the pass reports as
        "check 'A20' RAISED" and which costs the report every other finding.

        **The recorded hole**: ``over: da`` -- a bare YAML scalar -- IS read
        by ``transforms._joint_prior`` as ``tuple("da") == ('d', 'a')``
        (measured), so a document writing it and a ``kind: plan.*`` run does
        reach the package's refusal and this check stands down on it.  That
        is a decision, not an oversight: splitting a string into characters
        is an accident of ``tuple()`` rather than a grammar, and A20 saying
        "covers ['d', 'a']" to somebody who wrote ``over: da`` would name a
        coverage they did not write.  Recorded here so the next plan inherits
        a decision.
        """
        from rheplicant.config.preflight.fitting import _a20_joint_over

        def over(value):
            return _a20_joint_over(
                {"inference": {"joint_prior": {"jeffreys": {"over": value}}}})

        assert over(["d", 7, None]) == ("d",)
        assert over(["d", "a"]) == ("d", "a")
        for bad in (7, 3.5, "da", {"d": 1}, None, True):
            assert over(bad) == (), bad
        # ...and the message that reads it names only the names.
        [one] = _gates(preflight_document(
            inference={"parameters": COVERED,
                       "joint_prior": {"jeffreys": {"over": ["d", 7]}}},
            runs=[{"name": "fit", "kind": "plan.estimate"}]))
        assert "covers ['d']" in one.message

    def test_a_space_that_is_not_a_BOOL_is_left_to_the_executors_own_check(
            self):
        # `_run_fisher` (`exits.py:55-57`) refuses a non-bool `space:` by
        # name -- "space: is a bool; got 'true'" -- and that is the fault the
        # reader has.  Kills `run.get("space") is True` weakened to
        # `run.get("space")`: under it a YAML-quoted `space: "true"` earns
        # A21 (or A23), which names a joint prior or a missing prior when the
        # fault is a string where a bool goes.  Nothing else in this module
        # writes a truthy non-bool there, so the mutation survives it all.
        for value in ("true", 1, ["true"]):
            assert _gates(preflight_document(
                inference={"parameters": COVERED, "joint_prior": JOINT},
                runs=[{"name": "f", "kind": "fisher",
                       "space": value}])) == [], value
            assert _gates(preflight_document(
                inference={"parameters": {"w": NONLINEAR_W}},
                runs=[{"name": "f", "kind": "fisher",
                       "space": value}])) == [], value

    def test_a_joint_prior_with_no_fitting_run_is_not_refused(self):
        # A20 and A21 are about a PAIR.  Kills a check that reads
        # `inference:` alone -- which is what every function in this layer did
        # before this plan, and is the plan's finding 2.
        assert _gates(preflight_document(
            inference={"parameters": COVERED, "joint_prior": JOINT},
            runs=[{"kind": "forward"}])) == []

    def test_A20_fires_and_A23_does_not_on_one_document(self):
        # §2.6 item 4, as an assertion.  This document trips A20 AND would
        # trip A23's plan.sample-gradient leg (neither latent declares a
        # prior: of its own).  An implementation that ran A23 first tells the
        # user their latents "declare no prior", which contradicts A20's
        # refusal of the same document -- and A20's fix (drop joint_prior)
        # and A23's (declare a prior) are different edits.
        found = _gates(preflight_document(
            inference={"parameters": COVERED, "joint_prior": JOINT},
            runs=[{"name": "fit", "kind": "plan.sample", "n_sweeps": 8,
                   "seed": {"from": "runtime.seeds.s"},
                   "blocks": [{"names": ["d", "a"], "engine": "gradient"}]}]))
        assert [f.check for f in found] == ["A20"]

    def test_A21_fires_and_A23_does_not_on_one_document(self):
        # A21's half of the same pin, and it is a DIFFERENT line: the
        # `continue` after A20 and the one after A21 are two statements, and
        # deleting the second alone leaves the test above green while a
        # `fisher` document is told BOTH that its joint prior would sit
        # inside its own definition (drop space:) and that its latents
        # declare no prior (declare one) -- two refusals naming two different
        # edits for one document, the second contradicting the first.
        found = _gates(preflight_document(
            inference={"parameters": COVERED, "joint_prior": JOINT},
            runs=[{"name": "f", "kind": "fisher", "space": True}]))
        assert [f.check for f in found] == ["A21"]

    def test_the_registry_puts_A20_and_A21_before_A23(self):
        # The other half of the ordering pin, and the one that fails if a
        # later editor splits `_prior_gates` into three functions and
        # registers them in the wrong order.  `CHECKS` insertion order IS run
        # order (§3.1), so this is the property, not a proxy for it.
        ids = list(CHECKS)
        assert ids.index("A20") < ids.index("A23")
        assert ids.index("A21") < ids.index("A23")

    def test_nuts_counts_joint_prior_coverage_and_npe_does_not(self):
        # The asymmetry `_sampled_space` encodes, re-decided from text.
        # Kills one rule for both routes: measured, `to_numpyro_model` accepts
        # a covered latent (numpyro_bridge.py:66-79) and `simulate_pairs`
        # does not (npe.py:111-118), and the shipped suite has a document
        # (`joint_prior_document`) that runs one and refuses the other.
        document = preflight_document(
            inference={"parameters": COVERED, "joint_prior": JOINT},
            runs=[{"name": "chain", "kind": "nuts",
                   "seed": {"from": "runtime.seeds.chain"}},
                  {"name": "amortized", "kind": "npe"}])
        found = _gates(document)
        assert [f.check for f in found] == ["A23"]
        # ...and the SECOND run is blamed, in `where` and in the message.
        # Kills a hard-coded `runs[0]` and a `named` built from the kind:
        # the first run here is clean, so either would send the reader to a
        # run that is fine.
        assert found[0].where == "runs[1]"
        assert found[0].message.startswith("runs['amortized']: ")
        assert "['d', 'a']" in found[0].message
        # A23's FOUR routes share one sentence and only `because` tells them
        # apart.  Kills `{because}` deleted from the message, or a route
        # assigned the wrong clause -- under which every A23 test above still
        # passes and the reader is told the wrong reason.
        assert "consulting inference.joint_prior not at all" in found[0].message

    def test_a_gradient_block_of_plan_sample_needs_a_prior_on_every_member(
            self):
        # Measured at P3: ParameterSpaceError from `engines.py:462-470`,
        # "Block ... is stepped by the gradient engine, and sampling it needs
        # a prior on every member".  Kills a whole-space reading: only the
        # GRADIENT block's members need one.
        found = _gates(preflight_document(
            inference={"parameters": {"d": {**LINEAR_D, "prior": PRIOR},
                                      "w": NONLINEAR_W}},
            runs=[{"name": "fit", "kind": "plan.sample", "n_sweeps": 8,
                   "seed": {"from": "runtime.seeds.s"},
                   "blocks": [{"names": ["d"]}, {"names": ["w"]}]}]))
        assert [f.check for f in found] == ["A23"]
        assert "['w']" in found[0].message and "'d'" not in found[0].message
        assert "stepped by the gradient engine" in found[0].message

    def test_a_conjugate_block_of_plan_sample_needs_no_prior(self):
        # `require_priors` is called only for GRADIENT blocks
        # (`plan.py:1064-1066`).  Kills "every latent of a plan.sample needs a
        # prior", which would refuse a document the package runs.
        assert _gates(preflight_document(
            inference={"parameters": {"d": LINEAR_D, "a": LINEAR_A}},
            runs=[{"name": "fit", "kind": "plan.sample", "n_sweeps": 8,
                   "seed": {"from": "runtime.seeds.s"},
                   "blocks": [{"names": ["d", "a"]}]}])) == []

    def test_plan_estimate_needs_no_prior_at_all(self):
        # `require_priors` lives in `sample()`, not `estimate()` -- the
        # asymmetry the package's own message names ("fine for
        # plan.estimate(), meaningless in a draw").
        assert _gates(preflight_document(
            inference={"parameters": {"w": NONLINEAR_W}},
            runs=[{"name": "fit", "kind": "plan.estimate",
                   "blocks": [{"names": ["w"], "engine": "gradient"}]}])) == []

    def test_the_warm_start_of_a_plan_sample_needs_no_prior_either(self):
        # THE TWIN, measured and standing DOWN rather than guarded.  Task 7's
        # A16-A19 read `warm_start.blocks` because `exits.py:269` hands them
        # to the same `SamplingPlan`; A23 must not, because that plan is
        # `.estimate()`d and `require_priors` is called from `.sample()`
        # alone (`plan.py:1064-1066`) -- and `_WARM_KEYS` (`exits.py:171`)
        # plus `exits.py:250` allow no `kind:` there but `plan.estimate`.
        # Kills copying Task 7's `_t7_sites` walk into this check: that
        # refuses a document the package runs.
        assert _gates(preflight_document(
            inference={"parameters": {"d": {**LINEAR_D, "prior": PRIOR},
                                      "w": NONLINEAR_W}},
            runs=[{"name": "fit", "kind": "plan.sample", "n_sweeps": 8,
                   "seed": {"from": "runtime.seeds.s"},
                   "blocks": [{"names": ["d"]},
                              {"names": ["w"], "engine": "conjugate"}],
                   "warm_start": {"kind": "plan.estimate", "move": ["d"],
                                  "blocks": [{"names": ["d", "w"],
                                              "engine": "gradient"}]}}])) == []

    def test_fisher_space_true_needs_a_prior_on_every_latent(self):
        # Measured at P3: `rheplicant/inference/uncertainty.py:346-357`
        # (`inference/`, not `config/sections/`), "latent 'w' declares no
        # prior, so what it returns would be a posterior precision for every
        # latent but that one".
        found = _gates(preflight_document(
            inference={"parameters": {"d": {**LINEAR_D, "prior": PRIOR},
                                      "w": NONLINEAR_W}},
            runs=[{"name": "f", "kind": "fisher", "space": True}]))
        assert [f.check for f in found] == ["A23"]
        assert "['w']" in found[0].message
        assert "space: true asks for a posterior precision" in found[0].message

    def test_width_fisher_on_a_conjugate_run_is_RECORDED_and_not_guarded(self):
        """The one twin this task found and did not close, pinned so the
        decision is inherited rather than rediscovered.

        ``fisher_information(space=...)`` has TWO callers under ``config/``:
        ``exits.py:60`` (``kind: fisher`` with ``space: true``, which A21 and
        A23's fisher leg guard) and ``conjugate.py:194``, where
        ``width: fisher`` on a ``kind: conjugate.wiener`` run passes
        ``space=`` UNCONDITIONALLY.  Measured through ``run_document`` this
        session, on a two-latent block that IS the whole space:

        * with no ``prior_std:`` on the run, both documents die earlier and
          better -- ``wiener_solve needs a prior_std for ['d', 'a']`` from the
          conjugate solve, which is the fault the reader actually has;
        * with ``prior_std: {d: 1.0, a: 5.0}`` declared, the joint-prior
          document reaches ``uncertainty.py:313`` verbatim (*"fisher_
          information was given space= a ParameterSpace declaring
          JeffreysPrior(over=['d', 'a'])"*) and the prior-free one reaches
          ``:348`` (*"latent 'a' declares no prior"*) -- both after every
          beam.

        **Not closed, deliberately, and the reason is P-1's own boundary.**
        Whether the earlier and better ``prior_std`` refusal fires is a
        question about a run option whose coverage of the block this pass
        would have to re-derive, and getting it wrong puts a joint-prior
        sentence in front of the one naming the missing ``prior_std:``  --
        the pre-emption Task 5 and Task 6 each paid for.  Schema §6's A21 row
        also names ``kind: fisher`` with ``space: true`` and nothing else.
        Recorded for §6's ledger; this test is what makes a later closure a
        deliberate act.
        """
        assert _gates(preflight_document(
            inference={"parameters": COVERED, "joint_prior": JOINT},
            runs=[{"name": "w", "kind": "conjugate.wiener",
                   "names": ["d", "a"], "width": "fisher",
                   "prior_std": {"d": 1.0, "a": 5.0}}])) == []
        assert _gates(preflight_document(
            inference={"parameters": {"w": NONLINEAR_W}},
            runs=[{"name": "w", "kind": "conjugate.wiener",
                   "names": ["w"], "width": "fisher"}])) == []

    def test_a_declared_but_MALFORMED_prior_is_still_a_declared_prior(self):
        # `_parse_prior` (`parameters.py:69-75`) returns None for an ABSENT
        # `prior:` and for nothing else; anything present is refused in its
        # own words ("is a mapping naming one family").  Kills
        # `.get("prior") is None` weakened to `not .get("prior")`: an empty
        # `prior: {}` then reads as missing, and A23 tells the reader to
        # declare a prior they did declare, in front of the grammar's
        # sentence about the one they wrote.  Nothing else here writes a
        # falsy prior, so the mutation survives the rest of the module.
        for prior in ({}, "", 0, []):
            assert _gates(preflight_document(
                inference={"parameters": {"w": {**NONLINEAR_W,
                                                "prior": prior}}},
                runs=[{"name": "chain", "kind": "nuts"}])) == [], prior

    def test_a_fully_priored_document_trips_nothing(self):
        assert _gates(preflight_document(
            inference={"parameters": PRIORED},
            runs=[{"name": "chain", "kind": "nuts",
                   "seed": {"from": "runtime.seeds.chain"}},
                  {"name": "amortized", "kind": "npe"},
                  {"name": "f", "kind": "fisher", "space": True}])) == []

    def test_a_run_that_EXPECTS_a_refusal_is_left_alone(self):
        """``expect: refuse`` is an assertion ABOUT the refusal, and A23 as
        the brief wrote it takes that assertion away from the one document in
        the repo that makes it.

        ``posterior_helpers.joint_prior_document()`` carries two runs over ONE
        joint-prior space -- ``kind: npe`` under ``expect: refuse``, and
        ``kind: nuts`` running -- and it is what
        ``test_config_exits_npe.py::TestThePriorGate`` reads through
        ``run_document``.  A P-1 refusal makes the whole document unloadable,
        so ``results['amortized'].error`` could never exist: measured, without
        this guard that class goes red naming a check it was never about.
        Task 7's ``_blocks`` stands down for the same reason
        (``exits.py:293-303`` captures the executor's error as the product).
        """
        from tests.config.posterior_helpers import joint_prior_document

        assert _gates(preflight_document(
            inference={"parameters": COVERED, "joint_prior": JOINT},
            runs=[{"name": "amortized", "kind": "npe",
                   "expect": "refuse"}])) == []
        # ...and on the shipped document itself, whose nuts run is still read:
        # the guard is per RUN, not per document.  Kills "if any run expects a
        # refusal, return".
        assert _gates(joint_prior_document()) == []
        # `expect: ok` is not a licence, which is the other half.
        assert len(_gates(preflight_document(
            inference={"parameters": COVERED, "joint_prior": JOINT},
            runs=[{"name": "amortized", "kind": "npe",
                   "expect": "ok"}]))) == 1

    def test_a_block_naming_an_undeclared_latent_is_left_to_A16(self):
        # `_a23_prior_free` reads `latents.get(name, {}).get("prior")`, so an
        # UNDECLARED name reads as prior-free and A23 would report it beside
        # A16's "names 'zzz', which inference.parameters does not declare".
        # One typo, two refusals, two different fixes -- the shape `gated`
        # exists to stop one function along.  Kills passing a block's raw
        # `names:` to `_a23_prior_free`.
        document = preflight_document(
            inference={"parameters": {"d": {**LINEAR_D, "prior": PRIOR}}},
            runs=[{"name": "fit", "kind": "plan.sample", "n_sweeps": 8,
                   "seed": {"from": "runtime.seeds.s"},
                   "blocks": [{"names": ["d", "zzz"], "engine": "gradient"}]}])
        assert _gates(document) == []
        assert [f.check for f in _found(document)] == ["A16"]

    def test_the_three_ids_are_registered_and_the_pass_runs_them_once(self):
        # 2C's shape 3: a correct decision shipped with no test, so reverting
        # it stays green.  Deleting any one id from `@register` leaves every
        # test above passing, because they call `_prior_gates` directly.  And
        # a `preflight()` that failed to de-duplicate by identity would emit
        # this document's one refusal three times.
        from rheplicant.config.preflight.fitting import _prior_gates

        for check in ("A20", "A21", "A23"):
            assert CHECKS[check] is _prior_gates
        report = preflight(preflight_document(
            inference={"parameters": COVERED, "joint_prior": JOINT},
            runs=[{"name": "fit", "kind": "plan.estimate",
                   "blocks": [{"names": ["d", "a"]}]}]))
        assert [f.check for f in report.refusals()
                if f.check == "A20"] == ["A20"]

    def test_the_base_document_earns_none_of_the_three(self):
        # The fixture's own contract: a check that finds nothing on the base
        # has actually looked.  Kills every inverted trigger at once.
        assert not ({"A20", "A21", "A23"}
                    & preflight(preflight_document()).checks())

    def test_every_prior_gate_message_ends_with_its_own_check_tag(self):
        # Enforced from Task 3 on.  Kills a clause that forgets the tail or
        # carries another check's -- `raise_if_refused` quotes the message and
        # the tag is what a reader greps for.
        documents = [
            preflight_document(
                inference={"parameters": COVERED, "joint_prior": JOINT},
                runs=[{"name": "fit", "kind": "plan.estimate"}]),
            preflight_document(
                inference={"parameters": COVERED, "joint_prior": JOINT},
                runs=[{"name": "f", "kind": "fisher", "space": True}]),
            preflight_document(
                inference={"parameters": COVERED},
                runs=[{"name": "chain", "kind": "nuts"}]),
            preflight_document(
                inference={"parameters": COVERED},
                runs=[{"name": "amortized", "kind": "npe"}]),
            preflight_document(
                inference={"parameters": COVERED},
                runs=[{"name": "f", "kind": "fisher", "space": True}]),
            preflight_document(
                inference={"parameters": {"w": NONLINEAR_W}},
                runs=[{"name": "fit", "kind": "plan.sample", "n_sweeps": 8,
                       "seed": {"from": "runtime.seeds.s"},
                       "blocks": [{"names": ["w"]}]}]),
        ]
        seen = set()
        for document in documents:
            for finding in _gates(document):
                seen.add(finding.check)
                assert finding.severity == REFUSE
                assert finding.message.endswith(
                    f"(check {finding.check})."), finding.message
        assert seen == {"A20", "A21", "A23"}


class TestThePriorGateMessagesSayWhichEditToMake:
    """Clause by clause, because for a validation layer the message IS the
    product.

    Task 6's carry-forward: eight of nine surviving mutants lived in refusal
    TEXT rather than in predicate logic, and a ``match=`` on one fragment
    leaves every other clause free to be wrong.  Each assertion below reads a
    separate clause, so each can be mutated on its own.
    """

    def test_A20_names_the_covered_latents_the_kind_and_BOTH_ways_out(self):
        [one] = _gates(preflight_document(
            inference={"parameters": COVERED, "joint_prior": JOINT},
            runs=[{"name": "fit", "kind": "plan.sample", "n_sweeps": 8,
                   "seed": {"from": "runtime.seeds.s"}}]))
        assert one.message.startswith("runs['fit']: ")
        assert "inference.joint_prior covers ['d', 'a']" in one.message
        # the KIND, not the family: a user with a plan.sample must not be
        # told about plan.estimate.
        assert ("kind: plan.sample does not evaluate a joint prior"
                in one.message)
        # WHY it is silent rather than merely wrong -- the package's own
        # measured sentence, which is the reason this is a refusal at all.
        assert "contributes exactly zero" in one.message
        assert "converged chi-squared" in one.message
        # both ways out, by name
        assert "kind: nuts is the exit that evaluates it" in one.message
        assert "drop inference.joint_prior" in one.message

    def test_A21_says_the_prior_would_be_inside_its_own_definition(self):
        [one] = _gates(preflight_document(
            inference={"parameters": COVERED, "joint_prior": JOINT},
            runs=[{"name": "f", "kind": "fisher", "space": True}]))
        assert one.message.startswith("runs['f']: ")
        assert "inference.joint_prior covers ['d', 'a']" in one.message
        assert "inside its own definition" in one.message
        # the clause that says why nothing downstream would notice, which is
        # what makes this a refusal rather than a note.
        assert "positive definite" in one.message
        assert "Drop space: true" in one.message
        assert "kind: nuts" in one.message

    def test_the_nuts_clause_says_whether_a_joint_prior_is_there_at_all(self):
        # The `because` fork on the nuts leg, which no other test reads.
        # Kills one wording for both: on a document with NO joint_prior,
        # "this document declares no inference.joint_prior" is what tells the
        # reader that covering it is an option they have not used, and on one
        # that declares a joint prior covering somebody else the true
        # sentence is the other.
        [bare] = _gates(preflight_document(
            inference={"parameters": {"w": NONLINEAR_W}},
            runs=[{"name": "chain", "kind": "nuts"}]))
        assert ("this document declares no inference.joint_prior"
                in bare.message)
        [partial] = _gates(preflight_document(
            inference={"parameters": {**COVERED, "w": NONLINEAR_W},
                       "joint_prior": JOINT},
            runs=[{"name": "chain", "kind": "nuts"}]))
        assert "no inference.joint_prior covering them" in partial.message
        assert "['w']" in partial.message
        # ...and the covered pair is NOT named as missing, which is the whole
        # asymmetry: `to_numpyro_model` accepts them.
        assert "'d'" not in partial.message and "'a'" not in partial.message

    def test_the_fisher_clause_is_not_the_npe_clause(self):
        # Four routes, one sentence, and `because` is the only thing telling
        # them apart.  Kills a route handed another route's clause.
        [fisher] = _gates(preflight_document(
            inference={"parameters": {"w": NONLINEAR_W}},
            runs=[{"name": "f", "kind": "fisher", "space": True}]))
        [npe] = _gates(preflight_document(
            inference={"parameters": {"w": NONLINEAR_W}},
            runs=[{"name": "amortized", "kind": "npe"}]))
        assert "posterior precision" in fisher.message
        assert "SIMULATES a bank" in npe.message
        assert "SIMULATES" not in fisher.message
        assert "posterior precision" not in npe.message


class TestA23AgreesWithTheGATEItRunsInFrontOf:
    """The differential that keeps two validators for one property in step.

    ``posterior_support._sampled_space`` decides the same question from a
    BUILT space, and A23 may not call it -- its first line is ``space =
    _space(run, built)`` (``:76``), which P-1 has nothing to build from.  So
    the rule is re-derived from text, and this task therefore ships the shape
    the plan warns about: two functions deciding one property, in two
    voices.  ``_sampled_space`` is not dead -- a run declaring ``expect:
    refuse`` is skipped by A23 and still reaches it, which is exactly what
    ``posterior_helpers.joint_prior_document`` does -- so what is needed is
    not a deletion but a pin that the two AGREE.

    Three ``inference:`` blocks through both, on both routes.  A change to
    either rule alone turns this red on the case it changed; asserting only
    that A23 refuses something could not.

    It BUILDS documents, which nothing else in Tasks 7-9 does but Task 7's
    ``test_the_space_names_are_the_declared_parameter_keys``.  Measured: the
    whole class is well inside §0.1's ~2 s ceiling, and it is a TEST building
    them, never a check.
    """

    #: ``exit_helpers``' own blocks, so the space this drives is one the
    #: shipped suite already measures: priored, prior-free, and prior-free
    #: under a joint prior that covers both.
    @staticmethod
    def _blocks():
        from tests.config.exit_helpers import PRIOR_FREE_TWO, TWO_LATENTS

        return {
            "priored": dict(TWO_LATENTS),
            "prior-free": {**TWO_LATENTS, "parameters": PRIOR_FREE_TWO},
            "covered": {**TWO_LATENTS, "parameters": PRIOR_FREE_TWO,
                        "joint_prior": {"jeffreys": {"over": ["d", "a"]}}},
        }

    @pytest.mark.parametrize("route", ["nuts", "npe"])
    def test_the_text_rule_and_the_built_rule_refuse_the_same_documents(
            self, route):
        from rheplicant.config.errors import ConfigError
        from rheplicant.config.sections.posterior_support import _sampled_space
        from tests.config.exit_helpers import conjugate_built, spec

        for label, block in self._blocks().items():
            built = conjugate_built(inference=block)
            try:
                _sampled_space(spec(kind=route), built, route=route)
                package_refuses = False
            except ConfigError:
                package_refuses = True
            found = _gates(preflight_document(
                inference=block,
                runs=[{"name": "r", "kind": route,
                       "seed": {"from": "runtime.seeds.s"}}]))
            assert bool(found) is package_refuses, (route, label)
            if found:
                assert [f.check for f in found] == ["A23"], (route, label)


def _seedings(document):
    from rheplicant.config.preflight.fitting import _seeds

    return list(_seeds(document))


SEED = {"from": "runtime.seeds.s"}
NPE_SEEDS = {sub: {"seed": {"from": f"runtime.seeds.{sub}"}}
             for sub in ("bank", "create", "train", "sample")}


class TestSeeds:
    def test_plan_estimate_refuses_a_seed_in_the_EXECUTORS_own_words(self):
        # The move, as a byte-for-byte claim.  Kills a re-worded copy: §2.3
        # makes `Report.raise_if_refused()` re-raise the first refusal
        # verbatim, so a pass that says something else changes what every
        # `pytest.raises(ConfigError, match=...)` in tests/config sees.  The
        # string is compared against the extracted function's OWN output, so
        # the two cannot drift.
        from rheplicant.config.errors import ConfigError
        from rheplicant.config.sections.exits import _a29_estimate_takes_no_seed

        with pytest.raises(ConfigError) as caught:
            _a29_estimate_takes_no_seed("runs['fit']", {"seed": SEED})
        found = _seedings(preflight_document(
            inference={"parameters": THREE},
            runs=[{"name": "fit", "kind": "plan.estimate", "seed": SEED,
                   "blocks": [{"names": ["d", "a", "w"]}]}]))
        assert [f.check for f in found] == ["A29"]
        assert found[0].message == str(caught.value)
        assert "plan.estimate refuses a seed" in found[0].message

    def test_plan_sample_requires_one(self):
        found = _seedings(preflight_document(
            inference={"parameters": THREE},
            runs=[{"name": "fit", "kind": "plan.sample", "n_sweeps": 8,
                   "blocks": [{"names": ["d", "a", "w"]}]}]))
        assert [f.check for f in found] == ["A29"]
        assert "'seed' is required and has no default" in found[0].message

    def test_conjugate_gcr_requires_one_and_still_says_a29(self):
        # `test_config_exits_gcr.py:216` matches on "A29" and on
        # "conjugate.wiener".  Kills dropping either from the moved message:
        # that test drives `run_document`, which after Task 2's hook reaches
        # THIS refusal first.
        found = _seedings(preflight_document(
            inference={"parameters": THREE},
            runs=[{"name": "conjugate.gcr", "kind": "conjugate.gcr",
                   "n_draws": 4}]))
        assert [f.check for f in found] == ["A29"]
        assert found[0].message.startswith("runs['conjugate.gcr']: ")
        assert "seed: is required and has no default" in found[0].message
        assert "conjugate.wiener" in found[0].message
        assert "A29" in found[0].message

    def test_nuts_requires_one_BEFORE_to_numpyro_model_would_be_built(self):
        # The sentence this task exists for.  `nuts.py:300` reads the seed
        # after `:287` builds `to_numpyro_model`; the pass reads text.  The
        # brief's second assertion here was `"numpyro" not in sys.modules or
        # True` -- a tautology, and an honest one would be flaky, because
        # another test in the same session may already have imported numpyro.
        # The real guard for that property is Task 2's import-time invariant
        # (`test_config_preflight.py` runs `import rheplicant.config` in a
        # fresh interpreter and asserts numpyro is absent), so this test
        # asserts what it can actually decide and points at the one that can.
        found = _seedings(preflight_document(
            inference={"parameters": PRIORED},
            runs=[{"name": "chain", "kind": "nuts"}]))
        assert [f.check for f in found] == ["A29"]
        assert found[0].message.startswith("runs['chain']: ")
        assert "'seed' is required and has no default" in found[0].message

    def test_npe_refuses_a_run_level_seed_and_requires_four(self):
        # Both halves in ONE test, so a message appended to the run-level
        # branch that contradicts the per-subsection one is caught -- 2D's
        # `test_the_advice_the_gate_gives_depends_on_the_document` shape.
        found = _seedings(preflight_document(
            inference={"parameters": PRIORED,
                       "npe": {**NPE_SEEDS, "train": {}}},
            runs=[{"name": "amortized", "kind": "npe", "seed": SEED}]))
        assert [f.check for f in found] == ["A29", "A29"]
        assert "needs FOUR seeds" in found[0].message
        assert found[0].where == "runs[0]"
        assert found[1].where == "inference.npe.train"
        assert "'seed' is required and has no default" in found[1].message
        # the per-subsection message wears the subsection's own prefix, which
        # is what `_seeded` (`npe.py:227`) passes -- a reader told
        # `runs['amortized']:` would go and edit the run the FIRST finding
        # told them to strip a seed from.
        assert found[1].message.startswith("inference.npe.train: ")

    def test_each_of_the_four_subsections_is_read_and_not_only_train(self):
        # 2C's shape 4 one level in: a loop that reads one name, or that
        # `break`s on the first bad subsection.  Four separate documents plus
        # one carrying all four faults, because a `return` after the first
        # finding passes every single-fault document.
        for subsection in ("bank", "create", "train", "sample"):
            found = _seedings(preflight_document(
                inference={"parameters": PRIORED,
                           "npe": {**NPE_SEEDS, subsection: {}}},
                runs=[{"name": "amortized", "kind": "npe"}]))
            assert [f.where for f in found] == [
                f"inference.npe.{subsection}"], subsection
        found = _seedings(preflight_document(
            inference={"parameters": PRIORED,
                       "npe": {sub: {} for sub in NPE_SEEDS}},
            runs=[{"name": "amortized", "kind": "npe"}]))
        assert [f.where for f in found] == [
            "inference.npe.bank", "inference.npe.create",
            "inference.npe.train", "inference.npe.sample"]

    def test_the_npe_subsections_are_read_only_when_a_npe_run_is_declared(
            self):
        # `inference.npe:` may sit on a document whose runs do not use it --
        # `npe_document`'s own builder writes the section unconditionally.
        # Kills reading the section off `inference:` alone: that direction
        # refuses a document with no npe run at all, which is the plan's
        # finding 2 in the other direction.
        assert _seedings(preflight_document(
            inference={"parameters": PRIORED, "npe": {"train": {}}},
            runs=[{"kind": "forward"}])) == []

    def test_the_embed_subsection_is_not_swept_for_a_seed(self):
        # `inference.npe.embed:` is the fifth member of `_NPE_KEYS` and it
        # declares NO seed -- measured, `_seeded` is called at `npe.py:240`,
        # `:251`, `:263` and `:283`, all four of them a subsection this check
        # lists, and never for `embed`.  Kills a loop over the section's OWN
        # keys instead of `_A29_NPE_SUBSECTIONS`: that refuses a document the
        # package runs, telling the reader a seed is missing from a
        # subsection that takes none.
        assert _seedings(preflight_document(
            inference={"parameters": PRIORED,
                       "npe": {**NPE_SEEDS,
                               "embed": {"python": "mod:fn"}}},
            runs=[{"name": "a", "kind": "npe"}])) == []

    def test_a_missing_npe_subsection_is_left_to_the_section_that_owns_it(
            self):
        # Stand down: `npe._subsection` (`:199-204`) already refuses an absent
        # subsection, and its sentence is about the SUBSECTION being required
        # rather than about the seed -- "there is no subsection this section
        # can do without".  A29 emitted here would answer a question the user
        # has not reached yet.  Kills `_seed_name({}, where)` on an absent
        # body, which would tell them a seed is missing from a block they
        # have not written.
        assert _seedings(preflight_document(
            inference={"parameters": PRIORED, "npe": {"train": NPE_SEEDS[
                "train"]}},
            runs=[{"name": "amortized", "kind": "npe"}])) == []

    def test_condition_is_OUTSIDE_a29(self):
        # `conjugate.py:568-572` takes an OPTIONAL seed.  Kills "every exit
        # that touches a key called seed needs one": that would refuse a
        # `kind: condition` run the package runs, and the survey's own table
        # calls this out as correctly excluded.
        assert _seedings(preflight_document(
            inference={"parameters": THREE},
            runs=[{"name": "c", "kind": "condition"}])) == []

    def test_a_LITERAL_seed_goes_through_the_same_binding(self):
        # Measured: `seed: 3` on a plan.sample is refused with "seed must NAME
        # an entry of runtime.seeds -- {from: runtime.seeds.<name>} -- and got
        # 3".  Kills a presence-only check (`"seed" in options`), which passes
        # on this document and lets the run reach `draws.py:107` at P3.
        found = _seedings(preflight_document(
            inference={"parameters": THREE},
            runs=[{"name": "fit", "kind": "plan.sample", "n_sweeps": 8,
                   "seed": 3, "blocks": [{"names": ["d", "a", "w"]}]}]))
        assert [f.check for f in found] == ["A29"]
        assert "must NAME an entry of runtime.seeds" in found[0].message

    def test_a_gcr_seed_outside_runtime_seeds_is_refused_by_the_form_leg(self):
        # `conjugate.gcr` is in BOTH the gate chain and `_A29_SEEDED_KINDS`,
        # and the two decide different things: the gate decides PRESENCE and
        # `_seed_name` decides FORM.  Kills `gated` widened to "skip
        # `_seed_name` whenever a gate exists for this kind", under which a
        # `seed: {from: nowhere.x}` on a gcr run passes P-1 and dies at
        # `draws.py:114`, behind the beam.
        found = _seedings(preflight_document(
            inference={"parameters": THREE},
            runs=[{"name": "g", "kind": "conjugate.gcr", "n_draws": 4,
                   "seed": {"from": "nowhere.x"}}]))
        assert [f.check for f in found] == ["A29"]
        assert "it must be under runtime.seeds." in found[0].message

    def test_the_run_keys_come_from_runs_py(self):
        # `_seeds` builds `options` as "every key that is not a RUN key", so
        # `_RUN_KEYS` decides what reaches `_seed_name`.  The CONTENT
        # assertion is the test and it is stated honestly: no document can
        # distinguish `dict(run)` from the filtered form today, because no
        # RUN key is called `seed` -- what the pin buys is that a sixth key
        # added to `runs.py`'s tuple turns this red rather than arriving as
        # an OPTION at P-1 while travelling on the RunSpec at P3.
        from rheplicant.config.sections.runs import _RUN_KEYS

        assert set(_RUN_KEYS) == {"expect", "kind", "name", "on", "reuse",
                                  "variant"}
        # ...and every one of them may sit beside a good seed without
        # disturbing it.
        assert _seedings(preflight_document(
            inference={"parameters": THREE},
            runs=[{"name": "fit", "kind": "plan.sample", "n_sweeps": 8,
                   "on": "primary", "reuse": None, "variant": None,
                   "expect": "ok", "seed": SEED,
                   "blocks": [{"names": ["d", "a", "w"]}]}])) == []

    def test_the_SECOND_run_is_blamed_and_not_the_first(self):
        # Task 3's carry-forward rule 3, on this task's own loop: a suite
        # that only ever puts ONE offender in a document cannot tell
        # `runs[i]` from `runs[0]`, and Task 3 shipped one loop guarded and
        # two unguarded.  Measured: without this, `where = "runs[0]"` is a
        # mutant the whole module survives.  The first run here is a clean
        # forward exit and the second is a `nuts` with no seed.
        found = _seedings(preflight_document(
            inference={"parameters": PRIORED},
            runs=[{"name": "fwd", "kind": "forward"},
                  {"name": "chain", "kind": "nuts"}]))
        assert [f.where for f in found] == ["runs[1]"]
        assert found[0].message.startswith("runs['chain']: ")

    def test_a_seedless_gcr_run_is_described_ONCE(self):
        # `conjugate.gcr` is in `_A29_SEEDED_KINDS` AND in the gate chain, so
        # without `gated` one missing key is described twice -- once by
        # `_a29_gcr_needs_a_seed` and once by `_seed_name`, in two voices,
        # naming two different fixes.  Kills the fall-through.
        found = _seedings(preflight_document(
            inference={"parameters": THREE},
            runs=[{"name": "g", "kind": "conjugate.gcr", "n_draws": 4}]))
        assert len(found) == 1

    def test_a_run_that_EXPECTS_a_refusal_is_left_alone(self):
        # The same guard `_prior_gates` needs and for the same reason: a P-1
        # refusal makes the document unloadable, so `expect: refuse` -- which
        # is an assertion about what `execute_run` CAPTURES -- could never be
        # made.  Not in the brief.  Kills reading every run: that takes
        # `expect: refuse` away from every seed-shaped document in the repo.
        assert _seedings(preflight_document(
            inference={"parameters": THREE},
            runs=[{"name": "fit", "kind": "plan.estimate", "seed": SEED,
                   "expect": "refuse",
                   "blocks": [{"names": ["d", "a", "w"]}]}])) == []
        assert len(_seedings(preflight_document(
            inference={"parameters": THREE},
            runs=[{"name": "fit", "kind": "plan.estimate", "seed": SEED,
                   "expect": "ok",
                   "blocks": [{"names": ["d", "a", "w"]}]}]))) == 1

    def test_a_warm_starts_seed_is_left_to_the_key_sweep(self):
        # THE TWIN of `plan.estimate refuses a seed`: `exits.py:269` builds a
        # SECOND plan and `.estimate()`s it, from `warm_start:`, whose own
        # `kind:` must be `plan.estimate` (`exits.py:250`).  It is NOT A29's,
        # because `_WARM_KEYS` (`exits.py:171`) has no `seed` at all, so the
        # unknown-key sweep already says "warm_start does not take ['seed']"
        # -- a more specific sentence, and Task 3's `A1.runs` reaches the run
        # level of the same document before the beam.  Kills widening the
        # gate to `warm_start`, which would tell the reader about an
        # asymmetry between two exit kinds when the fault is a key that does
        # not exist.
        assert _seedings(preflight_document(
            inference={"parameters": THREE},
            runs=[{"name": "fit", "kind": "plan.sample", "n_sweeps": 8,
                   "seed": SEED, "blocks": [{"names": ["d", "a", "w"]}],
                   "warm_start": {"kind": "plan.estimate", "move": ["d"],
                                  "seed": SEED,
                                  "blocks": [{"names": ["d", "a", "w"]}]}}])
        ) == []

    def test_every_A29_message_carries_its_tag_and_the_moved_ones_keep_theirs(
            self):
        """A29 is the one check in this plan whose messages do NOT all END
        with ``(check A29).``, and that is §3.2(c) rather than an oversight.

        The three lifted refusals already carry the tag mid-sentence -- and
        ``plan.estimate``'s carries it *inside a longer parenthesis*
        (``"...estimate has no key parameter; check A29). Drop it..."``),
        which is why the assertion below reads ``"check A29"`` and not
        ``"(check A29)"``.  §2.3 makes a MOVED message verbatim, so appending
        a second tag would both double it and change a string four shipped
        tests read.  The two that come through ``draws._seed_name`` carry
        none, so those are appended to.  Kills a later edit that appends to
        all five (a doubled tag) and one that appends to none (a finding a
        reader cannot grep).
        """
        gated = [
            preflight_document(
                inference={"parameters": THREE},
                runs=[{"name": "fit", "kind": "plan.estimate",
                       "seed": SEED}]),
            preflight_document(
                inference={"parameters": THREE},
                runs=[{"name": "g", "kind": "conjugate.gcr"}]),
            preflight_document(
                inference={"parameters": PRIORED},
                runs=[{"name": "a", "kind": "npe", "seed": SEED}]),
        ]
        for document in gated:
            [one] = _seedings(document)
            assert one.message.count("check A29") == 1, one.message
            assert not one.message.endswith("(check A29)."), one.message
        derived = [
            preflight_document(
                inference={"parameters": THREE},
                runs=[{"name": "fit", "kind": "plan.sample"}]),
            preflight_document(
                inference={"parameters": PRIORED},
                runs=[{"name": "chain", "kind": "nuts"}]),
            preflight_document(
                inference={"parameters": PRIORED,
                           "npe": {**NPE_SEEDS, "train": {}}},
                runs=[{"name": "a", "kind": "npe"}]),
        ]
        for document in derived:
            [one] = _seedings(document)
            assert one.message.count("check A29") == 1, one.message
            assert one.message.endswith("(check A29)."), one.message

    def test_A29_is_registered_and_the_base_document_earns_none_of_it(self):
        from rheplicant.config.preflight.fitting import _seeds

        assert CHECKS["A29"] is _seeds
        assert "A29" not in preflight(preflight_document()).checks()

    def test_a_missing_seed_wins_against_a_beam_that_cannot_be_read(self):
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
        #
        # `startswith` and not `in`: the refusal must be FIRST, which is what
        # `raise_if_refused` quotes verbatim.  A check registered ahead of
        # this one and firing on the same document would leave "check A29"
        # in the message (the tail names every other refusal's `where`) while
        # the sentence the user reads is somebody else's.
        from rheplicant.config.document import load_document
        from rheplicant.config.errors import ConfigError

        document = preflight_document(
            inference={"parameters": THREE},
            runs=[{"name": "g", "kind": "conjugate.gcr", "n_draws": 4}],
            resources=UNREADABLE_BEAM)
        with pytest.raises(ConfigError) as caught:
            load_document(document)
        assert str(caught.value).startswith("runs['g']: ")
        assert "check A29" in str(caught.value)
        assert "no_such_beam" not in str(caught.value)


class TestTheSectionsStillCallTheirOwnRefusal:
    """§2.2: one name, one binding, **two call sites**.

    Extracting a refusal and leaving the section calling nothing is how the
    ``_number``-vs-``_whole`` divergence on the 2C ledger happened.  It is
    worse here than it looks, because after this task the pass reaches each
    of the three refusals FIRST on every document route -- so deleting the
    section's own call leaves ``tests/config/test_config_exits_*.py`` green
    and the P3 second opinion gone.  Each test below drives the section
    function directly, which is the only route that can still see it.

    All three refuse before they touch ``built``, so ``built=None`` is enough
    and nothing is constructed.
    """

    def _spec(self, kind, options, name=None):
        from rheplicant.config.sections.runs import RunSpec

        return RunSpec(name=name or kind, kind=kind, variant=None,
                       on="primary", expect="ok", options=options)

    def test_run_plan_still_refuses_an_estimate_seed(self):
        from rheplicant.config.errors import ConfigError
        from rheplicant.config.sections.exits import _run_plan

        with pytest.raises(ConfigError, match="plan.estimate refuses a seed"):
            _run_plan(self._spec("plan.estimate", {"seed": SEED}), None)

    def test_gcr_plan_still_refuses_a_missing_seed(self):
        from rheplicant.config.errors import ConfigError
        from rheplicant.config.sections.conjugate import _gcr_plan

        with pytest.raises(ConfigError, match="A29"):
            _gcr_plan(self._spec("conjugate.gcr", {"n_draws": 4}),
                      where="runs['conjugate.gcr']")

    def test_run_npe_still_refuses_a_run_level_seed(self):
        from rheplicant.config.errors import ConfigError
        from rheplicant.config.sections.npe import _run_npe

        with pytest.raises(ConfigError, match="needs FOUR seeds"):
            _run_npe(self._spec("npe", {"seed": SEED}, name="amortized"),
                     None)


class TestNoHostileDocumentCanAbortTheTaskEightChecks:
    """The §2.3 TRAP, for `_prior_gates` and `_seeds`.

    A check that RAISES aborts the pass and discards every other finding.
    Task 4 measured two shapes here: a section that is present but not a
    MAPPING, and a value used as a dict key or iterated.  Both readers below
    take user text straight out of the document -- `joint_prior.jeffreys.over`,
    `runs[].blocks[].names`, `inference.npe.<sub>` -- so both are exposed.
    """

    HOSTILE = [
        {},
        {"inference": None},
        {"inference": "nope"},
        {"inference": {"joint_prior": "nope"}},
        {"inference": {"joint_prior": {"jeffreys": "nope"}}},
        {"inference": {"joint_prior": {"jeffreys": {"over": "da"}}}},
        {"inference": {"joint_prior": {"jeffreys": {"over": [7, None]}}}},
        {"inference": {"joint_prior": {"jeffreys": {"over": {"d": 1}}}}},
        {"inference": {"joint_prior": {"jeffreys": {"over": 7}}}},
        {"inference": {"joint_prior": JOINT, "parameters": None}},
        {"inference": {"npe": "nope"}},
        {"inference": {"npe": {"train": "nope"}}},
        {"inference": {"npe": {"train": {"seed": ["a"]}}}},
        {"inference": {"npe": {"train": {"seed": {"from": 7}}}}},
        {"runs": "nope"},
        {"runs": 7},
        {"runs": [None]},
        {"runs": [{"kind": None}]},
        {"runs": [{"kind": 7, "seed": 3}]},
        {"runs": [{"kind": "nuts"}]},
        {"runs": [{"kind": "npe", "seed": {"from": []}}]},
        {"runs": [{"kind": "fisher", "space": "true"}]},
        {"runs": [{"kind": "fisher", "space": ["true"]}]},
        {"runs": [{"kind": "plan.sample", "blocks": 5}]},
        {"runs": [{"kind": "plan.sample", "blocks": "nope"}]},
        {"runs": [{"kind": "plan.sample", "blocks": [{"names": 5}]}]},
        {"runs": [{"kind": "plan.sample", "blocks": [{"names": {"d": 1}}]}]},
        {"runs": [{"kind": "plan.sample",
                   "blocks": [{"names": ["d"], "engine": ["gradient"]}]}]},
        {"runs": [{"kind": "plan.estimate", "expect": ["refuse"],
                   "seed": SEED}]},
        {"runs": [{"kind": "conjugate.gcr", "seed": {"from": "runtime.x"}}]},
        {"runs": {"kind": "nuts"}},
    ]

    @pytest.mark.parametrize("patch", HOSTILE,
                             ids=[str(index) for index in
                                  range(len(HOSTILE))])
    def test_neither_check_raises(self, patch):
        document = _hostile_document(patch)
        for finding in (*_gates(document), *_seedings(document)):
            assert finding.check in ("A20", "A21", "A23", "A29")

    @pytest.mark.parametrize("patch", HOSTILE,
                             ids=[str(index) for index in
                                  range(len(HOSTILE))])
    def test_the_whole_pass_survives_each_of_them(self, patch):
        # `_check_where` runs OUTSIDE the per-check `try`, so a `where` built
        # from user text could kill the pass even when the check returns
        # cleanly.  Both of this task's `where`s are `runs[<int>]` and
        # `inference.npe.<literal>`, and this is what says so.
        preflight(_hostile_document(patch))
