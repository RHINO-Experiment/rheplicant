"""inference: the section orchestrator -- checks, trainable, truth, sequence."""

import pytest

from rheplicant.config import ConfigError
from rheplicant.config.sections.inference import CheckSpec, InferenceBuild, build_inference
from rheplicant.config.sections.observation import build_observation
from rheplicant.config.sections.runtime import build_runtime
from tests.config.inference_helpers import (
    MODEL,
    NOISY_MODEL,
    context,
    state,
    twin,
)

#: ``MODEL`` plus a node whose one leaf is ``(n_freq,)`` rather than a scalar
#: -- the shape 3.2(d) is about, and the only kind of leaf on which a scalar
#: ``init`` and a symmetric equality can disagree.
BANDPASS_MODEL = {
    **MODEL,
    "bandpass": {"bandpass": {"value": [1.0] * 8, "unit": "dimensionless"}},
}


def observation(**extras):
    section = {
        "freq": {"grid": {"linspace": {"start": 60.0, "stop": 85.0, "num": 8,
                                       "endpoint": True}, "unit": "MHz"}},
        "time": {"grid": {"arange": {"start": 0.0, "step": 2.0, "num": 16},
                          "unit": "s"}},
    }
    section.update(extras)
    build, _ = build_observation(section, runtime=build_runtime({"seed": 1}))
    return build


def infer(section, model=None, ctx=None):
    ctx = ctx or context()
    return build_inference(section, twin=twin(model, ctx), state=state(),
                           observation=observation(), context=ctx)


PARAMS = {"g": {"init": 1.0, "linear": True, "into": "gain.gain"}}

#: The smallest legal ``inference.npe:``: four subsections, four named
#: seeds, two required counts, no optional knob at all.
NPE = {
    "bank": {"n_simulations": 8, "seed": {"from": "runtime.seeds.npe_bank"}},
    "create": {"seed": {"from": "runtime.seeds.npe_create"}},
    "train": {"seed": {"from": "runtime.seeds.npe_train"}},
    "sample": {"n_draws": 4, "seed": {"from": "runtime.seeds.npe_sample"}},
}


class TestValidateRunsAtLoad:
    """Check C17: ``ParameterSpace.validate``, which nothing called at load.

    The package ships the check and four call sites call it, and measured, all
    four are at EXIT time -- so a document whose bindings do not fit its twin
    loads clean, past every resource the layer built, and fails at the fit.
    It reads shapes only (measured: live device arrays 15 before and 15
    after, 0.77-1.11 ms), so running it at load costs nothing that is not
    paid anyway.

    **Why the call-count test exists, and why it is this class's subject.**
    Measured before a line was written: inserting the correct implementation
    leaves ``tests/config`` entirely green.  Nothing in the suite could tell
    the call being present from the call being absent, because every document
    the suite carries is one ``validate`` accepts.  A test that asserts a
    document still builds therefore proves nothing here; the only thing that
    can fail when the call is deleted is a test that counts the call.
    """

    def _recorder(self, monkeypatch):
        from rheplicant.inference import ParameterSpace

        seen = []
        monkeypatch.setattr(
            ParameterSpace, "validate",
            lambda self, pipeline: seen.append((self, pipeline)))
        return seen

    def test_validate_is_called_exactly_once_on_the_fit_twin(self,
                                                             monkeypatch):
        """R1's mutant, as a test: delete the call and this is what dies.

        ``is`` identity, not equality -- two ``eqx.Module`` twins with the
        same leaves compare equal while being different objects, so ``==``
        here would pass against the full twin as well.
        """
        seen = self._recorder(monkeypatch)
        build = infer({"parameters": PARAMS})
        assert len(seen) == 1
        space, pipeline = seen[0]
        assert space is build.space
        assert pipeline is build.fit_twin

    def test_it_is_handed_the_fit_twin_and_not_the_full_one(self,
                                                            monkeypatch):
        """The trap, on a document where the two are DIFFERENT objects.

        Measured: handing ``validate`` the full twin raises on every document
        that writes ``inference.twin.without:``, because the bindings were
        resolved against the repaired twin and the dropped node's leaves are
        gone -- which is every document ``exit_helpers`` builds.  So this
        pins the distinction on a document that has one.
        """
        seen = self._recorder(monkeypatch)
        ctx = context()
        full = twin(NOISY_MODEL, ctx)
        build = build_inference({"twin": {"without": ["noise"]},
                                 "parameters": PARAMS},
                                twin=full, state=state(),
                                observation=observation(), context=ctx)
        assert len(seen) == 1
        assert seen[0][1] is build.fit_twin
        assert seen[0][1] is not full

    def test_a_document_with_no_parameters_validates_nothing(self,
                                                             monkeypatch):
        """The guard the plan names: ``build_space`` may return ``None``.

        A document with no ``inference.parameters`` has no space, and a
        ``None.validate`` would take the whole load down.
        """
        seen = self._recorder(monkeypatch)
        build = infer({"noise": {"kind": "homoscedastic",
                                 "sigma": {"value": 0.5, "unit": "K"}}})
        assert build.space is None
        assert seen == []

    def test_it_runs_before_the_builders_that_evaluate_the_twin(self,
                                                                monkeypatch):
        """3.2(b): the call is placed BEFORE the two real forward passes.

        ``build_noise``'s ``source: prediction_at_init`` and
        ``build_observed``'s simulation branch both evaluate the twin. A
        space that does not fit is refused before either is paid for, which
        is the whole reason C17 is a section fix rather than a built-pass
        check.
        """
        order = []
        from rheplicant.inference import ParameterSpace

        monkeypatch.setattr(ParameterSpace, "validate",
                            lambda self, pipeline: order.append("validate"))
        import rheplicant.config.sections.inference as inference_module

        real_observed = inference_module.build_observed
        real_noise = inference_module.build_noise
        monkeypatch.setattr(
            inference_module, "build_observed",
            lambda *a, **k: (order.append("observed"),
                             real_observed(*a, **k))[1])
        monkeypatch.setattr(
            inference_module, "build_noise",
            lambda *a, **k: (order.append("noise"), real_noise(*a, **k))[1])
        infer({"parameters": PARAMS,
               "observed": {"from": "simulation"}})
        assert order == ["validate", "noise", "observed"]

    def test_the_forward_pass_is_never_reached_by_a_refused_space(self):
        """The same ordering, observed through what it SAVES.

        A scalar latent bound into an ``(n_freq,)`` leaf is refused by
        ``validate``; ``build_observed``'s simulation branch must therefore
        never run.  Restoring the call to after ``build_observed`` leaves the
        refusal intact and this assertion is what notices.
        """
        import rheplicant.config.sections.inference as inference_module

        reached = []
        real = inference_module.build_observed
        try:
            inference_module.build_observed = (
                lambda *a, **k: (reached.append(1), real(*a, **k))[1])
            with pytest.raises(ConfigError, match="check C17"):
                infer({"parameters": {"b": {"init": 1.0,
                                            "into": "bandpass.bandpass"}},
                       "observed": {"from": "simulation"}},
                      model=BANDPASS_MODEL)
        finally:
            inference_module.build_observed = real
        assert reached == []

    def test_a_config_built_space_never_carries_a_raw_bind(self):
        """What makes "``validate`` allocates nothing" TRUE rather than lucky.

        ``validate``'s one allocating branch is
        ``_reject_latents_the_raw_bind_ignores``, which perturbs each latent
        and binds twice per probe.  It is gated on ``raw_bind is not None``,
        and ``build_space`` has no route that sets one -- the only builder of
        a raw bind is ``ParameterSpace.raw(...)``, a Python API no document
        can reach.  Pinned here so that a later route to ``raw_bind`` cannot
        turn a shapes-only check into an allocating one in silence.
        """
        build = infer({"parameters": PARAMS})
        assert build.space.raw_bind is None

    def test_the_allocating_branch_is_not_entered(self, monkeypatch):
        """The same property, asserted on the branch rather than on the flag."""
        from rheplicant.inference import ParameterSpace

        def explode(self, pipeline):
            raise AssertionError(
                "validate entered its allocating branch at load time")

        monkeypatch.setattr(ParameterSpace,
                            "_reject_latents_the_raw_bind_ignores", explode)
        assert infer({"parameters": PARAMS}).space is not None

    def test_the_refusal_names_the_document_key_and_both_spellings(self):
        """S1, and 2.3's rule that the package's own sentence survives.

        The package's headline refusal names a latent tuple and a selector
        index -- ``Bind for ('b',) produces shape () for `into` selector 0``
        -- and no path in either spelling, which is why 0.3 E.7 rules "name
        both spellings" unsatisfiable ON that sentence and points at
        ``config/paths.py::resolve_path_on`` instead.  So the sentence is
        carried verbatim and the document keys and both spellings are added
        beside it.
        """
        with pytest.raises(ConfigError) as caught:
            infer({"parameters": {"b": {"init": 1.0,
                                        "into": "bandpass.bandpass"}}},
                  model=BANDPASS_MODEL)
        assert str(caught.value) == (
            "inference: the parameter space this document declares does not "
            "fit the twin it binds into, and the fit would be the first "
            "thing to say so. The package refuses it in its own words: Bind "
            "for ('b',) produces shape () for `into` selector 0, but that "
            "leaf has shape (8,). That sentence names latents and selector "
            "positions rather than document keys, so here is what this "
            "document bound: inference.parameters.b -> 'bandpass.bandpass', "
            "which the twin spells '.operator.stages[1].bandpass'. Each path "
            "is written twice -- as the document wrote it, then as "
            "jax.tree_util.keystr spells it, which is the form the package's "
            "other refusals quote (check C17)."
        )

    def test_the_bindings_route_appears_in_the_table_too(self):
        """The `inference.bindings:` leg of the table, which had no test.

        A latent bound through an ``inference.bindings`` entry rather than
        through ``parameters.into:`` is the second of the two routes B4's own
        twin list names, and it is the one whose row the table was silently
        free to drop: deleting the whole ``enumerate(bindings)`` loop left
        every test green while a document bound this way got a refusal whose
        *"here is what this document bound"* table was EMPTY.
        """
        with pytest.raises(ConfigError) as caught:
            infer({"parameters": {"b": {"init": 1.0}},
                   "bindings": [{"latents": ["b"],
                                 "into": "bandpass.bandpass"}]},
                  model=BANDPASS_MODEL)
        assert str(caught.value) == (
            "inference: the parameter space this document declares does not "
            "fit the twin it binds into, and the fit would be the first "
            "thing to say so. The package refuses it in its own words: Bind "
            "for ('b',) produces shape () for `into` selector 0, but that "
            "leaf has shape (8,). That sentence names latents and selector "
            "positions rather than document keys, so here is what this "
            "document bound: inference.bindings[0] -> 'bandpass.bandpass', "
            "which the twin spells '.operator.stages[1].bandpass'. Each path "
            "is written twice -- as the document wrote it, then as "
            "jax.tree_util.keystr spells it, which is the form the package's "
            "other refusals quote (check C17)."
        )

    def test_every_bound_path_is_listed_not_only_the_first(self):
        """Both routes, both rows, and the separator between them.

        Truncating the table to its first row left every test green, because
        every document that reached the message had exactly one binding. This
        one has two -- one per route -- so a table that reports the first row
        only, or that drops the ``bindings`` leg, is short exactly one row.
        """
        with pytest.raises(ConfigError) as caught:
            infer({"parameters": {"b": {"init": 1.0,
                                        "into": "bandpass.bandpass"},
                                  "g": {"init": 1.0}},
                   "bindings": [{"latents": ["g"], "into": "gain.gain"}]},
                  model=BANDPASS_MODEL)
        assert str(caught.value) == (
            "inference: the parameter space this document declares does not "
            "fit the twin it binds into, and the fit would be the first "
            "thing to say so. The package refuses it in its own words: Bind "
            "for ('b',) produces shape () for `into` selector 0, but that "
            "leaf has shape (8,). That sentence names latents and selector "
            "positions rather than document keys, so here is what this "
            "document bound: inference.parameters.b -> 'bandpass.bandpass', "
            "which the twin spells '.operator.stages[1].bandpass'; "
            "inference.bindings[0] -> 'gain.gain', which the twin spells "
            "'.operator.stages[2].gain'. Each path is written twice -- as "
            "the document wrote it, then as jax.tree_util.keystr spells it, "
            "which is the form the package's other refusals quote "
            "(check C17)."
        )

    def test_a_failure_that_is_not_the_packages_propagates_as_itself(self):
        """The handler catches ``ParameterSpaceError`` and nothing wider.

        C-12 is the stated reason this handler exists at all, so the class it
        names is load-bearing. Widening it to ``except Exception`` left every
        test green while an unrelated failure -- a ``TypeError`` from a
        malformed transform, an ``OSError`` from a read -- would be dressed
        up as *"the parameter space does not fit the twin"*, which is a
        diagnosis of something that is not wrong.
        """
        from rheplicant.inference import ParameterSpace

        original = ParameterSpace.validate
        try:
            ParameterSpace.validate = (
                lambda self, pipeline: (_ for _ in ()).throw(
                    RuntimeError("boom")))
            with pytest.raises(RuntimeError, match="boom"):
                infer({"parameters": PARAMS})
        finally:
            ParameterSpace.validate = original

    def test_it_is_a_ConfigError_and_the_package_class_stays_catchable(self):
        """0.2 C-12, settled: the gap is benign, and here is why.

        ``ParameterSpaceError`` is ``(DirtError, ValueError)`` -- a SIBLING
        of ``ConfigError``, not a subclass, so ``pytest.raises(ConfigError)``
        does not catch it and a config-layer caller would have missed it.
        Re-voicing closes that at the config boundary, and the original is
        still reachable by name on ``__cause__`` for anyone who wants it.
        """
        from rheplicant.core.errors import ParameterSpaceError

        assert not issubclass(ParameterSpaceError, ConfigError)
        with pytest.raises(ConfigError) as caught:
            infer({"parameters": {"b": {"init": 1.0,
                                        "into": "bandpass.bandpass"}}},
                  model=BANDPASS_MODEL)
        assert isinstance(caught.value.__cause__, ParameterSpaceError)

    def test_applying_the_refusals_own_advice_makes_the_document_build(self):
        """S4: the remedy the message implies, applied, and the document
        passes -- with no second refusal waiting behind it.

        The sentence names the leaf's shape ``(8,)``; an ``init`` at that
        shape is what it asks for.
        """
        build = infer({"parameters": {
            "b": {"init": {"list": [1.0] * 8}, "into": "bandpass.bandpass"}}},
            model=BANDPASS_MODEL)
        assert build.space is not None

    def test_the_stochastic_leg_blames_the_twin_and_names_a_document_remedy(
            self):
        """The OTHER leg C17 delivers, which had no test at all.

        §0.3 E.7.4: the call adds exactly two things, ``refuse_stochastic_
        stages`` and the shape/dtype comparisons. This is the first, and one
        wording cannot be true of both -- here the space fits perfectly and
        the fault is in the twin, so the shape leg's framing would be false
        in three separate clauses and would parade an innocent binding table
        as evidence.

        The package's own sentence survives verbatim inside the new one; it
        is taken FROM the package here rather than transcribed, so this also
        pins that the quote is a quote.
        """
        from rheplicant.core.errors import ParameterSpaceError
        from rheplicant.inference.parameters import refuse_stochastic_stages

        ctx = context()
        noisy = twin(NOISY_MODEL, ctx)
        with pytest.raises(ParameterSpaceError) as package:
            refuse_stochastic_stages(noisy, "This ParameterSpace")
        with pytest.raises(ConfigError) as caught:
            build_inference({"parameters": PARAMS}, twin=noisy, state=state(),
                            observation=observation(), context=ctx)
        assert str(caught.value) == (
            "inference: the twin this document fits with still draws its own "
            "randomness, so it is the twin at fault here and not the "
            "parameter space. The package refuses it in its own words: "
            f"{package.value} From a document the repair is "
            "inference.twin.without: ['noise'] -- that drops the stage from "
            "the twin the FIT uses while inference.observed's simulation "
            "still defaults to the full twin, so the scatter goes on "
            "entering the data it was written for (check C17)."
        )

    def test_the_stochastic_leg_says_none_of_the_shape_legs_four_falsehoods(
            self):
        """Named individually, because each was separately wrong.

        The shape leg's wording claims the space does not fit the twin (it
        fits), that the package named latents and selector positions (it
        named a node), and then prints the document's bindings as though one
        of them were the fault. An equality pin already excludes all three;
        these say which three, so a future edit that reintroduces one is
        told what it reintroduced.
        """
        ctx = context()
        with pytest.raises(ConfigError) as caught:
            build_inference({"parameters": PARAMS},
                            twin=twin(NOISY_MODEL, ctx), state=state(),
                            observation=observation(), context=ctx)
        message = str(caught.value)
        assert "does not fit the twin it binds into" not in message
        assert "names latents and selector positions" not in message
        assert "here is what this document bound" not in message
        assert "inference.parameters.g ->" not in message

    def test_applying_the_stochastic_legs_advice_makes_the_document_build(
            self):
        """R4: the remedy named is one a document can actually write.

        The package's surviving advice is ``Assembly.without(node_id)`` and
        *"rebuild the Pipeline without it"* -- Python APIs no YAML author can
        reach, which would have made this the fifth advice loop on this plan.
        The document remedy exists, and this applies exactly the one the
        message names, character for character.
        """
        ctx = context()
        build = build_inference({"twin": {"without": ["noise"]},
                                 "parameters": PARAMS},
                                twin=twin(NOISY_MODEL, ctx), state=state(),
                                observation=observation(), context=ctx)
        assert build.space is not None
        assert "noise" not in build.fit_twin.lit

    def test_the_advice_leaves_the_scatter_in_the_simulated_data(self):
        """The other half of the advice, which is a claim about physics.

        The message says the fit twin loses the stage *while*
        ``inference.observed``'s simulation still defaults to the full twin
        -- i.e. following the advice does not quietly produce noiseless data
        to fit against. If that were false the remedy would trade one silent
        wrongness for another, so it is asserted rather than asserted-in-
        prose: the simulated data must NOT equal the fit twin's own
        prediction.
        """
        import jax.numpy as jnp

        ctx = context()
        full = twin(NOISY_MODEL, ctx)
        build = build_inference({"twin": {"without": ["noise"]},
                                 "parameters": PARAMS,
                                 "observed": {"from": "simulation"}},
                                twin=full, state=state(),
                                observation=observation(), context=ctx)
        clean = build.space.bind(
            build.fit_twin, dict(build.space.initial_values()))(state()).data
        assert not jnp.allclose(build.observed.entries["primary"], clean)

    def test_a_document_wrong_in_a_more_specific_way_hears_that_instead(self):
        """S4's stand-down half: C17 must not pre-empt a better sentence.

        This document is wrong in C17's way (a scalar into an ``(8,)`` leaf)
        AND names a latent that does not exist. The second is decided in
        ``build_space``, one call earlier, and says exactly which name is
        undeclared -- so it is the sentence the reader gets.
        """
        with pytest.raises(ConfigError) as caught:
            infer({"parameters": {"b": {"init": 1.0,
                                        "into": "bandpass.bandpass"}},
                   "bindings": [{"latents": ["ghost"], "into": "gain.gain"}]},
                  model=BANDPASS_MODEL)
        assert "'ghost' is not a declared latent" in str(caught.value)
        assert "check C17" not in str(caught.value)

    def test_no_exit_that_calls_validate_today_is_disturbed(self):
        """S3's first named twin: every exit still validates for itself.

        ``forward_fn`` and the Bayesian bridge validate at their own entry;
        this task ADDS a call, it does not move one. A load-time call that
        let the exits skip theirs would be a regression on every pipeline
        built outside the config layer -- including every one built in
        Python, which never passes through ``build_inference`` at all.

        Asserted BEHAVIOURALLY: an earlier version of this test grepped
        ``inspect.getsource(forward_fn)`` for the call, which is a tripwire
        on the package's source text and cannot fail for anything this task's
        code could do. This builds a space the twin does not fit, hands it
        straight to ``forward_fn``, and requires the package to refuse it on
        its own -- with no config layer anywhere in the call.
        """
        from rheplicant.core.errors import ParameterSpaceError
        from rheplicant.inference import Bind, Latent, ParameterSpace

        ctx = context()
        fit = twin(BANDPASS_MODEL, ctx)
        mismatched = ParameterSpace(
            latents=(Latent("b", init=1.0),),
            bindings=(Bind("b", into=lambda p: p["bandpass"].bandpass),))
        with pytest.raises(ParameterSpaceError, match="produces shape"):
            mismatched.forward_fn(fit, state())

    def test_load_document_calls_it_once_on_the_fit_twin(self, monkeypatch):
        """§5's box asserted at the level §5 states it.

        §5 says *"``load_document`` calls ``ParameterSpace.validate`` exactly
        once, on the fit twin"*. Every other test in this class asserts that
        of ``build_inference``, one level below -- true, and not the same
        claim: a second call could be added by the hook, or the whole builder
        could stop being reached, without any of them moving.

        **On a document whose ``linearity`` gate does not run**, and that is
        Plan 3C's correction rather than a weakening. The post-flight C12
        calls ``check_linearity``, which runs ``_isolate``, which validates --
        so on a document at the default ``linearity: refuse`` the count is
        ``1 + <one per linear latent>``, measured. The BUILDER's call is still
        exactly one and is still on the fit twin, which is what this test is
        about; :meth:`test_load_document_validates_once_more_per_linear_latent`
        pins the priced pass's own calls beside it.
        """
        from rheplicant.config.document import load_document
        from rheplicant.inference import ParameterSpace
        from tests.config.exit_helpers import conjugate_document

        document = conjugate_document()
        document["inference"]["checks"] = {
            "linearity": {"mode": "skip",
                          "reason": "counted on its own, one test down"}}
        seen = []
        monkeypatch.setattr(
            ParameterSpace, "validate",
            lambda self, pipeline: seen.append((self, pipeline)))
        run = load_document(document)
        assert len(seen) == 1
        assert seen[0][0] is run.inference.space
        assert seen[0][1] is run.inference.fit_twin
        assert seen[0][1] is not run.twin

    def test_load_document_validates_once_more_per_linear_latent(
            self, monkeypatch):
        """What the priced pass adds, pinned as a NUMBER rather than left
        to drift.

        ``check_linearity`` runs ``_isolate`` before ``_require_inexact``
        (``linear.py:517`` then ``:518``) and ``_isolate`` validates the space
        against the pipeline -- so C12, which calls it once per latent
        declared ``linear: true``, costs one extra validate each. Measured on
        ``conjugate_document()`` (one linear latent): **2**; on the two-latent
        document: **3**.

        **The surviving invariant is "validate is called, and every call is on
        the FIT twin"**, and that second half is the one the 3B defect was
        about -- an ``(8,)`` latent bound into a scalar leaf was accepted
        because nothing validated at all. A count that grows with the linear
        latents does not weaken it; a call on ``run.twin`` would.

        **Kills** a priced check that validates against the raw twin, and a
        C12 that silently stopped running.
        """
        from rheplicant.config.document import load_document
        from rheplicant.inference import ParameterSpace
        from tests.config.exit_helpers import TWO_LATENTS, conjugate_document

        for label, inference, expected in (
            ("one linear latent", None, 2),
            ("two linear latents", dict(TWO_LATENTS), 3),
        ):
            seen = []
            document = (conjugate_document() if inference is None
                        else conjugate_document(inference=inference))
            monkeypatch.setattr(
                ParameterSpace, "validate",
                lambda self, pipeline, _s=seen: _s.append((self, pipeline)))
            run = load_document(document)
            linear = [name for name in run.inference.space.names
                      if run.inference.space.latent(name).linear]
            assert len(seen) == expected == 1 + len(linear), label
            assert all(space is run.inference.space for space, _ in seen), label
            assert all(pipeline is run.inference.fit_twin
                       for _, pipeline in seen), label
            assert run.inference.fit_twin is not run.twin, label


class TestSequence:
    def test_none_still_yields_a_build_with_the_twin_as_fit_twin(self):
        build = infer(None)
        assert isinstance(build, InferenceBuild)
        assert build.space is None
        assert build.noise.kind == "none"
        assert build.observed is None

    def test_the_whole_section_composes(self):
        build = infer({
            "twin": {"without": ["noise"]},
            "parameters": PARAMS,
            "noise": {"kind": "homoscedastic",
                      "sigma": {"value": 0.5, "unit": "K"}},
            "observed": {"from": "simulation", "at": {"g": 1.5}},
        }, model=NOISY_MODEL)
        assert "noise" not in build.fit_twin.lit
        assert build.space is not None
        assert build.observed.entries["primary"].shape == (16, 8)

    def test_bindings_resolve_against_the_repaired_twin(self):
        build = infer({
            "twin": {"replace": {"gain": {"gain": {"value": 1.0,
                                                   "unit": "dimensionless"}}}},
            "parameters": {"d": {"init": 0.5,
                                 "into": "global_signal.depth"}},
        })
        assert build.replaced == ("gain",)

    def test_a_binding_into_a_removed_node_is_refused(self):
        """Paths resolve against the FIT twin: a latent cannot bind into a
        node the twin repair just removed."""
        with pytest.raises(ConfigError, match=r"noise\.sigma"):
            infer({"twin": {"without": ["noise"]},
                   "parameters": {"s": {"init": 0.1, "into": "noise.sigma"}}},
                  model=NOISY_MODEL)

    def test_a_non_list_bindings_section_is_refused(self):
        """bindings: is a LIST -- a scalar or a dashless single mapping both
        fail as ConfigError, not as a raw TypeError or a blamed first key."""
        with pytest.raises(ConfigError, match="LIST"):
            infer({"parameters": {"g": {"init": 1.0}}, "bindings": 42})
        with pytest.raises(ConfigError, match="LIST"):
            infer({"parameters": {"g": {"init": 1.0}},
                   "bindings": {"latents": ["g"], "into": "gain.gain"}})

    def test_npe_parses_rather_than_being_deferred_by_name(self):
        """The 2D deferral is gone: the section is grammar now.

        Its own module (``test_config_section_npe.py``) owns the grammar;
        what this asserts is the SEQUENCE -- ``build_inference`` calls
        ``parse_npe`` and lands the result -- and that the half-written
        section this test used to hand to the deferral is now refused for
        its own reason. ``match`` names the subsection, because
        ``draws._seed_name``'s message is identical across all four.
        """
        build = infer({"npe": NPE})
        assert build.npe is not None
        assert build.npe.bank["n_simulations"] == 8
        with pytest.raises(
                ConfigError,
                match=r"inference\.npe\.bank: 'seed' is required"):
            infer({"npe": {**NPE, "bank": {"n_simulations": 8}}})

    def test_unknown_inference_keys_are_swept(self):
        with pytest.raises(ConfigError, match="observations"):
            infer({"observations": {}})


class TestFrozenSequencing:
    NOISE = {"kind": "radiometer_frozen", "source": "observed",
             "channel_width": {"value": 4.0, "unit": "Hz"},
             "integration_time": {"value": 4.0, "unit": "s"}}

    def test_source_observed_freezes_from_the_primary(self):
        build = infer({"parameters": PARAMS, "noise": self.NOISE,
                       "observed": {"from": "simulation"}})
        assert build.noise.sigma is not None
        assert build.noise.sigma.shape == (16, 8)

    def test_source_observed_without_observed_is_refused(self):
        with pytest.raises(ConfigError, match="observed"):
            infer({"parameters": PARAMS, "noise": self.NOISE})

    def test_source_prediction_at_init_evaluates_the_fit_twin_once(self):
        build = infer({"parameters": PARAMS,
                       "noise": {**self.NOISE,
                                 "source": "prediction_at_init"}})
        assert build.noise.sigma is not None

    def test_prediction_at_init_evaluates_the_repaired_twin(self):
        """With the stochastic node repaired away, the frozen sigma comes from
        the FIT twin's deterministic prediction -- |prediction|/sqrt(w*tau)."""
        import jax.numpy as jnp

        build = infer({"twin": {"without": ["noise"]},
                       "parameters": PARAMS,
                       "noise": {**self.NOISE,
                                 "source": "prediction_at_init"}},
                      model=NOISY_MODEL)
        bound = build.space.bind(build.fit_twin,
                                 dict(build.space.initial_values()))
        expected = jnp.abs(bound(state()).data) * 0.25  # 1/sqrt(4 Hz * 4 s)
        assert jnp.allclose(build.noise.sigma, expected)

    def test_source_observed_reads_the_primary_among_several(self):
        """Several named observations: the sigma is decided from the entry
        NAMED primary, not whichever the document happened to list first."""
        import jax.numpy as jnp

        build = infer({"parameters": PARAMS, "noise": self.NOISE,
                       "observed": {"other": {"from": "simulation"},
                                    "primary": {"from": "simulation",
                                                "at": {"g": 1.5}}}})
        assert build.observed.primary == "primary"
        expected = jnp.abs(build.observed.entries["primary"]) * 0.25
        decoy = jnp.abs(build.observed.entries["other"]) * 0.25
        assert jnp.allclose(build.noise.sigma, expected)
        assert not jnp.allclose(build.noise.sigma, decoy)

    #: NOT ``TWO_OBSERVED``.  ``exit_helpers.py`` binds a module-level
    #: ``TWO_OBSERVED`` with DIFFERENT contents (it carries ``parameters`` and
    #: ``noise`` as well), and two names for two things one section apart is
    #: the HOMOSCEDASTIC shape 2C paid for.  Nothing shadows today -- this
    #: module imports nothing from ``exit_helpers`` -- and the name is
    #: different anyway so that the day it does, it is a NameError and not a
    #: wrong array.
    OBSERVED_PAIR = {"primary": {"from": "simulation", "at": {"g": 1.5}},
                     "night": {"from": "simulation", "at": {"g": 3.0}}}

    def test_source_observed_freezes_one_sigma_per_observation(self):
        """Two observations, two sigmas, each decided from its OWN data.

        `g` scales the whole prediction, so `night` at 3.0 against the
        primary's 1.5 is exactly twice the primary's data in every channel
        (measured) and so is its sigma.  A build that froze once and copied
        the array under both names passes the first assertion and fails the
        last, which is why the last one is there.
        """
        import jax.numpy as jnp

        build = infer({"parameters": PARAMS, "noise": self.NOISE,
                       "observed": self.OBSERVED_PAIR})
        entries = build.observed.entries
        per = build.noise.by_observation
        assert sorted(per) == ["night", "primary"]
        for name in ("primary", "night"):
            assert jnp.allclose(per[name], jnp.abs(entries[name]) * 0.25)
        assert jnp.allclose(per["night"], 2.0 * per["primary"])
        assert not jnp.allclose(per["night"], per["primary"])

    def test_prediction_at_init_freezes_one_sigma_and_fans_nothing(self):
        """The other source reads the TWIN, so there is nothing to fan.

        Its reference is the prediction at the declared inits -- g = 1.0 --
        which is NEITHER observation's data: two thirds of the primary's
        magnitude and one third of night's, because both were simulated from
        it.  A fan written to freeze off `observed.entries` whatever the
        source says would silently move this sigma onto the data, and the two
        divisions below are what catch that.  Measured: 0.12373096 against
        the primary's own 0.18559645 at channel [0, 4].
        """
        import jax.numpy as jnp

        build = infer({"parameters": PARAMS,
                       "noise": {**self.NOISE,
                                 "source": "prediction_at_init"},
                       "observed": self.OBSERVED_PAIR})
        entries = build.observed.entries
        assert build.noise.by_observation is None
        assert build.noise.sigma is not None
        assert jnp.allclose(build.noise.sigma,
                            jnp.abs(entries["primary"]) * 0.25 / 1.5)
        assert jnp.allclose(build.noise.sigma,
                            jnp.abs(entries["night"]) * 0.25 / 3.0)
        assert not jnp.allclose(build.noise.sigma,
                                jnp.abs(entries["primary"]) * 0.25)

    def test_several_observations_with_no_primary_are_still_refused(self):
        """The fan does not make a primary optional.

        `sigma` is still the primary's -- it is what `decided_noise` answers
        with when no run is in hand -- so a document with several
        observations and no entry named `primary` has no default to freeze,
        and the refusal it already gets stands.  Recorded here because "the
        sigma is now per observation" reads exactly like a licence to drop
        it.
        """
        with pytest.raises(ConfigError, match="or several with no primary"):
            infer({"parameters": PARAMS, "noise": self.NOISE,
                   "observed": {"day": {"from": "simulation"},
                                "night": {"from": "simulation"}}})


class TestChecks:
    def test_modes_and_reasons(self):
        build = infer({"checks": {
            "identifiability": {"mode": "refuse", "rtol": 1.0e-8,
                                "report": True},
            "linearity": {"mode": "refuse"},
            "prior_sensitivity": {"mode": "skip", "reason": "campaign"}}})
        assert build.checks["identifiability"] == CheckSpec(
            mode="refuse", report=True, reason=None, rtol=1.0e-8)
        assert build.checks["prior_sensitivity"].reason == "campaign"

    def test_skip_without_its_reason_is_check_a37(self):
        with pytest.raises(ConfigError, match="reason"):
            infer({"checks": {"linearity": {"mode": "skip"}}})

    def test_an_unknown_check_or_mode_is_refused(self):
        with pytest.raises(ConfigError, match="identifiability"):
            infer({"checks": {"stationarity": {"mode": "warn"}}})
        with pytest.raises(ConfigError, match="report"):
            infer({"checks": {"linearity": {"mode": "sometimes"}}})

    def test_rtol_belongs_to_identifiability_alone(self):
        with pytest.raises(ConfigError, match="rtol"):
            infer({"checks": {"linearity": {"mode": "warn", "rtol": 1e-8}}})


class TestTrainable:
    def test_leaves_compile_to_a_filter_spec(self):
        import equinox as eqx

        build = infer({"trainable": {"leaves": ["gain.gain"]}})
        params, _ = eqx.partition(build.fit_twin, build.trainable)
        import jax

        assert len([x for x in jax.tree.leaves(params) if x is not None]) == 1

    def test_nodes_take_every_inexact_leaf_under_them(self):
        import equinox as eqx
        import jax

        build = infer({"trainable": {"nodes": ["global_signal"]}})
        params, _ = eqx.partition(build.fit_twin, build.trainable)
        assert len([x for x in jax.tree.leaves(params)
                    if x is not None]) == 3

    def test_all_true_is_every_inexact_array(self):
        import equinox as eqx

        build = infer({"trainable": {"all": True}})
        assert build.trainable is eqx.is_inexact_array

    def test_all_true_with_a_subset_is_a_contradiction(self):
        with pytest.raises(ConfigError, match="all"):
            infer({"trainable": {"all": True, "leaves": ["gain.gain"]}})

    def test_routes_resolve_against_the_repaired_twin(self):
        """nodes: and leaves: compile on the FIT twin -- a spec built over
        the full twin would not even share the repaired tree's structure."""
        import equinox as eqx
        import jax

        build = infer({"twin": {"without": ["noise"]},
                       "trainable": {"nodes": ["global_signal"],
                                     "leaves": ["gain.gain"]}},
                      model=NOISY_MODEL)
        params, _ = eqx.partition(build.fit_twin, build.trainable)
        assert len([x for x in jax.tree.leaves(params)
                    if x is not None]) == 4

    def test_an_unknown_node_or_leaf_fails_fast(self):
        with pytest.raises(KeyError, match="rfi_field"):
            infer({"trainable": {"nodes": ["rfi_field"]}})
        with pytest.raises(ConfigError):
            infer({"trainable": {"leaves": ["gain.n_bits"]}})


class TestTruth:
    def test_at_wins_and_identity_leaves_derive(self):
        build = infer({
            "parameters": {**PARAMS,
                           "d": {"init": 0.1,
                                 "into": "global_signal.depth"}},
            "observed": {"from": "simulation", "at": {"g": 1.5}},
        })
        assert float(build.truth["g"]) == pytest.approx(1.5)
        assert float(build.truth["d"]) == pytest.approx(0.5)  # the leaf value

    def test_a_transformed_latent_is_omitted_with_its_reason(self):
        build = infer({
            "parameters": {"log_g": {"init": 0.0, "into": "gain.gain",
                                     "transform": "exp"}},
            "observed": {"from": "simulation"},
        })
        assert "log_g" not in build.truth
        assert "transform" in build.truth_omitted["log_g"]

    def test_a_fanned_identity_latent_is_omitted_with_the_fan_reason(self):
        """One latent tied identically into two leaves: no single leaf holds
        its truth, and the omission says so rather than blaming transform
        None."""
        build = infer({
            "parameters": {"d": {"init": 0.5,
                                 "into": ["global_signal.depth",
                                          "gain.gain"]}},
            "observed": {"from": "simulation"},
        })
        assert "d" not in build.truth
        assert "several leaves" in build.truth_omitted["d"]

    def test_the_truth_section_overrides_everything(self):
        build = infer({
            "parameters": PARAMS,
            "observed": {"from": "simulation", "at": {"g": 1.5}},
            "truth": {"g": 1.7},
        })
        assert float(build.truth["g"]) == pytest.approx(1.7)

    def test_a_truth_override_clears_the_omission_record(self):
        """A latent omitted for its transform stops being omitted the moment
        truth: declares it -- one name never sits in both dicts."""
        build = infer({
            "parameters": {"log_g": {"init": 0.1, "into": "gain.gain",
                                     "transform": "exp"}},
            "observed": {"from": "simulation"},
            "truth": {"log_g": 0.3},
        })
        assert float(build.truth["log_g"]) == pytest.approx(0.3)
        assert "log_g" not in build.truth_omitted

    def test_file_data_derives_no_truth(self):
        build = infer({"parameters": PARAMS})
        assert build.truth == {}
