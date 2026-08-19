"""inference.npe: the grammar (schema 4.7.10). No run is executed here.

The module seam 2C settled and 2D keeps: "calls ``run_document`` vs.
doesn't". Nothing in this file does. Most tests drive :func:`parse_npe`
directly, because that is its whole contract; the last class drives
``build_inference`` instead, because the SEAM -- where the parse sits in the
sequence, and what it lands on -- is not visible from the parser alone.

``kind: npe`` is still refused by name; this section is grammar only, and a
document may declare it with nothing consuming it. That is 2B's and 2C's
pattern (grammar before exit) and it is deliberate.

**Every refusal in ``npe.py`` is a sibling of eleven others**, and several
differ only in their ``inference.npe.<sub>[.<key>]:`` prefix -- four
subsections give the same "is required" sentence, three keys give the same
"is greater than zero", three more the same "is in [0, 1)". Every ``match=``
below therefore carries the prefix. A ``match="is required"`` here would be
satisfied by any of five branches, which is 2C's shape 1.
"""

import inspect

import pytest

from rheplicant.config import ConfigError
from rheplicant.config.sections.inference import InferenceBuild
from rheplicant.config.sections.npe import (
    _BANK_KEYS,
    _CREATE_KEYS,
    _CREATE_OPTIONS,
    _SAMPLE_KEYS,
    _TRAIN_KEYS,
    _TRAIN_OPTIONS,
    _TRANSLATED,
    NpeSpec,
    parse_npe,
)
from tests.config.inference_helpers import context
from tests.config.test_config_section_inference import PARAMS, infer

#: The schema's own example (4.7.10) at this plan's sizes -- 0.1 of the
#: bank, 1/60 of the steps. Nothing here runs, but a document a reader
#: copies out of a test should be one they can afford to run.
SECTION = {
    "bank": {"n_simulations": 64,
             "seed": {"from": "runtime.seeds.npe_bank"}},
    "create": {"n_components": 1, "width": 16, "depth": 2,
               "min_scale": 1.0e-3,
               "seed": {"from": "runtime.seeds.npe_create"}},
    "train": {"n_steps": 50, "batch_size": 16, "learning_rate": 1.0e-3,
              "validation_fraction": 0.25, "beta1": 0.9, "beta2": 0.999,
              "eps": 1.0e-8,
              "seed": {"from": "runtime.seeds.npe_train"}},
    "sample": {"n_draws": 100,
               "seed": {"from": "runtime.seeds.npe_sample"}},
}

#: The four seeds, minimal: every subsection with its seed and its required
#: count and nothing else, so a test that adds ONE key is testing that key.
BARE = {
    "bank": {"n_simulations": 64,
             "seed": {"from": "runtime.seeds.npe_bank"}},
    "create": {"seed": {"from": "runtime.seeds.npe_create"}},
    "train": {"seed": {"from": "runtime.seeds.npe_train"}},
    "sample": {"n_draws": 8,
               "seed": {"from": "runtime.seeds.npe_sample"}},
}


def section(**edits):
    """:data:`BARE` with one subsection replaced or extended.

    ``section(create={"width": 0})`` MERGES over ``BARE["create"]``, so the
    seed survives and the test is about ``width``. ``section(create=None)``
    deletes the subsection.
    """
    built = {name: dict(spec) for name, spec in BARE.items()}
    for name, spec in edits.items():
        if spec is None:
            built.pop(name, None)
        elif isinstance(spec, dict) and name in built:
            built[name] = {**built[name], **spec}
        else:
            built[name] = spec
    return built


def parsed(**edits):
    return parse_npe(section(**edits), context())


class TestTheGrammarMatchesTheSignatures:
    """The most valuable test in this task: the grammar, checked mechanically.

    A key in this grammar that the package call does not accept is a
    ``TypeError`` on the first document that declares it, in Task 7 or 8 --
    two commits and a whole executor away from the line that introduced it.
    Checking the two by eye is how ``width_size:`` gets written down: 2C's
    own notes say "create's width=64 maps to equinox's width_size=", which is
    true of what ``create`` does INTERNALLY (npe.py:216) and false as an
    instruction to its caller.

    Kills: any grammar key that is not a parameter of its call; a rename the
    layer performs and does not declare; an optional key the package requires
    and a required key the package defaults.
    """

    def calls(self):
        """The four package entry points, imported inside the test.

        Not at module scope. Measured: after ``import rheplicant.config``,
        ``rheplicant.inference`` is absent from ``sys.modules``; importing
        it here would make collecting this file the thing that decides
        whether the layer's import invariant can be measured at all.
        """
        from rheplicant.inference.npe import (
            NeuralPosterior,
            simulate_pairs,
            train_posterior,
        )

        return {"bank": (_BANK_KEYS, simulate_pairs),
                "create": (_CREATE_KEYS, NeuralPosterior.create),
                "train": (_TRAIN_KEYS, train_posterior),
                "sample": (_SAMPLE_KEYS, NeuralPosterior.sample)}

    @staticmethod
    def _split(fn):
        """``(every parameter, the required ones, the defaulted ones)``."""
        params = inspect.signature(fn).parameters
        required = {name for name, p in params.items()
                    if p.default is p.empty}
        return set(params), required, set(params) - required

    def test_the_translation_table_is_these_two_and_no_others(self):
        # The literal is REPEATED here rather than imported, deliberately.
        # Every other test in this class reads _TRANSLATED to decide what a
        # key means; if a future author "fixed" a signature mismatch by
        # adding {"width": "width_size"} to that table, every one of them
        # would go green again. This is the assertion that does not.
        assert _TRANSLATED == {"seed": "key", "n_draws": "n_samples"}

    @pytest.mark.parametrize("subsection",
                             ["bank", "create", "train", "sample"])
    def test_every_key_is_a_parameter_of_the_call_it_feeds(self, subsection):
        keys, fn = self.calls()[subsection]
        names, _, _ = self._split(fn)
        forwarded = {_TRANSLATED.get(key, key) for key in keys}
        assert forwarded <= names, (
            f"inference.npe.{subsection}: offers {sorted(forwarded - names)}, "
            f"which {fn.__qualname__} does not accept"
        )

    def test_embed_is_a_parameter_of_create_and_not_of_the_others(self):
        # embed: is a TOP-LEVEL key of inference.npe:, so no subsection sweep
        # covers it; without this the one key that is not in any _*_KEYS set
        # is the one key nothing checks.
        calls = self.calls()
        _, _, optional = self._split(calls["create"][1])
        assert "embed" in optional
        for name in ("bank", "train", "sample"):
            everything, _, _ = self._split(calls[name][1])
            assert "embed" not in everything

    def test_create_takes_width_and_has_no_width_size_at_all(self):
        # Drift 1 of the plan's opening notes, as a regression test. A
        # grammar spelling this key width_size: passes the sweep, passes
        # every parse test, and TypeErrors in Task 7.
        from rheplicant.inference.npe import NeuralPosterior

        everything, _, _ = self._split(NeuralPosterior.create)
        assert "width" in everything
        assert "width_size" not in everything

    def test_sample_takes_n_samples_so_n_draws_must_be_translated(self):
        # The rename is real, not decorative: an executor that forwarded
        # n_draws= straight through would TypeError. Asserting BOTH halves,
        # because "n_samples is a parameter" alone is equally true of a
        # signature that also accepts n_draws.
        from rheplicant.inference.npe import NeuralPosterior

        everything, _, _ = self._split(NeuralPosterior.sample)
        assert "n_samples" in everything
        assert "n_draws" not in everything

    def test_the_optional_keys_are_exactly_the_ones_the_package_defaults(self):
        # The rule "config keys never restate package defaults" only means
        # anything if the keys this layer treats as optional really are the
        # ones with defaults. _CREATE_OPTIONS/_TRAIN_OPTIONS are what
        # _passthrough forwards.
        calls = self.calls()
        _, _, create_optional = self._split(calls["create"][1])
        _, _, train_optional = self._split(calls["train"][1])
        # EQUALITY, not <=.  The name of this test claims both directions
        # and `<=` is only one: measured, dropping `min_scale` from
        # `_CREATE_OPTIONS` leaves `<=` green in BOTH assertions (the SECTION
        # round-trip test catches it elsewhere, which is not the same as this
        # test catching it).  `embed` is the one optional parameter excluded,
        # because it travels on its own rather than through `_passthrough`;
        # `key` is keyword-only with NO default, so it is in `required` and
        # not here at all.
        assert set(_CREATE_OPTIONS) == create_optional - {"embed"}
        assert set(_TRAIN_OPTIONS) == train_optional

    def test_the_required_keys_are_the_ones_the_package_does_not_default(self):
        # The other direction, and the reason n_simulations:/n_draws:/every
        # seed: are required rather than optional.
        calls = self.calls()
        _, bank_required, _ = self._split(calls["bank"][1])
        _, sample_required, _ = self._split(calls["sample"][1])
        _, create_required, _ = self._split(calls["create"][1])
        _, train_required, _ = self._split(calls["train"][1])
        assert {"n_simulations", "key"} <= bank_required
        assert {"n_samples", "key"} <= sample_required
        assert "key" in create_required and "key" in train_required

    def test_the_silent_embed_is_the_packages_own_default_object(self):
        # NpeSpec.embed is jnp.ravel when the document is silent, which reads
        # like a restated default and is not one: it IS the object in
        # create's signature, so passing it and omitting it are the same
        # call. `is`, not `==`: two jnp.ravel wrappers would compare equal
        # and be different jit cache entries.
        from rheplicant.inference.npe import NeuralPosterior

        default = inspect.signature(
            NeuralPosterior.create).parameters["embed"].default
        assert parsed().embed is default


class TestTheSectionParses:
    def test_the_schema_shape_parses_and_every_declared_key_survives(self):
        spec = parse_npe(SECTION, context())
        assert isinstance(spec, NpeSpec)
        assert spec.bank["n_simulations"] == 64
        assert spec.create["n_components"] == 1
        assert spec.create["width"] == 16
        assert spec.create["depth"] == 2
        assert spec.create["min_scale"] == pytest.approx(1.0e-3)
        assert spec.train["n_steps"] == 50
        assert spec.train["batch_size"] == 16
        assert spec.train["learning_rate"] == pytest.approx(1.0e-3)
        assert spec.train["validation_fraction"] == pytest.approx(0.25)
        assert spec.train["beta1"] == pytest.approx(0.9)
        assert spec.train["beta2"] == pytest.approx(0.999)
        assert spec.train["eps"] == pytest.approx(1.0e-8)
        assert spec.sample["n_draws"] == 100

    def test_omitted_keys_receive_the_executed_package_defaults(self):
        spec = parsed(create={"width": 32})
        assert spec.create == {
            "seed": {"from": "runtime.seeds.npe_create"},
            "n_components": 4,
            "width": 32,
            "depth": 3,
            "min_scale": 1e-3,
        }
        assert spec.train == {
            "seed": {"from": "runtime.seeds.npe_train"},
            "n_steps": 3000,
            "batch_size": 256,
            "learning_rate": 1e-3,
            "validation_fraction": 0.1,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
        }

    def test_counts_arrive_as_ints_and_rates_as_floats(self):
        # int(2) and 2.0 reach a package call identically; 2 and 2.5 do not,
        # and _whole is what stands between them.
        spec = parse_npe(SECTION, context())
        assert isinstance(spec.bank["n_simulations"], int)
        assert isinstance(spec.sample["n_draws"], int)
        assert isinstance(spec.create["min_scale"], float)
        assert isinstance(spec.train["validation_fraction"], float)

    def test_the_section_itself_must_be_a_mapping(self):
        with pytest.raises(ConfigError,
                           match=r"inference\.npe: is a mapping with bank:"):
            parse_npe(["bank"], context())

    def test_an_unknown_top_level_key_is_swept(self):
        with pytest.raises(ConfigError,
                           match=r"inference\.npe: the npe section does not "
                                 r"take \['embedding'\]"):
            parse_npe({**BARE, "embedding": "ravel"}, context())

    @pytest.mark.parametrize("missing",
                             ["bank", "create", "train", "sample"])
    def test_each_of_the_four_subsections_is_required(self, missing):
        # Four sibling refusals whose only difference is the prefix; the
        # match carries it. Parametrized rather than written once, because
        # "the subsection is required" is exactly the branch an
        # implementation is most likely to write for one and forget for
        # three -- and three of the four are the ones nothing else reaches.
        with pytest.raises(ConfigError,
                           match=rf"inference\.npe\.{missing}: is required\."):
            parsed(**{missing: None})

    @pytest.mark.parametrize("subsection",
                             ["bank", "create", "train", "sample"])
    def test_each_subsection_must_be_a_mapping(self, subsection):
        with pytest.raises(ConfigError,
                           match=rf"inference\.npe\.{subsection}: is a "
                                 r"mapping; got 4"):
            parsed(**{subsection: 4})


class TestTheFourSeeds:
    """A29's last member: four calls, four PRNG keys, four named seeds.

    ``_seed_name``'s three messages are prefixed by the ``form`` argument and
    are otherwise word-for-word identical across the four subsections, so a
    ``match=`` that omits the prefix passes under any of them -- and an
    implementation that checked ONE seed and reused it for all four would go
    green on every one.
    """

    def test_the_four_seeds_are_carried_unresolved_and_stay_distinct(self):
        # The thesis of the subsection layout: four INDEPENDENT names. An
        # implementation that read a single seed and copied it into all four
        # dicts satisfies every refusal test in this class and fails here.
        # Unresolved, too: parse_npe reads a document, and a key is a draw.
        spec = parse_npe(SECTION, context())
        seeds = [spec.bank["seed"], spec.create["seed"],
                 spec.train["seed"], spec.sample["seed"]]
        assert seeds == [{"from": "runtime.seeds.npe_bank"},
                         {"from": "runtime.seeds.npe_create"},
                         {"from": "runtime.seeds.npe_train"},
                         {"from": "runtime.seeds.npe_sample"}]
        assert len({entry["from"] for entry in seeds}) == 4

    def test_no_seed_is_resolved_to_a_key_at_parse_time(self):
        # The context declares no runtime.seeds at all, and every one of
        # these four names would still RESOLVE (blake2s off the root seed).
        # That it does not is the point: a draw belongs to the run.
        assert context().seeds == {}
        spec = parse_npe(SECTION, context())
        for held in (spec.bank, spec.create, spec.train, spec.sample):
            assert isinstance(held["seed"], dict)

    @pytest.mark.parametrize("subsection",
                             ["bank", "create", "train", "sample"])
    def test_a_missing_seed_is_refused_naming_its_own_subsection(
            self, subsection):
        spec = dict(BARE[subsection])
        spec.pop("seed")
        with pytest.raises(ConfigError,
                           match=rf"inference\.npe\.{subsection}: 'seed' is "
                                 r"required and has no default"):
            parse_npe({**section(), subsection: spec}, context())

    @pytest.mark.parametrize("subsection",
                             ["bank", "create", "train", "sample"])
    def test_a_literal_seed_is_refused_naming_its_own_subsection(
            self, subsection):
        with pytest.raises(ConfigError,
                           match=rf"inference\.npe\.{subsection}: seed must "
                                 r"NAME an entry of runtime\.seeds"):
            parsed(**{subsection: {"seed": 11}})

    @pytest.mark.parametrize("subsection",
                             ["bank", "create", "train", "sample"])
    def test_a_seed_outside_the_namespace_is_refused_by_subsection(
            self, subsection):
        with pytest.raises(ConfigError,
                           match=rf"inference\.npe\.{subsection}: seed names "
                                 r"'seeds\.oops'; it must be under "
                                 r"runtime\.seeds\."):
            parsed(**{subsection: {"seed": {"from": "seeds.oops"}}})


class TestTheCacheKeyBelongsToPlan4:
    """Schema 4.7.10 shows ``bank: {..., cache: {file: {...}}}``; it does not
    ship. See the task's Executor's note -- the decision is recorded and
    overridable."""

    def test_cache_is_refused_by_name_and_the_message_says_plan_4(self):
        with pytest.raises(ConfigError,
                           match=r"inference\.npe\.bank: the bank does not "
                                 r"take \['cache'\].*Plan 4"):
            parsed(bank={"cache": {"file": {"path": "bank.npz"}}})

    def test_width_size_is_answered_with_the_key_that_does_exist(self):
        # A reader who took 2C's own note literally writes width_size:. The
        # sweep alone would offer the five create: keys and let them guess;
        # the hint names width: and says why the two words both exist.
        with pytest.raises(ConfigError,
                           match=r"inference\.npe\.create: the estimator does "
                                 r"not take \['width_size'\].*the config key "
                                 r"is width:"):
            parsed(create={"width_size": 64})

    def test_n_samples_under_sample_is_answered_with_n_draws(self):
        # The other half of the rename: a reader who read the package's
        # signature writes n_samples:, and the sweep hands back n_draws.
        with pytest.raises(ConfigError,
                           match=r"inference\.npe\.sample: the draw does not "
                                 r"take \['n_samples'\]; it takes "
                                 r"\['n_draws', 'seed'\]"):
            parsed(sample={"n_samples": 8})


class TestTheRequiredCounts:
    def test_n_simulations_is_required_and_the_message_names_simulate_pairs(
            self):
        spec = {"seed": BARE["bank"]["seed"]}
        with pytest.raises(ConfigError,
                           match=r"inference\.npe\.bank\.n_simulations: is "
                                 r"required -- simulate_pairs"):
            parse_npe({**section(), "bank": spec}, context())

    def test_n_draws_is_required_and_the_message_names_sample(self):
        spec = {"seed": BARE["sample"]["seed"]}
        with pytest.raises(ConfigError,
                           match=r"inference\.npe\.sample\.n_draws: is "
                                 r"required -- NeuralPosterior\.sample"):
            parse_npe({**section(), "sample": spec}, context())

    @pytest.mark.parametrize("subsection,key", [("bank", "n_simulations"),
                                                ("sample", "n_draws")])
    def test_a_required_count_that_is_not_whole_is_refused(self, subsection,
                                                           key):
        # int(2.5) is 2, so a bank declared 2.5 used to RUN as 2: the
        # document says one thing and the run does another.
        with pytest.raises(ConfigError,
                           match=rf"inference\.npe\.{subsection}\.{key}: is "
                                 r"an integer >= 1; got 2\.5"):
            parsed(**{subsection: {key: 2.5}})

    @pytest.mark.parametrize("subsection,key", [("bank", "n_simulations"),
                                                ("sample", "n_draws")])
    def test_a_required_count_of_zero_is_refused(self, subsection, key):
        with pytest.raises(ConfigError,
                           match=rf"inference\.npe\.{subsection}\.{key}: is "
                                 r"an integer >= 1; got 0"):
            parsed(**{subsection: {key: 0}})

    @pytest.mark.parametrize("subsection,key", [("bank", "n_simulations"),
                                                ("sample", "n_draws")])
    def test_true_is_not_a_count(self, subsection, key):
        # bool is an int in Python, so `n_draws: true` would otherwise draw
        # once. _whole refuses bool first, and this is the leg of that guard
        # nothing else reaches.
        with pytest.raises(ConfigError,
                           match=rf"inference\.npe\.{subsection}\.{key}: is "
                                 r"an integer >= 1; got True"):
            parsed(**{subsection: {key: True}})


class TestTheOptionalNumbers:
    """One test per key per failure mode. A grammar test that checks the
    happy path plus one refusal leaves every other leg untested (2C's shape
    7), and these eleven keys are eleven legs.

    ELEVEN is ``len(_CREATE_OPTIONS) + len(_TRAIN_OPTIONS)`` -- 4 + 7 -- and
    the number is written here because an earlier draft said twelve. That
    was not a typo with no consequence: a reader auditing this class against
    its own claim counts to twelve, finds twelve-ish legs and stops, and the
    one key with no refusal leg at all (``depth``, whose floor of 0 reads
    like "anything goes") stayed uncovered behind the wrong total. An
    adversarial mutation found it -- ``parsed["depth"] = spec["depth"]``, a
    raw passthrough, left all 89 tests green.
    """

    @pytest.mark.parametrize("subsection,key",
                             [("create", "n_components"),
                              ("create", "width"),
                              ("train", "n_steps"),
                              ("train", "batch_size")])
    def test_the_counts_are_whole_and_at_least_one(self, subsection, key):
        with pytest.raises(ConfigError,
                           match=rf"inference\.npe\.{subsection}\.{key}: is "
                                 r"an integer >= 1; got 0"):
            parsed(**{subsection: {key: 0}})
        with pytest.raises(ConfigError,
                           match=rf"inference\.npe\.{subsection}\.{key}: is "
                                 r"an integer >= 1; got 1\.5"):
            parsed(**{subsection: {key: 1.5}})

    def test_width_zero_is_refused_although_equinox_accepts_it(self):
        # Measured: eqx.nn.MLP(width_size=0, depth=2) builds, and two inputs
        # 100 apart come back as the identical vector -- there is no path
        # from input to output. The estimator still trains and still samples;
        # what it reports is the prior. width_size=1 is NOT structurally
        # dead (measured: it separates those inputs at key 2), so 1 is the
        # right floor and 0 is the refusal.
        with pytest.raises(ConfigError,
                           match=r"inference\.npe\.create\.width: is an "
                                 r"integer >= 1; got 0"):
            parsed(create={"width": 0})

    def test_depth_zero_is_accepted_because_equinox_accepts_it(self):
        # The other side of the same boundary, and the one that stops the
        # floor above from being copied onto depth: measured, depth=0 builds
        # a single linear layer and evaluates. A layer that refused it would
        # reject a document the package runs.
        assert parsed(create={"depth": 0}).create["depth"] == 0

    @pytest.mark.parametrize("value,shown", [(-1, r"-1"), (1.5, r"1\.5"),
                                             (True, r"True")])
    def test_depth_is_still_checked_although_its_floor_is_zero(self, value,
                                                               shown):
        # The leg the acceptance test above cannot supply. A floor of 0
        # reads like "anything goes", and both of depth's other tests assert
        # ACCEPTANCE, so neither can tell _whole(..., 0) from a raw
        # passthrough -- measured, `parsed["depth"] = spec["depth"]` leaves
        # this whole module green without this test.
        #
        # What reaches equinox unguarded, measured on eqx.nn.MLP:
        #   -1     IndexError: index is out of bounds for axis 0 with size 0
        #   2.5    TypeError: 'float' object cannot be interpreted as an int
        #   "deep" TypeError: can only concatenate str (not "int") to str
        #   True   BUILDS SILENTLY, two layers -- identical to depth: 1,
        #          because bool is an int in Python. That is the one this
        #          leg exists for: the other three are loud.
        with pytest.raises(ConfigError,
                           match=rf"inference\.npe\.create\.depth: is an "
                                 rf"integer >= 0; got {shown}"):
            parsed(create={"depth": value})

    @pytest.mark.parametrize("subsection,key",
                             [("create", "min_scale"),
                              ("train", "learning_rate"),
                              ("train", "eps")])
    def test_the_positive_reals_refuse_zero_and_negatives(self, subsection,
                                                          key):
        for value in (0.0, -1.0):
            with pytest.raises(ConfigError,
                               match=rf"inference\.npe\.{subsection}\.{key}: "
                                     r"is greater than zero"):
                parsed(**{subsection: {key: value}})

    @pytest.mark.parametrize("key",
                             ["validation_fraction", "beta1", "beta2"])
    def test_the_fractions_refuse_one_and_above_and_negatives(self, key):
        for value in (1.0, 1.5, -0.1):
            with pytest.raises(ConfigError,
                               match=rf"inference\.npe\.train\.{key}: is in "
                                     r"\[0, 1\)"):
                parsed(train={key: value})

    def test_validation_fraction_zero_is_accepted(self):
        # The package's own guard is `0.0 <= x < 1.0` and 0.0 is the
        # documented "train on everything" setting (npe.py:328-331). A
        # config layer that refused it would reject the faster path the
        # package offers on purpose.
        assert parsed(train={"validation_fraction": 0.0}
                      ).train["validation_fraction"] == 0.0

    @pytest.mark.parametrize("subsection,key",
                             [("create", "min_scale"),
                              ("train", "learning_rate"),
                              ("train", "eps"),
                              ("train", "validation_fraction"),
                              ("train", "beta1"),
                              ("train", "beta2")])
    def test_a_real_that_is_not_a_number_is_refused(self, subsection, key):
        with pytest.raises(ConfigError,
                           match=rf"inference\.npe\.{subsection}\.{key}: is a "
                                 r"number; got 'fast'"):
            parsed(**{subsection: {key: "fast"}})

    @pytest.mark.parametrize("subsection,key",
                             [("create", "min_scale"),
                              ("train", "learning_rate"),
                              ("train", "beta1")])
    def test_true_is_not_a_real_either(self, subsection, key):
        # bool passes isinstance(x, int), so True would otherwise be 1.0 --
        # a legal beta1 and a legal learning_rate. The leg exists for that.
        with pytest.raises(ConfigError,
                           match=rf"inference\.npe\.{subsection}\.{key}: is a "
                                 r"number; got True"):
            parsed(**{subsection: {key: True}})

    def test_an_unknown_key_in_each_subsection_is_swept(self):
        # The LABEL is asserted, not skipped over with `.*`. Three of the
        # four are pinned by the hint tests above; "training" is pinned
        # nowhere else, and a `.*` here sits exactly where the label goes --
        # measured, label="training" -> label="the bank" left this module
        # green while a stray key under train: was answered in the bank's
        # vocabulary.
        for subsection, label, stray in (("bank", "the bank", "n_sims"),
                                         ("create", "the estimator",
                                          "n_layers"),
                                         ("train", "training", "lr"),
                                         ("sample", "the draw", "draws")):
            with pytest.raises(ConfigError,
                               match=rf"inference\.npe\.{subsection}: {label} "
                                     rf"does not take \['{stray}'\]"):
                parsed(**{subsection: {stray: 1}})


class TestTheEmbedding:
    """``embed:`` resolves to a callable HERE, and that is the whole point.

    A ``{python:}`` the process cannot import, or cannot call the way
    ``jax.vmap(embed)(data)`` calls it (npe.py:209), is refused when the
    document is read -- not after ``simulate_pairs`` has spent the run's
    entire budget building a bank the estimator will never see.
    """

    def test_the_word_ravel_and_silence_both_give_jnp_ravel(self):
        import jax.numpy as jnp

        assert parsed().embed is jnp.ravel
        assert parsed(embed="ravel").embed is jnp.ravel

    def test_a_python_target_resolves_to_the_imported_attribute(self):
        import jax.numpy as jnp

        assert parsed(embed={"python": "jax.numpy:ravel"}).embed is jnp.ravel

    def test_a_callable_that_cannot_take_one_datum_is_refused(self):
        # operator.add is importable, is a callable, and binds two
        # positional arguments. Without the probe it reaches the user as a
        # raw TypeError from inside jax.vmap, naming no section -- the same
        # shape gradient's objective: and optimize's loss: already guard.
        with pytest.raises(ConfigError,
                           match=r"inference\.npe\.embed: 'operator:add' "
                                 r"cannot be called as \(datum\)"):
            parsed(embed={"python": "operator:add"})

    def test_an_uncallable_target_is_refused_by_a_check_of_its_own(self):
        # THE LEG _binds CANNOT REACH. jnp.pi is a float; inspect describes
        # no signature for it, so _binds returns (True, None) and passes it
        # through -- by design, because guessing there would refuse working
        # jax wrappers. Measured: without the explicit callable() check this
        # target resolves cleanly and dies inside jax.vmap with a raw
        # TypeError naming no section. gradient's objective: and mmodes'
        # sky: have the same hole today (Plan 3's ledger).
        with pytest.raises(ConfigError,
                           match=r"inference\.npe\.embed: 'jax\.numpy:pi' is "
                                 r"a float and embed: takes a callable"):
            parsed(embed={"python": "jax.numpy:pi"})

    def test_a_missing_attribute_is_refused_by_the_hatch(self):
        # import_target speaks for itself here, exactly as it does for
        # gradient's objective: -- one seam, one message, and this test
        # exists so a later author does not wrap it in a second one.
        with pytest.raises(ConfigError,
                           match=r"'jax\.numpy' has no attribute 'flatten'"):
            parsed(embed={"python": "jax.numpy:flatten"})

    def test_an_unimportable_module_is_refused_by_the_hatch(self):
        with pytest.raises(ConfigError,
                           match=r"cannot import 'nosuchmodule'"):
            parsed(embed={"python": "nosuchmodule:fn"})

    def test_args_beside_python_is_refused_with_what_it_would_have_meant(
            self):
        # {python: "mod:factory", args: {...}} reads as reasonable and is
        # the hatch's spelling for CALLING the target. embed: hands over a
        # callable, so there is no spelling for a factory here, and the
        # refusal says that rather than "not a mapping".
        with pytest.raises(ConfigError,
                           match=r"inference\.npe\.embed: \['args'\] rides "
                                 r"beside python:"):
            parsed(embed={"python": "jax.numpy:ravel", "args": {}})

    def test_literal_beside_python_is_refused_the_same_way(self):
        # The twin leg. `args` and `literal` are siblings in the hatch and
        # a guard written for one is routinely left open on the other.
        with pytest.raises(ConfigError,
                           match=r"inference\.npe\.embed: \['literal'\] rides "
                                 r"beside python:"):
            parsed(embed={"python": "jax.numpy:ravel", "literal": {}})

    @pytest.mark.parametrize("node", ["flatten", 4, ["ravel"],
                                      {"ref": "resources.arrays.embed"}])
    def test_anything_else_is_refused_naming_the_two_forms(self, node):
        with pytest.raises(ConfigError,
                           match=r"inference\.npe\.embed: is 'ravel' or "
                                 r"\{python: 'mod:fn'\}"):
            parsed(embed=node)


class TestTheSeamInBuildInference:
    """Where the parse sits in ``build_inference``, and what it lands on.

    ``inference.py``'s docstring says its order is load-bearing. ``parse_npe``
    resolves no data, so it can go first -- and going first is what makes the
    section's refusals cheap: a document with a bad ``inference.npe:`` is
    refused before a twin is repaired, a space is built or a forward model is
    evaluated.
    """

    def test_the_section_now_parses_rather_than_being_deferred(self):
        # Replaces test_npe_is_plan_2d_by_name. `kind: npe` is STILL refused
        # -- Task 8 promotes it -- and this is the section, not the kind.
        build = infer({"npe": SECTION})
        assert isinstance(build, InferenceBuild)
        assert isinstance(build.npe, NpeSpec)
        assert build.npe.bank["n_simulations"] == 64

    def test_a_document_that_declares_no_npe_carries_none(self):
        assert infer(None).npe is None
        assert infer({"parameters": PARAMS}).npe is None

    def test_the_section_is_refused_before_the_twin_is_repaired(self):
        # The discriminator is the exception TYPE, not the text. A bad
        # `twin.without:` raises AssemblyError out of build_fit_twin, which
        # is not a ConfigError; a bad `npe:` raises ConfigError. Declare
        # both and the winner says which ran first -- so a parse_npe placed
        # after build_fit_twin fails this test loudly, and no rewording of
        # either message can make it pass.
        with pytest.raises(ConfigError,
                           match=r"inference\.npe\.bank\.n_simulations: is an "
                                 r"integer >= 1; got 0"):
            infer({"twin": {"without": ["nonesuch"]},
                   "npe": section(bank={"n_simulations": 0})})

    def test_the_inference_sweep_still_speaks_first(self):
        # check_unknown_keys("inference", ...) stays ahead of parse_npe: a
        # document that misspells a SIBLING section should be told about the
        # misspelling, not about npe's insides.
        with pytest.raises(ConfigError, match="observations"):
            infer({"observations": {}, "npe": section(sample=None)})

    def test_npe_is_a_defaulted_field_so_nothing_else_constructs_it(self):
        # InferenceBuild is a NamedTuple and gained a field; a field WITHOUT
        # a default would have made every construction site a TypeError.
        # There is exactly one, and this is the property that kept it that
        # way.
        #
        # POSITION-INDEPENDENT ON PURPOSE.  An earlier draft asserted
        # `_fields[-1] == "npe"` and `_field_defaults == {"npe": None}`, and
        # Task 5 appends a second defaulted field (`refs`, section 3.1), which
        # turns both into `AssertionError: assert 'refs' == 'npe'` in a task
        # whose Files list does not own this module.  The property this test
        # is FOR is "npe is defaulted", and that is what it now says.
        assert "npe" in InferenceBuild._field_defaults
        assert InferenceBuild._field_defaults["npe"] is None
