"""A26 and A49 -- ``preflight/noise.py``, and what they used to lose to.

**Every message here is pinned by EQUALITY on its whole text**, against a
literal written out in this file.  That is the point of the task and not a
style: measured before it, swapping A26's message with its mirror leg's
ENTIRELY left ``tests/config`` at exit 0, because the two shipped pins are
``match="axis"`` and ``match="1-D"`` and each matches both sentences.  Those
two pins stay green and unchanged in ``test_config_section_noise.py``; they
are simply not able to tell right from wrong on their own.

**Registry and report assertions are subset-shaped only.**  Five other wave-1
branches register checks that run on these same documents, so ``ids(doc) ==
frozenset({...})`` is green here and red after the merge.  Every "and nothing
else" statement is scoped to :data:`MINE` through :func:`silent_here`, which
is ``test_preflight_model.py:110``'s idiom.

**The build is still asked too.**  A hoist has three parts and the third is
that the section keeps calling the same function at build time (plan §2.2), so
several tests below assert the P-1 finding and the P2 refusal say the same
sentence -- and :class:`TestTheOneBinding` asserts that sentence is written in
exactly one module under ``src/``.
"""

import copy

import pytest

from rheplicant.config.document import load_document
from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import REFUSE
from rheplicant.config.preflight import CHECKS, preflight
from rheplicant.config.preflight.document import _task3_over_layers
from rheplicant.config.preflight.noise import (
    _a26_in,
    _a26_rank,
    _a49_in,
    _b6_is_layered,
    _b6_over_layers,
    _logdet,
    _sigma_axis,
)
from rheplicant.config.sections.noise import build_noise
from rheplicant.config.sections.observation import build_observation
from rheplicant.config.sections.runtime import build_runtime
from tests.config.inference_helpers import context
from tests.config.message_binding import assert_bound_once, modules_carrying
from tests.config.preflight_helpers import (
    UNREADABLE_BEAM,
    findings,
    ids,
    only,
    preflight_document,
)

#: The ids THIS module is about.  An "and nothing else" assertion over the
#: whole report would name this module the day any sibling task's check first
#: fires on one of these documents, which is a merge hazard rather than a
#: property of A26 or A49.
MINE = frozenset({"A26", "A49"})


# --- the documents ---------------------------------------------------------

#: A ``kind: radiometer`` block with its ``include_logdet`` REMOVED, which is
#: A49's required direction.  Built off ``exit_helpers.RADIOMETER`` rather than
#: written out, so a change to the shared block cannot leave this one behind
#: carrying a bandwidth nothing else uses.
RADIOMETER_NO_LOGDET = {"kind": "radiometer",
                        "channel_width": {"value": 1.0, "unit": "MHz"},
                        "integration_time": {"value": 2.0, "unit": "s"}}

#: A ``kind: radiometer_frozen`` block, complete -- and then given a key its
#: own kind does not take.  ``radiometer_frozen`` is S3's first named twin:
#: its sigma is decided from the DATA and is constant thereafter, so the term
#: ``include_logdet`` weighs is not prediction-dependent and the key is
#: refused rather than required.
FROZEN = {"kind": "radiometer_frozen", "source": "observed",
          "channel_width": {"value": 16.0, "unit": "Hz"},
          "integration_time": {"value": 1.0, "unit": "s"}}

#: A one-dimensional sigma, the spelling schema §6 writes.
ONE_D = {"kind": "homoscedastic", "sigma": {"ones": ["n_freq"], "unit": "K"}}

#: The same document with the fix its message names, in the place that works.
ONE_D_FIXED = {**ONE_D, "axis": "freq"}

#: ``flags:`` -- S3's second named twin.  ``FlaggedNoise`` FORWARDS
#: ``depends_on_prediction`` (``inference/noise.py``), so wrapping a model in
#: it moves no A49 row, and ``flags`` is a legal key for both kinds that carry
#: one.  Every ``flags:`` document below therefore expects exactly what its
#: unflagged partner expects.
FLAGS = {"from": "observation"}
AUX_FLAGS = {"aux": {"flags": {"zeros": ["n_time", "n_freq"]}}}


def with_noise(noise, **patch):
    """The base document with ``inference.noise`` replaced by ``noise``.

    ``preflight_document``'s merge is one level deep, so this keeps
    ``parameters:`` and ``observed:`` and replaces the noise block whole --
    which is what these two checks read and all they read.
    """
    return preflight_document(inference={"noise": noise}, **patch)


def with_variants(noise, variants):
    """``with_noise``, with ``variants:`` REPLACED rather than merged.

    ``preflight_document``'s merge is one level deep and the base fixture
    already declares a variant of its own (``unity_gain``), so a keyword patch
    ADDS to it.  Every test below that counts layers or asserts "no variants
    at all" needs the section it wrote and nothing else.
    """
    document = with_noise(noise)
    if variants is None:
        document.pop("variants", None)
    else:
        document["variants"] = variants
    return document


def observation(**sections):
    """8 channels by 16 samples, ``test_config_section_noise.py``'s own.

    A local builder rather than a shared one because this is the argument
    ``build_noise`` takes and not a document; ``inference_helpers.context``
    supplies the matching scope.
    """
    section = {
        "freq": {"grid": {"linspace": {"start": 60.0, "stop": 85.0, "num": 8,
                                       "endpoint": True}, "unit": "MHz"}},
        "time": {"grid": {"arange": {"start": 0.0, "step": 2.0, "num": 16},
                          "unit": "s"}},
        **sections,
    }
    build, _ = build_observation(section, runtime=build_runtime({"seed": 1}))
    return build


def built(noise, *, seeds=None, **sections):
    """``build_noise`` on one block -- the second opinion, at P2.

    ``seeds`` goes to the CONTEXT and ``**sections`` to the observation, which
    is where each of them lives: a ``{normal:}`` sigma names an entry of
    ``runtime.seeds`` and ``seed_for`` reads it off the context.
    """
    return build_noise(noise, observation=observation(**sections),
                       context=context(seeds=dict(seeds or {})))


def ids_here(document):
    return ids(document) & MINE


def silent_here(document):
    """Did this document earn nothing from THIS module's own two checks?"""
    return ids_here(document) == frozenset()


# --- the messages, whole ---------------------------------------------------
#
# Copied from `src/rheplicant/config/sections/noise.py`, which is where all
# three are written.  `TestTheOneBinding` is what asserts that "where they are
# written" is one place and not two.

A26_MESSAGE = (
    "inference.noise.sigma: is 1-D, and a 1-D sigma reads equally well along "
    "either axis of (n_time, n_freq) data; declare axis: time or axis: freq "
    "(check A26)."
)

A49_REQUIRED = (
    "inference.noise.include_logdet: is required for a prediction-dependent "
    "noise model and has no default. False is the documented GLS variant -- a "
    "DIFFERENT estimator, biased high by (1 + f^2) "
    "(inference/noise.py:56-68) -- and a lost declaration comes back True "
    "with no error."
)

A49_HINT = (
    "include_logdet is required exactly when the sigma depends on the "
    "prediction (kind: radiometer) and refused otherwise -- for a constant "
    "sigma it changes nothing (A49)"
)

A49_REFUSED_ON_FROZEN = (
    "inference.noise: kind: radiometer_frozen does not take "
    "['include_logdet']; it takes ['channel_width', 'floor', "
    "'integration_time', 'kind', 'source']. " + A49_HINT
)

A49_REFUSED_ON_HOMOSCEDASTIC = (
    "inference.noise: kind: homoscedastic does not take ['include_logdet']; "
    "it takes ['axis', 'flags', 'kind', 'sigma']. " + A49_HINT
)

#: The mirror leg, which is NOT hoisted: it interpolates the RESOLVED extents,
#: which belong to the axes slot.  Written out here so that
#: ``test_the_mirror_leg_was_not_hoisted`` can assert where it lives.
A26_MIRROR = ("inference.noise.axis: says how to read a 1-D sigma; this one "
              "has shape (16, 8).")

#: The more specific refusal A26 must not pre-empt.
AXIS_GRAMMAR = "inference.noise.axis: is none, time or freq; got 'sideways'."

#: The other one: a homoscedastic block with no sigma at all.
SIGMA_REQUIRED = ("inference.noise: kind: homoscedastic requires sigma: -- a "
                  "value node.")


class TestTheRowsArriveBeforeTheBeam:
    """The headline, and the only reason either check moved.

    ``build_noise`` runs inside ``build_inference``, the last builder
    ``load_document`` calls, so both of these lost to a beam file that does
    not exist.  Measured on this very fixture at ``ea4839b`` with the two
    checks unregistered: all three legs came back *"No file at
    'no_such_beam.npy'."*
    """

    def _refusal(self, noise):
        with pytest.raises(ConfigError) as caught:
            load_document(with_noise(noise, resources=UNREADABLE_BEAM))
        return str(caught.value)

    def test_a_one_d_sigma_is_reported_and_not_the_missing_beam(self):
        assert self._refusal(ONE_D) == A26_MESSAGE

    def test_a_missing_include_logdet_is_reported_and_not_the_beam(self):
        assert self._refusal(RADIOMETER_NO_LOGDET) == A49_REQUIRED

    def test_a_refused_include_logdet_is_reported_and_not_the_beam(self):
        assert self._refusal({**FROZEN, "include_logdet": True}) == (
            A49_REFUSED_ON_FROZEN)

    def test_the_beam_still_wins_when_the_noise_block_is_correct(self):
        """The anti-vacuity partner: without it every assertion above would
        pass against a fixture whose beam had quietly become readable, and
        three tests about PHASE would be testing nothing at all."""
        assert self._refusal(ONE_D_FIXED).startswith(
            "No file at 'no_such_beam.npy'.")


class TestA26TheOneDSigma:
    """A26: a 1-D sigma with no ``axis:`` beside it."""

    def test_the_finding_whole(self):
        found = only(with_noise(ONE_D), "A26")
        assert found.severity == REFUSE
        assert found.where == "inference.noise.sigma"
        assert found.message == A26_MESSAGE

    def test_the_build_says_the_same_sentence(self):
        """The hoist's third part (§2.2): the section keeps calling it.

        ``test_config_section_noise.py::test_a_1d_sigma_without_axis_is_check
        _a26`` still pins this with ``match="axis"`` and is unchanged; this is
        the equality the ``match=`` cannot express.
        """
        with pytest.raises(ConfigError) as caught:
            built(ONE_D)
        assert str(caught.value) == A26_MESSAGE

    #: Every spelling whose RANK the text carries.  ``linspace``/``arange``/
    #: ``modulo`` are unconditionally 1-D even when ``num`` is a shape symbol,
    #: which is why A26 is answered on rank and never on shape; ``from_grid``
    #: hands back an observation axis; and ``{value: [...]}`` and
    #: ``{list: [...]}`` are the two the plan names nowhere -- measured, both
    #: reach ``build_noise`` with ``ndim == 1``.
    ONE_D_SPELLINGS = {
        "ones": {"ones": ["n_freq"], "unit": "K"},
        "zeros": {"zeros": ["n_freq"], "unit": "K"},
        "full": {"full": {"shape": ["n_freq"], "value": 0.5}, "unit": "K"},
        "list": {"list": [0.1, 0.2, 0.3], "unit": "K"},
        "value_list": {"value": [0.1, 0.2], "unit": "K"},
        "linspace": {"linspace": {"start": 1.0, "stop": 2.0, "num": "n_freq",
                                  "endpoint": True}, "unit": "K"},
        "arange": {"arange": {"start": 1.0, "step": 1.0, "num": "n_freq"},
                   "unit": "K"},
        "modulo": {"modulo": {"num": "n_freq", "period": 2}, "unit": "K"},
        "from_grid": {"from_grid": "freq"},
        "normal": {"normal": {"shape": ["n_freq"], "seed": {"from":
                                                            "runtime.seeds.a"}}},
        "stack": {"stack": [{"value": 0.1}, {"value": 0.2}], "unit": "K"},
    }

    @pytest.mark.parametrize("spelling", sorted(ONE_D_SPELLINGS))
    def test_every_spelling_that_carries_a_rank_is_refused(self, spelling):
        """Kills the naive reading -- "only the array constructors carry a
        shape" -- which answers None for six of these eleven and lets each of
        them through to lose to the beam again."""
        document = with_noise({"kind": "homoscedastic",
                               "sigma": self.ONE_D_SPELLINGS[spelling]},
                              runtime={"seed": 7, "seeds": {"a": 3}})
        assert only(document, "A26").message == A26_MESSAGE

    #: Every spelling that is not 1-D, and three that stop being 1-D.
    #:
    #: The last four are the ones a ``shape:``-only reader gets wrong.  A draw
    #: is ``loc + scale * normal(key, shape)`` -- plain arithmetic, which
    #: BROADCASTS -- so an operand that is itself an array outranks the shape.
    #: ``drawn_with_a_2_D_loc`` is the measured REGRESSION: it LOADS at
    #: ``ea4839b`` and the first version of this module refused it.
    NOT_ONE_D = {
        "scalar": {"value": 0.5, "unit": "K"},
        "bare_number": 0.5,
        "shorthand": "0.5 K",
        "two_d_ones": {"ones": ["n_time", "n_freq"], "unit": "K"},
        "two_d_list": {"list": [[0.1, 0.2], [0.3, 0.4]], "unit": "K"},
        "column": {"ones": ["n_freq"], "unit": "K", "column": True},
        "stack_of_rows": {"stack": [{"ones": ["n_freq"]},
                                    {"ones": ["n_freq"]}], "unit": "K"},
        "drawn_with_a_2_D_loc": {"normal": {
            "shape": ["n_freq"], "seed": {"from": "runtime.seeds.a"},
            "loc": {"ones": ["n_time", "n_freq"]}}},
        "drawn_with_a_2_D_scale": {"normal": {
            "shape": ["n_freq"], "seed": {"from": "runtime.seeds.a"},
            "scale": {"ones": ["n_time", "n_freq"]}}},
        "drawn_with_a_2_D_low": {"uniform": {
            "shape": ["n_freq"], "seed": {"from": "runtime.seeds.a"},
            "low": {"ones": ["n_time", "n_freq"]}}},
        "filled_with_a_2_D_value": {"full": {"shape": ["n_freq"],
                                             "value": [[1.0, 2.0]]}},
    }

    @pytest.mark.parametrize("spelling", sorted(NOT_ONE_D))
    def test_a_sigma_that_is_not_one_d_earns_nothing(self, spelling):
        """``column: true`` is the one that is not obvious: it is applied LAST
        by ``modifiers.py`` and forces ``(n,)`` to ``(n, 1)``, so a check that
        read the form key and stopped would refuse a document that builds."""
        assert silent_here(with_noise(
            {"kind": "homoscedastic", "sigma": self.NOT_ONE_D[spelling]},
            runtime={"seed": 7, "seeds": {"a": 3}}))

    def test_a_column_sigma_really_does_build(self):
        """The anti-vacuity partner for the ``column`` row above: standing
        down would be right for the wrong reason if the document were refused
        for something else."""
        assert built({"kind": "homoscedastic",
                      "sigma": {"ones": ["n_freq"], "unit": "K",
                                "column": True}}).model.sigma.shape == (8, 1)

    def test_a_drawn_sigma_with_a_2_D_loc_really_does_build(self):
        """The anti-vacuity partner for the regression row, and the ONLY test
        that can see the fix.

        Measured: this document LOADS at ``ea4839b`` with a resolved sigma of
        shape ``(16, 8)`` -- ``draws.py`` builds a normal as
        ``loc + scale * jax.random.normal(key, shape)`` and the 2-D ``loc:``
        broadcasts the ``(8,)`` draw up.  Reading ``shape:`` as the rank made
        A26 refuse it, which is a check refusing a document the layer accepts,
        and applying A26's own remedy then earned the mirror leg -- a fifth
        advice loop on top of a false positive.

        Named for what it asserts rather than for the check, because the check
        must say NOTHING here; the ``(16, 8)`` is the whole evidence.
        """
        drawn = {"kind": "homoscedastic",
                 "sigma": self.NOT_ONE_D["drawn_with_a_2_D_loc"]}
        assert built(drawn, seeds={"a": 3}).model.sigma.shape == (16, 8)

    def test_a_drawn_sigma_with_a_scalar_loc_is_still_A26s(self):
        """The other side of the gate: an operand that IS rank 0 leaves
        ``shape:`` in charge, so the draw is still a 1-D sigma.

        Kills a fix that stood down on every ``normal:``/``uniform:``/``full:``
        rather than on the ones whose operands broadcast -- which would be the
        easy over-correction and would lose four of the eleven spellings above.
        The three spellings are the whole value grammar ``_resolve_operand``
        accepts for an operand: a bare number, the shorthand, and a value node.
        """
        for operand in (1.0, "1.0 K", {"value": 1.0}):
            sigma = {"normal": {"shape": ["n_freq"],
                                "seed": {"from": "runtime.seeds.a"},
                                "loc": operand}}
            document = with_noise({"kind": "homoscedastic", "sigma": sigma},
                                  runtime={"seed": 7, "seeds": {"a": 3}})
            assert only(document, "A26").message == A26_MESSAGE, operand


class TestA26StandsDownOnWhatTheTextCannotSee:
    """§3.2 (c): refusing on "I could not tell" refuses documents that build.

    ``{ref:}``, ``{file:}``, ``{from:}`` and ``{python:}`` carry no shape in
    the document's text.  A26 says nothing about them here and ``build_noise``
    still refuses the 1-D ones at P2, in the same words -- so the false
    negative is a LATE refusal and never a missed one.
    """

    OPAQUE = {
        "ref": {"ref": "resources.arrays.flat"},
        "file": {"file": {"path": "no_such_sigma.npy", "format": "npy"}},
        "python": {"python": "numpy:zeros", "args": [8]},
        "from": {"from": "channel_spacing"},
    }

    @pytest.mark.parametrize("spelling", sorted(OPAQUE))
    def test_a_sigma_whose_shape_the_text_cannot_see_says_nothing(
            self, spelling):
        assert silent_here(with_noise({"kind": "homoscedastic",
                                       "sigma": self.OPAQUE[spelling]}))

    def test_the_stand_down_is_capable_of_failing(self):
        """The same document with a spelling that DOES carry a rank fires.

        Without this the class above passes on a check that was deleted, on a
        check whose gate is inverted, and on a fixture that stopped reaching
        ``inference.noise`` at all.
        """
        assert ids_here(with_noise(ONE_D)) == frozenset({"A26"})

    def test_a_ref_to_a_one_d_array_is_still_refused_at_the_build(self):
        """What the stand-down COSTS, measured rather than asserted in prose:
        the document is refused, by the same sentence, one phase later."""
        with pytest.raises(ConfigError) as caught:
            load_document(with_noise(
                {"kind": "homoscedastic",
                 "sigma": {"ref": "resources.arrays.flat"}}))
        assert str(caught.value) == A26_MESSAGE


class TestA26DoesNotPreEmpt:
    """S4's first half: a document wrong in A26's way AND wrong in a way
    something else says better."""

    def test_an_axis_outside_the_three_hears_the_grammar_instead(self):
        """Kills the naive gate ``if axis not in ("time", "freq")``.

        That implementation fires A26 here -- telling a reader who has already
        written an ``axis:`` to "declare axis: time or axis: freq" -- in front
        of the refusal that names what they actually wrote.
        """
        document = with_noise({**ONE_D, "axis": "sideways"})
        assert silent_here(document)
        with pytest.raises(ConfigError) as caught:
            built({**ONE_D, "axis": "sideways"})
        assert str(caught.value) == AXIS_GRAMMAR

    def test_a_sigma_under_a_kind_that_does_not_take_one_is_not_A26s(self):
        """A26's ``kind == "homoscedastic"`` gate, which nothing tested.

        Measured: dropping the gate leaves this module, ``test_config_section
        _noise.py`` AND the whole of ``pytest tests/config -n 16`` green, and
        it is not an equivalent mutation -- the ungated check fires A26 on the
        block below, where the shipped code is silent.  It would pre-empt the
        ONLY message that tells the reader the key is in the wrong block:
        ``sigma:`` is not a ``radiometer`` key at all, and "declare axis: time
        or axis: freq" is advice about a key that should not be there.

        A stand-down with no test is a stand-down one refactor from becoming a
        pre-emption (R4).
        """
        noise = {"kind": "radiometer",
                 "sigma": {"ones": ["n_freq"], "unit": "K"},
                 "channel_width": {"value": 1.0, "unit": "MHz"},
                 "integration_time": {"value": 2.0, "unit": "s"},
                 "include_logdet": True}
        assert silent_here(with_noise(noise))
        with pytest.raises(ConfigError) as caught:
            built(noise)
        assert str(caught.value).startswith(
            "inference.noise: kind: radiometer does not take ['sigma']")

    def test_a_homoscedastic_block_with_no_sigma_hears_the_requirement(self):
        document = with_noise({"kind": "homoscedastic"})
        assert silent_here(document)
        with pytest.raises(ConfigError) as caught:
            built({"kind": "homoscedastic"})
        assert str(caught.value) == SIGMA_REQUIRED

    def test_an_empty_list_sigma_is_the_value_grammars_own_refusal(self):
        assert silent_here(with_noise({"kind": "homoscedastic",
                                       "sigma": {"list": []}}))


class TestA26sAdviceIsAmbiguous:
    """S4's second half, and the finding plan §0.3 E.4 predicted.

    A26's remedy is *"declare axis: time or axis: freq"* and the document has
    two places to write it.  ``axis:`` is a legal value-node MODIFIER -- the
    grammar records it and never applies it (``modifiers.py``'s own docstring
    says so) -- while ``build_noise`` reads the SIBLING key.  So the reader who
    puts it inside the sigma node earns the identical sentence a second time,
    with nothing telling them which of the two placements was meant.

    **The message is not reworded here.**  It is a hoist and a hoist keeps its
    words verbatim (§2.3); rewording it is a decision for the task that owns
    ``docs/config-inference.md``.  What this task ships is the measurement.
    """

    def test_the_advice_written_inside_the_sigma_node_re_earns_it(self):
        inside = {"kind": "homoscedastic",
                  "sigma": {"ones": ["n_freq"], "unit": "K", "axis": "freq"}}
        assert only(with_noise(inside), "A26").message == A26_MESSAGE

    def test_the_build_agrees_that_the_inside_placement_does_not_help(self):
        """The ambiguity is the LAYER's and not this pass's -- which is what
        makes it a finding to record rather than a check to fix."""
        inside = {"kind": "homoscedastic",
                  "sigma": {"ones": ["n_freq"], "unit": "K", "axis": "freq"}}
        with pytest.raises(ConfigError) as caught:
            built(inside)
        assert str(caught.value) == A26_MESSAGE

    def test_the_sibling_placement_is_the_one_that_builds(self):
        assert silent_here(with_noise(ONE_D_FIXED))
        assert built(ONE_D_FIXED).model.sigma.shape == (1, 8)

    def test_axis_time_is_the_other_half_of_the_remedy(self):
        fixed = {"kind": "homoscedastic",
                 "sigma": {"ones": ["n_time"], "unit": "K"}, "axis": "time"}
        assert silent_here(with_noise(fixed))
        assert built(fixed).model.sigma.shape == (16, 1)


class TestA49BothDirections:
    """A49: ``include_logdet`` present exactly when the sigma needs it."""

    def test_the_required_direction_whole(self):
        found = only(with_noise(RADIOMETER_NO_LOGDET), "A49")
        assert found.severity == REFUSE
        assert found.where == "inference.noise.include_logdet"
        assert found.message == A49_REQUIRED

    def test_the_refused_direction_on_a_frozen_kind_whole(self):
        found = only(with_noise({**FROZEN, "include_logdet": True}), "A49")
        assert found.where == "inference.noise.include_logdet"
        assert found.message == A49_REFUSED_ON_FROZEN

    def test_the_refused_direction_on_a_constant_sigma_whole(self):
        found = only(with_noise({"kind": "homoscedastic",
                                 "sigma": {"value": 0.5, "unit": "K"},
                                 "include_logdet": True}), "A49")
        assert found.message == A49_REFUSED_ON_HOMOSCEDASTIC

    def test_the_build_says_the_same_two_sentences(self):
        """The hoist's third part, both directions.  ``test_config_section_
        noise.py``'s three ``match="include_logdet"`` pins are unchanged; each
        of them matches all three of these sentences, which is why the
        equality is here."""
        with pytest.raises(ConfigError) as caught:
            built(RADIOMETER_NO_LOGDET)
        assert str(caught.value) == A49_REQUIRED
        with pytest.raises(ConfigError) as caught:
            built({**FROZEN, "include_logdet": True})
        assert str(caught.value) == A49_REFUSED_ON_FROZEN

    def test_a_truthy_non_bool_is_still_a_lost_declaration(self):
        """Kills ``if not section.get("include_logdet")``.

        ``include_logdet: 1`` is a lost declaration, not a yes: the truthy
        implementation takes it as a yes and the estimator changes with
        nothing said.  ``test_config_section_noise.py`` pins this at the
        build; this is the same property at P-1.
        """
        found = only(with_noise({**RADIOMETER_NO_LOGDET,
                                 "include_logdet": 1}), "A49")
        assert found.message == A49_REQUIRED

    @pytest.mark.parametrize("declared", [True, False])
    def test_a_declared_bool_clears_it(self, declared):
        document = with_noise({**RADIOMETER_NO_LOGDET,
                               "include_logdet": declared})
        assert silent_here(document)
        assert built({**RADIOMETER_NO_LOGDET,
                      "include_logdet": declared}).include_logdet is declared

    def test_dropping_the_key_clears_the_refused_direction(self):
        """S4: this check's own advice, applied.  The hint says the key is
        "refused otherwise", so the remedy is to delete it -- and the document
        then loads rather than trading one refusal for another."""
        assert silent_here(with_noise(FROZEN))
        assert built(FROZEN).frozen["source"] == "observed"


class TestA49DoesNotPreEmpt:
    """S4's stand-down half for A49."""

    def test_an_unknown_kind_hears_build_noise_name_the_four(self):
        """``inference.noise.kind: banana`` is ``build_noise``'s vocabulary
        refusal.  A second voice for one typo is worse than a late one, and
        the naive membership test would also raise ``KeyError`` here."""
        assert silent_here(with_noise({"kind": "banana",
                                       "include_logdet": True}))

    def test_a_stray_key_that_is_not_include_logdet_is_not_A49s(self):
        """Kills the ungated sweep.

        Running ``check_unknown_keys`` unconditionally would claim every
        stray key under ``inference.noise`` as an A49 finding -- a different
        check's subject wearing this one's id -- and would make A49 fire on
        documents whose ``include_logdet`` is perfectly correct.
        """
        assert silent_here(with_noise({"kind": "homoscedastic",
                                       "sigma": {"value": 0.5, "unit": "K"},
                                       "flors": 1}))

    def test_a_kind_that_is_not_even_a_string_does_not_abort_the_pass(self):
        """§2.3's TRAP, and it is not hypothetical: ``kind: [radiometer]`` is
        unhashable, ``kind not in _KIND_KEYS`` raises ``TypeError``, and the
        pass turns that into "check 'A49' RAISED" -- which DISCARDS every
        other finding in the report.  ``test_preflight_fitting.py``'s hostile
        battery is what found it; this is the same document named."""
        for kind in ([("radiometer")], {"radiometer": 1}, 5, None):
            document = with_noise({"kind": kind, "include_logdet": True})
            assert silent_here(document)

    def test_a_malformed_noise_section_decides_nothing(self):
        for noise in (None, "nope", [], {}):
            assert silent_here(with_noise(noise)), noise

    def test_an_absent_inference_section_decides_nothing(self):
        assert silent_here(preflight_document(inference=None))


class TestTheTwins:
    """S3.  The three the task body names, plus the two this task found."""

    def test_radiometer_frozen_is_the_kind_that_looks_prediction_dependent(
            self):
        """Named twin 1.  Its sigma is decided from the data ONCE and is
        constant thereafter, so ``include_logdet`` is refused on it and not
        required -- the opposite of ``radiometer``, one word away."""
        assert only(with_noise({**FROZEN, "include_logdet": True}),
                    "A49").message == A49_REFUSED_ON_FROZEN
        assert silent_here(with_noise(FROZEN))

    @pytest.mark.parametrize(
        ("noise", "expected"),
        [({"kind": "radiometer", "channel_width": {"value": 1.0,
                                                   "unit": "MHz"},
           "integration_time": {"value": 2.0, "unit": "s"},
           "flags": FLAGS}, frozenset({"A49"})),
         ({"kind": "radiometer", "channel_width": {"value": 1.0,
                                                   "unit": "MHz"},
           "integration_time": {"value": 2.0, "unit": "s"},
           "include_logdet": True, "flags": FLAGS}, frozenset()),
         ({"kind": "homoscedastic", "sigma": {"ones": ["n_freq"],
                                              "unit": "K"},
           "flags": FLAGS}, frozenset({"A26"})),
         ({**ONE_D_FIXED, "flags": FLAGS}, frozenset())],
    )
    def test_a_flags_entry_never_moves_a_row(self, noise, expected):
        """Named twin 2.  ``FlaggedNoise`` FORWARDS ``depends_on_prediction``
        (``inference/noise.py``), so wrapping a model in it changes neither
        which kinds need ``include_logdet`` nor whether a sigma is 1-D --
        and ``flags`` is a legal key for both kinds, so a check keyed on "the
        block has exactly these keys" would get all four of these wrong."""
        assert ids_here(with_noise(noise,
                                   observation=AUX_FLAGS)) == expected

    def test_the_observed_realise_sigma_is_a_RECORDED_FALSE_NEGATIVE(self):
        """Named twin 3, and the answer is "not guarded", on purpose.

        ``inference.observed.<n>.realise`` has a ``kind: homoscedastic`` of
        its own that resolves a ``sigma:`` and hands it to the same
        ``HomoscedasticNoise``.  Its key set is ``{kind, sigma, seed}`` --
        **there is no ``axis:`` in that grammar at all** -- so A26's remedy
        ("declare axis: time or axis: freq") is advice the layer would refuse,
        which is exactly the loop R4 exists to stop.

        Measured, and this is the cost of not guarding it: on a SQUARE 8x8
        grid a length-8 sigma under ``realise:`` is as ambiguous as the one
        A26 refuses under ``noise:``, and the document LOADS.  Adding
        ``axis:`` to that grammar is a section change with its own decision;
        §7 records it by name.
        """
        square = {"time": {"grid": {"arange": {"start": 0.0, "step": 2.0,
                                               "num": 8}, "unit": "s"}}}
        observed = {"from": "simulation", "at": {"g": 1.5}, "twin": "full",
                    "realise": {"kind": "homoscedastic",
                                "sigma": {"ones": ["n_freq"], "unit": "K"},
                                "seed": {"from": "runtime.seeds.a"}}}
        document = preflight_document(
            observation=square, inference={"observed": observed},
            runtime={"seed": 7, "seeds": {"a": 3}})
        assert silent_here(document)
        load_document(document)   # it builds; that IS the false negative

    def test_the_twin_replace_route_carries_no_inference_noise(self):
        """Plan §0.3 E.10, answered: these two checks do NOT walk
        ``inference.twin.replace`` and that is not a false negative.

        The ruling is about text checks that walk ``model:``.  Neither check
        here reads ``model:``: ``inference.noise`` is the LIKELIHOOD's noise,
        and ``inference.twin.replace`` is a mapping of graph node id -> node
        spec (``sections/twin.py``), so the two never meet.  Measured: a
        document that replaces the ``noise`` NODE keeps whatever
        ``inference.noise`` said, and both answers are unchanged by the
        replacement.
        """
        replace = {"replace": {"noise": {"type": "RadiometerNoiseOperator",
                                         "channel_width": {"value": 1.0,
                                                           "unit": "MHz"},
                                         "integration_time": {"value": 2.0,
                                                              "unit": "s"}}}}
        assert ids_here(preflight_document(
            inference={"noise": ONE_D, "twin": replace})) == frozenset({"A26"})
        assert ids_here(preflight_document(
            inference={"noise": ONE_D_FIXED,
                       "twin": replace})) == frozenset()

    def test_a_variant_that_patches_the_noise_block_is_reported(self):
        """The twin this task found: ``variants:``.

        3A's ``A1.horizon`` walks ``resources.beams`` by layer for exactly
        this reason -- a variant IS a different document, and the refusal it
        earns waits until somebody selects it.  The base document here is
        CLEAN, so the finding can only have come from the layer walk.

        ``~axis`` and not ``axis: none``: a variant patch MERGES, so a patch
        that merely restates the sigma leaves the base's ``axis: freq`` in
        place and the variant is as clean as the base.  Measured -- that was
        this test's first draft and it found nothing.
        """
        document = with_variants(ONE_D_FIXED,
                                 {"night": {"inference": {"noise": {
                                     "~axis": None}}}})
        found = only(document, "A26")
        assert found.message == f"variants.night: {A26_MESSAGE}"
        assert found.where == "variants.night.inference.noise.sigma"

    def test_a_variant_that_patches_the_logdet_is_reported_too(self):
        document = with_variants({**RADIOMETER_NO_LOGDET,
                                  "include_logdet": True},
                                 {"gls": {"inference": {"noise": {
                                     "~include_logdet": None}}}})
        assert only(document, "A49").message == (
            f"variants.gls: {A49_REQUIRED}")

    def test_the_base_documents_own_finding_is_said_once(self):
        """``_task3_over_layers``' de-duplication, on this section: a base
        fault plus four variants would otherwise be five sentences, four of
        them blaming a variant that did not introduce it."""
        document = with_variants(ONE_D, {
            f"v{index}": {"inference": {"parameters": {}}}
            for index in range(4)})
        assert only(document, "A26").message == A26_MESSAGE


class TestTheLayerGate:
    """``_b6_is_layered`` is a COST gate and must change no answer.

    ``_task3_over_layers`` deep-copies the whole document once per variant per
    check that layers, and ``test_config_preflight.py``'s cold budget is the
    tightest guard in this suite.  So the walk is skipped when no declared
    patch can reach ``inference:`` at all -- which is decidable from the
    patch's own top-level keys, because layering is one level deep by design
    (``layering.py``).
    """

    #: Documents whose gate answers differ, so the equivalence below is driven
    #: through both branches.  ``~inference`` is the delete spelling and a gate
    #: that forgot it would skip the walk on a variant that removes the
    #: section outright.
    BATTERY = {
        "no variants at all": {},
        "a variant that touches model only": {
            "v": {"model": {"gain": {"gain": {"value": 2.0,
                                              "unit": "dimensionless"}}}}},
        "a variant that rewrites the noise": {
            "v": {"inference": {"noise": ONE_D}}},
        "a variant that deletes inference": {"v": {"~inference": None}},
        "a variant that is not a mapping": {"v": 5},
        "variants that are not a mapping": None,
    }

    def _document(self, variants):
        return with_variants(ONE_D_FIXED, variants)

    @pytest.mark.parametrize("case", sorted(BATTERY))
    @pytest.mark.parametrize("per_layer", [_a26_in, _a49_in],
                             ids=["A26", "A49"])
    def test_the_layer_gate_changes_no_finding(self, case, per_layer):
        """The gate against the walker it is allowed to skip, side by side."""
        document = self._document(self.BATTERY[case])
        assert (tuple(_b6_over_layers(document, per_layer))
                == tuple(_task3_over_layers(document, per_layer)))

    def test_the_layer_gate_actually_skips_the_deepcopy(self, monkeypatch):
        """The anti-vacuity partner, counted rather than TIMED.

        A wall-clock assertion here would be a benchmark of whatever else is
        running on the machine; the property is that ``apply_variant`` is not
        CALLED, and that is exact.  A gate stuck at True passes every
        equivalence above and fails this one.
        """
        import rheplicant.config.layering as layering

        calls: list[str] = []
        real = layering.apply_variant
        monkeypatch.setattr(layering, "apply_variant",
                            lambda document, name: (calls.append(name),
                                                    real(document, name))[1])

        untouched = with_variants(ONE_D_FIXED, {"a": {"model": {}},
                                                "b": {"model": {}}})
        tuple(_b6_over_layers(untouched, _a26_in))
        assert calls == []

        touched = with_variants(ONE_D_FIXED, {"a": {"inference": {}},
                                              "b": {"model": {}}})
        tuple(_b6_over_layers(touched, _a26_in))
        assert calls == ["a", "b"]

    @pytest.mark.parametrize(
        ("variants", "layered"),
        [(None, False),
         ({"v": {"model": {}}}, False),
         ({"v": {"inference": {}}}, True),
         ({"v": {"~inference": None}}, True),
         ({"a": {"model": {}}, "b": {"inference": {}}}, True),
         ("nope", False)],
    )
    def test_the_gate_reads_the_patches_own_top_level(self, variants,
                                                      layered):
        document = with_noise(ONE_D_FIXED)
        if variants is not None:
            document["variants"] = variants
        else:
            document.pop("variants", None)
        assert _b6_is_layered(document) is layered


class TestTheRankReader:
    """``_a26_rank`` on its own, where a document cannot reach every branch."""

    @pytest.mark.parametrize(
        ("node", "rank"),
        [(0.5, 0), (True, 0), ("0.5 K", 0),
         ({"value": 0.5}, 0), ({"value": [1.0, 2.0]}, 1),
         ({"value": [[1.0], [2.0]]}, 2),
         ({"zeros": []}, 0), ({"ones": ["n_freq"]}, 1),
         ({"ones": ["n_time", "n_freq"]}, 2),
         ({"full": {"shape": ["n_freq"], "value": 1.0}}, 1),
         ({"list": [1.0]}, 1), ({"list": [[1.0, 2.0]]}, 2),
         ({"linspace": {}}, 1), ({"arange": {}}, 1), ({"modulo": {}}, 1),
         ({"from_grid": "time"}, 1),
         ({"normal": {"shape": []}}, 0),
         ({"uniform": {"shape": ["n_time", "n_freq"]}}, 2),
         ({"stack": [{"value": 1.0}]}, 1),
         ({"stack": [{"ones": ["n_freq"]}]}, 2),
         # `modifiers.py` applies `column:` on truthiness, so `column: false`
         # leaves the value 1-D and A26 must still fire.  Kills widening
         # `node.get("column")` to `"column" in node`, which is otherwise
         # green across the whole of `tests/config`.
         ({"ones": ["n_freq"], "column": False}, 1),
         ({"ones": ["n_freq"], "column": 0}, 1),
         # a draw's operands broadcast, so `shape:` is not the last word
         ({"normal": {"shape": ["n_freq"], "loc": 1.0}}, 1),
         ({"normal": {"shape": ["n_freq"], "loc": "1.0 K"}}, 1),
         ({"normal": {"shape": ["n_freq"], "loc": {"value": 1.0}}}, 1),
         ({"normal": {"shape": ["n_freq"],
                      "loc": {"ones": ["n_time", "n_freq"]}}}, None),
         ({"normal": {"shape": ["n_freq"],
                      "scale": {"ones": ["n_time", "n_freq"]}}}, None),
         ({"normal": {"shape": ["n_freq"], "loc": {"ref": "resources.x"}}},
          None),
         ({"uniform": {"shape": ["n_freq"], "low": 0.0, "high": 1.0}}, 1),
         ({"uniform": {"shape": ["n_freq"],
                       "high": {"ones": ["n_time", "n_freq"]}}}, None),
         ({"full": {"shape": ["n_freq"], "value": 1.0}}, 1),
         ({"full": {"shape": ["n_freq"], "value": [[1.0]]}}, None),
         # every one of these is a stand-down, not a rank
         ({"ref": "resources.arrays.flat"}, None),
         ({"file": {"path": "x.npy"}}, None),
         ({"from": "channel_spacing"}, None),
         ({"python": "numpy:zeros"}, None),
         ({"from_switch_order": {"resource": "resources.s_params"}}, None),
         ({"basis_fit": {}}, None),
         ({"ones": ["n_freq"], "column": True}, None),
         ({"ones": ["n_freq"], "list": [1.0]}, None),
         ({"unit": "K"}, None),
         ({"ones": "n_freq"}, None),
         ({"full": ["n_freq"]}, None),
         ({"list": []}, None),
         ({"stack": []}, None),
         ({"value": "not a number"}, None),
         ([1.0, 2.0], None),
         (None, None)],
    )
    def test_the_rank_the_text_declares(self, node, rank):
        assert _a26_rank(node) == rank


class TestTheRegistryAndTheOrder:
    """The four legal registry forms, and the one ordering claim that was
    made about this task and is FALSE."""

    def test_each_id_is_bound_to_its_own_function(self):
        assert CHECKS["A26"] is _sigma_axis
        assert CHECKS["A49"] is _logdet

    def test_the_foot_import_is_what_registers_them(self):
        """R1, and the ONE mutant this module could not kill on its own.

        Measured: deleting ``preflight/__init__.py``'s ``noise`` foot import
        leaves every other test in this file GREEN, because the file's own
        ``from rheplicant.config.preflight.noise import ...`` runs the
        ``@register`` decorators itself.  ``test_config_preflight.py::
        TestTheFootImportCannotRot`` does catch it -- but that is a guard in a
        module this task does not own, and a task whose registration is
        pinned only by somebody else's census is one refactor away from
        registering nothing.

        A SUBPROCESS is the only honest form: within this session the module
        is already imported, so nothing in-process can distinguish "registered
        by the foot import" from "registered by the test's own import".
        """
        import subprocess
        import sys

        done = subprocess.run(
            [sys.executable, "-c",
             "from rheplicant.config.preflight import CHECKS\n"
             "print(sorted(k for k in CHECKS if k in ('A26', 'A49')))\n"
             "print(CHECKS['A26'].__module__, CHECKS['A49'].__module__)"],
            capture_output=True, text=True)
        assert done.returncode == 0, done.stdout + done.stderr
        registered, modules = done.stdout.split("\n")[:2]
        assert registered == "['A26', 'A49']"
        assert modules == ("rheplicant.config.preflight.noise "
                           "rheplicant.config.preflight.noise")

    def test_both_ids_reach_a_document_through_the_pass(self):
        assert {"A26"} <= ids(with_noise(ONE_D))
        assert {"A49"} <= ids(with_noise(RADIOMETER_NO_LOGDET))
        assert "A26" not in ids(with_noise(ONE_D_FIXED))

    def test_A27_still_speaks_first_beside_a_conjugate_run(self):
        """Plan §0.3 E.4 ruling 1, re-measured on this branch.

        The task body claimed hoisting A49 would REVERSE an order 3A recorded
        -- that a ``radiometer`` with no ``include_logdet`` would start
        arriving before A27 beside a ``conjugate.wiener`` run.  It does not:
        the foot import is alphabetical, ``fitting`` sorts before ``noise``,
        and ``raise_if_refused`` hands back the first refusal.  What the hoist
        buys is that A49 is now IN the report at all on a document whose beam
        does not exist -- visible in the tail rather than never reached.

        Asserted as the order of two NAMED ids within the report and not as a
        registration index or a length: five sibling branches are registering
        into the same dict.
        """
        document = with_noise(RADIOMETER_NO_LOGDET, runs=[
            {"kind": "conjugate.wiener", "width": "none", "names": ["g"]}])
        spoken = [one.check for one in findings(document)
                  if one.check in {"A27", "A49"}]
        assert spoken == ["A27", "A49"]


class TestTheOneBinding:
    """§3.2 (h): a hoisted rule has ONE binding, as a command.

    Two modules carrying one sentence is two validators for one property and
    they drift -- the ``_number``-vs-``_whole`` divergence on the 2C ledger.
    Each literal below is this task's own row; there is no shared table,
    because a shared table's failure mode is a *passing* test after a merge
    keeps one side of it.
    """

    @pytest.mark.parametrize(
        "literal", [A26_MESSAGE, A49_REQUIRED, A49_HINT],
        ids=["A26", "A49-required", "A49-hint"])
    def test_each_hoisted_sentence_is_written_in_exactly_one_module(
            self, literal):
        assert_bound_once(literal)

    def test_the_hoisted_sentences_live_in_the_section_that_owns_them(self):
        """``assert_bound_once`` alone is satisfied by a hoist that MOVED the
        words into ``preflight/`` and left the section calling nothing, which
        is the other half of §2.2 -- the section keeps its own opinion."""
        for literal in (A26_MESSAGE, A49_REQUIRED, A49_HINT):
            assert modules_carrying(literal) == ("config/sections/noise.py",)

    def test_the_mirror_leg_was_not_hoisted(self):
        """Plan §0.3 E.4 ruling 3, pinned.

        ``inference.noise.axis: says how to read a 1-D sigma; this one has
        shape (16, 8).`` interpolates the RESOLVED extents, which are the axes
        slot's inputs.  A P-1 copy of it would have to resolve ``n_time`` and
        ``n_freq``, which is the slot boundary this plan exists to hold.
        """
        assert modules_carrying("says how to read a 1-D sigma") == (
            "config/sections/noise.py",)

    def test_the_two_refusals_A26_defers_to_stayed_where_they_are(self):
        """The two sentences A26 stands down FOR, checked for the same thing.

        A pass that pre-empted either would most naturally do it by copying
        the words, and neither is on ``assert_bound_once``'s list above --
        those three are the hoisted ones.  Only two literals, because each
        call re-parses every module under ``src/rheplicant/``: the three
        hoisted rows are already pinned to one module by name above, and
        ``A26_MIRROR`` by the test beside this one.
        """
        # A clause and not the rendered sentence for the first: it
        # interpolates the token the reader wrote, and the walker folds every
        # interpolation to one character -- so the whole text harvests as
        # nothing at all and the assertion would be measuring its own quoting.
        for literal in ("inference.noise.axis: is none, time or freq; got",
                        SIGMA_REQUIRED):
            assert modules_carrying(literal) == ("config/sections/noise.py",)


class TestTheDocumentsThisTaskDidNotBreak:
    """The shipped documents these two checks now run over.

    Every one of them is a real fixture that other modules assert about, and a
    check that fired on one would land as a failure in a file this task never
    opened.
    """

    def test_the_base_fixture_earns_neither(self):
        assert silent_here(preflight_document())

    def test_the_shared_radiometer_block_earns_neither(self):
        from tests.config.exit_helpers import GCR_RADIOMETER, RADIOMETER

        for block in (RADIOMETER, GCR_RADIOMETER):
            assert silent_here(with_noise(copy.deepcopy(block)))

    def test_the_validation_pages_worked_document_earns_neither(self):
        """``docs/config-validation.md``'s document writes an
        ``inference.noise`` block with ``include_logdet:`` and is EXECUTED by
        ``test_config_surface.py``, which asserts it earns exactly A27, A30
        and A33.  Its block is correct, so A49 stands down -- and the day it
        stops being correct this says so here rather than in a file this task
        does not own."""
        from tests.config.test_config_surface import (
            TestTheValidationPageDocument,
        )

        page = TestTheValidationPageDocument()._document()
        assert preflight(page).checks() & MINE == frozenset()
