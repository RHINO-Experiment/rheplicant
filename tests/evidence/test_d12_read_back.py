"""D12's precondition: read the COMMITTED archives, written before the switch.

D12 rules that rheplicant keeps its own evidence containers and converts at
each arithmetic call, and it attaches a precondition in as many words: the
read-back fixture must be written with today's code and **committed** before
the switch, so Wave D's regression reads bytes from before rather than bytes
its own writer just produced. A fixture that rebuilt the archive at test time
could only ever report that the new writer agrees with the new reader, which
is not the question.

So this file asserts field by field against values written down HERE, not
against a freshly built term. Every one of those values is off its default, on
purpose: the defect the format exists to prevent is
``eqx.tree_serialise_leaves`` taking a static field from whatever template it
was handed, and a term whose statics are all defaults round-trips correctly
under exactly that bug.

Regenerate with::

    JAX_ENABLE_X64=1 .venv/bin/python tests/evidence/fixtures/make_d12_archives.py

...but do not regenerate to make this file pass after a change to the evidence
layer. Going green that way is how the precondition would be discharged
without being met.
"""

from __future__ import annotations

import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.archive import load_memory
from tests.evidence.fixtures.make_d12_archives import ARCHIVES, MANIFESTS, factorization

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


class TestTheFixtureIsActuallyThere:
    """A read-back regression whose input is missing is a skipped test wearing
    a passing one's clothes, and the plan makes the fixture's PRESENCE the
    gate on Wave D starting at all."""

    @pytest.mark.parametrize(("name", "shape"), ARCHIVES)
    def test_the_binary_is_committed(self, name, shape):
        path = FIXTURES / name
        assert path.exists(), (
            f"{name} is missing ({shape}). It is D12's precondition and Wave D "
            "may not start without it; regenerate with "
            "tests/evidence/fixtures/make_d12_archives.py and COMMIT the result."
        )
        assert path.stat().st_size > 0

    @pytest.mark.parametrize("name", MANIFESTS)
    def test_the_manifest_is_committed_beside_it(self, name):
        """In this format the manifest is the reconstruction spec, not a
        sidecar: a binary without it is unreadable rather than degraded."""
        assert (FIXTURES / name).exists()

    def test_the_two_forms_really_differ_in_the_field_they_exist_for(self):
        """Anti-vacuity for the whole file. Two archives that happened to be
        identical would satisfy every assertion below twice and pin nothing
        about ``template_projections``, which is the field the format is
        sharpest on -- ``None`` and a length-zero array are the same claim to a
        reader and different pytrees to equinox.
        """
        described = [
            json.loads((FIXTURES / name).read_text())["terms"][0]
            for name in MANIFESTS
        ]
        assert described[0]["n_template_projections"] == 2
        assert described[1]["n_template_projections"] is None
        assert described[0]["template_names"] == ["gain_ripple", "ground_pickup"]
        assert described[1]["template_names"] == []


class TestTheStaticFieldsComeBackFromTheManifest:
    """Field by field, against values written in THIS file.

    Not against a freshly built term: the point is that the reader agrees with
    a writer it cannot consult, and comparing against a live object would let
    both move together.
    """

    @staticmethod
    def _term(name):
        return load_memory(FIXTURES / name, factorization()).archive[0]

    @pytest.mark.parametrize(("name", "_shape"), ARCHIVES)
    def test_the_identifying_statics(self, name, _shape):
        term = self._term(name)
        assert term.epoch_id == "night-042"
        assert term.n_observed == 777
        assert term.exact is False
        assert term.include_logdet is False
        assert term.noise_frozen_at == "gls"

    @pytest.mark.parametrize(("name", "_shape"), ARCHIVES)
    def test_the_support_and_the_provenance(self, name, _shape):
        """``inputs`` is what the memory's conditional-independence refusal
        reads, so a campaign that lost it would cheerfully sum two nights that
        shared a calibration solution."""
        term = self._term(name)
        assert term.support == {"depth": (-2.0, 2.0), "width": (-1.0, 3.0)}
        assert term.residual_dof == 13
        assert term.inputs == (
            ("beam_model", "sha256:b3ee"),
            ("cal_solution", "sha256:0f17"),
        )

    @pytest.mark.parametrize(("name", "_shape"), ARCHIVES)
    def test_the_arrays(self, name, _shape):
        term = self._term(name)
        assert np.allclose(
            np.asarray(term.info.factor), [[1.5, 0.25], [0.0, 0.75]]
        )
        assert np.allclose(np.asarray(term.info.target), [0.5, -0.25])
        assert float(term.info.offset) == pytest.approx(-3.25)
        assert float(term.residual_chi2) == pytest.approx(7.5)
        assert term.info.names == ("depth", "width")

    def test_the_templated_form_keeps_its_projections(self):
        term = self._term("d12_with_templates.rhep")
        assert term.template_names == ("gain_ripple", "ground_pickup")
        assert np.allclose(np.asarray(term.template_projections), [1.25, -0.5])

    def test_the_untemplated_form_comes_back_as_None_and_not_an_empty_array(self):
        """The distinction the format spends a paragraph on. An empty array
        here would be a different pytree, and every later leaf would be read
        from the wrong offset -- so this asserts the TYPE, not just emptiness.
        """
        term = self._term("d12_without_templates.rhep")
        assert term.template_projections is None
        assert term.template_names == ()

    @pytest.mark.parametrize(("name", "_shape"), ARCHIVES)
    def test_the_dtype_is_the_one_the_writer_used(self, name, _shape):
        """The manifest records the writer's arithmetic, and this session is
        the x64 one, so a float32 read would mean the dtype came from a
        template rather than from the file."""
        term = self._term(name)
        assert term.info.factor.dtype == jnp.float64


class TestTheManifestIsNotBOUNDToItsBinary:
    """A weakness found while writing this fixture, MEASURED and pinned rather
    than fixed here (ledger D39).

    This format's first sentence is "write a memory to disk so that reading it
    back cannot lie about it", and the manifest is its reconstruction spec.
    But nothing ties a manifest to the binary it was written for: pair
    ``x.rhep`` with a different archive's ``x.json`` and the two directions
    behave differently, and only one of them complains.

    Measured on the two committed fixtures:

    ==============================  ==========================================
    pairing                         result
    ==============================  ==========================================
    templated binary, plain spec    **loads**, and ``template_projections``
                                    comes back ``None`` -- the two values in
                                    the file are silently discarded. Every
                                    other field is correct.
    plain binary, templated spec    ``TreePathError`` at
                                    ``template_projections``
    ==============================  ==========================================

    The asymmetry has a cause: ``template_projections`` is the LAST dynamic
    leaf, so a spec that expects one fewer leaf simply stops early, while one
    that expects one more runs off the end. The docstring's warning about
    "every later leaf read from the wrong offset" is about a field with
    something after it; here there is nothing after it, so the consequence is
    a quiet loss instead of a loud corruption.

    **Not fixed here**, and the reason is scope rather than difficulty:
    binding the pair (a digest of the binary in the manifest, or a length) is
    a change to the on-disk FORMAT, and the evidence layer is Wave D's. Doing
    it under a precondition's name would be switching the thing the
    precondition exists to protect. Pinned instead, in both directions, so the
    switch cannot change it without saying so.
    """

    @staticmethod
    def _paired(binary, manifest, into):
        import shutil

        into.mkdir(exist_ok=True)
        shutil.copy(FIXTURES / binary, into / "x.rhep")
        shutil.copy(FIXTURES / manifest, into / "x.json")
        return into / "x.rhep"

    def test_a_plain_spec_silently_DROPS_the_projections(self, tmp_path):
        path = self._paired(
            "d12_with_templates.rhep", "d12_without_templates.json", tmp_path
        )
        term = load_memory(path, factorization()).archive[0]
        # The finding. Not an assertion that this is right -- an assertion that
        # this is what happens, so D39 can be discharged deliberately.
        assert term.template_projections is None
        assert term.template_names == ()
        # ... while everything else came back correctly, which is what makes it
        # quiet rather than obviously broken.
        assert float(term.info.offset) == pytest.approx(-3.25)
        assert term.n_observed == 777

    def test_a_templated_spec_on_a_plain_binary_DOES_fail(self, tmp_path):
        """The other direction, and it is the one that behaves: a spec
        expecting one more leaf than the file holds runs off the end."""
        path = self._paired(
            "d12_without_templates.rhep", "d12_with_templates.json", tmp_path
        )
        with pytest.raises(Exception, match="template_projections|TreePath"):
            load_memory(path, factorization())

    def test_a_missing_manifest_is_named_as_such(self):
        """The refusal `archive.py` argues for, and it works: a binary without
        its manifest is unreadable rather than degraded, and the message says
        which. Kept beside the two above because it is the same question --
        what the format does when the pair is broken -- answered well in one
        case and not in the other."""
        import shutil

        tmp = FIXTURES / "_orphan_check"
        tmp.mkdir(exist_ok=True)
        try:
            shutil.copy(FIXTURES / "d12_with_templates.rhep", tmp / "orphan.rhep")
            with pytest.raises(StateValidationError, match="manifest"):
                load_memory(tmp / "orphan.rhep", factorization())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def test_this_session_is_the_one_the_fixture_was_written_in():
    """The manifest records the writer's x64 state and ``load_memory`` refuses
    a mismatch, so this file only means anything in the x64 session. Asserted
    rather than assumed: run outside it, every test above would fail on the
    dtype and none would say why.
    """
    assert jax.config.jax_enable_x64 is True
    for name in MANIFESTS:
        assert json.loads((FIXTURES / name).read_text())["jax_enable_x64"] is True
