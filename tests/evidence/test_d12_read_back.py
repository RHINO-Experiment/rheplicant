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
    """The manifest still is not BOUND to its binary — but the silent half is
    closed (ledger D39, step 1 of the discharge).

    This format's first sentence is "write a memory to disk so that reading it
    back cannot lie about it", and the manifest is its reconstruction spec.
    Nothing ties a manifest to the binary it was written for, and until
    2026-08-29 the two mispairings behaved differently with only one of them
    complaining:

    ==============================  ==========================================
    pairing                         result, as measured 2026-08-27
    ==============================  ==========================================
    templated binary, plain spec    **loaded**, and ``template_projections``
                                    came back ``None`` -- the two values in
                                    the file silently discarded, every other
                                    field correct.
    plain binary, templated spec    ``TreePathError`` at
                                    ``template_projections``
    ==============================  ==========================================

    The asymmetry had a cause: ``template_projections`` is the LAST dynamic
    leaf, so a spec expecting one fewer leaf simply stopped early, while one
    expecting one more ran off the end. The module docstring's warning about
    "every later leaf read from the wrong offset" is about a field with
    something after it; here there was nothing after it, so the consequence was
    a quiet loss instead of a loud corruption.

    **The quiet half is now refused**: ``load_memory`` deserialises from an
    open handle and compares the bytes consumed against the file's size, so a
    manifest that stops early is caught by the bytes it never accounted for
    (measured on these fixtures: 144 of 1976).

    **Deliberately NOT done: a digest.** The obvious fix -- put a hash of the
    binary in the manifest -- requires a ``_FORMAT_VERSION`` bump, and this
    reader refuses any version that is not exactly its own (``!=``, not
    ``<``). This layer's whole premise is that *the raw data is gone*, so a
    bump with no backward-compatible read makes every existing archive
    permanently unreadable with nothing to re-archive from. The order that is
    safe is: close the silent half (done), make the reader version-tolerant,
    and only then bind. What a digest would still add over the byte check is
    the case the byte check cannot see -- two archives of the SAME SHAPE, where
    the counts match and the contents do not, which is also the likeliest real
    mispairing (two nights of one campaign). That remains open.
    """

    @staticmethod
    def _paired(binary, manifest, into):
        import shutil

        into.mkdir(exist_ok=True)
        shutil.copy(FIXTURES / binary, into / "x.rhep")
        shutil.copy(FIXTURES / manifest, into / "x.json")
        return into / "x.rhep"

    def test_a_plain_spec_is_refused_by_the_bytes_it_never_read(self, tmp_path):
        """The half that used to be silent. Refused by arithmetic, not by luck.

        The check is bytes-consumed against file-size, so it does not depend on
        ``template_projections`` being the last leaf -- any manifest that
        under-describes its binary is caught, wherever the shortfall is.
        """
        path = self._paired(
            "d12_with_templates.rhep", "d12_without_templates.json", tmp_path
        )
        with pytest.raises(StateValidationError, match="were never read"):
            load_memory(path, factorization())

    def test_the_refusal_names_the_shortfall(self, tmp_path):
        """A byte count is the one thing that tells a reader WHICH pair is wrong.

        Without it the message is "these do not match", which is true of every
        mispairing and actionable for none.
        """
        path = self._paired(
            "d12_with_templates.rhep", "d12_without_templates.json", tmp_path
        )
        with pytest.raises(StateValidationError) as caught:
            load_memory(path, factorization())
        assert "144 bytes" in str(caught.value), str(caught.value)

    def test_an_older_version_is_refused_with_what_to_do_about_it(self, tmp_path):
        """The refusal a version bump would hand every existing archive.

        This reader accepts exactly its own version (``!=``, not ``<=``), and
        that cannot be relaxed: the versions differ in BYTE LAYOUT, so an older
        binary read through this template runs off the end or reads leaves at
        the wrong offset. The module docstring records that for version 1 by
        measurement.

        So the message is the only mitigation available, and it has to carry
        the one instruction that works -- go back to the writer and re-archive
        -- plus the constraint that makes this matter at all: this layer's
        premise is that the raw data is gone, so a version it cannot read is
        work nobody can recover. That is why the policy in the message is "a
        bump ships with a converter, or it does not ship".
        """
        import json

        path = self._paired(
            "d12_with_templates.rhep", "d12_with_templates.json", tmp_path
        )
        manifest = tmp_path / "x.json"
        spec = json.loads(manifest.read_text())
        spec["format_version"] = spec["format_version"] - 1
        manifest.write_text(json.dumps(spec))

        with pytest.raises(StateValidationError) as caught:
            load_memory(path, factorization())
        message = str(caught.value)
        assert "re-archive" in message, message
        assert "converter" in message, message
        assert "BYTE LAYOUT" in message, message

    def test_a_newer_version_is_refused_differently(self, tmp_path):
        """The two directions need different advice, so they must not share a
        sentence: an older archive is converted forward, a newer one is met by
        upgrading. Telling a user to re-archive something their reader is too
        old to open would send them in a circle."""
        import json

        path = self._paired(
            "d12_with_templates.rhep", "d12_with_templates.json", tmp_path
        )
        manifest = tmp_path / "x.json"
        spec = json.loads(manifest.read_text())
        spec["format_version"] = spec["format_version"] + 1
        manifest.write_text(json.dumps(spec))

        with pytest.raises(StateValidationError) as caught:
            load_memory(path, factorization())
        message = str(caught.value)
        assert "Upgrade" in message, message
        assert "re-archive" not in message.split("Upgrade")[1], message

    def test_the_matched_pair_still_loads(self, tmp_path):
        """ANTI-VACUITY. A check that refused every archive would pass the two
        cases above and destroy the format."""
        path = self._paired(
            "d12_with_templates.rhep", "d12_with_templates.json", tmp_path
        )
        term = load_memory(path, factorization()).archive[0]
        assert term.template_projections is not None
        assert term.template_names != ()
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
