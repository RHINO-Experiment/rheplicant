"""A35, the feature half: a distribution that is here and is too old for the route.

The presence half is ``test_preflight_depends.py``; this is the split §0.3 E.1
names in advance.

**Why there is a second half at all.**  ``import limtod_jax`` SUCCEEDS on
limTOD 1.6 and the failure is a missing attribute, so a check that only asked
"is it installed" would close four routes and leave the measured one open.
``radio/beams.py::_require_limtod_jax`` answers it with ``hasattr(module,
feature)`` and this check mirrors that -- under two constraints that this
module is where they are pinned:

* **``hasattr`` is legal for ``limtod_jax`` and for nothing else.**  Measured
  on a complete limTOD 1.10.0, ``hasattr(limTOD, "uvbeam")`` and
  ``hasattr(limTOD, "cstbeam")`` are both **False** -- they are submodules --
  so a table row asking limTOD for either would refuse every install in
  existence.  :class:`TestHasattrIsLegalForOneModuleOnly` is that measurement,
  run rather than quoted.
* **the attribute may only be asked of a module the process already holds.**
  Asking it of one that is not imported means importing it, and ``import
  limTOD`` costs **1.02 s** in this worktree while the whole pass is budgeted
  at 0.05 s.  So the leg is silent in a fresh process and speaks in a session
  that has already built a projector -- :class:`TestTheLegNeverImports` states
  that as a property, in both directions, because a false negative nobody
  measured is indistinguishable from a check that works.
"""

import importlib.util
import pathlib
import subprocess
import sys
import textwrap
import types

import pytest

from rheplicant.config.preflight.depends import _CONDITIONAL, _FEATURES
from tests.config.preflight_helpers import only, preflight_document

_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Every attribute the tables ask ``limtod_jax`` for -- **derived, never
#: restated**.  A literal here is a second source of truth, and the failure it
#: hides is exact: set one row's attribute to ``None`` (downgrading it to a
#: presence-only check, which is the very false negative §6 widened A35 to
#: close) and a hard-coded tuple keeps asserting that the real module carries a
#: symbol the table no longer asks for.  Reading it off ``_FEATURES`` makes
#: that mutation shrink this tuple instead, which the per-row negatives below
#: then notice.
LIMTOD_JAX_FEATURES = tuple(sorted(
    {attribute for requirements in _FEATURES.values()
     for _, module, attribute in requirements
     if module == "limtod_jax" and attribute}
    | {requirement[2] for _, requirement in _CONDITIONAL.values()
       if requirement[1] == "limtod_jax" and requirement[2]}))


def stand_in(name, *attributes, spec=True):
    """A module object holding ``attributes`` and nothing else.

    ``spec=True`` gives it a ``__spec__``, which is §0.3 E.2(7)'s recipe for
    "old": without one, ``importlib.util.find_spec`` raises ``ValueError``, and
    a check that reached ``find_spec`` first would be wrapped by the pass as
    *"pre-flight check 'A35' RAISED ValueError"* -- so the test would silently
    measure "the check crashed" instead of "the feature is missing".
    ``spec=False`` is what
    :meth:`TestTheOldRecipe.test_a_stand_in_without_a_spec_does_not_abort_the_pass`
    drives, because "the recipe is needed" and "the pass survives without it"
    are two different claims and only one of them was measured for the plan.
    """
    module = types.ModuleType(name)
    if spec:
        module.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    else:
        module.__spec__ = None
    for attribute in attributes:
        setattr(module, attribute, object())
    return module


def truncating_beam():
    """A document whose beam asks for ``limtod_jax.horizon_truncated_beam``."""
    return preflight_document(resources={"beams": {
        "horn": {"format": "npy", "path": "b.npy", "nside": 4,
                 "normalize": "pixel_sum", "frame": "beam_local",
                 "horizon": {"mode": "truncate_map"}}}})


def driftscan(**extra):
    """A document with one driftscan projector, ``extra`` written onto it."""
    return preflight_document(resources={"projectors": {
        "p": {"engine": "driftscan", "lmax": 8, "normalize_beam": True,
              "beam": {"ref": "resources.beams.horn"}, **extra}}})


def analysing_transform():
    """A document whose latent transform asks for ``limtod_jax.map2alm_iter``."""
    return preflight_document(inference={"parameters": {
        "g": {"init": 1.0, "into": "gain.gain",
              "transform": {"beam_analysis": {"nside": 4, "lmax": 8}}}}})


#: ``(the symbol, the document that asks for it, where it is asked)`` -- one
#: row per attribute requirement in the two tables.  Read as a check on the
#: parametrization: ``test_every_attribute_row_has_a_case`` asserts this covers
#: :data:`LIMTOD_JAX_FEATURES` exactly, so a fifth attribute row added to
#: ``_FEATURES`` arrives here or fails.
ATTRIBUTE_ROWS = (
    ("horizon_truncated_beam", truncating_beam, "resources.beams.horn"),
    ("driftscan", driftscan, "resources.projectors.p"),
    ("map2alm_iter", analysing_transform, "inference.parameters.g"),
    ("check_uniform_grid", lambda: driftscan(uniform_sampling=True),
     "resources.projectors.p"),
)


class TestTheVersionCase:
    def test_a_limtod_jax_without_the_symbol_is_refused_by_name(self, monkeypatch):
        """The measured case the schema widened A35 for.

        The stand-in imports -- it is in ``sys.modules`` and carries a
        ``__spec__`` -- so every presence-only implementation of A35 says
        nothing here.  This is the test that dies under one.
        """
        monkeypatch.setitem(sys.modules, "limtod_jax", stand_in("limtod_jax"))
        finding = only(truncating_beam(), "A35")
        assert finding.message == (
            "resources.beams.horn: horizon.mode: truncate_map needs "
            "limtod_jax.horizon_truncated_beam, and the limtod_jax this process holds "
            "does not carry it -- the module imports and the symbol is missing, which is "
            "what a limTOD older than this route looks like. limTOD is a hard dependency "
            "of this package rather than an extra, so a missing one means the install is "
            'broken or limTOD was removed: pip install "limTOD[jax]>=1.10". Said from '
            "the document's text, so that a missing dependency arrives before the run "
            "rather than as an ImportError in the middle of one (check A35).")

    def test_a_limtod_jax_that_carries_it_says_nothing(self, monkeypatch):
        """The anti-vacuity partner: the leg reads the ATTRIBUTE, not the
        stand-in.  Without this, an implementation that refused every
        ``sys.modules`` stand-in would pass the test above."""
        monkeypatch.setitem(sys.modules, "limtod_jax",
                            stand_in("limtod_jax", *LIMTOD_JAX_FEATURES))
        assert not [one for one in _findings(truncating_beam())
                    if one.check == "A35"]

    def test_every_attribute_row_has_a_case_below(self):
        """The parametrization is a second list; this is what keeps it honest.

        A fifth attribute requirement added to ``_FEATURES`` with no document
        beside it would otherwise be covered by nothing while every test here
        stayed green.
        """
        assert tuple(sorted(symbol for symbol, _, _ in ATTRIBUTE_ROWS)) == \
            LIMTOD_JAX_FEATURES

    @pytest.mark.parametrize("symbol, build, where", ATTRIBUTE_ROWS,
                             ids=[row[0] for row in ATTRIBUTE_ROWS])
    def test_each_row_loses_its_own_symbol_and_says_so(self, monkeypatch,
                                                       symbol, build, where):
        """**Per ROW**, not per module -- the distinction that matters.

        A whole-module mutant ("delete the ``hasattr`` leg") is killed by any
        one row's test, so it proves nothing about the other three: measured on
        this commit before these cases existed, setting
        ``("transform","beam_analysis")``'s or ``("engine","driftscan")``'s
        attribute to ``None`` -- silently downgrading that route to a
        presence-only check -- left the whole suite green.  That downgrade is
        exactly the installed-but-old ``limtod_jax`` the schema widened A35
        for, so each row now carries its own negative: a stand-in holding every
        symbol EXCEPT this row's must earn A35 on this row's route, naming this
        row's symbol.
        """
        others = [feature for feature in LIMTOD_JAX_FEATURES
                  if feature != symbol]
        monkeypatch.setitem(sys.modules, "limtod_jax",
                            stand_in("limtod_jax", *others))
        finding = only(build(), "A35")
        assert finding.where == where
        assert f"needs limtod_jax.{symbol}," in finding.message

    @pytest.mark.parametrize("feature", LIMTOD_JAX_FEATURES)
    def test_the_real_limtod_jax_carries_every_symbol_the_table_asks_for(
            self, feature):
        """The other direction, and the one that would make this check a live
        defect: a table naming a symbol limTOD never had would refuse every
        install.  Skipped rather than importing limTOD from cold -- the module
        is in ``sys.modules`` whenever anything in the session has built a
        projector, and the measurement is recorded in this module's docstring
        for the run where it is not."""
        limtod_jax = sys.modules.get("limtod_jax")
        if limtod_jax is None:
            if importlib.util.find_spec("limtod_jax") is None:
                pytest.skip("limtod_jax is not installed in this environment")
            limtod_jax = pytest.importorskip("limtod_jax")
        assert hasattr(limtod_jax, feature)


class TestHasattrIsLegalForOneModuleOnly:
    def test_no_requirement_asks_limTOD_itself_for_an_attribute(self):
        """§0.3 E.2(2), as a property of the table.

        A row ``("limTOD", "limTOD", "uvbeam")`` reads perfectly and refuses
        every install there has ever been, because ``limTOD.uvbeam`` is a
        SUBMODULE and ``hasattr`` on an unimported submodule is ``False``.
        """
        for requirements in _FEATURES.values():
            for distribution, module, attribute in requirements:
                if module == "limTOD":
                    assert attribute is None, (distribution, attribute)

    def test_only_limtod_jax_is_asked_for_an_attribute_at_all(self):
        every = [requirement for requirements in _FEATURES.values()
                 for requirement in requirements]
        every += [requirement for _, requirement in _CONDITIONAL.values()]
        assert {module for _, module, attribute in every
                if attribute is not None} == {"limtod_jax"}

    def test_the_submodules_really_are_invisible_to_hasattr(self):
        """The measurement the ruling rests on, RUN rather than quoted -- and
        run in a FRESH PROCESS, which is the part that had to be discovered.

        Measured on limTOD 1.10.0 in a clean interpreter: ``uvbeam`` and
        ``cstbeam`` are both ``False``, while ``sky_model`` is ``True``,
        because the package's own ``__init__`` binds that one.  So two of the
        three are invisible and the rule cannot be "ask limTOD for the
        submodule" either way.

        **In this test session the answer is not even stable.**  Importing
        ``limTOD.uvbeam`` -- which ``kinds/beams.py::_uvbeam_maps`` does, and
        which ``test_config_kind_beams.py`` reaches -- BINDS ``uvbeam`` as an
        attribute of the parent package, so ``hasattr(limTOD, "uvbeam")``
        becomes ``True`` for the rest of the process.  Written in-process this
        assertion failed once in three ``-n 16`` runs and passed twice, which
        is R10's hazard exactly.  That instability is not a nuisance around the
        ruling; it IS the ruling: a gate whose answer depends on what the
        process happened to import earlier is worse than no gate.
        """
        if importlib.util.find_spec("limTOD") is None:  # pragma: no cover
            pytest.skip("limTOD is not installed in this environment")
        done = subprocess.run(
            [sys.executable, "-c",
             "import limTOD;"
             " print(hasattr(limTOD, 'uvbeam'), hasattr(limTOD, 'cstbeam'),"
             " hasattr(limTOD, 'sky_model'))"],
            capture_output=True, text=True, cwd=str(_ROOT))
        assert done.returncode == 0, done.stdout + done.stderr
        assert done.stdout.split()[:3] == ["False", "False", "True"], done.stdout

    def test_the_distribution_is_spelled_limTOD_and_not_limtod_jax(self):
        """Why :data:`Requirement` carries the distribution and the module
        separately.  ``importlib.metadata.version("limtod-jax")`` raises
        ``PackageNotFoundError``; the module ``limtod_jax`` ships inside the
        ``limTOD`` distribution, and an install line naming the wrong one is
        advice that cannot be followed."""
        import importlib.metadata

        with pytest.raises(importlib.metadata.PackageNotFoundError):
            importlib.metadata.version("limtod-jax")
        assert {requirement[0] for requirements in _FEATURES.values()
                for requirement in requirements
                if requirement[1] == "limtod_jax"} == {"limTOD"}


class TestTheConditionalRequirement:
    """The one requirement a value cannot decide on its own.

    ``DriftScanProjector.from_beam_maps`` calls ``_limtod_jax(uniform_sampling)``
    (``radio/sky/driftscan.py:299``) and only the ``uniform=True`` branch asks
    for ``check_uniform_grid``.
    """

    def test_uniform_sampling_adds_the_fft_fast_path_requirement(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "limtod_jax",
                            stand_in("limtod_jax", "driftscan",
                                     "horizon_truncated_beam", "map2alm_iter"))
        finding = only(driftscan(uniform_sampling=True), "A35")
        assert "limtod_jax.check_uniform_grid" in finding.message

    def test_a_projector_without_it_is_not_asked_for_it(self, monkeypatch):
        """The mutant this kills is folding the conditional into ``_FEATURES``:
        that refuses a document the package builds, on an install that runs it
        fine."""
        monkeypatch.setitem(sys.modules, "limtod_jax",
                            stand_in("limtod_jax", "driftscan",
                                     "horizon_truncated_beam", "map2alm_iter"))
        assert not [one for one in _findings(driftscan()) if one.check == "A35"]

    def test_the_sibling_is_read_off_the_entry_and_not_the_document(self):
        """The table says which key, so a rename in ``kinds/projectors.py``
        that left this behind is visible here rather than silent."""
        key, requirement = _CONDITIONAL[("engine", "driftscan")]
        assert key == "uniform_sampling"
        assert requirement == ("limTOD", "limtod_jax", "check_uniform_grid")

    def test_one_absent_module_is_said_once_however_many_rows_want_it(
            self, monkeypatch):
        """``uniform_sampling: true`` puts TWO limtod_jax requirements on one
        entry, and an absent module has no attribute to distinguish them -- so
        without the de-duplication in ``_in_layer`` the reader gets the same
        sentence twice and ``only()`` cannot be used about this check at all."""
        monkeypatch.setitem(sys.modules, "limtod_jax", None)
        assert only(driftscan(uniform_sampling=True), "A35")


class TestABrokenInstallIsNotAnAbsentOne:
    """§0.3 E.2(7)'s third recipe, and the claim it is there to make honest.

    ``find_spec`` answers *findable*, not *importable*.  A distribution that is
    installed but broken -- a failing loader, a missing shared object -- is
    found, so A35 stands down and the shipped gate reports the real
    ``ImportError``.  That is the correct behaviour: this pass must not execute
    a module to find out, and a check that refused on "I could not import it"
    would be running the import it exists to avoid.
    """

    @pytest.fixture
    def broken(self):
        class _Loader:
            @staticmethod
            def create_module(spec):
                return None

            @staticmethod
            def exec_module(module):
                raise ImportError("the shared object is missing")

        class _Finder:
            @staticmethod
            def find_spec(name, path=None, target=None):
                if name == "h5py":
                    return importlib.util.spec_from_loader(name, _Loader())
                return None

        sys.modules.pop("h5py", None)
        sys.meta_path.insert(0, _Finder)
        try:
            yield
        finally:
            sys.meta_path.remove(_Finder)
            sys.modules.pop("h5py", None)

    def test_a_broken_install_is_found_and_stood_down_on(self, broken):
        del broken
        document = preflight_document(observation={
            "from_file": {"format": "rhino_hdf5", "path": "obs.h5",
                          "freq_unit": "MHz"}})
        assert not [one for one in _findings(document) if one.check == "A35"]

    def test_and_the_import_really_does_fail(self, broken):
        """The anti-vacuity half.  Without it, a finder that quietly did
        nothing would make the test above green while proving nothing about
        what ``find_spec`` claims."""
        del broken
        with pytest.raises(ImportError, match="the shared object is missing"):
            import h5py  # noqa: F401

    def test_the_same_document_with_h5py_absent_IS_refused(self, monkeypatch):
        """And the difference between the two verdicts, on one document."""
        monkeypatch.setitem(sys.modules, "h5py", None)
        document = preflight_document(observation={
            "from_file": {"format": "rhino_hdf5", "path": "obs.h5",
                          "freq_unit": "MHz"}})
        assert only(document, "A35").where == "observation.from_file"


class TestTheOldRecipe:
    def test_a_stand_in_without_a_spec_does_not_abort_the_pass(self, monkeypatch):
        """§0.3 E.2(7) warns that a stand-in with no ``__spec__`` makes
        ``find_spec`` raise ``ValueError``, which the pass wraps as *"pre-flight
        check 'A35' RAISED"* -- aborting it and hiding every later finding.

        ``_verdict`` reads ``sys.modules`` BEFORE ``find_spec`` precisely so
        that a live module never reaches that path, and this is the assertion
        that the choice holds: the same stand-in, ``__spec__`` deleted, still
        gets the feature verdict and no ``RAISED`` sentence appears.
        """
        monkeypatch.setitem(sys.modules, "limtod_jax",
                            stand_in("limtod_jax", spec=False))
        findings = _findings(truncating_beam())
        assert not [one for one in findings if "RAISED" in one.message]
        assert only(truncating_beam(), "A35").message.startswith(
            "resources.beams.horn: horizon.mode: truncate_map needs "
            "limtod_jax.horizon_truncated_beam,")

    def test_find_spec_still_raises_on_one_without_a_spec(self):
        """The measurement behind the recipe, kept because it is what makes the
        paragraph above a decision rather than a coincidence: the day
        ``_verdict`` is rewritten to call ``find_spec`` first, this is the
        behaviour it would meet."""
        module = stand_in("zz_no_spec", spec=False)
        sys.modules["zz_no_spec"] = module
        try:
            with pytest.raises(ValueError, match="__spec__ is None"):
                importlib.util.find_spec("zz_no_spec")
        finally:
            del sys.modules["zz_no_spec"]


#: The child of :class:`TestTheLegNeverImports`.  A FRESH process, because the
#: property is about a module that is absent from ``sys.modules`` and a test
#: session has imported half the package by the time it runs.
_CHILD = textwrap.dedent('''
    import sys

    from rheplicant.config.preflight import preflight
    from tests.config.preflight_helpers import preflight_document

    document = preflight_document(resources={"beams": {"horn": {
        "format": "npy", "path": "b.npy", "nside": 4,
        "normalize": "pixel_sum", "frame": "beam_local",
        "horizon": {"mode": "truncate_map"}}}})
    report = preflight(document)
    print("limtod_jax" in sys.modules)
    print(sorted(one.check for one in report.findings))
''')


class TestTheLegNeverImports:
    def test_a_fresh_process_neither_imports_limtod_jax_nor_reports_it(self):
        """**The declared false negative, named rather than left to be found.**

        On a document asking for ``limtod_jax.horizon_truncated_beam``, a fresh
        process holds no ``limtod_jax``, so the attribute cannot be asked and
        A35 says nothing.  Both halves are asserted together because they are
        one decision: the silence is the price of the module staying out of
        ``sys.modules``, and ``_require_limtod_jax`` -- which raises an
        ImportError naming the version -- is the backstop.

        A change that made the leg speak here would necessarily have imported
        limtod_jax, and the first line of this assertion is what says so.
        """
        done = subprocess.run([sys.executable, "-c", _CHILD],
                              capture_output=True, text=True, cwd=str(_ROOT))
        assert done.returncode == 0, done.stdout + done.stderr
        imported, checks = done.stdout.splitlines()[:2]
        assert imported == "False", "the pass imported limtod_jax"
        assert "A35" not in checks, checks


def _findings(document):
    from rheplicant.config.preflight import preflight

    return preflight(document).findings
