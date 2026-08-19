"""A35, the presence half: a distribution the document asks for and is not here.

The feature half -- an installed module that is too old for the route -- is
``test_preflight_depends_features.py``, the split §0.3 E.1 names in advance.

**Every absence in this module is SIMULATED and none of them reads the real
environment.**  The two are different tests and conflating them is how a suite
becomes a report about the machine it ran on: this worktree's venv carries
healpy, pygdsm, pyuvdata, limTOD, limtod_jax, rhino_cal_jax, h5py and numpyro
and does NOT carry MomentRFI, while the main checkout's venv carries MomentRFI
and does NOT carry pyuvdata -- so a test that asserted "``format: uvbeam``
earns A35" against the live environment would be green in one tree and red in
the other.  The recipe is :func:`blocked`: ``sys.modules[name] = None``, which
is CPython's own way of making an import fail, and which
``importlib.util.find_spec`` reports as absence.

**MomentRFI is no exception, and it used to be one.**  This module shipped
declaring MomentRFI "the one distribution no test environment has" and leaving
two tests to read the real environment for it.  **That premise was false when
it was written**: the repository's own primary venv
(``/Users/zzhang/projects/e-RHINO/.venv``) has MomentRFI installed as an
editable, so both tests failed there with *"A35 produced 0 findings"* while
passing in every worktree -- a suite that was a report about which checkout it
ran in.  Measured, then fixed: presence is simulated in BOTH directions here,
by :func:`blocked` for the absence and by :func:`_stand_in` for the presence,
and :data:`PRESENCE_ALWAYS_SIMULATED` names the distribution that gets the
treatment.
"""

import pathlib
import subprocess
import sys

import pytest

from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import REFUSE
from rheplicant.config.kinds.beams import BEAM_FORMATS
from rheplicant.config.kinds.projectors import ENGINES
from rheplicant.config.kinds.s_params import S_PARAM_KINDS
from rheplicant.config.kinds.sky_models import SKY_KINDS
from rheplicant.config.preflight import CHECKS
from rheplicant.config.preflight.depends import (
    _ALTERNATIVE,
    _CONDITIONAL,
    _FEATURES,
    _INSTALL,
    _TRIGGER,
    _extras,
)
from rheplicant.config.sections.runs import _KINDS
from tests.config.preflight_helpers import ids, only, preflight_document

#: The repository root, so the subprocess below can import ``tests.config``
#: whatever directory pytest was started from.
_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Every top-level module name the table probes.  Read off the table rather
#: than written out, so a requirement added without a thought about the
#: environment still gets blocked by the sweeping tests below.
MODULES = tuple(sorted({requirement[1]
                        for requirements in _FEATURES.values()
                        for requirement in requirements}))

#: The distribution this module never asks the environment about, in either
#: direction.
#:
#: **It was called ``ABSENT_FOR_REAL`` and described as "the one distribution
#: no test environment has".  That was measured to be false**: the
#: repository's primary venv carries MomentRFI as an editable install, and the
#: two tests that trusted the claim failed there and nowhere else.  Every
#: other row in :data:`ROUTES` can be handled by :func:`present_or_skip`,
#: which skips honestly when a distribution is missing; MomentRFI cannot,
#: because it is present in one of this repository's two environments and
#: absent in the other, so *either* real-environment branch loses its coverage
#: in one of them.  Simulating both directions is what makes the verdict the
#: same in both.
PRESENCE_ALWAYS_SIMULATED = "MomentRFI"


def blocked(monkeypatch, *names):
    """Make ``names`` absent, the way §0.3 E.2(7) names.

    ``sys.modules[name] = None`` is CPython's own import block: ``import name``
    raises ``ImportError`` and ``find_spec(name)`` returns ``None``.  It is
    used rather than a fake meta-path finder because a finder simulates a
    BROKEN install, not a missing one, and the two are different verdicts --
    ``test_a_broken_install_is_not_an_absent_one`` in the feature module is
    that distinction.
    """
    for name in names:
        monkeypatch.setitem(sys.modules, name, None)


def present_or_skip(module):
    """Skip unless ``module`` is findable -- **without importing it**.

    Not ``pytest.importorskip``, and the reason is measured rather than
    stylistic.  ``find_spec`` is the exact predicate A35 reads, so this skips
    on precisely the condition the check would answer differently on; and
    importing the real thing is not free -- pyuvdata and pygdsm are seconds
    each, against §0.1's ~2 s ceiling for any single test.  The environments
    genuinely differ: the main checkout's venv carries no pyuvdata at all.
    """
    import importlib.util

    if module in sys.modules and sys.modules[module] is not None:
        return
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"{module} is not installed in this environment")


def a35_wheres(document):
    """Every place A35 pointed at, sorted.

    Tests assert about the ENTRY they are about rather than about ``ids()``,
    for the reason §0.3 E.11 gives: the shared base document carries
    ``inference.parameters.g.prior``, which is itself an A35 token, so a
    document with every module blocked is never A35-free and ``"A35" not in
    ids(doc)`` would be an assertion about the fixture rather than about the
    row under test.
    """
    from rheplicant.config.preflight import preflight

    return sorted(one.where for one in preflight(document).findings
                  if one.check == "A35")


#: An ``inference:`` patch whose latent declares NO prior, for the rows that
#: block numpyro and are not about the prior route.
NO_PRIOR = {"parameters": {"g": {"init": 1.0, "into": "gain.gain"}}}


def beam(**spec):
    """A ``resources:`` patch carrying one beam entry and nothing else."""
    return {"beams": {"horn": {"nside": 4, "normalize": "pixel_sum", **spec}}}


#: The eight routes measured in the plan's own table, plus the four §0.3 E.2
#: added.  Each row is ``(id, the patch, the where, the module to block)``.
ROUTES = (
    ("beam-format-cst",
     {"resources": beam(format="cst", directory="cst", phi0_deg=0.0,
                        phi_sense="ccw")},
     "resources.beams.horn", "limTOD"),
    ("beam-format-uvbeam",
     {"resources": beam(format="uvbeam", path="b.beamfits")},
     "resources.beams.horn", "pyuvdata"),
    ("beam-format-healpix",
     {"resources": beam(format="healpix", path="b.fits", order="ring",
                        freq={"ones": ["n_freq"]}, frame="beam_local")},
     "resources.beams.horn", "healpy"),
    ("beam-format-gaussian",
     {"resources": beam(format="gaussian", fwhm_deg=10.0, frame="beam_local")},
     "resources.beams.horn", "healpy"),
    ("beam-horizon-truncate-map",
     {"resources": beam(format="npy", path="b.npy", frame="beam_local",
                        horizon={"mode": "truncate_map"})},
     "resources.beams.horn", "limtod_jax"),
    ("projector-engine-driftscan",
     {"resources": {"projectors": {"p": {"engine": "driftscan", "lmax": 8,
                                         "normalize_beam": True,
                                         "beam": {"ref": "resources.beams.horn"}}}}},
     "resources.projectors.p", "limtod_jax"),
    ("projector-engine-general-pointing",
     {"resources": {"projectors": {"p": {"engine": "general_pointing", "lmax": 8,
                                         "nside": 4, "normalize_beam": True,
                                         "beam": {"ref": "resources.beams.horn"}}}}},
     "resources.projectors.p", "limtod_jax"),
    ("s-params-termination",
     {"resources": {"s_params": {"z": {"kind": "termination",
                                       "termination": "open"}}}},
     "resources.s_params.z", "rhino_cal_jax"),
    ("sky-model-gdsm",
     {"resources": {"sky_models": {"s": {"kind": "gdsm", "nside": 8}}}},
     "resources.sky_models.s", "pygdsm"),
    ("observation-from-file",
     {"observation": {"from_file": {"format": "rhino_hdf5", "path": "obs.h5",
                                    "freq_unit": "MHz"}}},
     "observation.from_file", "h5py"),
    ("model-noise-wave",
     {"model": {"noise_wave": {"type": "NoiseWaveOperator"}}},
     "model.noise_wave", "rhino_cal_jax"),
    ("model-flagging-momentrfi",
     {"model": {"flagging": {"type": "MomentRFIFlaggingOperator"}}},
     "model.flagging", "MomentRFI"),
    ("transform-beam-analysis",
     {"inference": {"parameters": {"g": {"init": 1.0, "into": "gain.gain",
                                         "transform": {"beam_analysis":
                                                       {"nside": 4, "lmax": 8}}}}}},
     "inference.parameters.g", "limtod_jax"),
    ("run-kind-nuts",
     {"runs": [{"kind": "nuts", "name": "n"}], "inference": NO_PRIOR},
     "runs[0]", "numpyro"),
    ("inference-parameter-prior",
     {"inference": {"parameters": {"g": {"init": 1.0, "into": "gain.gain",
                                         "prior": {"normal": {"loc": 1.0,
                                                              "scale": 0.5}}}}}},
     "inference.parameters.g", "numpyro"),
)
ROUTE_IDS = [row[0] for row in ROUTES]

#: Every ``(token, value)`` that needs nothing optional.  Written out rather
#: than derived from ``_FEATURES``, deliberately: derived, it would agree with
#: the table by construction and assert nothing at all.
EMPTY = frozenset({
    ("format", "npy"), ("format", "npz"), ("format", "inline"),
    ("format", "python"),
    ("horizon", "none"), ("horizon", "projector_mask"),
    ("engine", "matrix"),
    ("s_params", "touchstone"),
    ("sky_model", "uniform"), ("sky_model", "power_law"),
    ("sky_model", "maps"), ("sky_model", "python"),
    ("run", "npe"),
})


class TestTheRegistration:
    def test_A35_is_this_module_s_function(self):
        assert CHECKS["A35"] is _extras

    def test_A35_is_WIRED_into_the_pass_and_not_merely_decorated(self):
        """The mutation the test above cannot see, and it is the one that ships.

        ``CHECKS["A35"] is _extras`` passes because THIS MODULE's own
        ``from ...depends import _extras`` runs the decorator.  Deleting the
        ``@register`` decorator and deleting the foot-import LINE are different
        mutations, and only the second decides whether A35 runs for a user --
        measured on this commit: with the foot import removed, all 86 tests of
        this task were green while
        ``python -c "from rheplicant.config.preflight import CHECKS;
        print('A35' in CHECKS)"`` printed **False**.

        Six wave-1 branches each insert one line into
        ``preflight/__init__.py``.  A merge that drops mine is caught by
        ``test_config_preflight.py::TestTheFootImportCannotRot`` -- a guard in a
        file no wave-1 task owns -- and, until this test, by nothing in the task
        itself.

        A SUBPROCESS, because in this process the import has already happened
        and cannot be un-happened.
        """
        done = subprocess.run(
            [sys.executable, "-c",
             "from rheplicant.config.preflight import CHECKS\n"
             "assert 'A35' in CHECKS, sorted(CHECKS)\n"],
            capture_output=True, text=True, cwd=str(_ROOT))
        assert done.returncode == 0, (
            "A35 is decorated but not wired: importing the package does not "
            "import preflight/depends.py, so the pass never runs it.\n"
            + done.stdout + done.stderr)

    def test_this_module_head_imports_no_sibling_under_preflight(self):
        """**Not** the precedence claim §0.3 C.5 asks for -- the mechanism under
        it, which is the only half this branch can honestly assert.

        C.5 says the alphabetical foot block decides registration order, and
        that ``depends`` sorting first therefore makes A35 run first.
        **Measured here, that reasoning is wrong even where its conclusion
        holds.**  What decides the order is which module is IMPORTED first, and
        a foot-imported module that head-imports a sibling registers that
        sibling's checks ahead of its own -- a sibling task measured exactly
        that: a module sorting first in the foot block whose own check landed
        at index 29 of 40.

        **This docstring used to say "A35 is at index 0 in this tree
        (``list(CHECKS)[:4] == ['A35', 'A1.runs', 'A1.horizon',
        'A1.variants']``, measured in a fresh process)". It was true when it
        was written and is now false.** Re-measured in a fresh process at the
        merged tree: ``len(CHECKS)`` is **49**, ``list(CHECKS)[:4]`` is
        ``['A1.runs', 'A1.horizon', 'A1.variants', 'A38']``, and **A35 is at
        index 30**.  Five sibling tasks landed foot imports after this one was
        written and every one of them moved it.

        Which is the finding, not an embarrassment: **the sentence that
        rotted is the sentence explaining why the index must not be
        asserted**, and it rotted by exactly the mechanism it describes.  The
        test itself never touched the index -- its assertion is the ``ast``
        walk below -- so a suite of 7264 tests could not see the measurement
        going stale.  A measured number written into prose has no guard; only
        the assertion does.  The number is kept here rather than deleted
        because the *mechanism* claim still needs an illustration, and it is
        dated so the next reader knows to re-take it.

        The mechanism, which is what the assertion below holds: every helper
        this module borrows from ``preflight/document.py`` and
        ``preflight/model.py`` is imported inside a function, so importing
        ``depends`` registers A35 and nothing else.  The position itself is
        not asserted, because a sibling landing a module that sorts before
        ``depends`` and head-imports ``document`` moves it -- green here, red
        at merge, which is exactly what R8 bans.  That is not hypothetical any
        more; it is what happened.
        """
        import ast

        source = pathlib.Path(_extras.__globals__["__file__"]).read_text()
        module = ast.parse(source)
        head = [node for node in module.body
                if isinstance(node, (ast.Import, ast.ImportFrom))]
        named = {node.module for node in head if isinstance(node, ast.ImportFrom)}
        named |= {alias.name for node in head if isinstance(node, ast.Import)
                  for alias in node.names}
        siblings = {name for name in named
                    if name and name.startswith("rheplicant.config.preflight.")}
        assert siblings == set(), (
            f"{sorted(siblings)} are head-imported, so their checks register "
            "before this module's own and A35 no longer leads the pass")

    def test_the_base_document_earns_no_A35(self):
        """The fixture is the one every other test is a patch of.

        It carries ``inference.parameters.g.prior`` -- a real A35 token -- so
        this is not vacuous: it asserts that a token with its distribution
        PRESENT says nothing, which is the property that keeps A35 out of every
        other module's report.
        """
        assert "A35" not in ids(preflight_document())


class TestTheTableIsTheSourcesOwn:
    """The table is a second source of truth, so it is compared to the first.

    A ninth beam format or a fourth engine landing in ``kinds/`` with no row
    here would leave A35 silently blind to it; these are what make that a
    failing test rather than a discovery three plans later.
    """

    @pytest.mark.parametrize("token, declared", [
        ("format", BEAM_FORMATS),
        ("engine", ENGINES),
        ("s_params", S_PARAM_KINDS),
        ("sky_model", SKY_KINDS),
    ], ids=["beam-formats", "projector-engines", "s-param-kinds", "sky-kinds"])
    def test_the_table_covers_every_value_the_source_declares(self, token, declared):
        covered = {value for name, value in _FEATURES if name == token}
        assert covered == set(declared)

    def test_the_horizon_modes_are_build_beams_own_three(self):
        """``kinds/beams.py`` writes these as literals in an ``elif``, with no
        constant to import -- so they are restated here and this is what goes
        red if a fourth mode is added."""
        covered = {value for name, value in _FEATURES if name == "horizon"}
        assert covered == {"none", "truncate_map", "projector_mask"}

    def test_every_run_kind_named_is_an_exit_this_layer_runs(self):
        """Only ``nuts`` and ``npe`` have a verdict; the other fourteen exits
        need nothing optional and are not listed, because listing sixteen rows
        of ``()`` would be noise.  What IS asserted is that no row here names a
        kind ``parse_runs`` would refuse -- a stale row is a check that can
        never fire."""
        named = {value for name, value in _FEATURES if name == "run"}
        assert named == {"nuts", "npe"}
        assert named <= set(_KINDS)

    def test_the_ten_section_tokens_each_have_a_phrase(self):
        """§0.3 E.2(3): the section-token is one of ten literal words, never a
        dotted path with a hole in it.  Both halves are asserted -- the words
        themselves, and that every token in the table has a sentence to say."""
        assert set(_TRIGGER) == {
            "format", "horizon", "engine", "s_params", "sky_model", "file",
            "node", "transform", "run", "prior"}
        assert {name for name, _ in _FEATURES} == set(_TRIGGER)
        assert all("." not in token for token in _TRIGGER)

    def test_every_distribution_named_has_a_followable_install_line(self):
        """R4: advice that cannot be followed is not advice.  Both directions,
        because a stale ``_INSTALL`` row is advice about a requirement nobody
        has."""
        named = {requirement[0]
                 for requirements in _FEATURES.values()
                 for requirement in requirements}
        named |= {requirement[0] for _, requirement in _CONDITIONAL.values()}
        assert named == set(_INSTALL)

    def test_no_requirement_probes_a_dotted_module(self):
        """§0.3 E.2(1) mechanised.  One ``find_spec`` on a DOTTED name imports
        the parent package -- measured, 26 ms to 1180.8 ms against a 50 ms
        bound -- so every module in the table is a top-level name and every
        submodule requirement is probed by its distribution instead."""
        for requirements in _FEATURES.values():
            for _, module, _ in requirements:
                assert "." not in module, module

    def test_the_rows_that_need_nothing_still_need_nothing(self):
        """Every empty row, in BOTH directions.

        §0.3 E.2(6) names two of them -- ``s_params: touchstone`` and
        ``runs[].kind: npe`` -- and each has a test of its own in
        :class:`TestTheStandDowns` and :class:`TestTheTwins`.  The other eleven
        had nothing: measured, giving ``("format","npy")`` a healpy requirement
        survived both new modules AND the whole of ``tests/config``, because
        healpy is installed here and A35 therefore stays silent -- including on
        ``preflight_helpers.UNREADABLE_BEAM``, which is an npy beam.  On any
        install without healpy that spurious row refuses a document that
        builds.

        The equality is what makes it total: a row moving OUT of ``EMPTY``
        (gaining a requirement) and a row moving IN (losing a real one) are
        both failures here.
        """
        assert {key for key, value in _FEATURES.items() if not value} == EMPTY
        for key in EMPTY:
            assert key not in _CONDITIONAL, key

    def test_no_gdsm_extra_is_advised(self):
        """§0.3 E.2(4): there is no ``rheplicant[gdsm]``.  The shipped gate
        advises ``limTOD[gdsm]`` and so must this one, or the remedy is a pip
        command that fails."""
        assert 'pip install "limTOD[gdsm]"' in _INSTALL["pygdsm"]
        assert "There is no rheplicant[gdsm]." in _INSTALL["pygdsm"]
        assert 'install "rheplicant[gdsm]"' not in _INSTALL["pygdsm"]


class TestEveryRouteThatReachesTheUserBadlyToday:
    @pytest.mark.parametrize("patch, where, module",
                             [row[1:] for row in ROUTES], ids=ROUTE_IDS)
    def test_the_route_earns_exactly_one_A35_naming_its_own_place(
            self, monkeypatch, patch, where, module):
        """One finding per place, pointing at the entry the reader must edit.

        ``only`` rather than an ``in`` assertion: a walk that forgot to break
        would fire twice on one entry, and no membership test can see that.
        The driftscan row is the one that would -- it carries two requirements
        on one module once ``uniform_sampling:`` is written, which is what
        ``_in_layer``'s de-duplication is for.
        """
        blocked(monkeypatch, module)
        finding = only(preflight_document(**patch), "A35")
        assert finding.severity == REFUSE
        assert finding.where == where
        assert finding.message.startswith(f"{where}: ")
        assert finding.message.endswith("(check A35).")

    @pytest.mark.parametrize("patch, where, module",
                             [row[1:] for row in ROUTES], ids=ROUTE_IDS)
    def test_the_same_route_is_silent_when_the_distribution_is_there(
            self, monkeypatch, patch, where, module):
        """The anti-vacuity partner of the test above, and the reason every
        row of :data:`ROUTES` names the module it blocks.

        Without this, an A35 that fired on the TOKEN rather than on the token
        AND the absence would pass every positive test in this module while
        refusing every document in the world.  ``MomentRFI`` is given a
        stand-in rather than the real thing, because it is the one requirement
        present in one of this repository's two environments and absent in the
        other: ``present_or_skip`` would drop this row's coverage in whichever
        tree lacks it, and asserting it is there would fail in that tree
        outright.  The stand-in gives the same verdict in both -- and it is
        itself the assertion that A35 reads the environment and not the
        document alone.
        """
        if module == PRESENCE_ALWAYS_SIMULATED:
            monkeypatch.setitem(sys.modules, module, _stand_in(module))
        else:
            present_or_skip(module)
        assert where not in a35_wheres(preflight_document(**patch))


def _stand_in(name):
    """A module object ``find_spec`` will report as present.

    The ``__spec__`` is not decoration: without one ``find_spec`` raises
    ``ValueError``, the pass wraps that as *"pre-flight check 'A35' RAISED"*,
    and a test written without it silently measures "the check crashed"
    instead of "the module is there" (§0.3 E.2(7)).
    """
    import importlib.util
    import types

    module = types.ModuleType(name)
    module.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    return module


class TestTheWholeSentence:
    """S1: equality on the whole text, never ``match=``.

    A35 is INVENTED, so there is no pre-existing pin to keep green and no
    one-binding row -- these four literals are the only thing standing between
    a reworded message and a green suite.  Four rather than fifteen because the
    sentence is BUILT from the tables, and these four cover every clause the
    builder can add: the plain one, the two-distribution one, the one carrying
    the submodule note, and the one carrying an alternative route.
    """

    def test_the_healpix_beam_says_this(self, monkeypatch):
        blocked(monkeypatch, "healpy")
        finding = only(preflight_document(**ROUTES[2][1]), "A35")
        assert finding.message == (
            "resources.beams.horn: format: healpix needs the healpy distribution, and "
            "healpy is not importable in this environment. healpy arrives with limTOD's "
            "own dependencies, so a missing one means the install is incomplete: pip "
            'install "limTOD[jax]>=1.10". Said from the document\'s text, so that a '
            "missing dependency arrives before the run rather than as an ImportError in "
            "the middle of one (check A35).")

    def test_the_cst_beam_says_which_submodule_it_did_not_probe(self, monkeypatch):
        """§0.3 E.2(1): a submodule requirement is probed by its top-level
        distribution only, **and the message says so**.  A reader with limTOD
        installed who still meets an ``ImportError`` from ``limTOD.cstbeam``
        must not be left thinking this check had already cleared it."""
        blocked(monkeypatch, "limTOD")
        finding = only(preflight_document(**ROUTES[0][1]), "A35")
        assert finding.message == (
            "resources.beams.horn: format: cst needs the limTOD distribution, and limTOD "
            "is not importable in this environment. limTOD is a hard dependency of this "
            "package rather than an extra, so a missing one means the install is broken "
            'or limTOD was removed: pip install "limTOD[jax]>=1.10". Only the top-level '
            "limTOD is probed here: the submodules this layer reaches (limTOD.cstbeam, "
            "limTOD.uvbeam, limTOD.sky_model) are settled by their own gates when the "
            "resource is built, because probing one at this phase would import limTOD -- "
            "measured at 1180.8 ms against this pass's 50 ms budget. Said from the "
            "document's text, so that a missing dependency arrives before the run rather "
            "than as an ImportError in the middle of one (check A35).")

    def test_the_momentrfi_node_says_what_needs_none_of_it(self, monkeypatch):
        """The alternative clause, and the sentence :class:`TestApplyingTheAdvice`
        then follows.

        ``blocked`` rather than the real environment, and that is a
        correction.  This line read ``del monkeypatch  # MomentRFI is absent
        for real; nothing to simulate.``, which was true of every worktree and
        false of the repository's own primary venv -- where MomentRFI is
        installed, A35 correctly stands down, and ``only`` then asserts on
        zero findings: *"A35 produced 0 findings on this document, not one:
        []"*.  See :data:`PRESENCE_ALWAYS_SIMULATED`.
        """
        blocked(monkeypatch, PRESENCE_ALWAYS_SIMULATED)
        finding = only(preflight_document(**ROUTES[11][1]), "A35")
        assert finding.message == (
            "model.flagging: MomentRFIFlaggingOperator needs the MomentRFI distribution, "
            "and MomentRFI is not importable in this environment. MomentRFI is the 'rfi' "
            "extra and is not on PyPI, so the extra names the requirement rather than "
            'resolving it: uv pip install "MomentRFI @ '
            'git+https://github.com/zzhang0123/MomentRFI". The threshold-based '
            "FlaggingOperator needs none of it. Said from the document's text, so that a "
            "missing dependency arrives before the run rather than as an ImportError in "
            "the middle of one (check A35).")

    def test_a_uvbeam_missing_both_hears_about_both(self, monkeypatch):
        """Two distributions for one route, which is the whole reason
        ``_FEATURES`` maps to a TUPLE of requirements.  A table mapping one
        token to one requirement passes every other test in this module."""
        blocked(monkeypatch, "pyuvdata", "limTOD")
        findings = [one for one in _findings(preflight_document(**ROUTES[1][1]))
                    if one.check == "A35"]
        assert [one.message.split(" needs ")[1].split(",")[0] for one in findings] == [
            "the pyuvdata distribution", "the limTOD distribution"]


def _findings(document):
    from rheplicant.config.preflight import preflight

    return preflight(document).findings


#: Every ``_INSTALL`` line, verbatim, beside the extra ``pyproject.toml`` must
#: declare for it to be followable.  ``None`` where the requirement is not an
#: extra of this package at all.
INSTALL_LINES = (
    ("limTOD", None,
     "limTOD is a hard dependency of this package rather than an extra, so a missing "
     'one means the install is broken or limTOD was removed: pip install '
     '"limTOD[jax]>=1.10".'),
    ("healpy", None,
     "healpy arrives with limTOD's own dependencies, so a missing one means the "
     'install is incomplete: pip install "limTOD[jax]>=1.10".'),
    ("pygdsm", None,
     "pygdsm is optional and arrives through limTOD's extra rather than through this "
     'package\'s: pip install "limTOD[gdsm]". There is no rheplicant[gdsm].'),
    ("pyuvdata", "uvbeam",
     "pyuvdata is the 'uvbeam' extra; the limTOD bridge itself ships with limTOD, so "
     'only the file reader is missing: uv pip install -e ".[uvbeam]".'),
    ("h5py", "rhino",
     "h5py is the 'rhino' extra and it does resolve from an index: uv pip install -e "
     '".[rhino]".'),
    ("numpyro", "numpyro",
     "numpyro is the 'numpyro' extra: pip install 'rheplicant[numpyro]'."),
    ("rhino-cal-jax", "cal",
     "rhino-cal-jax is the 'cal' extra and is not on PyPI, so the extra names the "
     "requirement rather than resolving it, and the branch matters: uv pip install "
     '"rhino-cal-jax @ '
     'git+https://github.com/RHINO-Experiment/rhino-cal@feat/rhino-cal-jax".'),
    ("MomentRFI", "rfi",
     "MomentRFI is the 'rfi' extra and is not on PyPI, so the extra names the "
     'requirement rather than resolving it: uv pip install "MomentRFI @ '
     'git+https://github.com/zzhang0123/MomentRFI".'),
)


class TestTheMessageTablesThemselves:
    """R2 on the TABLES, not only on the builder's clause shapes.

    The four pins in :class:`TestTheWholeSentence` cover every shape the
    builder can assemble -- and measured, they leave the CONTENTS free: eight
    message mutants survived this commit's first version, three of them R4
    failures shipping green.  ``_INSTALL["h5py"]`` rewritten to advise
    ``uv pip install -e ".[hdf5-reader]"`` -- an extra ``pyproject.toml`` does
    not declare -- passed both new modules and the whole of ``tests/config``.

    The tables are where a reviewer-invisible edit lands, so each is pinned by
    equality on its whole contents, and each install line is checked against
    the extra it names.
    """

    def test_every_install_line_is_pinned_and_no_line_is_unpinned(self):
        assert {name for name, _, _ in INSTALL_LINES} == set(_INSTALL)

    @pytest.mark.parametrize("distribution, extra, text", INSTALL_LINES,
                             ids=[row[0] for row in INSTALL_LINES])
    def test_the_install_line_says_this(self, distribution, extra, text):
        assert _INSTALL[distribution] == text

    @pytest.mark.parametrize("distribution, extra, text", INSTALL_LINES,
                             ids=[row[0] for row in INSTALL_LINES])
    def test_the_extra_it_names_is_one_pyproject_declares(self, distribution,
                                                          extra, text):
        """R4, mechanically: advice that cannot be followed is not advice.

        Reads ``pyproject.toml``'s own ``[project.optional-dependencies]``
        rather than a list here, so an extra renamed there and not here fails
        on the side that is wrong.  The three rows with no extra of ours --
        limTOD, healpy, pygdsm -- assert the opposite: their advice must point
        at NO extra of this package, which is §0.3 E.2(4)'s "there is no
        rheplicant[gdsm]".

        Two rows -- ``cal`` and ``rfi`` -- name their extra in prose and then
        give a git URL rather than a ``pip install .[extra]``, because those
        extras NAME the requirement rather than resolving it (pyproject says so
        in its own comments).  So the text must carry the extra either quoted
        or as a pip extra spec, and both spellings count.
        """
        import tomllib

        with (_ROOT / "pyproject.toml").open("rb") as handle:
            declared = set(tomllib.load(handle)["project"]["optional-dependencies"])
        if extra is None:
            assert not any(f"[{name}]" in text for name in declared), (
                f"{distribution}'s advice points at an extra of this package, "
                "and it has none")
        else:
            assert extra in declared, (
                f"{distribution} is advised as the {extra!r} extra and "
                f"pyproject.toml declares {sorted(declared)}")
            assert f"'{extra}'" in text or f"[{extra}]" in text

    def test_every_trigger_phrase_says_this(self):
        assert _TRIGGER == {
            "format": "format: {value}",
            "horizon": "horizon.mode: {value}",
            "engine": "engine: {value}",
            "s_params": "kind: {value}",
            "sky_model": "kind: {value}",
            "file": "format: {value}",
            "node": "{value}",
            "transform": "transform: {value}",
            "run": "kind: {value}",
            "prior": "a declared prior",
        }

    def test_every_alternative_says_this(self):
        """Each names a route that needs nothing optional, quoted from the
        package's own gate.  A rewritten one that named a route which is NOT
        dependency-free would be an R4 loop, and survived until this pin."""
        matrix = ("engine: matrix takes a precomputed sky->TOD matrix and needs no "
                  "optional dependency (fixed pointing and beam only).")
        assert _ALTERNATIVE == {
            ("engine", "driftscan"): matrix,
            ("engine", "general_pointing"): matrix,
            ("node", "MomentRFIFlaggingOperator"):
                "The threshold-based FlaggingOperator needs none of it.",
        }
        assert _FEATURES[("engine", "matrix")] == ()


class TestTheWholeSentenceOnEveryDistribution:
    """One whole-text pin per remaining ``_INSTALL`` row, driven through the
    route that names it -- so the table and the builder are pinned together
    rather than only apart.
    """

    def test_the_from_file_route_says_this(self, monkeypatch):
        blocked(monkeypatch, "h5py")
        finding = only(preflight_document(**ROUTES[9][1]), "A35")
        assert finding.message == (
            "observation.from_file: format: rhino_hdf5 needs the h5py distribution, and "
            "h5py is not importable in this environment. h5py is the 'rhino' extra and "
            'it does resolve from an index: uv pip install -e ".[rhino]". Said from the '
            "document's text, so that a missing dependency arrives before the run rather "
            "than as an ImportError in the middle of one (check A35).")

    def test_the_nuts_run_says_this(self, monkeypatch):
        """Also the only pin on ``_TRIGGER["run"]``'s sentence in a real
        message: rewriting it to ``run: {value}`` survived everything."""
        blocked(monkeypatch, "numpyro")
        finding = only(preflight_document(**ROUTES[13][1]), "A35")
        assert finding.message == (
            "runs[0]: kind: nuts needs the numpyro distribution, and numpyro is not "
            "importable in this environment. numpyro is the 'numpyro' extra: pip install "
            "'rheplicant[numpyro]'. Said from the document's text, so that a missing "
            "dependency arrives before the run rather than as an ImportError in the "
            "middle of one (check A35).")

    def test_the_declared_prior_says_this(self, monkeypatch):
        """``_TRIGGER["prior"]`` is the one phrase that names no key, because
        the route is a key's PRESENCE.  Reworded, it survived everything."""
        blocked(monkeypatch, "numpyro")
        finding = only(preflight_document(**ROUTES[14][1]), "A35")
        assert finding.message == (
            "inference.parameters.g: a declared prior needs the numpyro distribution, "
            "and numpyro is not importable in this environment. numpyro is the 'numpyro' "
            "extra: pip install 'rheplicant[numpyro]'. Said from the document's text, so "
            "that a missing dependency arrives before the run rather than as an "
            "ImportError in the middle of one (check A35).")

    def test_the_uvbeam_route_says_this(self, monkeypatch):
        blocked(monkeypatch, "pyuvdata")
        finding = only(preflight_document(**ROUTES[1][1]), "A35")
        assert finding.message == (
            "resources.beams.horn: format: uvbeam needs the pyuvdata distribution, and "
            "pyuvdata is not importable in this environment. pyuvdata is the 'uvbeam' "
            "extra; the limTOD bridge itself ships with limTOD, so only the file reader "
            'is missing: uv pip install -e ".[uvbeam]". Said from the document\'s text, '
            "so that a missing dependency arrives before the run rather than as an "
            "ImportError in the middle of one (check A35).")

    def test_the_noise_wave_node_names_the_DISTRIBUTION_and_the_MODULE(
            self, monkeypatch):
        """**The whole reason ``Requirement`` is a triple**, pinned.

        ``rhino-cal-jax`` is the distribution and ``rhino_cal_jax`` is the
        module, and the two rows where they differ are this one and limTOD /
        limtod_jax.  Collapsing the head to name the module twice -- an edit no
        reader would question -- survived every other test in this task.
        """
        blocked(monkeypatch, "rhino_cal_jax")
        finding = only(preflight_document(**ROUTES[10][1]), "A35")
        assert finding.message == (
            "model.noise_wave: NoiseWaveOperator needs the rhino-cal-jax distribution, "
            "and rhino_cal_jax is not importable in this environment. rhino-cal-jax is "
            "the 'cal' extra and is not on PyPI, so the extra names the requirement "
            "rather than resolving it, and the branch matters: uv pip install "
            '"rhino-cal-jax @ '
            'git+https://github.com/RHINO-Experiment/rhino-cal@feat/rhino-cal-jax". '
            "Said from the document's text, so that a missing dependency arrives before "
            "the run rather than as an ImportError in the middle of one (check A35).")


class TestTheTwins:
    """S3.  Each of these is a second route to a rule the first one guards."""

    def test_an_UNSELECTED_variant_that_introduces_the_route_is_walked(
            self, monkeypatch):
        """§0.3 F.5(1), and **this is the only property the layer walk buys**.

        The document's BASE carries no beam, so a base-only check finds
        nothing; the walk is what reports a route that only ``variants.big``
        introduces, on a pass where no variant was selected.  Measured cost of
        buying it: ``_task3_over_layers`` builds every layer eagerly through
        ``apply_variant``, which deep-copies the document, and on 3A's
        cold-cost document that is **3.65 ms** -- see this task's report.
        """
        blocked(monkeypatch, "healpy")
        document = preflight_document(
            variants={"big": {"resources": beam(format="gaussian", fwhm_deg=10.0,
                                                frame="beam_local")}})
        finding = only(document, "A35")
        assert finding.message.startswith("variants.big: resources.beams.horn: ")

    def test_a_SELECTED_variant_reaches_the_check_without_the_layer_walk(
            self, monkeypatch):
        """The other half of the same twin, and the one that is NOT the walk's.

        ``config/document.py::_assemble`` applies the selected variant BEFORE
        the pre-flight pass runs, so a check that only ever read the document
        it is handed already sees a route the chosen variant introduced.  This
        test proves that route works; the test above proves the un-selected one
        does; and stating which is which is what stops the layer walk being
        credited with both.
        """
        from rheplicant.config import ConfigError, load_document

        blocked(monkeypatch, "healpy")
        document = preflight_document(
            variants={"big": {"resources": beam(format="gaussian", fwhm_deg=10.0,
                                                frame="beam_local")}})
        with pytest.raises(ConfigError) as excinfo:
            load_document(document, variant="big")
        assert "needs the healpy distribution" in str(excinfo.value)

    def test_the_same_route_in_base_and_variant_is_said_once(self, monkeypatch):
        """``_task3_over_layers``' own contract, exercised on this check: a
        document with the fault in its base and four variants must not hand the
        reader the same sentence five times."""
        blocked(monkeypatch, "healpy")
        patch = beam(format="gaussian", fwhm_deg=10.0, frame="beam_local")
        document = preflight_document(
            resources=patch,
            variants={f"v{index}": {"runtime": {"seed": index}} for index in range(4)})
        assert only(document, "A35").where == "resources.beams.horn"

    def test_a_variant_s_own_route_is_not_counted_twice(self, monkeypatch):
        """Why ``_routes`` skips the top-level ``variants:`` section.

        ``_task3_over_layers`` has ALREADY applied each variant, so a walk that
        also descended into the raw ``variants:`` block would meet the variant's
        own node twice -- once as a variant layer and once as text sitting in
        the base -- and the two carry different ``where``s, so the finding-level
        de-duplication cannot collapse them.

        Measured on a document with a ``{file:}`` node in the base AND in a
        variant: shipped, two findings; with the skip removed, **three**, the
        variant's reported twice -- which also breaks ``only()`` for every
        caller.  It survived the whole of ``tests/config`` until this test.
        """
        blocked(monkeypatch, "h5py")
        node = {"file": {"path": "o.h5", "format": "rhino_hdf5"}}
        document = preflight_document(
            resources={"arrays": {"base": dict(node)}},
            variants={"big": {"resources": {"arrays": {"rec": dict(node)}}}})
        assert a35_wheres(document) == ["resources.arrays.base",
                                        "variants.big.resources.arrays.rec"]

    def test_inference_twin_replace_reaches_the_same_builder(self, monkeypatch):
        """§0.3 E.10, and it is a real candidate rather than a formality:
        ``twin.py:69`` sends ``replace.<node>`` into ``build_node_operator``,
        the same function ``model.<node>`` reaches, and ``preflight/model.py::
        _nodes`` cannot see it.  The message names the section it FOUND, which
        a verbatim hoist of A13's ``f"model.{node_id}: "`` could not."""
        blocked(monkeypatch, "rhino_cal_jax")
        document = preflight_document(
            inference={"twin": {"without": ["noise"],
                                "replace": {"noise_wave":
                                            {"type": "NoiseWaveOperator"}}}})
        finding = only(document, "A35")
        assert finding.where == "inference.twin.replace.noise_wave"
        assert finding.message.startswith(
            "inference.twin.replace.noise_wave: NoiseWaveOperator needs ")

    def test_a_file_value_node_is_the_from_file_twin(self, monkeypatch):
        """A10's twin is A35's too: ``observation.from_file`` is not the only
        h5py route.  ``parse_from_file`` resolves ``{"file": dict(spec)}``
        itself (``sections/ingest.py:113``), so ANY value node writing
        ``{file: {format: rhino_hdf5}}`` reaches the same reader -- measured
        at ``resources.arrays.<n>``, which is where this one writes it."""
        blocked(monkeypatch, "h5py")
        document = preflight_document(
            resources={"arrays": {"rec": {"file": {"path": "obs.h5",
                                                   "format": "rhino_hdf5"}}}})
        finding = only(document, "A35")
        assert finding.where == "resources.arrays.rec"

    def test_a_binding_s_transform_is_the_latent_s_transform_s_twin(
            self, monkeypatch):
        """``parse_transform`` has two call sites, not one.

        ``sections/transforms.py:354`` reads a LATENT's own ``transform:`` and
        ``:393`` reads a BINDING entry's, and ``{beam_analysis: ...}`` under
        either reaches ``limtod_jax.map2alm_iter``.  The module docstring
        claims both; until this test only the parameters site was driven, and
        deleting the bindings walk survived everything.
        """
        blocked(monkeypatch, "limtod_jax")
        document = preflight_document(inference={
            "parameters": {"g": {"init": 1.0}},
            "bindings": [{"latents": ["g"], "into": "gain.gain",
                          "transform": {"beam_analysis": {"nside": 4, "lmax": 8}}}]})
        finding = only(document, "A35")
        assert finding.message.startswith(
            "inference.bindings[0]: transform: beam_analysis needs ")

    def test_a_file_node_inside_a_LIST_is_pointed_at_by_its_index(
            self, monkeypatch):
        """The ``[index]`` leg of the path builder, which nothing else drives.

        Every other route lands on a mapping key, so a ``_joined`` that
        spelled list positions with a dot -- ``runs.0.at.g`` -- answered every
        existing test correctly.  ``config/paths.py``'s grammar is what a
        reader pastes back into their document, and it writes ``runs[0]``.
        """
        blocked(monkeypatch, "h5py")
        document = preflight_document(runs=[{
            "kind": "forward", "name": "f",
            "at": {"g": {"file": {"path": "o.h5", "format": "rhino_hdf5"}}}}])
        finding = only(document, "A35")
        assert finding.message.startswith("runs[0].at.g: format: rhino_hdf5 needs ")

    def test_a_single_run_written_as_a_MAPPING_is_walked(self, monkeypatch):
        """``parse_runs`` accepts one exit as a bare mapping (``runs.py:122``:
        ``if isinstance(section, Mapping): section = [section]``), so a walk
        that only handled a list would miss every single-run document written
        that way -- and the schema's own pages use the form."""
        blocked(monkeypatch, "numpyro")
        document = preflight_document(runs={"kind": "nuts", "name": "n"},
                                      inference=NO_PRIOR)
        finding = only(document, "A35")
        assert finding.where == "runs[0]"

    def test_the_python_spelling_of_a_node_is_the_type_spelling_s_twin(
            self, monkeypatch):
        """``{python: 'rheplicant.radio:NoiseWaveOperator'}`` builds the same
        class as ``{type: NoiseWaveOperator}`` and 3A's tests already exercise
        it.  Resolved through ``preflight/model.py::_t5_radio_class``, which
        imports nothing -- the class is already in ``sys.modules`` because
        ``import rheplicant.config`` imports ``rheplicant.radio``."""
        blocked(monkeypatch, "rhino_cal_jax")
        document = preflight_document(
            model={"noise_wave": {"python": "rheplicant.radio:NoiseWaveOperator"}})
        finding = only(document, "A35")
        assert finding.message.startswith("model.noise_wave: NoiseWaveOperator needs ")

    def test_gaussian_and_healpix_are_one_twin_inside_one_function(
            self, monkeypatch):
        """``_maps_for`` dispatches both, three lines apart: ``_healpix_maps``
        opens with a bare ``import healpy as hp`` and ``_gaussian`` calls
        ``_require_healpy``.  A35 must not inherit that asymmetry."""
        blocked(monkeypatch, "healpy")
        document = preflight_document(resources={"beams": {
            "ga": {"format": "gaussian", "nside": 4, "normalize": "pixel_sum",
                   "fwhm_deg": 10.0, "frame": "beam_local"},
            "hp": {"format": "healpix", "nside": 4, "normalize": "pixel_sum",
                   "path": "b.fits", "order": "ring", "frame": "beam_local",
                   "freq": {"ones": ["n_freq"]}}}})
        wheres = sorted(one.where for one in _findings(document)
                        if one.check == "A35")
        assert wheres == ["resources.beams.ga", "resources.beams.hp"]

    def test_the_two_flagging_classes_are_not_one(self, monkeypatch):
        """Both register at node ``flagging`` and only one needs anything."""
        del monkeypatch
        document = preflight_document(
            model={"flagging": {"type": "FlaggingOperator",
                                "threshold": {"value": 3.0,
                                               "unit": "adc_count"}}})
        assert "A35" not in ids(document)

    def test_the_nuts_exit_and_the_npe_one_are_not_one(self, monkeypatch):
        """§0.3 E.2(6): ``kind: npe`` needs nothing optional.  Blocking numpyro
        for both halves is what makes the negative half mean something."""
        blocked(monkeypatch, "numpyro")
        base = {"inference": {"parameters": {"g": {"init": 1.0,
                                                   "into": "gain.gain"}}}}
        assert only(preflight_document(runs=[{"kind": "nuts", "name": "n"}],
                                       **base), "A35")
        assert "A35" not in ids(preflight_document(
            runs=[{"kind": "npe", "name": "n"}], **base))


#: ``token -> a patch putting ``value`` where that token is read``.  Nine of
#: the ten, and the two absentees are absent for reasons rather than by
#: oversight -- both were driven with every hostile shape below during review
#: and neither could escape a ``TypeError``:
#:
#: * ``prior``'s "value" is the literal ``"declared"`` this module writes,
#:   never anything the document supplies, so it cannot reach the tuple lookup
#:   at all.  :meth:`TestNoHostileDocumentCanAbortA35.
#:   test_the_prior_token_reads_no_user_value` is that statement as an
#:   assertion rather than a claim.
#: * ``transform``'s "value" is a mapping KEY (``for word in block``), and a
#:   mapping key is hashable by construction -- Python refuses to build
#:   ``{["x"]: ...}`` before this check ever sees it.  So the unhashable shapes
#:   that are the whole point of this battery are unreachable there.
HOSTILE_TOKENS = {
    "format": lambda value: {"resources": beam(format=value)},
    "horizon": lambda value: {"resources": beam(format="npy", path="b.npy",
                                                frame="beam_local",
                                                horizon={"mode": value})},
    "engine": lambda value: {"resources": {"projectors": {"p": {"engine": value}}}},
    "s_params": lambda value: {"resources": {"s_params": {"z": {"kind": value}}}},
    "sky_model": lambda value: {"resources": {"sky_models": {"s": {"kind": value}}}},
    "file": lambda value: {"observation": {"from_file": {"format": value,
                                                         "path": "o.h5"}}},
    "file-value-node": lambda value: {
        "resources": {"arrays": {"rec": {"file": {"path": "o.h5",
                                                  "format": value}}}}},
    "node": lambda value: {"model": {"flagging": {"type": value}}},
    "run": lambda value: {"runs": [{"kind": value, "name": "r"}]},
    "variants-inner": lambda value: {
        "variants": {"big": {"resources": beam(format=value)}}},
}

#: The supported evidence shapes a YAML author can put where a token's value
#: belongs.  The unhashable ones are the point:
#: ``_FEATURES.get((token, value))`` is a dict
#: lookup on a tuple carrying user text, and ``{'a': 1}`` or ``['x']`` in that
#: tuple raises ``TypeError: unhashable type`` before any check logic runs.
#: YAML sets are tested separately because the evidence boundary rejects them
#: before any check runs.
HOSTILE_VALUES = (["x"], {"x": 1}, None, 7, True, [["x"]], [{"x": 1}],
                  (1, 2), 1.5)


class TestNoHostileDocumentCanAbortA35:
    """The §2.3 TRAP, and A35 is the worst possible place for it.

    ``_requirements`` does ``_FEATURES.get((token, value))`` -- a dict lookup
    on a tuple one of whose members is copied straight out of the user's
    document.  A list or a mapping there raises ``TypeError: unhashable type``,
    which ``passes.py:207`` turns into *"pre-flight check 'A35' RAISED
    TypeError"*: **the pass aborts and every later finding is discarded.**  A35
    runs first, so it discards all of them.

    ``depends.py``'s ``if not isinstance(value, str): continue`` is what stops
    that, and measured on this commit's first version, removing it let **51**
    documents escape a ``TypeError`` while all 86 of this task's tests stayed
    green -- the only thing that noticed was two of the twenty-three cases of
    ``test_preflight_fitting.py``'s own battery, which happen to land on one of
    A35's ten tokens.  The other nine tokens were covered by nothing.
    """

    @pytest.mark.parametrize("value", HOSTILE_VALUES,
                             ids=[str(index) for index in
                                  range(len(HOSTILE_VALUES))])
    @pytest.mark.parametrize("token", sorted(HOSTILE_TOKENS))
    def test_the_whole_pass_survives_it(self, token, value):
        """The pass RETURNS, and no finding is a report about A35 crashing.

        Both halves are needed: ``preflight`` does not re-raise a check's
        exception, it converts it -- so a test that only called the pass would
        be green on exactly the failure this is about.
        """
        from rheplicant.config.preflight import preflight

        report = preflight(preflight_document(**HOSTILE_TOKENS[token](value)))
        assert not [one for one in report.findings if "'A35' RAISED" in one.message]

    @pytest.mark.parametrize("value", HOSTILE_VALUES,
                             ids=[str(index) for index in
                                  range(len(HOSTILE_VALUES))])
    @pytest.mark.parametrize("token", sorted(HOSTILE_TOKENS))
    def test_it_survives_with_every_distribution_blocked_too(
            self, monkeypatch, token, value):
        """The same sweep on the environment that makes A35 speak.

        With everything present most rows return before building a message, so
        a guard could be missing further down the function and this battery
        would not reach it.
        """
        from rheplicant.config.preflight import preflight

        blocked(monkeypatch, *MODULES)
        report = preflight(preflight_document(**HOSTILE_TOKENS[token](value)))
        assert not [one for one in report.findings if "'A35' RAISED" in one.message]

    @pytest.mark.parametrize("token", sorted(HOSTILE_TOKENS))
    def test_an_unsupported_set_is_rejected_at_the_evidence_boundary(
            self, token):
        """A set is not recursively frozen evidence, so checks never see it."""
        from rheplicant.config.preflight import preflight

        with pytest.raises(
                ConfigError,
                match=r"initial_merge document: unsupported evidence leaf type set"):
            preflight(preflight_document(**HOSTILE_TOKENS[token]({"x"})))

    def test_the_prior_token_reads_no_user_value(self):
        """Why ``prior`` is not in the battery, said rather than left out.

        Its route yields the literal ``"declared"``; the document decides only
        whether the key is THERE.  A hostile ``prior:`` therefore cannot reach
        the tuple lookup, and the ``isinstance`` guard is not what protects it.
        """
        from rheplicant.config.preflight.depends import _routes

        document = preflight_document(inference={"parameters": {
            "g": {"init": 1.0, "into": "gain.gain", "prior": ["nonsense"]}}})
        values = [value for _, token, value, _ in _routes(document)
                  if token == "prior"]
        assert values == ["declared"]


class TestTheStandDowns:
    """§2.3 and §3.2(c): refusing on "I could not tell" refuses documents that
    build, and a test asserting "nothing is emitted" must be shown able to
    fail."""

    def test_a_touchstone_s_param_needs_nothing_optional(self, monkeypatch):
        """§0.3 E.2(6).  ``build_s_param`` returns at ``kind: touchstone``
        BEFORE ``_require_cal``, so the cal extra is not this row's business.
        Anti-vacuity: the same entry under ``kind: termination`` DOES fire, on
        the same blocked environment -- which is what makes the silence above
        a statement about the value rather than about the walk never reaching
        ``resources.s_params``."""
        blocked(monkeypatch, *MODULES)
        entry = {"z": {"kind": "touchstone",
                       "file": {"path": "z.s1p", "format": "touchstone"}}}
        assert "resources.s_params.z" not in a35_wheres(
            preflight_document(resources={"s_params": entry}))
        assert "resources.s_params.z" in a35_wheres(preflight_document(
            resources={"s_params": {"z": {"kind": "termination",
                                          "termination": "open"}}}))

    def test_a_matrix_projector_needs_nothing_optional(self, monkeypatch):
        """The engine that returns before every limTOD call.  Its sibling
        ``driftscan`` fires on the same blocked environment."""
        blocked(monkeypatch, *MODULES)
        matrix = {"projectors": {"p": {"engine": "matrix",
                                       "matrix": {"ones": [4, 4]},
                                       "provenance": {"by": "a test"}}}}
        assert "resources.projectors.p" not in a35_wheres(
            preflight_document(resources=matrix))
        assert "resources.projectors.p" in a35_wheres(preflight_document(
            resources={"projectors": {"p": {"engine": "driftscan", "lmax": 8,
                                            "normalize_beam": True,
                                            "beam": {"ref": "resources.beams.horn"}}}}))

    def test_a_driftscan_projector_does_not_need_healpy(self, monkeypatch):
        """**The test that makes this task's one departure from §0.3 survive
        its own reasoning.**

        §0.3 E.2(5) instructs that ``projectors.engine: driftscan`` be added as
        a route, and the obvious reading gives it a healpy row beside its
        limtod_jax one.  It must not have one: measured,
        ``DriftScanProjector.from_beam_maps`` analyses the beam with
        ``limtod_jax.map2alm_iter`` and reaches healpy nowhere -- an
        independent reviewer confirmed it by BUILDING a driftscan projector
        with ``sys.modules["healpy"] = None`` and getting ``beam_alms (2, 45)``
        back.  ``kinds/projectors.py::_analyse`` is the only ``_require_healpy``
        site and it sits inside the ``general_pointing`` branch, behind
        ``beam_alms is None``.

        A comment in the source is not enough to defend that: the next reader
        who follows the ruling literally re-adds the row, and measured, doing
        so leaves BOTH new modules and the whole of ``tests/config`` green --
        because healpy is installed here, so A35 stays silent either way.  This
        test is red for that edit and only for it.
        """
        entry = {"projectors": {"p": {"engine": "driftscan", "lmax": 8,
                                      "normalize_beam": True,
                                      "beam": {"ref": "resources.beams.horn"}}}}
        blocked(monkeypatch, "healpy")
        assert "resources.projectors.p" not in a35_wheres(
            preflight_document(resources=entry))
        blocked(monkeypatch, "limtod_jax")
        assert "resources.projectors.p" in a35_wheres(
            preflight_document(resources=entry))

    def test_a_general_pointing_projector_is_not_asked_for_healpy_either(
            self, monkeypatch):
        """The sibling, and the reason its row is empty of healpy too.

        ``general_pointing`` DOES reach ``_analyse`` -- but only when
        ``beam_alms:`` is absent, which is a requirement conditional on a
        sibling being ABSENT and ``_CONDITIONAL`` cannot express.  A row here
        would refuse a document carrying ``beam_alms:``, which builds without
        healpy.  ``_require_healpy`` -- already a ``ConfigError`` naming healpy
        -- is the recorded backstop.
        """
        blocked(monkeypatch, "healpy")
        entry = {"projectors": {"g": {"engine": "general_pointing", "lmax": 8,
                                      "nside": 4, "normalize_beam": True,
                                      "beam_alms": {"ones": [45]}}}}
        assert "resources.projectors.g" not in a35_wheres(
            preflight_document(resources=entry))

    def test_a_value_the_table_has_no_row_for_is_left_to_its_builder(
            self, monkeypatch):
        """A typo is ``build_beam``'s refusal to make, naming every format it
        knows -- not this check's, which would pre-empt it with advice about a
        distribution nobody asked for.  The anti-vacuity partner is the same
        entry spelled right."""
        blocked(monkeypatch, *MODULES)
        assert "resources.beams.horn" not in a35_wheres(preflight_document(
            resources=beam(format="helpix", path="b.fits")))
        assert "resources.beams.horn" in a35_wheres(preflight_document(
            resources=beam(format="healpix", path="b.fits", order="ring",
                           frame="beam_local", freq={"ones": ["n_freq"]})))

    def test_a_resources_entry_that_does_not_resolve_is_stood_down_on(
            self, monkeypatch):
        """``resolved_specs`` is TOTAL and DROPS a malformed entry rather than
        raising; a check that read the raw text instead would refuse an entry
        whose ``format:`` comes from its parent, and one that let the
        ``ConfigError`` out would abort the pass and hide every later finding.

        Here the parent does not exist, so the entry never resolves, A35 says
        nothing, and ``load_document`` says the right sentence at the right
        phase.
        """
        blocked(monkeypatch, "healpy")
        document = preflight_document(resources={"beams": {
            "horn": {"extends": "no_such_parent", "format": "gaussian",
                     "nside": 4, "normalize": "pixel_sum", "fwhm_deg": 10.0,
                     "frame": "beam_local"}}})
        assert a35_wheres(document) == []

    def test_a_present_distribution_lets_the_specific_refusal_win(self):
        """S4's stand-down: a document wrong in this check's WAY and wrong in a
        way something else says better.

        ``format: uvbeam`` is exactly A35's subject -- two distributions, the
        route the plan's table calls out -- and with pyuvdata installed A35 has
        nothing to say, so ``build_beam``'s own sentence about ``phi0_deg`` on
        a uvbeam entry is what the reader gets.  A35 runs FIRST of every check
        in this package, so an A35 that fired on the token alone would pre-empt
        it and this is the test that would go red.
        """
        pytest.importorskip("pyuvdata")
        from rheplicant.config import load_document

        document = preflight_document(**{
            "resources": beam(format="uvbeam", path="b.beamfits", phi0_deg=0.0)})
        assert "A35" not in ids(document)
        with pytest.raises(ConfigError) as excinfo:
            load_document(document)
        assert "are not written for format: uvbeam" in str(excinfo.value)


class TestApplyingTheAdvice:
    """S4's second half, on the distribution whose remedy a user can act on.

    MomentRFI is not on PyPI, so the message has to name a git install and a
    document-level alternative rather than an extra -- and the second is the
    one a test can take.

    **The absence is simulated here, and this class used to say the opposite.**
    Its docstring read *"No simulation anywhere in this class: MomentRFI is not
    on PyPI and no test environment carries it"*; the repository's primary venv
    carries it, and the first test below failed there and nowhere else.  Being
    on PyPI and being installed are different questions, and only the first is
    still true.  See :data:`PRESENCE_ALWAYS_SIMULATED`.
    """

    def test_the_document_that_earns_it_would_otherwise_load_and_die_later(
            self, monkeypatch):
        """What A35 buys, stated as the thing it replaces.

        With MomentRFI present the document LOADS -- constructing
        ``MomentRFIFlaggingOperator`` imports nothing -- and the ``ImportError``
        arrives at forward-evaluation time from inside a ``jax.pure_callback``.
        A35 is the whole of what stands between the two.
        """
        blocked(monkeypatch, PRESENCE_ALWAYS_SIMULATED)
        document = preflight_document(
            model={"flagging": {"type": "MomentRFIFlaggingOperator"}})
        assert only(document, "A35")

    @pytest.mark.parametrize("installed", [False, True],
                             ids=["simulated-absent", "simulated-present"])
    def test_both_verdicts_are_reachable_whatever_this_venv_carries(
            self, monkeypatch, installed):
        """Both sides of the only dispatch A35 makes, pinned in one test.

        The check branches on one question -- is the distribution findable --
        and the failure this closes was that the branch taken depended on
        which checkout the suite ran in rather than on anything the test said.
        So both sides are driven here from simulations, and the pair is the
        statement that this document's verdict is a function of the simulation
        and of nothing else.  Whichever venv runs it, both legs execute and
        both are asserted.

        What it does NOT cover, said plainly: it cannot stop a *sibling* test
        from reading the real environment again -- it only pins that the
        simulations reach both answers.  The sibling above is the one that
        regressed, and its own ``blocked`` call is what holds it.
        """
        if installed:
            monkeypatch.setitem(sys.modules, PRESENCE_ALWAYS_SIMULATED,
                                _stand_in(PRESENCE_ALWAYS_SIMULATED))
        else:
            blocked(monkeypatch, PRESENCE_ALWAYS_SIMULATED)
        document = preflight_document(
            model={"flagging": {"type": "MomentRFIFlaggingOperator"}})
        assert a35_wheres(document) == ([] if installed else ["model.flagging"])

    def test_taking_the_advice_leaves_a_document_that_loads(self):
        """R4: the remedy the message names, applied, reaches a document this
        layer accepts.  The other remedy -- installing MomentRFI -- is not one
        a test can take, which is exactly why the message carries a second."""
        from rheplicant.config import load_document

        document = preflight_document(
            model={"flagging": {"type": "FlaggingOperator",
                                "threshold": {"value": 3.0,
                                               "unit": "adc_count"}}})
        assert "A35" not in ids(document)
        load_document(document)
