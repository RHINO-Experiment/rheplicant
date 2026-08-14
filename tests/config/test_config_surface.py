"""What rheplicant.config exports, and the boundary it keeps."""

import pathlib
import re

import pytest

import rheplicant
import rheplicant.config

#: ``tests/config/`` -> the repository root -> ``docs/``.
_DOCS = pathlib.Path(__file__).resolve().parents[2] / "docs"


def _page(name: str) -> str:
    return (_DOCS / name).read_text()


def _section(text: str, heading: str) -> str:
    """The body under a ``## heading``, up to the next ``## ``."""
    _, marker, after = text.partition(f"\n{heading}\n")
    assert marker, f"{heading!r} is no longer a heading on the page"
    body, _, _ = after.partition("\n## ")
    return body


def _paragraphs(text: str) -> list[str]:
    return [block for block in re.split(r"\n[ \t]*\n", text) if block.strip()]


def _page_document(heading: str) -> dict:
    """The first ``yaml`` fence under a heading, parsed as a reader reads it."""
    yaml = pytest.importorskip(
        "yaml",
        reason="PyYAML rides in with myst-parser, in the docs extra",
    )
    _, _, after = _page("config-inference.md").partition(heading)
    assert after, f"{heading!r} is no longer a heading on the page"
    fence = re.search(r"```yaml\n(.*?)```", after, re.DOTALL)
    assert fence, f"no yaml fence under {heading!r}"
    return yaml.safe_load(fence.group(1))


def _kinds_named_in(paragraph: str, kinds) -> set[str]:
    """Every run kind a paragraph names, family globs included.

    ``conjugate.*`` is how both pages have always spelled the three conjugate
    kinds at once, so matching backticked tokens exactly would read a
    sentence naming all three as naming none -- and the sentence this guards
    was exactly that shape.
    """
    hits = set()
    for token in re.findall(r"`([^`]+)`", paragraph):
        token = token.rstrip("*:")
        for kind in kinds:
            if token == kind or (token.endswith(".") and kind.startswith(token)):
                hits.add(kind)
    return hits


class TestTheSurface:
    def test_everything_in_all_is_importable_from_the_package(self):
        for name in rheplicant.config.__all__:
            assert hasattr(rheplicant.config, name), name

    def test_the_config_layer_is_not_re_exported_from_the_top_level(self):
        """rheplicant.__all__ is the package's advertised surface and every
        name in it is a modelling object. A config layer that leaked into it
        would make `from rheplicant import *` import a document parser."""
        assert not set(rheplicant.config.__all__) & set(rheplicant.__all__)

    def test_the_registry_views_are_live_rather_than_snapshots(self):
        """FILE_FORMATS and DERIVATIONS are re-exported because a caller asking
        "what can this document say?" should not have to import the private
        module that happens to hold the table. They must stay the LiveNames
        views -- a tuple() taken here would freeze at import time and go short
        the moment Plan 1B registers into the same tables."""
        from rheplicant.config.derive import _DERIVATIONS
        from rheplicant.config.files import _READERS

        assert set(rheplicant.config.FILE_FORMATS) == set(_READERS)
        assert set(rheplicant.config.DERIVATIONS) == set(_DERIVATIONS)


class TestTheLayerBoundaryIsMechanical:
    def test_no_config_module_is_imported_by_core_radio_or_inference(self):
        """The other direction is guarded by tests/core/test_layering.py for
        core. This is the whole-package half: nothing below config may reach
        up into it, or config stops being removable."""
        src = pathlib.Path(rheplicant.__file__).parent
        offenders = [
            str(path.relative_to(src))
            for path in src.rglob("*.py")
            if "config" not in path.parts
            and (
                "from rheplicant.config" in path.read_text()
                or "import rheplicant.config" in path.read_text()
            )
        ]
        assert not offenders, offenders


class TestEveryFormHasAResolver:
    def test_the_registry_covers_the_declared_grammar(self):
        """VALUE_FORMS is the grammar; _RESOLVERS is what is implemented. The
        two must not drift apart silently -- a form declared and unregistered
        is a key a user can write that does nothing recognisable."""
        from rheplicant.config.values import _RESOLVERS, VALUE_FORMS

        declared = set(VALUE_FORMS) - {"value"}  # form 1 is handled inline
        implemented = set(_RESOLVERS)
        deferred = set()  # nothing deferred: every declared form has a resolver
        assert declared - implemented == deferred, declared - implemented
        assert implemented - declared == set()


class TestPlan1BOnTheSurface:
    def test_the_path_and_resource_entry_points_are_exported(self):
        import rheplicant.config as config

        for name in ("compile_path", "resolve_path_on", "build_resources", "RESOURCE_KINDS"):
            assert name in config.__all__, name

    def test_every_registry_is_reachable_from_the_package(self):
        """Four registries, and a reader who wants to know what is available
        should not have to import four private modules to find out."""
        import rheplicant.config as config

        for name in ("VALUE_FORMS", "FILE_FORMATS", "DERIVATIONS", "RESOURCE_KINDS"):
            assert name in config.__all__, name


class TestThePlan2ASurface:
    def test_the_document_layer_is_exported(self):
        import rheplicant.config as config

        for name in ("ConfiguredRun", "apply_variant", "load_document",
                     "recursive_update", "run_forward"):
            assert name in config.__all__
            assert getattr(config, name) is not None

    def test_importing_the_package_registers_the_object_readers(self):
        import rheplicant.config as config

        assert "rhino_hdf5" in config.FILE_FORMATS
        assert "eqx_leaves" in config.FILE_FORMATS


class TestThePlan2BSurface:
    def test_the_inference_and_runs_layer_is_exported(self):
        import rheplicant.config as config

        for name in ("InferenceBuild", "RunResult", "run_document"):
            assert name in config.__all__, name
            assert getattr(config, name) is not None


class TestThePlan2CSurface:
    """Plan 2C adds no public name, and this is where that is decided.

    The conjugate family, the diagnostics, ``reuse:`` and ``beam_analysis``
    are document *vocabulary*: a caller reaches every one of them by writing
    YAML and calling ``run_document``, which Plan 2B already exported. The
    registry that dispatches them is wiring, not surface -- an exported
    ``EXECUTORS`` would be a mutable handle on the table that decides what
    every document means.

    So the surface is pinned whole rather than sampled. A whole-list
    assertion is deliberately the brittle kind: adding a public name is meant
    to be a two-line change, the name and this list, and 2B's own review
    found the README's count stale precisely because nothing was brittle
    about it.
    """

    #: Every name ``rheplicant.config`` advertised at the end of Plan 2B, and
    #: therefore at the end of Plan 2C. Sorted here for reading; compared
    #: sorted, because ``__all__`` groups constants before classes.
    SURFACE = (
        "ACCEPTED_UNITS", "BuiltResources", "ConfigError", "ConfiguredRun",
        "DERIVATIONS", "FILE_FORMATS", "FieldSpec", "InferenceBuild",
        "RESOURCE_KINDS", "ResolutionContext", "ResolvedPath",
        "ResolvedValue", "RunResult", "SHAPE_SYMBOLS", "ShapeScope", "Unit",
        "VALUE_FORMS", "VALUE_MODIFIERS", "apply_variant", "build_resources",
        "canonical_unit", "compile_path", "convert_to_canonical", "deliver",
        "field_specs", "load_document", "parse_path", "recursive_update",
        "resolve_extent", "resolve_path_on", "resolve_value", "run_document",
        "run_forward",
    )

    def test_plan_2c_adds_no_name_to_the_surface(self):
        import rheplicant.config as config

        assert sorted(config.__all__) == sorted(self.SURFACE), (
            "rheplicant.config.__all__ changed. Plan 2C's decision was that "
            "nine new run kinds are document vocabulary rather than public "
            "API; if a later plan hands the caller a new object, add it here "
            "and to the docstring paragraph that says what the layer does."
        )

    def test_the_exit_registry_stays_wiring_rather_than_surface(self):
        """The dispatch table is not something a caller may hold."""
        import rheplicant.config as config

        for name in ("EXECUTORS", "register", "reuse_of", "RunSpec",
                     "execute_run"):
            assert name not in config.__all__, name

    def test_every_declared_kind_is_reachable_from_a_document(self):
        """Declared, parseable and registered -- three places to half-ship.

        ``test_config_exit_support.py`` compares ``_KINDS`` and ``EXECUTORS``
        as sets. This walks the route a document walks instead: ``parse_runs``
        is what ``run_document`` calls, and it tests the deferral tuples
        BEFORE ``_KINDS``, so a kind can be declared, registered and
        unit-tested and still be refused by name in every document anyone
        writes.
        """
        from rheplicant.config.sections import exits  # registers every kind
        from rheplicant.config.sections.exit_support import EXECUTORS
        from rheplicant.config.sections.runs import _KINDS, parse_runs

        assert exits is not None
        for kind in _KINDS:
            (spec,) = parse_runs([{"kind": kind}])
            assert spec.kind == kind, f"{kind} is not reachable by a document"
            assert kind in EXECUTORS, f"{kind} declares no executor"
        assert len(_KINDS) == len(set(_KINDS)) == 16, sorted(_KINDS)

    def test_nothing_is_deferred_to_plan_2c_any_more(self):
        """The audit of Tasks 2-11: the deferral tuple itself must be gone.

        An empty ``_KINDS_2C`` left behind would keep a dead refusal branch
        that reads, to the next author, as a kind still owed.
        """
        from rheplicant.config.sections import runs

        assert not hasattr(runs, "_KINDS_2C"), (
            "runs._KINDS_2C still exists; the last task to move a kind out of "
            "it deletes the tuple and its refusal branch."
        )

    def test_plan_2d_adds_no_name_to_the_surface_either(self):
        """The two new products stay off ``__all__``, deliberately.

        2B exported ``RunResult`` and ``InferenceBuild`` because a caller
        HOLDS both; 2C exported nothing, because a run kind is something a
        document says.  ``NutsProduct`` and ``NpeProduct`` are the second
        shape: a caller receives an instance through
        ``run_document(...)[name].product`` and never constructs, annotates
        or ``isinstance``-checks one -- and exporting the CLASS would make
        the field layout of a config-internal NamedTuple public API, which
        is what a later plan adding ``report:``/``timings:`` to it would then
        have to break.

        The one counter-argument, answered mechanically rather than in
        prose: ``NpeProduct.posterior`` hands back a trained estimator a
        caller may well want to ``log_prob`` against, and wanting the OBJECT
        is not wanting this wrapper -- ``NeuralPosterior`` is already on
        ``rheplicant.inference.__all__`` (asserted below), so the type the
        caller needs is exported by the layer that owns it.

        Whole-list, like ``test_plan_2c_adds_no_name_to_the_surface``: this
        asserts that neither name reached the package, which a sampled check
        of ``__all__`` alone would miss if one were bound to the module and
        left out of the list.
        """
        import rheplicant.config as config
        import rheplicant.inference as inference
        from rheplicant.config.sections.npe import NpeProduct
        from rheplicant.config.sections.nuts import NutsProduct

        for product in (NutsProduct, NpeProduct):
            assert product.__name__ not in config.__all__, product.__name__
            assert not hasattr(config, product.__name__), product.__name__
        assert "NeuralPosterior" in inference.__all__


class TestThePagesSayWhatTheLayerDoes:
    """The documentation guards check links and counts, never claims.

    That gap is not hypothetical. ``docs/config-inference.md`` said the nine
    2C kinds "are refused by name" for eleven commits after the first of them
    shipped, ``docs/config-sections.md`` said the same, and every
    documentation test stayed green throughout -- anchors resolved and counts
    matched, because both of those really were fine.

    **What this class does NOT check, so the next author does not assume it
    does.** It checks which kinds are named as *deferred*, which transforms
    are named as *registered*, and one claim about what ``kind: none``
    supports. It checks nothing else about what a kind REQUIRES -- not a
    swept key, not a required key, not which section a value comes from. Two
    false sentences of exactly that shape survived this class's first draft:
    ``prior_std:`` was said to become required only when *every* member of a
    block lacked a prior (it is *any*), and ``kind: none`` was said to be
    "legal only for forward and optimize" when six exits run under it. Both
    were found by reading the source, not by a red test. A new per-kind claim
    on either page is unguarded unless someone writes the guard.
    """

    #: The plans that are still in the future.  "Plan 2D" left this tuple when
    #: 2D landed: a paragraph that names a shipped kind beside the plan that
    #: SHIPPED it is honest history, and scanning for it made this guard
    #: assert a contradiction the moment the last 2D kind reached ``_KINDS``.
    #: The docstring below is written against that; it moves with this tuple.
    FUTURE_PLANS = ("Plan 4",)

    PAGES = ("config-inference.md", "config-sections.md")

    def _rows(self):
        table = _section(_page("config-inference.md"), "## Transforms")
        return [line for line in table.splitlines()
                if line.startswith("|") and not set(line) <= set("|- ")]

    def test_the_transform_table_lists_every_registered_transform(self):
        """Both directions, because each has its own way of going stale.

        A registered transform missing from the table is shipped and
        undiscoverable -- ``beam_analysis`` was, for three commits. A table
        row naming no registered transform is worse: a reader writes it and
        gets a refusal quoting a vocabulary that does not include the word
        the page told them to use.
        """
        from rheplicant.config.sections.transforms import _MAPPING, _NAMED

        registered = set(_NAMED) | set(_MAPPING)
        rows = self._rows()
        assert len(rows) > 5, f"the Transforms table stopped parsing: {rows}"

        listed = set()
        for row in rows[1:]:  # [0] is the header
            first_cell = row.split("|")[1]
            for token in re.findall(r"`([^`]+)`", first_cell):
                name = token.split(":")[0].strip()
                assert name in registered, (
                    f"the Transforms table offers {name!r}, which "
                    "transforms.py does not register -- a reader who writes "
                    "it is refused in a vocabulary that omits it."
                )
                listed.add(name)
        assert listed == registered, (
            f"registered but not on the page: {sorted(registered - listed)}"
        )

    #: ``inference.npe:`` subsection -> the name of the frozenset ``npe.py``
    #: sweeps for it.  ``embed:`` is deliberately absent: it takes a VALUE
    #: (``ravel`` or a ``{python:}`` hook), not a mapping of keys, so it has
    #: no frozenset to compare a row against -- but it must still HAVE a row,
    #: which ``test_every_npe_subsection_has_a_row`` asserts separately.
    _NPE_TABLES = {
        "bank": "_BANK_KEYS",
        "create": "_CREATE_KEYS",
        "train": "_TRAIN_KEYS",
        "sample": "_SAMPLE_KEYS",
    }

    def _npe_rows(self) -> dict[str, set[str]]:
        """``{subsection: {key names}}`` read off the page's own table."""
        table = _section(_page("config-inference.md"), "## The npe section")
        rows = {}
        for line in table.splitlines():
            if not line.startswith("|") or set(line) <= set("|- "):
                continue
            cells = line.split("|")
            names = re.findall(r"`([^`]+)`", cells[1])
            if not names or names[0].rstrip(":") == "Subsection":
                continue
            keys = {token.rstrip(":")
                    for token in re.findall(r"`([^`]+)`", cells[3])}
            rows[names[0].rstrip(":")] = keys
        assert len(rows) > 3, f"the npe table stopped parsing: {rows}"
        return rows

    def test_every_npe_subsection_has_a_row(self):
        """The five ``parse_npe`` sweeps, against the five the page offers.

        Both directions, for the reason the Transforms table above gives: a
        subsection the parser accepts and the page omits is shipped and
        undiscoverable, and a row naming a subsection the parser does not
        accept sends a reader to a refusal quoting a vocabulary without it.
        """
        from rheplicant.config.sections.npe import _NPE_KEYS

        assert set(self._npe_rows()) == set(_NPE_KEYS)

    def test_each_npe_rows_keys_are_the_keys_the_parser_sweeps(self):
        """The page's Keys column is the parser's own frozenset, per row.

        This is the guard the ``inference.npe:`` section would otherwise not
        have.  Every other claim Task 11 put on that page is executed (the
        worked document), driven (``kind: none``) or read off the module
        (the deferral sentences); the key lists were prose, and prose is
        what went false for eleven commits the last time.  It kills a knob
        added to ``_CREATE_OPTIONS``/``_TRAIN_OPTIONS`` without a page entry,
        a knob dropped from the package and left on the page, and the
        transposition of two rows' key lists -- none of which any anchor or
        count check can see.

        ``seed:`` is in every frozenset here and on every row: it is
        translated to the package's ``key=`` rather than passed through, and
        a row that omitted it would be describing a subsection a reader
        cannot seed.
        """
        from rheplicant.config.sections import npe

        rows = self._npe_rows()
        for subsection, table in self._NPE_TABLES.items():
            assert rows[subsection] == set(getattr(npe, table)), (
                f"the npe table's {subsection}: row offers "
                f"{sorted(rows[subsection])}; npe.{table} sweeps "
                f"{sorted(getattr(npe, table))}."
            )

    def test_no_page_says_a_shipped_kind_arrives_with_a_later_plan(self):
        """A shipped kind may not share a paragraph with a FUTURE plan.

        Paragraph-scoped rather than page-scoped: both pages legitimately
        name later plans elsewhere, and a page-wide search would be satisfied
        by any mention anywhere.

        Narrower than the sentence it was written for, and worth saying so.
        The false sentence named "Plan 2C", and only ``FUTURE_PLANS`` is
        scanned -- a page claiming a kind arrives with a plan that has
        already shipped it would still pass.  "Plan 2D" was in that tuple
        until 2D landed and left it at Task 11: after the last 2D kind
        reached ``_KINDS``, every honest sentence about what 2D shipped names
        a shipped kind beside "Plan 2D", and a guard that called that an
        offence would have made the page choose between being accurate and
        being green.

        The per-page anti-vacuity floor is not a formality.  A tuple narrowed
        to a string no paragraph contains -- a typo, or a plan number that
        retires the way "Plan 2D" just did -- makes ``continue`` fire on every
        paragraph and leaves this guard passing while reading nothing.  It is
        asserted PER PAGE rather than once at the end because ``_paragraphs``
        splits on blank lines only: a deferral sentence that wraps between
        ``Plan`` and ``4`` matches no filter, and a whole-run counter would be
        held up by the other page while this one went unread.
        """
        from rheplicant.config.sections.runs import _KINDS

        offenders = []
        for name in self.PAGES:
            scanned = 0
            for paragraph in _paragraphs(_page(name)):
                if not any(plan in paragraph for plan in self.FUTURE_PLANS):
                    continue
                scanned += 1
                named = _kinds_named_in(paragraph, _KINDS)
                if named:
                    offenders.append(f"{name}: {sorted(named)} in "
                                     f"{paragraph.strip()[:120]!r}")
            assert scanned >= 1, (
                f"{name}: no paragraph named any of {self.FUTURE_PLANS}, so "
                "this guard read nothing. Either the pages stopped deferring "
                "anything -- in which case delete the guard -- or the tuple "
                "is stale, or a deferral sentence now wraps mid-name."
            )
        assert not offenders, (
            "these kinds ship today and the page says they arrive later:\n  "
            + "\n  ".join(offenders)
        )

    def test_both_pages_still_name_what_is_genuinely_deferred(self):
        """The other direction: silence is not honesty either.

        Deleting the deferral sentence would satisfy the test above and leave
        a reader to discover ``kind: compare`` by being refused.  Only
        ``_KINDS_PLAN4`` is left to defer: ``_KINDS_2C`` went with ``predict``
        and ``_KINDS_2D`` with ``npe``.
        """
        from rheplicant.config.sections.runs import _KINDS_PLAN4

        deferred = set(_KINDS_PLAN4)
        for name in self.PAGES:
            covered = set()
            for paragraph in _paragraphs(_page(name)):
                if "Plan 2D" in paragraph or "Plan 4" in paragraph:
                    covered |= _kinds_named_in(paragraph, deferred)
            assert covered == deferred, (
                f"{name} names {sorted(covered)} as deferred; the module "
                f"defers {sorted(deferred)}."
            )

    #: What each kind needs written beside it to get as far as the noise
    #: check. ``seed:`` names an entry ``runtime.seeds`` does not declare,
    #: which is derived from the root seed rather than refused.
    _UNDER_NONE = {
        "forward": ({}, True),
        "optimize": ({"optimizer": "gradient", "learning_rate": 1e-3,
                      "n_steps": 2}, True),
        "identifiability": ({"names": ["g"]}, True),
        "score_directions": ({"names": ["g"]}, True),
        "fisher": ({}, False),
        "plan.estimate": ({"blocks": [{"names": ["g"]}]}, False),
        "plan.sample": ({"blocks": [{"names": ["g"]}], "n_sweeps": 2,
                         "warmup": 1, "check_identifiability": False,
                         "seed": {"from": "runtime.seeds.probe"}}, False),
        "conjugate.wiener": ({"names": ["g"], "prior_std": {"g": 10.0},
                              "width": "none"}, False),
        "conjugate.gcr": ({"names": ["g"], "prior_std": {"g": 10.0},
                           "n_draws": 2,
                           "seed": {"from": "runtime.seeds.probe"}}, False),
        "conjugate.gls": ({"names": ["g"], "prior_std": {"g": 10.0}}, False),
        "condition": ({"names": ["g"], "prior_std": {"g": 10.0}}, False),
        # num_warmup/num_samples/seed are required and the sweep refuses a run
        # without them, so they are written here to get PAST the grammar and
        # as far as the noise check -- 2 and 2 because this row is about which
        # refusal fires, not about a posterior.
        "nuts": ({"num_warmup": 2, "num_samples": 2,
                  "seed": {"from": "runtime.seeds.probe"}}, False),
        "npe": ({}, False),
    }

    #: What a kind needs in ``inference:`` BESIDE the noiseless block to reach
    #: the noise check at all.  ``npe`` reads its whole grammar from
    #: ``inference.npe:``; without one it is refused for a reason that has
    #: nothing to do with ``kind: none``, and the row above would pass on the
    #: wrong message (2C shape 1).  A side table rather than a third element
    #: on every row: thirteen rows would be rewritten to add one, and the diff
    #: would hide which row this task actually cares about.  The section is
    #: read OFF THE PAGE, so there is one spelling of it in this repo.
    _UNDER_NONE_INFERENCE = {"npe": "npe"}

    def _kind_none_sides(self):
        """The bullet's two halves: what still runs, and what is refused."""
        noise = _section(_page("config-inference.md"), "## Noise")
        _, marker, after = noise.partition("- `kind: none`")
        assert marker, "the kind: none bullet is gone from the Noise section"
        bullet, _, _ = after.partition("\n- ")
        runs, split, refused = bullet.partition("are refused naming this kind")
        assert split, (
            "the kind: none bullet no longer separates the exits that run "
            "from the exits that are refused, so this test cannot tell which "
            "side of the claim a kind is on. Keep the two halves in one "
            "bullet, split by that phrase, or rewrite this test."
        )
        return runs, refused

    def test_the_kind_none_bullet_is_the_list_the_exits_really_make(self):
        """Read the page, then run it. The bullet said two; six run.

        "legal only for `forward` and `optimize`" was true when 2B wrote it
        and false the moment `identifiability` shipped, and it understates
        rather than overstates -- which is why no reader complained and no
        test noticed. A reader acts on this one: it is what decides whether a
        cheap diagnostic has to be given a noise model it does not use.

        Both directions and both SIDES: the kinds measured to run must be on
        the running half of the sentence, and the kinds measured to refuse on
        the refused half, so swapping the two lists fails here. ``mmodes``
        and ``gradient`` are named on the page and not driven here -- one
        needs limTOD and a projector, the other is per-objective; both are
        measured in their own modules.
        """
        from rheplicant.config import run_document
        from rheplicant.config.errors import ConfigError

        document = _page_document(TestTheWorkedDocumentOnThePage.HEADING)
        runs_side, refused_side = self._kind_none_sides()
        posterior_page = _page_document(
            TestThePosteriorDocumentOnThePage.HEADING)["inference"]

        for kind, (options, should_run) in self._UNDER_NONE.items():
            block = {**document["inference"], "noise": {"kind": "none"}}
            extra = self._UNDER_NONE_INFERENCE.get(kind)
            if extra is not None:
                block[extra] = posterior_page[extra]
            noiseless = {
                **document,
                "inference": block,
                "runs": [{"name": "probe", "kind": kind, **options}],
            }
            side = runs_side if should_run else refused_side
            assert f"`{kind}`" in side, (
                f"the kind: none bullet puts {kind!r} on the wrong side, or "
                "does not name it at all"
            )
            if should_run:
                assert run_document(noiseless)["probe"].product is not None
            else:
                with pytest.raises(
                    ConfigError, match="weighs residuals with inference.noise"
                ):
                    run_document(noiseless)


class TestTheWorkedDocumentOnThePage:
    """``docs/config-inference.md``'s conjugate example is executed here.

    2B's page shipped with a document nobody had run; its final review ran
    the YAML by hand and found errors in it. A page that carries a document
    is a page making a promise, so the promise is a test: the fence is read
    off the page as a reader would read it, and the numbers below are the
    ones this document really produces (measured against the shipped
    package, not derived from the prose).
    """

    HEADING = "## A conjugate document"

    def _document(self):
        return _page_document(self.HEADING)

    def test_the_page_s_document_runs_and_recovers_the_injected_gain(self):
        """The mean, not merely a mean -- but a smoke pin, not a kill.

        ``observed.at`` injects ``g = 1.5`` and the solve lands at 1.5225,
        which is the truth pulled a little by the prior's ``loc: 1.0`` and by
        the realised noise; an executor that never solved would hand back the
        declared ``init: 1.0``. Pinned to the same tolerance
        ``tests/config/test_config_exits_plan.py`` uses for the same document
        under ``plan.estimate``.

        What this does NOT kill, said out loud so no one relies on it:
        the page's latent declares ``prior: {normal: {loc: 1.0, scale:
        10.0}}``, numerically identical to the run's ``prior_std: {g: 10.0}``
        / ``prior_mean: {g: 1.0}``, so an executor that IGNORED both keys
        returns 1.5224889516830444 -- the same value to the last bit
        (measured). Nor can the two be made load-bearing on THIS document:
        ``_reconcile`` refuses a supplied keyword that disagrees with the
        declared prior, and dropping the declaration to ``prior: null``
        makes the ``width: fisher`` run raise. The keys reaching the solve is
        Task 3's business, and
        ``test_the_mapping_form_keeps_each_width_on_its_own_latent`` and
        ``test_a_scalar_prior_std_over_several_latents_is_check_a51`` are
        where it is killed. A SWAP is killed by the package, not here: on
        this document, whose declared prior repeats the two run keys, a
        swapped ``prior_mean``/``prior_std`` is refused outright rather than
        coming back as a mean near 1.0.
        """
        from rheplicant.config import run_document

        results = run_document(self._document())

        assert list(results) == ["identifiable", "mean", "at_init", "spread"], (
            "runs must come back in declaration order"
        )
        mean = results["mean"].product
        assert float(mean["mean"]["g"]) == pytest.approx(1.5, abs=0.05)

    def test_width_fisher_puts_an_error_bar_on_that_mean(self):
        """``width:`` is the whole reason the page tells a user to write it.

        0.0159 is the posterior sigma this document has; the prior's own
        scale is 10.0, so a route that returned the PRIOR width instead of
        the data's would be off by nearly three orders of magnitude and
        would still be a finite, correctly-shaped number.
        """
        import numpy as np

        from rheplicant.config import run_document

        covariance = run_document(self._document())["mean"].product["covariance"]
        sigma = float(np.sqrt(np.asarray(covariance.matrix)[0, 0]))
        assert sigma == pytest.approx(0.0159, rel=0.05)

    def test_the_diagnostic_run_answers_before_the_fit_is_paid_for(self):
        """One latent against a 16x8 grid: rank 1, nothing in the null space.

        ``n_data`` is the flattened ``(n_time, n_freq)`` of the document's
        own grids, 16 x 8 = 128; ``rank 1 / nullity 0`` is the verdict the
        width in the previous test depends on. A smoke pin, not a
        discriminator: the full twin and the fit twin predict the same shape
        for this document, so ``n_data`` is 128 either way (measured), and
        dropping the declared ``names:`` leaves all four values unchanged.
        The fit-twin choice is killed in Task 7's own tests, not here.
        """
        from rheplicant.config import run_document

        report = run_document(self._document())["identifiable"].product
        assert (report.rank, report.nullity) == (1, 0)
        assert (report.n_par, report.n_data) == (1, 128)

    def test_the_page_s_predict_run_carries_that_width_into_data_space(self):
        """``reuse:`` on the page, executed rather than described.

        The delta method turns the ``fisher`` run's 1x1 covariance into one
        standard deviation per sample, so the product is shaped like the data
        and not like the parameter block -- ``n_data`` from the diagnostic run
        is the same 128. The magnitude is what discriminates: 0.0079 K is the
        propagated width, where the parameter's own sigma is 0.0159 and the
        declared prior's is 10.0, and a route that handed back either instead
        would be finite, correctly shaped and positive.
        """
        import numpy as np

        from rheplicant.config import run_document

        results = run_document(self._document())
        spread = np.asarray(results["spread"].product)

        assert spread.shape == (16, 8)
        assert spread.size == results["identifiable"].product.n_data
        assert np.all(np.isfinite(spread)) and np.all(spread > 0.0)
        assert float(spread.max()) == pytest.approx(0.00786, rel=0.05)

    def test_the_page_s_reuse_may_only_look_backwards(self):
        """The one claim on the page that is about order rather than value.

        The page tells the reader to note where ``at_init`` sits in the list.
        Moving the ``predict`` above it -- the same two runs, the same keys --
        must refuse, or that sentence is decoration. ``match=`` targets the
        forward-reference branch alone: ``reuse_of``'s neighbours say
        "reuse: <run name> is required" for a missing key and "refused (...)"
        for a run that raised, and neither contains this phrase.
        """
        from rheplicant.config import run_document
        from rheplicant.config.errors import ConfigError

        document = self._document()
        runs = list(document["runs"])
        names = [run["name"] for run in runs]
        forward = {**document,
                   "runs": [*runs[:2], runs[3], runs[2]]}

        assert names == ["identifiable", "mean", "at_init", "spread"], names
        with pytest.raises(ConfigError, match="names no earlier run"):
            run_document(forward)


class TestThePosteriorDocumentOnThePage:
    """``docs/config-inference.md``'s posterior example is executed here.

    The same promise ``TestTheWorkedDocumentOnThePage`` makes for the
    conjugate example: a page that carries a document is a page making a
    promise, and 2B's final review found errors in the one nobody had run.
    This one is more expensive -- all four runs together are 4.0 s of
    class-scoped setup, measured with ``--durations``, against 0.9 s for the
    whole conjugate document -- so it is run ONCE for the class rather than
    once per assertion.

    What it deliberately does NOT pin: any number the npe run produces.  Fifty
    training steps on sixty-four simulations is an estimator that has not
    converged -- measured on this page's own YAML, ``g`` comes back with a
    mean of 1.14 and a standard deviation of 8.6, against an injected truth of
    1.5 -- and pinning a posterior from it would be pinning noise.  (An
    earlier draft of this docstring said "a mean near 1.52 and a standard
    deviation of 6.2"; neither number came from a drive of this document, and
    replacing them is why Step 11.6 exists.)  The nuts run's mean IS pinned,
    because it agrees with the conjugate page's own answer and that agreement
    is a cross-check between two independent exits rather than a
    self-comparison.
    """

    HEADING = "## A posterior document"

    @pytest.fixture(scope="class")
    @classmethod
    def results(cls):
        # @classmethod is required, not stylistic: pytest raises
        # PytestRemovedIn10Warning for a class-scoped fixture defined as an
        # instance method, and this suite is otherwise warning-clean.
        # tests/config/test_config_exits_predict.py:371 and :566 use the same
        # form, for the same reason.
        from rheplicant.config import run_document

        return run_document(_page_document(cls.HEADING))

    def test_the_pages_chain_agrees_with_the_pages_conjugate_mean(self,
                                                                  results):
        """Two exits, one document, one answer.

        The conjugate page recovers ``g = 1.5224889516830444`` by an exact
        linear-Gaussian solve; this page's NUTS chain reaches 1.521574 by
        gradient-based sampling from the same likelihood (both measured, on
        the two documents as they stand on the page).  Agreement between two
        routes that share no code below ``inference.noise`` is the assertion,
        and it is a genuine cross-check rather than a self-comparison: the
        posterior document's ``observed`` array is BIT-IDENTICAL to the
        conjugate page's (``seed_for`` derives an undeclared name as
        ``_digest(name) ^ root``, so adding a ``seeds:`` mapping for OTHER
        names does not perturb ``observed_noise``).

        A chain that never moved would sit at the declared ``init: 1.0``,
        which ``abs=0.05`` excludes.  **A chain started by
        ``init_to_uniform`` would NOT be excluded** -- measured on this
        page's own YAML by monkeypatching ``nuts._init_strategy``, 200 warmup
        and 200 samples, ``init_to_declared`` gives 1.521574 and
        ``init_to_uniform`` gives 1.521562: indistinguishable, both far
        inside this tolerance.  An earlier draft of this docstring claimed
        the opposite and cited the bridge's ``r_hat`` 840, which belongs to
        the ring toy and not to this one-latent document -- here the two
        strategies give ``r_hat`` 0.9965 and 0.9967.  ``init:`` is separated
        from ``uniform`` by the spy in ``test_config_exits_nuts.py``, not by
        this number, and Task 5's body already records exactly this for a
        one-latent document.
        """
        import numpy as np

        chain = results["chain"].product
        assert sorted(chain.samples) == ["g"]
        assert "prediction" not in chain.samples
        assert chain.n_draw == 200 and chain.n_chain == 1
        assert float(np.mean(np.asarray(chain.samples["g"]))) == \
            pytest.approx(1.52, abs=0.05)

    def test_the_page_s_npe_run_returns_the_shape_it_promises(self, results):
        """Shapes, keys and the estimator -- not a posterior.

        ``NeuralPosterior.sample`` returns a flat ``(n_draws, n_params)``
        array and this layer unravels it back to ``{latent: (n_draws,
        *shape)}``, which is the ONLY form ``predict`` can read.  The page
        says ``n_draws: 12``; this is that sentence executed.  ``best_step``
        is asserted to be a plain ``int`` because ``train_posterior`` returns
        it as a traced array, and one that reached a message or a report
        would print as ``Array(50, dtype=int32)``.
        """
        import numpy as np

        posterior = results["amortized"].product
        assert sorted(posterior.samples) == ["g"]
        assert posterior.n_draw == 12
        assert np.asarray(posterior.samples["g"]).shape == (12,)
        assert type(posterior.best_step) is int
        assert np.asarray(posterior.train_loss).shape == (50,)

    def test_both_predicts_come_back_shaped_like_the_data(self, results):
        """``reuse:`` on the page, on the two routes 2D adds.

        ``(16, 8)`` is the document's own grid, and the leading axis is the
        draw count each run kept -- 50 for the thinned chain, 12 for the
        amortized posterior.  A ``predict`` that pushed the mean once would
        come back ``(16, 8)`` on both, and one that dropped ``n_draw:`` would
        come back with 200 rows.
        """
        import numpy as np

        chain_spread = np.asarray(results["chain_spread"].product)
        npe_spread = np.asarray(results["npe_spread"].product)
        assert chain_spread.shape == (50, 16, 8)
        assert npe_spread.shape == (12, 16, 8)
        assert np.all(np.isfinite(chain_spread))
        assert np.all(np.isfinite(npe_spread))

    def test_the_runs_come_back_in_declaration_order(self, results):
        """The page's own reading order, and the order ``reuse:`` needs.

        Both predicts look backwards, so the page's list is not decoration:
        moving either above the run it names is refused.
        """
        assert list(results) == ["chain", "chain_spread", "amortized",
                                 "npe_spread"]
