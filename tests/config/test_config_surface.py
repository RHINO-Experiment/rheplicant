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
        assert len(_KINDS) == len(set(_KINDS)) == 15, sorted(_KINDS)

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
        assert runs._KINDS_2D == ("npe",)


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

    def test_no_page_says_a_shipped_kind_arrives_with_a_later_plan(self):
        """A shipped kind may not share a paragraph with a FUTURE plan.

        Paragraph-scoped rather than page-scoped: both pages legitimately
        name later plans elsewhere, and a page-wide search would be satisfied
        by any mention anywhere.

        Narrower than the sentence it was written for, and worth saying so.
        The false sentence named "Plan 2C", and only "Plan 2D" and "Plan 4"
        are scanned here -- a page claiming a kind arrives with the plan that
        has already shipped it would still pass. That is defensible while 2C
        is the current plan and indefensible after 2D lands, at which point
        "Plan 2D" stops being a future plan and this list moves on with it.
        """
        from rheplicant.config.sections.runs import _KINDS

        offenders = []
        for name in self.PAGES:
            for paragraph in _paragraphs(_page(name)):
                if "Plan 2D" not in paragraph and "Plan 4" not in paragraph:
                    continue
                named = _kinds_named_in(paragraph, _KINDS)
                if named:
                    offenders.append(f"{name}: {sorted(named)} in "
                                     f"{paragraph.strip()[:120]!r}")
        assert not offenders, (
            "these kinds ship today and the page says they arrive later:\n  "
            + "\n  ".join(offenders)
        )

    def test_both_pages_still_name_what_is_genuinely_deferred(self):
        """The other direction: silence is not honesty either.

        Deleting the deferral sentence would satisfy the test above and leave
        a reader to discover ``kind: nuts`` by being refused.
        """
        from rheplicant.config.sections.runs import _KINDS_2D, _KINDS_PLAN4

        deferred = set(_KINDS_2D) | set(_KINDS_PLAN4)
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
    }

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

        for kind, (options, should_run) in self._UNDER_NONE.items():
            noiseless = {
                **document,
                "inference": {**document["inference"], "noise": {"kind": "none"}},
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
