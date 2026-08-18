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


def _page_document(heading: str, page: str = "config-inference.md") -> dict:
    """The first ``yaml`` fence under a heading, parsed as a reader reads it.

    ``page`` was ``config-inference.md`` written into the body until Plan 3A;
    it is a parameter now because ``config-validation.md`` carries a document
    too, and a second copy of this parse is the duplication §2.2 forbids. The
    default is the page every existing caller already reads, so none of them
    changes.
    """
    yaml = pytest.importorskip(
        "yaml",
        reason="PyYAML rides in with myst-parser, in the docs extra",
    )
    _, _, after = _page(page).partition(heading)
    assert after, f"{heading!r} is no longer a heading on {page}"
    fence = re.search(r"```yaml\n(.*?)```", after, re.DOTALL)
    assert fence, f"no yaml fence under {heading!r} on {page}"
    return yaml.safe_load(fence.group(1))


def _block(text: str, heading: str) -> str:
    """The body under a heading of ANY level, up to the NEXT heading.

    :func:`_section` is 3A's and stops only at the next ``## ``, which is what
    a ``## `` section wants. Plan 3C's gate section carries ``### ``
    subsections, and a guard that read one of them with ``_section`` would
    swallow every sibling below it -- so a table asserted "under the defaults
    heading" would in fact be matched anywhere in the section, and moving a row
    between subsections would change nothing. This stops at ``\\n#`` of any
    depth.
    """
    _, marker, after = text.partition(f"\n{heading}\n")
    assert marker, f"{heading!r} is no longer a heading on the page"
    return re.split(r"\n#{1,6} ", after)[0]


def _rows(body: str) -> list[list[str]]:
    """Every data row of every markdown table in ``body``, as cell lists.

    **Not "the first table"**: it walks every ``|``-prefixed line in ``body``,
    drops only rows that are entirely ``-``/``:``/space (the ``|---|`` rule),
    and returns everything else past the very first row collected -- which
    means a SECOND table's own header row is not recognised as a header and
    is returned as if it were data. Every guarded block in this module holds
    exactly one pipe-free table, so this has no live consequence today; if
    that ever stops being true, this function needs splitting, not its
    caller's arity assertion loosened. A cell is stripped. Escaped pipes
    (``\\|``) are **not** handled -- a literal ``\\|`` inside a cell splits
    that cell in two. Returns ``[]`` when there is no table, which every
    caller turns into a red test with its own message rather than a vacuous
    pass.
    """
    found = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if all(set(cell) <= set("-: ") for cell in cells):
            continue          # the |---|---| rule
        found.append(cells)
    return found[1:] if found else []


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

    def test_plan_3a_adds_the_four_names_a_caller_touches(self):
        """The pass, its report, its element and its warning class.

        A caller CALLS preflight, RECEIVES a Report, READS a Finding and must
        be able to NAME ConfigWarning in a filterwarnings call. Each is the
        same footing as a name already here: load_document, ConfiguredRun,
        ResolvedValue, ConfigError.
        """
        import rheplicant.config as config

        for name in ("ConfigWarning", "Finding", "Report", "preflight"):
            assert name in config.__all__, name
            assert hasattr(config, name), name

    def test_the_check_registry_stays_wiring_rather_than_surface(self):
        """The exact argument 2C made for EXECUTORS, in the docstring above.

        A check id is something the SCHEMA says. A caller that could register
        one would be extending §6 from outside, and the id space is the
        schema's. Pinned both ways -- not in ``__all__``, and not reachable as
        an attribute -- because ``from ... import *`` and ``config.register``
        are two different leaks.
        """
        import rheplicant.config as config

        for name in ("CHECKS", "register"):
            assert name not in config.__all__, name
            assert not hasattr(config, name), name

    def test_config_warning_is_a_userwarning_so_a_caller_can_filter_it(self):
        """The whole reason it is exported rather than module-local.

        A caller who cannot name the class cannot turn these into errors, and
        ``category=UserWarning`` would also catch numpyro's.
        """
        import warnings

        import rheplicant.config as config

        assert issubclass(config.ConfigWarning, UserWarning)
        assert config.ConfigWarning is not UserWarning
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=config.ConfigWarning)
            with pytest.raises(config.ConfigWarning):
                warnings.warn("x", config.ConfigWarning, stacklevel=2)

    def test_exporting_preflight_shadows_the_subpackage_and_that_is_pinned(
            self):
        """THE side effect of binding the function in ``config/__init__.py``.

        Measured: after this task both ``from rheplicant.config import
        preflight`` and ``import rheplicant.config.preflight as x`` hand back
        the **function** -- the getattr-first path -- and only
        ``sys.modules[...]`` and ``importlib.import_module(...)`` return the
        module.

        Kills any later module reaching for ``config.preflight.__file__``:
        that is an ``AttributeError`` at import, and the one place that
        already wanted it (``test_config_preflight.py``'s ``_PREFLIGHT_DIR``)
        goes through ``importlib.import_module`` for exactly this reason.
        """
        import importlib
        import sys
        import types

        import rheplicant.config as config

        assert callable(config.preflight)
        assert not isinstance(config.preflight, types.ModuleType)
        module = importlib.import_module("rheplicant.config.preflight")
        assert isinstance(module, types.ModuleType)
        assert module is sys.modules["rheplicant.config.preflight"]
        assert module.preflight is config.preflight

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

    def test_exported_layering_functions_are_the_neutral_objects(self):
        """Catches package-level imports preserving an obsolete wrapper."""
        import rheplicant.config as config
        from _rheplicant_bootstrap import layering as neutral

        assert config.apply_variant is neutral.apply_variant
        assert config.recursive_update is neutral.recursive_update

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

    #: What Plan 3A adds, and the four are argued one by one in
    #: ``config/__init__.py``'s own docstring: a caller CALLS the pass,
    #: RECEIVES its report, READS an element of it, and NAMES the warning
    #: class in a ``filterwarnings``. Kept as its own tuple rather than
    #: folded into ``SURFACE`` so the whole-list assertion below still says
    #: which plan put each name there.
    SURFACE_3A = ("ConfigWarning", "Finding", "Report", "preflight")

    #: What Plan 3B adds: **nothing**, and
    #: ``TestPlan3BsWiringAndItsSurface::test_plan_3b_adds_no_name_to_the_
    #: surface_and_here_is_why`` carries the mechanical argument. An empty
    #: tuple rather than an absent one, so the sum below reads as a per-plan
    #: ledger and a later reader does not have to infer that 3B was skipped.
    SURFACE_3B: tuple[str, ...] = ()

    #: What Plan 3C adds, argued in ``config/__init__.py``'s own docstring: a
    #: caller CALLS ``gates`` on the ``inference.checks:`` mapping it already
    #: holds and READS the ``Gate``s it hands back -- the same
    #: call/receive/read shape that put ``preflight``/``Report``/``Finding``
    #: here. ``priced`` is deliberately absent, for 3B's reason: it is a pass
    #: ``load_document`` runs for the caller, and its answer arrives on
    #: ``ConfiguredRun.report``.
    SURFACE_3C = ("Gate", "gates")

    def test_the_surface_is_2b_s_list_plus_the_names_each_later_plan_added(
            self):
        import rheplicant.config as config

        assert sorted(config.__all__) == sorted(self.SURFACE
                                                + self.SURFACE_3A
                                                + self.SURFACE_3B
                                                + self.SURFACE_3C), (
            "rheplicant.config.__all__ changed. Plan 2C's decision was that "
            "nine new run kinds are document vocabulary rather than public "
            "API, and 2D's that a product received through run_document is "
            "not one either; Plan 3A added exactly four, 3B none and 3C two. "
            "If a later plan hands the caller a new object, add it here and "
            "to the docstring paragraph that says what the layer does."
        )

    def test_the_exit_registry_stays_wiring_rather_than_surface(self):
        """The dispatch table is not something a caller may hold."""
        import rheplicant.config as config

        for name in ("EXECUTORS", "PARSERS", "PRE_EXECUTORS",
                     "DEFERRED_CHECKS", "register", "handler_for",
                     "parse_run", "parsed_options", "ParsedOptions",
                     "ParsedRun", "RunParseContext", "ExitHandler",
                     "reuse_of", "RunSpec", "execute_run"):
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

        Whole-list, like
        ``test_the_surface_is_2b_s_list_plus_the_names_each_later_plan_added``:
        this asserts that neither name reached the package, which a sampled check
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
        # `n_sweeps: 8, warmup: 4` and not `2, 1`: this row is about WHICH
        # refusal fires, and a run keeping one draw is now refused by A24
        # (`preflight/fitting.py::_counts`) before the noise check is
        # reached at all -- correctly, since the package refuses it too
        # (`plan.py:1055`), just three phases later.  Four kept draws is
        # `MIN_DRAWS` exactly, so the document gets past P-1 and the row
        # measures the sentence it was written to measure.
        "plan.sample": ({"blocks": [{"names": ["g"]}], "n_sweeps": 8,
                         "warmup": 4, "check_identifiability": False,
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

    #: Number words the pages use to count kinds. Not a general table -- these
    #: are the sizes the ``## Runs`` subsections actually come in, and a word
    #: outside it means a subsection was resized past what this guard knows.
    _NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                     "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                     "eleven": 11, "twelve": 12, "thirteen": 13,
                     "fourteen": 14, "fifteen": 15, "sixteen": 16}

    #: A kind's own bullet under a ``### `` subsection: ``- `forward` — ``.
    #: The BULLET HEADS, never the body -- measured, the ``predict`` bullet
    #: names fisher/plan.sample/nuts/npe in its prose and the ``conjugate.gls``
    #: bullet names its two siblings, so a body-wide scan is not a partition.
    #:
    #: An earlier version of this comment said such a scan "would have made
    #: every assertion below pass while measuring nothing". That undersells
    #: the guard and is false: measured, widening this pattern to
    #: ``r"`([a-z][a-z._]*)`"`` turns TWO assertions red --
    #: ``test_the_runs_subsections_partition_every_declared_kind`` (on the
    #: duplicate and the set equality) and
    #: ``test_every_subsection_heading_that_counts_counts_right`` (on the
    #: lengths). The reason to read heads is that the partition would be
    #: meaningless, not that nothing would notice.
    _KIND_BULLET = re.compile(r"^- `([a-z][a-z._]*)` — ", re.MULTILINE)

    #: The same bullets with their BODIES, ``{kind: text}``. Terminated by the
    #: next bullet, the next ``### `` heading, or the end of ``## Runs``.
    _KIND_BULLET_BODY = re.compile(
        r"^- `([a-z][a-z._]*)` — (.*?)(?=^- `|^### |\Z)",
        re.MULTILINE | re.DOTALL)

    def _runs_subsections(self):
        """``[(heading, [kinds bulleted])]`` for ``## Runs``."""
        runs = _section(_page("config-inference.md"), "## Runs")
        out = []
        for chunk in runs.split("\n### ")[1:]:
            heading, _, body = chunk.partition("\n")
            out.append((heading.strip(), self._KIND_BULLET.findall(body)))
        return out

    def test_the_runs_subsections_partition_every_declared_kind(self):
        """The page's own division of ``runs:``, against ``_KINDS``.

        A kind that ships and is bulleted nowhere is undiscoverable -- the
        reader has to be refused to find out it exists. A bullet naming no
        declared kind sends a reader to write a word ``parse_runs`` rejects.
        And a kind bulleted TWICE would let both halves of the count check
        below pass while the page said sixteen and listed seventeen.
        """
        from rheplicant.config.sections.runs import _KINDS

        sections = self._runs_subsections()
        assert len(sections) >= 4, (
            f"## Runs parsed into {len(sections)} subsections; the '### ' "
            "split or the bullet pattern has stopped matching."
        )
        bulleted = [kind for _, kinds in sections for kind in kinds]
        assert len(bulleted) == len(set(bulleted)), (
            f"a kind is bulleted twice: {sorted(bulleted)}"
        )
        assert set(bulleted) == set(_KINDS), (
            f"bulleted but not declared: {sorted(set(bulleted) - set(_KINDS))}; "
            f"declared but not bulleted: {sorted(set(_KINDS) - set(bulleted))}"
        )

    def test_every_subsection_heading_that_counts_counts_right(self):
        """"The five that fit" is a claim, and nothing was checking it.

        Not every heading carries a number -- "The conjugate family" does not,
        and is covered by the partition above -- so the scan is over the ones
        that do, with a floor so a rewording that dropped every number word
        cannot leave this passing.
        """
        counted = 0
        for heading, kinds in self._runs_subsections():
            words = [word for word in self._NUMBER_WORDS
                     if re.search(rf"\b{word}\b", heading, re.IGNORECASE)]
            if not words:
                continue
            counted += 1
            assert len(words) == 1, f"{heading!r} carries {words}"
            assert self._NUMBER_WORDS[words[0]] == len(kinds), (
                f"{heading!r} says {words[0]} and bullets {len(kinds)}: "
                f"{kinds}"
            )
        assert counted >= 3, (
            f"only {counted} '### ' headings under ## Runs carry a number "
            "word; either they were reworded or the scan has drifted."
        )

    def test_config_sections_states_how_many_kinds_runs_holds(self):
        """``config-sections.md`` counts them in words and nothing read it.

        It is right today -- 2D updated it from fourteen to sixteen -- and it
        was right by hand, which is the condition this repo has twice paid for
        (README's test count drifted by 759; the D range by three).
        """
        from rheplicant.config.sections.runs import _KINDS

        stated = re.search(r"\[the (\w+) kinds it runs\]",
                           _page("config-sections.md"))
        assert stated, (
            "config-sections.md no longer states how many kinds `runs:` holds "
            "in the '[the N kinds it runs](...)' form this guard reads."
        )
        word = stated.group(1).lower()
        assert word in self._NUMBER_WORDS, f"unknown number word {word!r}"
        assert self._NUMBER_WORDS[word] == len(_KINDS), (
            f"config-sections.md says {word} kinds; runs.py declares "
            f"{len(_KINDS)}."
        )

    def _kind_bullets(self) -> dict:
        """``{kind: the whole text of its bullet}`` under ``## Runs``."""
        runs = _section(_page("config-inference.md"), "## Runs")
        found = dict(self._KIND_BULLET_BODY.findall(runs))
        assert len(found) > 5, f"the bullet-body scan stopped matching: {found}"
        return found

    def test_every_exit_that_reads_the_noise_as_a_rule_documents_a28(self):
        """A28 appeared NOWHERE under ``docs/`` while five sentences shipped it.

        ``config-inference.md`` documented A27 correctly on the
        ``conjugate.gls`` bullet and its mirror was simply absent, so a user
        refused by A28 had the check id and no page naming it. The gate is
        read from the package: ``_T10_ITERATES`` is the set of kinds that read
        ``inference.noise`` as a RULE and is what gates A28's three legs, so a
        fourth kind joining it arrives here as a red test rather than as an
        undocumented refusal.

        Per BULLET and not per page: a page-wide search for "A28" passes as
        soon as any one of the three carries it, which is the hole this closes.
        """
        from rheplicant.config.preflight.fitting import _T10_ITERATES

        bullets = self._kind_bullets()
        missing = sorted(kind for kind in _T10_ITERATES
                         if "A28" not in bullets.get(kind, ""))
        assert not missing, (
            f"these kinds read inference.noise as a rule and their bullets on "
            f"config-inference.md do not mention check A28: {missing}"
        )

    def test_every_config_page_is_reachable_from_a_toctree(self):
        """A page in no toctree is a sphinx warning and nothing else.

        Measured: deleting ``config-validation`` from ``docs/index.md`` moves
        the nitpicky count from 35 to 36 and turns NO test red -- the only
        gate is a clean ``sphinx -n`` build, which no test runs and which a
        contributor has to remember. ``test_docs_links.py`` checks that a page
        is tracked by git and that its anchors resolve, never that anyone can
        reach it.
        """
        index = _page("index.md")
        listed = set(re.findall(r"^(config-[a-z-]+)$", index, re.MULTILINE))
        pages = {path.stem for path in _DOCS.glob("config-*.md")}
        assert pages, "no config-*.md pages found; the glob has drifted"
        assert pages <= listed, (
            f"these pages are in no toctree in docs/index.md and will each "
            f"cost one nitpicky sphinx warning: {sorted(pages - listed)}"
        )

    def test_the_validation_pages_report_table_is_the_report_api(self):
        """The method table is prose about an object the suite can ask.

        A row for a method ``Report`` does not have sends a reader to an
        ``AttributeError``; a method the page omits is one they will not
        find. Both directions, against ``Report``'s own public surface --
        which is how this test found ``of(severity)`` missing from the table
        on its first run, and with it the ``report`` severity, which the page
        named nowhere.
        """
        from rheplicant.config import Report

        body = _section(_page("config-validation.md"), "## What a Report carries")
        listed = set(re.findall(r"\| `report\.(\w+)\([^)]*\)` \|", body))
        assert listed, "the Report method table stopped parsing"
        public = {name for name in vars(Report)
                  if not name.startswith("_") and callable(getattr(Report, name))}
        assert listed == public, (
            f"on the page and not on Report: {sorted(listed - public)}; on "
            f"Report and not on the page: {sorted(public - listed)}"
        )

    def test_the_validation_page_counts_a_findings_fields(self):
        """"A ``Finding`` is four fields" -- and it names all four.

        The count and the names together, because either alone drifts: a
        fifth field added to ``findings.py`` leaves the word "four" false, and
        a field renamed leaves the page pointing at a name that is gone.
        """
        import dataclasses

        from rheplicant.config import Finding

        body = _section(_page("config-validation.md"), "## What a Report carries")
        opens = re.search(r"A `Finding` is (\w+) fields: (.*?)\n\n", body,
                          re.DOTALL)
        assert opens, "config-validation.md no longer counts a Finding's fields"
        word = opens.group(1).lower()
        assert word in self._NUMBER_WORDS, f"unknown number word {word!r}"
        fields = {field.name for field in dataclasses.fields(Finding)}
        assert self._NUMBER_WORDS[word] == len(fields), (
            f"the page says {word} fields; Finding has {len(fields)}."
        )
        named = set(re.findall(r"`(\w+)`", opens.group(2)))
        assert fields <= named, (
            f"the page counts {word} fields and names {sorted(named & fields)}; "
            f"it does not name {sorted(fields - named)}."
        )

    def test_the_validation_page_counts_the_sources_it_lists(self):
        """"reads three things and no fourth" and "Three sources", once each.

        Self-consistency rather than a check against the package: the three
        sources are §2.4's scope boundary, which no shipped constant holds.
        What this kills is the two sentences drifting away from the list they
        introduce -- a fourth bullet added under *What it decides* leaves both
        words saying three, and nothing else here would notice.

        **What stays PROSE on that page, said plainly so no one assumes
        otherwise.** "It never constructs an operator, never resolves a value
        node, never opens a file" is enforced by ``test_config_preflight.py``'s
        static call and import bans, not by anything reading the sentence.
        "``where`` is never a path into the package" is enforced by
        ``preflight._check_where`` and asserted behaviourally
        (``test_every_where_on_it_is_a_path_into_the_document``); inverting
        the SENTENCE alone turns nothing red. So is every clause of *What it
        cannot decide*, and the 90.9 % figure in the opening paragraph, which
        is §2.7's pinned measurement and re-derivable by no test in this
        plan's budget.
        """
        page = _page("config-validation.md")
        body = _section(page, "## What it decides, and from what")
        bullets = re.findall(r"^- \*\*", body, re.MULTILINE)
        assert len(bullets) >= 2, (
            f"the source bullets stopped parsing: {len(bullets)} found"
        )
        claims = [
            (r"reads (\w+) things and no fourth", "the pass's own sentence"),
            (r"^(\w+) sources, and the third", "the list's lead-in"),
        ]
        for pattern, what in claims:
            found = re.search(pattern, page, re.MULTILINE | re.IGNORECASE)
            assert found, (
                f"config-validation.md no longer counts its sources in {what}, "
                f"in the {pattern!r} form this guard reads."
            )
            word = found.group(1).lower()
            assert word in self._NUMBER_WORDS, f"{what}: unknown word {word!r}"
            assert self._NUMBER_WORDS[word] == len(bullets), (
                f"{what} says {word} and the section bullets {len(bullets)}."
            )

    def test_the_validation_page_names_the_sections_structurally_refused(self):
        """``config-validation.md`` lists them; ``_structural`` decides them.

        The sentence is *"``outputs:``, ``defaults:`` and ``plugins:``, which
        arrive with Plan 4, and ``campaign:``, which is reserved with
        capability 4"*, and the two halves are two different tables:
        ``_NOT_YET`` carries the first three with the plan each arrives with,
        while ``campaign`` has a clause of its own naming §8.2. A page that
        called all four "deferred" would be describing one table where there
        are two, and a section added to either and left off the page is one a
        reader discovers by being refused.
        """
        from rheplicant.config.preflight import _NOT_YET, _SECTIONS

        body = _section(_page("config-validation.md"), "## The pre-flight pass")
        # ``\s+`` and not a space: the sentence wraps between "is" and
        # "structural", and a literal partition reads nothing and asserts
        # nothing -- which is how this guard would go green while the page
        # said anything at all.
        opens = re.search(r"The exception is\s+structural", body)
        assert opens, (
            "config-validation.md no longer explains the structural exception "
            "in the sentence this guard reads."
        )
        tail = body[opens.end():]
        named = {token.rstrip(":")
                 for token in re.findall(r"`([a-z_]+):`", tail)}
        assert named == set(_NOT_YET) | {"campaign"}, (
            f"the page names {sorted(named)} as refused before any check "
            f"runs; _structural refuses {sorted(set(_NOT_YET) | {'campaign'})}."
        )
        assert named <= set(_SECTIONS), sorted(named - set(_SECTIONS))
        # And the sentence COUNTS them in words, which the set comparison
        # above cannot see: a section added to either table would be added to
        # the list and leave "four" behind.
        counted = re.search(r"(\w+) whole sections this layer does not read",
                            tail)
        assert counted, (
            "the page no longer counts the sections it names in the "
            "'N whole sections this layer does not read' form."
        )
        word = counted.group(1).lower()
        assert word in self._NUMBER_WORDS, f"unknown number word {word!r}"
        assert self._NUMBER_WORDS[word] == len(named), (
            f"config-validation.md says {word} sections and names "
            f"{len(named)}: {sorted(named)}."
        )


class TestTheCountsProseStatesAboutThisLayer:
    """A count in prose that no run checks is this project's own failure mode.

    ``tests/test_readme_counts.py`` closed it for the README's test count and
    ``tests/test_docs_links.py`` for the D range; the two counts below are the
    ones Plan 3A wrote and nothing else reads. The changelog's number is the
    interesting one: it is neither ``len(CHECKS)`` (34 slots, high by the
    dotted ``A1.*`` keys) nor anything else the registry hands back directly,
    so the plan's own body warned that "this one has no guard".
    """

    #: The schema §6 rows Plan 3A decides, from its scope table -- A1, A38,
    #: A39; A2, A3, A4, A6, A7, A32; A5, A8, A31; A14, A15; A16-A19; A20,
    #: A21, A23, A29; A24, A25; A27, A28; A30, A33; A41, A42, A52. Written
    #: out because it is a historical fact about one plan rather than a live
    #: property of the registry: Plan 3B registers into the same ``CHECKS``,
    #: so an equality against the registry would go red on work that is not
    #: wrong. What IS asserted against the registry is that every one of them
    #: is still there.
    PLAN_3A = frozenset({
        "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A14", "A15", "A16",
        "A17", "A18", "A19", "A20", "A21", "A23", "A24", "A25", "A27", "A28",
        "A29", "A30", "A31", "A32", "A33", "A38", "A39", "A41", "A42", "A52",
    })

    def test_every_check_plan_3a_claims_is_registered(self):
        """The direction that goes silently wrong: a check quietly dropped.

        ``Finding.check`` is the bare id and a registry SLOT may be dotted
        (``A1.runs``), so the comparison is over the bare ids. Measured at
        this commit the two are EQUAL -- 31 each, 34 slots -- but only the
        subset is asserted, because 3B adds to the same table.

        **The direction this is open on, for whoever writes Plan 3B.** The
        DROP direction is defended in depth and not only here: measured, all
        34 slots are pinned by their owning task's own module -- 29 by
        IDENTITY (``CHECKS[id] is <fn>``, several of them through a loop over
        the ids one function claims) and Task 3's five (``A1.runs``,
        ``A1.variants``, ``A1.horizon``, ``A38``, ``A39``) by MEMBERSHIP
        (``set(_IDS) <= set(CHECKS)``, ``test_preflight_document.py:112``).
        So removing an id from the registry AND from ``PLAN_3A`` together
        still goes red there. The INFLATE direction does not:
        registering an extra id, adding it to ``PLAN_3A`` and bumping the
        changelog to "thirty-two" exits 0. It is unreachable today --
        swapping this ``<=`` for ``==`` still passes, measured -- and it
        becomes live the moment 3B registers its first check, which is why
        the equality is not asserted. If 3B wants the tighter form back, the
        way to have it is a per-plan id set beside this one, not a widening
        of this one.
        """
        from rheplicant.config.preflight import CHECKS

        bare = {slot.split(".")[0] for slot in CHECKS}
        assert self.PLAN_3A <= bare, (
            f"these schema ids are in Plan 3A's scope and no function "
            f"registers them: {sorted(self.PLAN_3A - bare)}"
        )

    #: The schema §6 rows Plan 3B decides, across all THREE registries -- the
    #: text pass's ``CHECKS``, the axes pass's ``AXIS_CHECKS`` and the built
    #: pass's ``BUILT_CHECKS``. A per-plan set BESIDE :data:`PLAN_3A` and not
    #: a widening of it, which is what ``test_every_check_plan_3a_claims_is_
    #: registered``'s docstring asked Plan 3B for by name.
    #:
    #: A13 is in both this set and two registries: ``A13.text`` decides the
    #: tone's text legs in the pre-flight pass and ``A13.grid`` decides its
    #: grid bounds in the axes pass. It is ONE schema row, so it is one entry
    #: here, and that is why the count below is not the sum of three
    #: registry lengths.
    #:
    #: What is NOT here, and deliberately: B2 and B7 (pinning tests -- the
    #: shipped graph is a tree and one resource is one object, so there is
    #: nothing to register), and Task 8's C7, C11, C17 and B4, which are
    #: refusals fixed or added IN PLACE inside ``config/sections/`` and
    #: register no slot at all. A set built from "what this plan worked on"
    #: rather than "what this plan registered" would be a set no registry can
    #: check.
    PLAN_3B = frozenset({
        "A10", "A11", "A12", "A13", "A26", "A35", "A40", "A43", "A44", "A45",
        "A46", "A47", "A48", "A49", "A50", "B5", "B9", "C1", "C2", "C3", "C8",
        "C9",
    })

    def test_every_check_plan_3b_claims_is_registered(self):
        """The same direction as 3A's, over the three registries 3B ships.

        Subset and not equality, for the reason 3A's own docstring gives: a
        later plan registers into the same tables. What equality would buy is
        bought instead by the disjointness assertion below -- an id that
        drifted from one plan's set to the other's is caught there without
        making either set a hostage to the next plan's work.
        """
        from rheplicant.config.inflight import AXIS_CHECKS, BUILT_CHECKS
        from rheplicant.config.preflight import CHECKS

        bare = {slot.split(".")[0]
                for registry in (CHECKS, AXIS_CHECKS, BUILT_CHECKS)
                for slot in registry}
        assert self.PLAN_3B <= bare, (
            f"these schema ids are in Plan 3B's scope and no function "
            f"registers them in any of the three passes: "
            f"{sorted(self.PLAN_3B - bare)}"
        )

    #: The schema §6 rows Plan 3C decides, across all FOUR registries -- the
    #: three above plus the post-flight pass's own ``CHECKS``. Derived once
    #: and written down, exactly as its two siblings are: measured at this
    #: commit, the union of the four registries' bare ids is 60, of which
    #: ``PLAN_3A`` holds 31 and ``PLAN_3B`` 22, leaving these seven.
    #:
    #: **A1 is NOT here even though Plan 3C registers ``A1.checks``.** A1 is
    #: "unknown key anywhere", one schema row, and Plan 3A claimed it; a plan
    #: adding a slot under a row somebody else already decided does not
    #: re-claim the row. The RULE -- one schema row is one entry, however many
    #: registries or slots implement it -- is the same rule 3B's own set
    #: applies to A13 (one row, two registries, one entry); but A13's two
    #: slots are BOTH 3B's, so that citation does not exercise the CROSS-PLAN
    #: case A1 is. A1 is this set's own instance of that case, and it is what
    #: keeps the three-way disjointness below meaningful rather than a
    #: formality.
    #:
    #: C18 and C19 did not exist in schema §6 before this plan; Task 1 added
    #: both rows in the commit that added their ids, so the census is green at
    #: every commit rather than only at the end.
    PLAN_3C = frozenset({"A37", "C12", "C13", "C15", "C16", "C18", "C19"})

    def test_every_check_plan_3c_claims_is_registered(self):
        """The same direction as 3A's and 3B's, over all four registries.

        Subset and not equality, for the reason 3A's own docstring gives.
        ``C15`` is in the AXES registry and not the text pass, deliberately:
        the only text reader for ``n_freq``
        (``preflight/values.py::_a41_scope``) answers ``None`` for a
        non-``linspace``/``arange``/``modulo``/list grid, for a symbolic
        ``num:`` and for every ingested run, so a pre-flight C15 would be
        silent by construction on a whole class of documents. The axes pass
        carries ``context.shape_scope`` and still runs ahead of
        ``build_resources``.
        """
        from rheplicant.config.inflight import AXIS_CHECKS, BUILT_CHECKS
        from rheplicant.config.postflight import CHECKS as PRICED
        from rheplicant.config.preflight import CHECKS

        bare = {slot.split(".")[0]
                for registry in (CHECKS, AXIS_CHECKS, BUILT_CHECKS, PRICED)
                for slot in registry}
        assert self.PLAN_3C <= bare, (
            f"these schema ids are in Plan 3C's scope and no function "
            f"registers them in any of the four passes: "
            f"{sorted(self.PLAN_3C - bare)}"
        )

    def test_the_three_plans_claim_disjoint_rows(self):
        """No plan re-decided a row another had already taken.

        This is what makes the three sets addable and the changelog's three
        numbers independent. It is also the assertion that goes red if a
        later reader "fixes" a drifted id by adding it to two lists.

        **Pairwise and not a triple intersection.** ``A & B & C`` is empty the
        moment any one pair is disjoint, so a row shared by exactly two plans
        -- which is the mistake a reader actually makes -- would pass it.
        """
        for left, right in (("PLAN_3A", "PLAN_3B"), ("PLAN_3A", "PLAN_3C"),
                            ("PLAN_3B", "PLAN_3C")):
            shared = getattr(self, left) & getattr(self, right)
            assert shared == frozenset(), (
                f"{left} and {right} both claim {sorted(shared)}. A schema "
                "row is decided by one plan; adding a drifted id to both "
                "lists makes each plan's changelog count wrong by one."
            )

    #: ``TestThePagesSayWhatTheLayerDoes``' table, plus the neighbourhood the
    #: changelog's own number sits in. Extended rather than replaced: a
    #: reworded "thirty-two" must fail as a WRONG COUNT (32 != 31), where an
    #: unknown word would fail as a broken scan and read as this guard's bug.
    _WORDS = {**TestThePagesSayWhatTheLayerDoes._NUMBER_WORDS,
              "twenty-nine": 29, "thirty": 30, "thirty-one": 31,
              "thirty-two": 32, "thirty-three": 33}

    #: Each plan's changelog entry, by its own ``### `` heading, and the
    #: sentence that entry states its count in.
    #:
    #: **The heading is what anchors the search, and that is the whole point.**
    #: ``CHANGELOG.md`` is newest-at-top and ``re.search`` returns the
    #: *leftmost* match, so a guard that searches the whole file reads
    #: whichever entry happens to be highest -- and every later plan writes
    #: above every earlier one. Measured by review: an entry saying
    #: *"Thirty-one schema §6 checks now decide something entirely
    #: different."* placed above these two leaves the whole module at exit 0
    #: while Plan 3A's guard reads **that entry's** number.
    _ENTRIES = {
        "3A": ("### Everything a document can be refused for before it costs "
               "anything",
               r"([A-Za-z-]+)\s+schema §6 checks now decide"),
        "3B": ("### The rest of what text decides, and two slots for what it "
               "cannot",
               r"registers\s+(\d+)\s+schema §6 ids\s+across\s+the\s+three"
               r"\s+passes"),
        "3C": ("### The checks that cost something, and the gate that decides "
               "whether to pay",
               r"puts a price on\s+(\d+)\s+schema §6 rows"),
    }

    def _entry(self, plan: str) -> tuple[str, str]:
        """``(the text of that plan's entry, the whole file)``.

        The entry runs from its own ``### `` heading to the next heading at
        the same level or above -- so a count stated in a *neighbouring*
        entry is outside the slice and cannot be mistaken for this one's.

        **Two guards and not one, because ``in`` is weaker than the slice.**
        The membership test below accepts a heading that has been *reworded*
        around this one -- ``"### The rest of what text decides, and two
        slots for what it cannot (2026)"`` contains the string but is not the
        line ``partition`` needs. That produces an EMPTY slice, and an empty
        slice makes ``changelog.index("")`` return 0, so the callers below go
        red pointing at a sentence-collision that does not exist. The second
        assertion turns that into the true message: *the heading moved.*
        Same failure mode as the tautology this class replaced, one layer
        down -- the guard survives, but what it tells the reader is false.
        """
        heading, _ = self._ENTRIES[plan]
        changelog = (_DOCS.parent / "CHANGELOG.md").read_text()
        assert heading in changelog, (
            f"CHANGELOG.md no longer carries Plan {plan}'s entry under "
            f"{heading!r}. This guard finds that plan's count by its own "
            "heading; renaming the heading without moving the guard leaves "
            "the guard reading whichever entry sorts highest."
        )
        after = changelog.partition(f"\n{heading}\n")[2]
        entry = re.split(r"\n#{1,3} ", after)[0]
        assert entry.strip(), (
            f"Plan {plan}'s heading {heading!r} appears in CHANGELOG.md but "
            "not as a heading LINE of its own -- it has been reworded or has "
            "something appended to it, so this guard slices an empty entry "
            "and every count below would fail for the wrong reason. Update "
            "_ENTRIES to the new wording."
        )
        return entry, changelog

    def test_the_changelog_counts_the_checks_plan_3a_decided(self):
        """"Thirty-one schema §6 checks" was prose with no reader.

        Neither registry number is it: 34 SLOTS is high by the three dotted
        ``A1.*`` keys and by ``A14.cal_loads``. The number is the count of
        schema ROWS, which is what ``PLAN_3A`` holds -- so this asserts the
        word against that, and the test above asserts that against the
        registry. ``\\s+`` and not a space: the sentence wraps between the
        number and the word it counts, and a line-anchored scan would read
        nothing and stay green.

        **Searched inside Plan 3A's own entry, not over the file.** See
        :data:`_ENTRIES`: the unanchored form read whichever entry sat
        highest, which every later plan changes.
        """
        entry, _ = self._entry("3A")
        stated = re.search(self._ENTRIES["3A"][1], entry)
        assert stated, (
            "Plan 3A's CHANGELOG entry no longer states how many schema §6 "
            "checks it decides, in the 'N schema §6 checks now decide' form "
            "this guard reads."
        )
        word = stated.group(1).lower()
        assert word in self._WORDS, f"unknown number word {word!r}"
        assert self._WORDS[word] == len(self.PLAN_3A), (
            f"CHANGELOG.md says {word} checks; Plan 3A's scope table has "
            f"{len(self.PLAN_3A)}."
        )

    def test_the_changelog_counts_the_rows_plan_3b_registered(self):
        """Plan 3B's own number, in a sentence form of its own, and in DIGITS.

        Digits and a different sentence, for two measured reasons.
        :data:`_WORDS` stops at "thirty-three", so a number word outside its
        table would fail as ``unknown number word`` -- which reads as this
        guard's bug rather than as a wrong count -- and ``\\d+`` needs no word
        table at all. And reusing 3A's wording in a *newer* entry is the trap
        :data:`_ENTRIES` describes; a distinct sentence means the two guards
        cannot confuse each other's numbers even if one day the anchoring
        breaks.

        The number is the count of schema ROWS, over the three registries and
        de-duplicated -- A13 is one row decided in two passes. ``\\s+`` and not
        a space, for the reason the sibling above gives: the sentence wraps.
        """
        entry, _ = self._entry("3B")
        stated = re.search(self._ENTRIES["3B"][1], entry)
        assert stated, (
            "Plan 3B's CHANGELOG entry no longer states how many schema §6 "
            "rows it registers, in the 'registers N schema §6 ids across the "
            "three passes' form this guard reads."
        )
        assert int(stated.group(1)) == len(self.PLAN_3B), (
            f"CHANGELOG.md says {stated.group(1)} rows; Plan 3B's scope set "
            f"has {len(self.PLAN_3B)}."
        )

    def test_the_changelog_counts_the_rows_plan_3c_priced(self):
        """Plan 3C's number, in a THIRD sentence form, and in DIGITS.

        A third form and not a reuse of either sibling's, for the reason
        :data:`_ENTRIES` gives and 3B's own docstring repeats: two entries
        sharing a wording make each guard read whichever sits highest, and
        3C's entry is written *above* both. Digits for 3B's reason --
        :data:`_WORDS` stops at "thirty-three", so a number word outside its
        table fails as ``unknown number word``, which reads as this guard's
        bug rather than as a wrong count.

        The number is the count of schema ROWS this plan decides, over all
        four registries and de-duplicated. ``\\s+`` and not a space: the
        sentence wraps.
        """
        entry, _ = self._entry("3C")
        stated = re.search(self._ENTRIES["3C"][1], entry)
        assert stated, (
            "Plan 3C's CHANGELOG entry no longer states how many schema §6 "
            "rows it prices, in the 'puts a price on N schema §6 rows' form "
            "this guard reads."
        )
        assert int(stated.group(1)) == len(self.PLAN_3C), (
            f"CHANGELOG.md says {stated.group(1)} rows; Plan 3C's scope set "
            f"has {len(self.PLAN_3C)}."
        )

    @pytest.mark.parametrize("plan", sorted(_ENTRIES))
    def test_each_count_sentence_appears_in_exactly_one_entry(self, plan):
        """No entry other than this plan's states its count in this plan's
        words.

        **This replaces an assertion that could not fail.** The first version
        sliced the file at the leftmost match and asserted there was no match
        before it -- true by construction, because ``re.search`` returns the
        leftmost match. Review demonstrated the consequence: a foreign entry
        above Plan 3A's, reusing 3A's exact wording, left the whole module at
        exit 0 while 3A's guard read that entry's number.

        What is asserted now is the real property, and it is checked over the
        WHOLE FILE rather than over a prefix: this plan's sentence occurs
        exactly once, and the one occurrence is inside this plan's own entry.
        A later plan reusing either wording -- above, below, anywhere -- is a
        red test with a message naming what it collided with.

        Plan 3C writes above both of these. This is what it will hear.
        """
        heading, pattern = self._ENTRIES[plan]
        entry, changelog = self._entry(plan)
        everywhere = [match.start() for match in re.finditer(pattern,
                                                             changelog)]
        assert len(everywhere) == 1, (
            f"Plan {plan}'s count sentence appears {len(everywhere)} times in "
            f"CHANGELOG.md. It is how that plan's guard finds its own number, "
            "so a second entry using the same words makes the guard read "
            "whichever sits highest. Give the newer entry a sentence form of "
            "its own, as Plan 3B's does."
        )
        start = changelog.index(entry)
        assert start <= everywhere[0] < start + len(entry), (
            f"Plan {plan}'s count sentence is not inside its own entry "
            f"({heading!r}); the guard is reading a neighbour's number."
        )


class TestPlan3BsWiringAndItsSurface:
    """What Plan 3B added to the wiring, and what it deliberately did not add
    to the surface.

    Three subjects, all of them decisions the plan asked to be pinned "either
    way" rather than left to a reader's inference.
    """

    def test_plan_3b_adds_no_name_to_the_surface_and_here_is_why(self):
        """The export decision, pinned with its reason rather than asserted.

        Plan 3B ships two new passes and two new registries and exports
        **nothing**. The reason is mechanical, not stylistic: ``axes()`` takes
        an ``Axes``, whose ``runtime`` and ``observation`` fields come from
        ``build_runtime`` and ``build_observation`` -- and **neither builder
        is exported**, so an exported ``axes()`` would be a public entry point
        a caller could not construct an argument for. ``built()`` is worse
        again: its payload carries the twin, the state and the resources.

        ``AXIS_CHECKS`` and ``BUILT_CHECKS`` follow ``CHECKS``, which
        ``test_the_check_registry_stays_wiring_rather_than_surface`` already
        keeps off the surface: a registry is a mutable handle on the table
        that decides what every document means.

        The whole-list assertion in ``TestThePlan2CSurface`` is what would go
        red if a name were added; this says why none was, and asserts the
        premise -- so if a later plan DOES export ``build_runtime``, the
        argument recorded here stops being true and this test says so.
        """
        import rheplicant.config as config

        for builder in ("build_runtime", "build_observation"):
            assert builder not in config.__all__, (
                f"{builder} is exported now, so the reason recorded here for "
                "keeping axes() off the surface no longer holds; re-decide it "
                "rather than letting this comment rot."
            )
        for name in ("axes", "built", "AXIS_CHECKS", "BUILT_CHECKS", "Axes",
                     "Built"):
            assert name not in config.__all__, name

    def test_the_three_prior_gates_are_one_function_in_the_order_written(self):
        """3A's load-bearing anchors A20, A21 and A23 keep their order.

        **What this actually pins, said plainly, because the plan's own
        framing of it was wrong.** A20, A21 and A23 are bound by ONE
        ``@register("A20", "A21", "A23")`` on ONE function
        (``preflight/fitting.py::_prior_gates``), and ``passes.sweep``
        de-duplicates by ``id(fn)`` -- so the pass calls that function once
        and there is no "A20 runs, then A23 runs" sequence in the pass at all.
        The ordering that matters is INSIDE the function's body, where A20 and
        A21 ``continue`` past runs whose A23 legs are unreachable, and the
        function's own docstring records it as structural.

        So the index comparison below pins the decorator's argument order --
        a real property, and the one a reader looking up a slot meets -- and
        nothing more. The identity assertion is the load-bearing half.

        **Why an index comparison is legal here when the cross-module one was
        withdrawn.** §0.3 F.2's exemption for a relative registration-index
        assertion was granted and then refuted at Plan 3B's wave-1 merge:
        ``test_preflight_ingest.py``'s ``A28 < A10 < A2`` went red because
        ``preflight/beam_spill.py`` head-imports ``document`` and ``model``,
        so a module's position in the alphabetical foot block decides nothing.
        That failure mode cannot reach this assertion: one ``@register`` call
        in one module fixes all three positions relative to each other, and no
        sibling's imports can interleave them.
        """
        from rheplicant.config.preflight import CHECKS

        assert CHECKS["A20"] is CHECKS["A21"] is CHECKS["A23"], (
            "A20, A21 and A23 are no longer one function. The 'A20 and A21 "
            "make two of A23's legs moot' reasoning in _prior_gates' "
            "docstring is about one body's control flow; split across "
            "functions it becomes a claim about run order, which is not what "
            "the registry gives you."
        )
        order = list(CHECKS)
        assert order.index("A20") < order.index("A21") < order.index("A23")

    def test_the_one_binding_walker_is_called_from_the_modules_that_hoist(
            self):
        """A FLOOR on ``assert_bound_once`` call sites, found by a glob.

        §3.2(h)'s shared table was replaced by "each hoisting task
        parametrizes a test in its own module over its own literals", which
        is the right shape and has one failure mode: a task can simply not
        write one, and nothing central notices. This is the central notice --
        deliberately a floor and not an equality, so a later plan adding rows
        is never a red test here.

        **Discovered by a glob over ``tests/config/*.py``**, never a module
        list: a maintained list is indistinguishable from a glob while the
        set does not change, and this project has already paid for that
        (§0.3 F.5(6)). ``message_binding.py`` itself is excluded -- it holds
        the definition and its own anti-vacuity case, neither of which is a
        task pinning a hoisted literal.

        **The floor undercounts on purpose.** Most call sites sit inside a
        ``parametrize`` over a tuple of literals, so eleven call sites walk
        considerably more than eleven messages. Counting the literals would
        mean importing seven modules and reading private tuples out of them,
        which is a second copy of each task's own list.

        **Counted over ``ast`` and not with ``str.count``, and that is
        measured.** A mutant that replaced ``assert_bound_once(literal)`` with
        ``pass  # assert_bound_once(literal)`` SURVIVED the substring form:
        the text is still in the file, in a comment. A call node is a call.
        """
        import ast

        def _calls(source: str) -> int:
            found = 0
            for node in ast.walk(ast.parse(source)):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "id", None) or getattr(
                    node.func, "attr", None)
                found += name == "assert_bound_once"
            return found

        helper = "message_binding.py"
        # Off this file's OWN location, never off the cwd: a relative
        # "tests/config" glob answers an empty list when pytest is run from
        # anywhere else, and an empty list would make the floor unreachable
        # rather than red -- which is why the anti-vacuity line is below.
        modules = sorted(pathlib.Path(__file__).resolve().parent.glob("*.py"))
        assert any(path.name == helper for path in modules), (
            "the glob found no message_binding.py, so it is not looking "
            "where it thinks it is and every count below is zero"
        )
        sites = {path.name: _calls(path.read_text())
                 for path in modules if path.name != helper}
        carrying = {name: n for name, n in sites.items() if n}
        assert sum(carrying.values()) >= 11, (
            f"only {sum(carrying.values())} assert_bound_once call sites: "
            f"{carrying}. Plans 3A and 3B leave eleven across seven modules; "
            "a hoisted message with no one-binding pin is the "
            "_number-vs-_whole divergence with nothing comparing the two."
        )
        assert len(carrying) >= 7, (
            f"only {len(carrying)} modules call the walker: {sorted(carrying)}"
        )


class TestPlan3CsSurfaceAndItsPage:
    """What Plan 3C put on the surface, and the page that has to agree.

    Two names land -- ``gates`` and ``Gate`` -- and one deliberately does not.
    Everything below either asserts that decision or executes a claim the page
    makes about the gate, so that the page cannot drift from
    ``config/gating.py`` without a red test naming the drift.
    """

    PAGE = "config-validation.md"

    #: The gate's own sections, by their exact headings. Written here rather
    #: than inlined so that a heading renamed on the page fails ONCE, with
    #: ``_block``'s own "no longer a heading" message, instead of once per
    #: assertion with a message about a table that is really about a heading.
    COSTS = "## The post-flight pass, and what it costs"
    GATE = "## A gate: what runs, what a failure costs, and what is recorded"
    STATES_TABLE = "### Six effective states, four of them writable"
    CROSS = "### The cross-product, as one table"
    IN_CODE = "### What a gate is, in code"
    SLOTS = "## The three later slots, and what each one buys"
    SPELLINGS = "### A refused document produces no record at all"

    #: :data:`TestThePagesSayWhatTheLayerDoes._NUMBER_WORDS` stops at sixteen
    #: and the cross-product is eighteen cells. Extended here rather than
    #: widened there, for the reason
    #: :data:`TestTheCountsProseStatesAboutThisLayer._WORDS` gives: a reworded
    #: count must fail as a WRONG COUNT, where an unknown word fails as a
    #: broken scan and reads as this guard's own bug.
    _WORDS = {**TestThePagesSayWhatTheLayerDoes._NUMBER_WORDS,
              "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20}

    #: How the page's ``report:`` column expands into the cells it stands for.
    #: THREE of the five spellings expand, and each for its own reason:
    #: ``either`` says the outcome does not depend on ``report:``, ``—`` says
    #: the column is meaningless for a gate that never ran, and ``ignored``
    #: says :func:`~rheplicant.config.gating.verdict` reads past it. The two
    #: literal spellings pin one value each.
    _REPORT_CELL = {"either": (True, False), "true": (True,),
                    "false": (False,), "ignored": (True, False),
                    "—": (True, False)}

    #: And how the ``it failed`` column does NOT expand. ``—`` is one cell and
    #: not two: a gate that never ran has no failure to have had, so pairing
    #: it with both truth values would count a state of the world that cannot
    #: occur. This asymmetry is the whole of why the table is eighteen cells
    #: and not twenty-four.
    _FAILED_CELL = {"yes": True, "no": False, "—": False}

    def _cross_product_rows(self):
        rows = _rows(_block(_page(self.PAGE), self.CROSS))
        assert rows, (
            f"{self.PAGE}'s cross-product table stopped parsing under "
            f"{self.CROSS!r}; every assertion over it would be vacuous."
        )
        return rows

    # -- the surface ------------------------------------------------------

    def test_plan_3c_adds_gates_and_gate_and_here_is_why(self):
        """The two names, on the same footing as ``preflight``/``Report``.

        A caller CALLS ``gates`` on the ``inference.checks:`` mapping their
        own document already holds, and READS the ``Gate``s it hands back.
        Both halves are asserted rather than described: the function must be
        callable on a raw mapping with no build behind it, which is the
        property that distinguishes it from ``axes``/``built``/``priced``.
        """
        import rheplicant.config as config

        for name in ("Gate", "gates"):
            assert name in config.__all__, name
            assert hasattr(config, name), name
        resolved = config.gates({"linearity": {"mode": "warn"}})
        assert set(resolved) == set(config.gates(None))
        assert all(isinstance(gate, config.Gate)
                   for gate in resolved.values())

    def test_the_priced_pass_stays_wiring_rather_than_surface(self):
        """``priced`` is NOT exported, and the reason is Plan 3B's.

        It is a pass ``load_document`` runs on the caller's behalf: its
        payload carries the built twin, the state, the resources and the
        resolved gates, and by the time a caller could construct one the
        answer is already on ``ConfiguredRun.report``. ``preflight`` remains
        the one pass a front-end calls itself, because it is the one that
        answers before anything is built.

        Pinned both ways, as ``CHECKS``/``register`` are: absent from
        ``__all__`` is about ``import *``, and absent as an attribute is about
        ``config.priced``.
        """
        import rheplicant.config as config

        for name in ("priced", "Priced", "PostCheck", "verdict",
                     "check_gates", "auto_skipped", "MODES", "STATES",
                     "DEFAULT_MODE", "CHECK_ID", "CHECK_NAMES", "AUTO_SKIP",
                     "AUTO_SKIP_ID", "OFF"):
            assert name not in config.__all__, name
            assert not hasattr(config, name), name

    def test_the_defaults_are_reachable_through_the_exported_function(self):
        """``gates(None)`` applies them, so no caller needs ``DEFAULT_MODE``.

        This is the premise of keeping the defaults table module-local. If it
        stopped holding -- if ``gates(None)`` returned ``{}`` the way
        ``sections/inference.py::_checks(None)`` used to -- the argument for
        not exporting ``DEFAULT_MODE`` would be gone and this says so.
        """
        from rheplicant.config import gates
        from rheplicant.config.gating import CHECK_NAMES, DEFAULT_MODE

        applied = {name: gate.state for name, gate in gates(None).items()}
        assert applied == DEFAULT_MODE, (
            f"gates(None) applies {applied}; DEFAULT_MODE is {DEFAULT_MODE}. "
            "A caller told to ask gates(None) for the defaults is being told "
            "something false, and DEFAULT_MODE would have to be exported."
        )
        assert set(applied) == set(CHECK_NAMES)

    def test_a_state_this_package_does_not_know_does_not_run(self):
        """``Gate.runs`` is a POSITIVE test and gating.py says why. Measured
        at review: negating it survives the whole of tests/config.

        ``state in ("refuse", "warn", "report")`` and
        ``state not in ("skip", OFF, AUTO_SKIP)`` agree over every member of
        :data:`~rheplicant.config.gating.STATES` -- six states, and the two
        forms answer the same for each -- which is exactly why a mutation to
        the negated form survived the whole suite: nothing in ``STATES`` can
        tell them apart. But ``STATES`` is not the full domain. ``gates()``
        never validates ``mode`` (see this class's precondition test), and
        ``gates`` is public, so ``gates({"linearity": {"mode": <typo>}})``
        builds exactly the gate below -- a state outside ``STATES`` -- and the
        two forms disagree on it: the positive form stands it down, the
        negated form runs it.
        """
        from rheplicant.config import Gate
        from rheplicant.config.gating import STATES

        unknown = Gate(name="linearity", state="a_state_no_plan_has_added",
                       record=False, reason=None, rtol=None)
        assert unknown.state not in STATES
        assert unknown.runs() is False, (
            "an unrecognised state RUNS. gating.py's docstring: a state added "
            "later must default to NOT running -- a lost check is silent, a "
            "refusal that should not have happened is loud. This is also "
            "reachable from the public surface: gates({'linearity': "
            "{'mode': <typo>}}) builds exactly this gate."
        )

    # -- the page ---------------------------------------------------------

    def test_the_page_lists_every_effective_state_and_its_spelling(self):
        """Six states, four writable, and the page's word for each.

        Read off ``STATES`` and ``MODES`` so that a seventh state -- or a
        fifth writable word -- moves the page rather than sitting undocumented
        in a module nobody reads. The count in the HEADING is scanned too, and
        it is the one that most needs it: it is also this table's locator, so
        a page corrected to seven states and a heading left saying six keeps
        every other assertion here passing.
        """
        from rheplicant.config.gating import MODES, STATES

        rows = _rows(_block(_page(self.PAGE), self.STATES_TABLE))
        assert rows, f"{self.PAGE}'s effective-state table stopped parsing"
        listed = [row[0].strip("`") for row in rows]
        assert listed == list(STATES), (
            f"the page lists {listed}; gating.STATES is {list(STATES)}. The "
            "order is the page's own claim -- the four writable ones, then "
            "the two that are not."
        )
        writable = {row[0].strip("`") for row in rows
                    if "mode:" in row[1]}
        assert writable == set(MODES), (
            f"the page shows {sorted(writable)} as writable as a mode:; "
            f"gating.MODES is {sorted(MODES)}."
        )
        for word, what in ((r"(\w+) effective states", "the heading's count"),
                           (r"states, (\w+) of them writable",
                            "the heading's writable count")):
            found = re.search(word, self.STATES_TABLE, re.IGNORECASE)
            assert found, f"{what} is no longer in {self.STATES_TABLE!r}"
            said = found.group(1).lower()
            assert said in self._WORDS, f"{what}: unknown word {said!r}"
        assert self._WORDS[re.search(r"(\w+) effective states",
                                     self.STATES_TABLE,
                                     re.IGNORECASE).group(1).lower()] == len(STATES)
        assert self._WORDS[re.search(r"states, (\w+) of them writable",
                                     self.STATES_TABLE,
                                     re.IGNORECASE).group(1).lower()] == len(MODES)

    def test_the_pages_cross_product_table_is_EXECUTED(self):
        """Every cell of the page's table is driven through ``verdict``.

        This is the load-bearing guard on this page and it is deliberately not
        a shape check. §4.7.8 shipped the ``mode``-versus-``report:``
        ambiguity in prose three times; a table that merely *parses* would
        ship it a fourth. So each row is expanded into the cells it stands
        for, a ``Gate`` is built for each, ``verdict`` is called, and the
        severity and check id the page promises are compared against the
        finding it actually returns.

        ``name="linearity"`` throughout: the cross-product is a property of
        the STATE and not of which check is in it, and ``CHECK_ID`` is read
        rather than written so the expected id moves if the mapping does. The
        ``auto_skip`` row promises a DIFFERENT id, and the page writes it in
        backticks; that is read off the cell rather than hardcoded here.
        """
        from rheplicant.config.findings import REFUSE, REPORT, WARN
        from rheplicant.config.gating import CHECK_ID, Gate, verdict

        severities = {"REFUSE": REFUSE, "WARN": WARN, "REPORT": REPORT}
        driven = 0
        for state, ran, failed, record, promised in self._cross_product_rows():
            state = state.strip("`")
            # The "the check ran?" column is a claim too, and ``Gate.runs()``
            # is what decides it. Asserted here rather than in a test of its
            # own because it is what makes the ``—`` cells legible: a row that
            # says "no" is exactly a row whose failed and report: columns
            # stand for nothing.
            probe = Gate(name="linearity", state=state, record=False,
                         reason="the document said so", rtol=None)
            assert probe.runs() == (ran == "yes"), (
                f"the page says the {state!r} row {'runs' if ran == 'yes' else 'does not run'}; "
                f"Gate.runs() says {probe.runs()}."
            )
            assert record in self._REPORT_CELL, (
                f"the report: cell {record!r} on the {state!r} row is not one "
                f"of {sorted(self._REPORT_CELL)}; this guard cannot know how "
                "many cells it stands for."
            )
            assert failed in self._FAILED_CELL, (
                f"the failed cell {failed!r} on the {state!r} row is not one "
                f"of {sorted(self._FAILED_CELL)}."
            )
            named = re.search(r"\*\*(REFUSE|WARN|REPORT)\*\*", promised)
            spelled = re.search(r"`(C\d+)`", promised)
            assert named or "*none*" in promised, (
                f"the {state!r}/{failed!r}/{record!r} row promises "
                f"{promised!r}, which is neither a **SEVERITY** nor *none*."
            )
            for keeps in self._REPORT_CELL[record]:
                gate = Gate(name="linearity", state=state, record=keeps,
                            reason=("the document said so"
                                    if state in ("skip", "auto_skip")
                                    else None),
                            rtol=None)
                got = verdict(gate, failed=self._FAILED_CELL[failed],
                              where="inference.parameters.g",
                              message="the numbers")
                driven += 1
                if named is None:
                    assert got is None, (
                        f"the page promises no finding for state={state!r}, "
                        f"failed={failed!r}, report={keeps}; verdict returns "
                        f"{got}."
                    )
                    continue
                assert got is not None, (
                    f"the page promises **{named.group(1)}** for "
                    f"state={state!r}, failed={failed!r}, report={keeps}; "
                    "verdict returns nothing."
                )
                assert got.severity == severities[named.group(1)], (
                    f"the page promises **{named.group(1)}** for "
                    f"state={state!r}, failed={failed!r}, report={keeps}; the "
                    f"finding is {got.severity!r}."
                )
                expected = spelled.group(1) if spelled else CHECK_ID["linearity"]
                assert got.check == expected, (
                    f"the page promises the finding to carry {expected!r} for "
                    f"state={state!r}; it carries {got.check!r}."
                )
        assert driven == 18, (
            f"the page's table expands to {driven} cells and this guard was "
            "written against eighteen. If a state or a column spelling was "
            "added, move the count HERE and in the page's own sentence -- do "
            "not delete the assertion."
        )

    def test_the_page_counts_the_cells_its_table_stands_for(self):
        """"eighteen cells, not twelve rows", against the table itself.

        The sentence is the whole reason the table is legible: a reader who
        counts rows concludes ``{mode: skip, report: true}`` has a cell, and
        it does not. Both numbers are read off the page and both are compared
        -- the row count against the table, the cell count against the
        expansion -- because a page corrected in one and not the other is the
        drift this guards.
        """
        body = _block(_page(self.PAGE), self.CROSS)
        rows = self._cross_product_rows()
        stated = re.search(r"(\w+) cells, not (\w+) rows", body,
                           re.IGNORECASE)
        assert stated, (
            f"{self.PAGE} no longer counts its cross-product in the "
            "'N cells, not M rows' form this guard reads. Reword the pattern "
            "or the page, but do not leave the count unread."
        )
        cells, written = (word.lower() for word in stated.groups())
        for word in (cells, written):
            assert word in self._WORDS, f"unknown number word {word!r}"
        assert self._WORDS[written] == len(rows), (
            f"the page says {written} rows and the table has {len(rows)}."
        )
        expanded = sum(len(self._REPORT_CELL[row[3]]) for row in rows)
        assert self._WORDS[cells] == expanded, (
            f"the page says {cells} cells and the table expands to {expanded}."
        )

    def test_the_page_names_the_id_an_auto_skip_reports_under(self):
        """C14 is bound by ``verdict``, never registered, so nothing else
        would notice it moving.

        ``AUTO_SKIP_ID`` is not in any registry -- that is the point of it, so
        that a user grepping the record for C14 finds every check that was
        asked for and could not be decided -- which means the schema-census
        test cannot see it either. This is the only guard between the constant
        and the page.
        """
        from rheplicant.config.gating import AUTO_SKIP_ID

        # ``_section`` and not ``_block``: the id is named in the prose under
        # the states table AND promised again in the cross-product's last
        # row, and either is a legitimate home for it. What must not happen is
        # the whole gate section going quiet about it.
        body = _section(_page(self.PAGE), self.GATE)
        assert f"`{AUTO_SKIP_ID}`" in body, (
            f"the gate section no longer names {AUTO_SKIP_ID!r} as the id an "
            "auto-skip reports under."
        )

    def test_the_page_counts_a_gates_fields_and_names_them(self):
        """"A `Gate` is five fields" -- and it names all five.

        The count and the names together, for the reason the ``Finding``
        sibling above gives: a sixth field leaves the word "five" false, and a
        renamed field leaves the page pointing at a name that is gone.
        """
        from rheplicant.config import Gate

        body = _block(_page(self.PAGE), self.IN_CODE)
        opens = re.search(r"A `Gate` is (\w+) fields: (.*?)\n\n", body,
                          re.DOTALL)
        assert opens, f"{self.PAGE} no longer counts a Gate's fields"
        word = opens.group(1).lower()
        assert word in self._WORDS, f"unknown number word {word!r}"
        assert self._WORDS[word] == len(Gate._fields), (
            f"the page says {word} fields; Gate has {len(Gate._fields)}."
        )
        named = set(re.findall(r"`(\w+)`", opens.group(2)))
        assert set(Gate._fields) <= named, (
            f"the page counts {word} fields and does not name "
            f"{sorted(set(Gate._fields) - named)}."
        )

    def test_the_precondition_gates_advertises_holds_as_written(self):
        """``gates`` is advertised as free and pure over RAW text -- the
        schema's "Validate" panel holding a section the user is still typing
        -- and neither the page nor ``config/__init__.py`` used to name a
        precondition. Measured at review: on a section that has not passed
        ``check_gates``, a non-numeric ``rtol:`` does not stand the check
        down, it RAISES; and an unknown ``mode:`` word does not raise, it
        becomes a ``Gate`` whose ``state`` is not in ``STATES`` at all
        (``runs()`` reports it as standing down, per
        ``test_a_state_this_package_does_not_know_does_not_run`` above).
        Both behaviours are pinned here, driven rather than read, so the
        sentence describing them cannot rot silently on either page.
        """
        import rheplicant.config as config
        from rheplicant.config.gating import STATES

        with pytest.raises(ValueError):
            config.gates({"identifiability": {"rtol": "1e-8x"}})

        typo = config.gates(
            {"linearity": {"mode": "a_typo_no_plan_has_added"}})["linearity"]
        assert typo.state not in STATES
        assert typo.runs() is False

        needle = "already passed the pre-flight grammar"
        body = _block(_page(self.PAGE), self.IN_CODE)
        assert needle in body, (
            f"{self.PAGE} no longer states gates()'s precondition under "
            f"{self.IN_CODE!r}."
        )
        assert needle in (config.__doc__ or ""), (
            "rheplicant.config's own module docstring no longer states "
            "gates()'s precondition -- the page and the package have drifted "
            "apart again."
        )

    def test_the_page_counts_the_spellings_of_report_it_tabulates(self):
        """The one count on this page taken from a sentence rather than a
        table. Measured at review: "Three" and "Four" both passed -- nothing
        compared the sentence to the table it introduces.

        **Also checks ``gating.py``'s own copy of the same sentence, by
        equality against the page's.** The two docstrings carry the same
        claim over the same four-row list, in two unrelated syntaxes (a
        markdown table here, an RST grid table there) that no single parser
        reads -- two ROUTES to the same fact, which is exactly the shape the
        project's own defect pattern names: closing the route through one
        without checking the other is how a twin survives. Reverting either
        file's count alone, with the other left correct, is measured to
        reproduce the review's finding (``EXIT=0`` over the full suite either
        way) if only one of the two assertions below exists.
        """
        body = _block(_page(self.PAGE), self.SPELLINGS)
        stated = re.search(r"\*\*(\w+) unrelated things in this layer are "
                           r'spelled "report"', body)
        assert stated, f"{self.PAGE} no longer counts the report spellings"
        word = stated.group(1).lower()
        assert word in self._WORDS, f"unknown number word {word!r}"
        rows = _rows(body)
        assert rows, "the report-spellings table stopped parsing"
        assert self._WORDS[word] == len(rows), (
            f"the page says {word} spellings and tabulates {len(rows)}."
        )

        from rheplicant.config import gating

        module_stated = re.search(
            r"\*\*(\w+) unrelated things in this layer are "
            r'spelled "report"', gating.__doc__ or "")
        assert module_stated, (
            "gating.py's own module docstring no longer counts the report "
            "spellings in the same words the page does."
        )
        assert module_stated.group(1).lower() == word, (
            f"the page says {word!r} spellings; gating.py's own docstring "
            f"says {module_stated.group(1).lower()!r}. The two copies of "
            "this sentence have drifted apart."
        )

    def test_the_cost_table_carries_each_checks_id_and_its_default(self):
        """The page's own check table, against ``CHECK_ID`` and ``gates(None)``.

        Both columns, because they go stale in different ways: an id drifts
        when a schema row is renumbered, and a default drifts the moment
        somebody decides a check is cheap enough to turn on. Equality over the
        NAME set as well, so a fourth check name cannot arrive with no row.
        """
        from rheplicant.config import gates
        from rheplicant.config.gating import CHECK_ID

        rows = _rows(_block(_page(self.PAGE), self.COSTS))
        assert rows, f"{self.PAGE}'s post-flight cost table stopped parsing"
        table = {row[0].strip("`"): (row[2].strip("`"), row[3].strip("`"))
                 for row in rows}
        applied = gates(None)
        assert set(table) == set(CHECK_ID), (
            f"the page tabulates {sorted(table)}; gating.CHECK_ID knows "
            f"{sorted(CHECK_ID)}."
        )
        for name, (check, default) in table.items():
            assert check == CHECK_ID[name], (
                f"the page gives {name} the id {check!r}; CHECK_ID says "
                f"{CHECK_ID[name]!r}."
            )
            assert default == applied[name].state, (
                f"the page says {name} defaults to {default!r}; gates(None) "
                f"applies {applied[name].state!r}."
            )

    def test_the_linearity_interaction_paragraph_is_pinned_and_executed(self):
        """MAJOR 3 (Plan 3C fix round): the page used to mention the probe
        scales only in a table cell, and nothing user-facing said
        ``linearity`` refuses a converter document at the defaults whether or
        not it actually saturates -- ``digitising.py``'s own docstring
        already carried this measurement, but only a reader of the source.

        **Guarded the way this class guards its other prose**
        (``test_the_pages_cross_product_table_is_EXECUTED`` is the precedent):
        the paragraph's own numbers are pinned by substring AND the claim is
        EXECUTED against the real check, so a rewrite that quietly changed
        the probe scales, the escape, or the underlying behaviour goes red
        here rather than reading right forever.
        """
        from rheplicant.config.postflight import priced
        from tests.config.inflight_helpers import priced_run
        from tests.config.preflight_helpers import unsaturated_linear_case

        body = _block(_page(self.PAGE), self.COSTS)
        assert "(1e-3, 1, 1e3)" in body, (
            f"{self.PAGE} no longer names the probe scales beside the "
            "linearity row."
        )
        assert "mode: skip" in body and "reason:" in body, (
            f"{self.PAGE} no longer names linearity's own escape beside the "
            "shipped-interaction paragraph."
        )
        assert "5.32e+00" in body and "12.116166" in body, (
            f"{self.PAGE}'s shipped-interaction paragraph no longer carries "
            "the measured departure/peak it claims."
        )

        # Execute the claim itself: the most benign ADC this package can
        # build, no linearity decline, no C16 finding, C12 refuses anyway.
        document = unsaturated_linear_case()
        payload = priced_run(document)
        assert payload.run.report.refusals() == (), (
            "the built pass already refuses this document; the paragraph's "
            "premise -- a document that otherwise loads clean -- no longer "
            "holds."
        )
        findings = priced(payload)
        checks = {found.check: found for found in findings.findings}
        assert "C16" not in checks, (
            "model.adc: {scale: 1.0, n_bits: 12} now saturates on the base "
            "document -- the paragraph's own worked example ('clips "
            "nothing') is stale."
        )
        assert "C12" in checks and checks["C12"].severity == "refuse", (
            "the base document, with an unsaturating ADC and a declared "
            "linear: true latent, no longer earns a C12 refusal -- the "
            "page's central claim ('refused... whether or not it "
            "saturates') is stale."
        )
        assert "5.32e+00" in checks["C12"].message, (
            "C12's own message no longer carries the departure the page "
            "quotes."
        )

    def test_the_slots_section_counts_the_passes_that_actually_exist(self):
        """A counting heading, a table and an ordering list, all derived.

        This section shipped with 3B counting **two** later slots and was
        guarded by nothing, so a fourth pass made three separate claims false
        at once and no run said so. Every one of the three is now read off the
        registries: a fifth pass with a fifth registry turns all three red.

        The ordering list is FOUR and the table is THREE, and that difference
        is real rather than an off-by-one: the list starts at the text pass,
        which is not a *later* slot.
        """
        from rheplicant.config.inflight import AXIS_CHECKS, BUILT_CHECKS
        from rheplicant.config.postflight import CHECKS as PRICED
        from rheplicant.config.preflight import CHECKS

        passes = (CHECKS, AXIS_CHECKS, BUILT_CHECKS, PRICED)
        body = _block(_page(self.PAGE), self.SLOTS)
        heading = re.search(r"## The (\w+) later slots", self.SLOTS)
        assert heading, "the slots heading no longer counts its slots"
        word = heading.group(1).lower()
        assert word in self._WORDS, f"unknown number word {word!r}"
        assert self._WORDS[word] == len(passes) - 1, (
            f"the heading says {word} later slots; there are "
            f"{len(passes) - 1} registries after the text pass."
        )
        rows = _rows(body)
        assert len(rows) == len(passes) - 1, (
            f"the slots table has {len(rows)} rows for {len(passes) - 1} "
            "later passes."
        )
        ordering = re.findall(r"^- (\w[\w -]*?) pass —", body, re.MULTILINE)
        assert len(ordering) == len(passes), (
            f"the ordering list names {ordering} for {len(passes)} passes. "
            "It starts at the TEXT pass, so it is one longer than the table."
        )

    def test_the_checks_section_gates_rather_than_records(self):
        """``docs/config-inference.md``'s ``## Checks``, which had no guard.

        It ended *"The section is grammar plus record in 2B; its gating is
        Plan 3's validate"* and survived five plans saying so, because nothing
        read it. What is asserted now is what a reader needs and what actually
        drifts: the three names, the four modes, and each check's real
        default -- taken from ``gates(None)``, not from a sentence.

        The stale sentence itself is pinned by absence as well. That is
        normally a weak shape, and it is defensible exactly here: this is the
        one sentence in this layer's prose with a five-plan record of being
        restored by a reader "fixing" the paragraph around it.
        """
        from rheplicant.config import gates
        from rheplicant.config.gating import CHECK_NAMES, MODES

        body = _block(_page("config-inference.md"), "## Checks")
        assert "its gating is Plan 3's validate" not in body, (
            "config-inference.md's Checks section is claiming again that the "
            "gating is future work. It is not: gating.py decides it and the "
            "post-flight pass runs it."
        )
        modes = re.search(r"`mode: ([^`]+)`", body)
        assert modes, "the Checks section no longer spells the mode words"
        assert [word.strip() for word in modes.group(1).split("|")] == list(MODES), (
            f"the page spells the modes {modes.group(1)!r}; gating.MODES is "
            f"{list(MODES)}."
        )
        rows = _rows(body)
        assert rows, "config-inference.md's Checks table stopped parsing"
        table = {row[0].strip("`"): row[-1].strip("`") for row in rows}
        assert set(table) == set(CHECK_NAMES), (
            f"the page tabulates {sorted(table)}; gating.CHECK_NAMES is "
            f"{sorted(CHECK_NAMES)}."
        )
        applied = gates(None)
        for name, default in table.items():
            assert default == applied[name].state, (
                f"config-inference.md says {name} defaults to {default!r}; "
                f"gates(None) applies {applied[name].state!r}."
            )


#: ``docs/config-validation.md``'s check id -> [(every phrase the page's own
#: bullet must carry for this way out, the patch applying it)]. Both halves
#: are load-bearing: the phrases alone would let the patch drift from the
#: advice, and the patch alone would let the advice be reworded into
#: something that does not work. Two entries for A27 because the bullet
#: offers two ways out.
#:
#: The phrases are a TUPLE and not one string, and that is measured rather
#: than tidy: with only ``"kind: conjugate.gls"`` pinned, deleting the page's
#: "drop ``width:`` with it" clause SURVIVED, because the patch went on
#: dropping the key the page no longer mentioned.
#:
#: Module scope rather than a class attribute: a ``parametrize`` expression
#: in a class body cannot see the class namespace from inside a nested
#: comprehension, and the failure is a collection-time ``NameError`` that
#: takes the whole module down.
_PAGE_FIXES = {
    "A27": [
        # `width:` goes with it: measured, changing the kind alone swaps A27
        # for A1 -- `width` is not a `conjugate.gls` option and Task 3's run
        # sweep says so. The page names that, so the patch does it.
        (("kind: conjugate.gls", "drop `width:`"),
         lambda doc: {**doc, "runs": [{k: v for k, v in doc["runs"][0].items()
                                       if k != "width"}
                                      | {"kind": "conjugate.gls"}]}),
        # THREE edits, and the page says so since Plan 3B. Writing only the
        # kind leaves `include_logdet:` on a kind whose key set does not carry
        # it (A49, and `build_noise` before A49 was hoisted), and leaves
        # `source:` unwritten, which `radiometer_frozen` has no default for.
        (("inference.noise.kind: radiometer_frozen", "keeps this exit",
          "drop it (check A49)", "source: observed"),
         lambda doc: {**doc, "inference": {
             **doc["inference"],
             "noise": {**{k: v for k, v in doc["inference"]["noise"].items()
                          if k != "include_logdet"},
                       "kind": "radiometer_frozen",
                       "source": "observed"}}}),
    ],
    "A30": [
        (("inference.twin: {without: [noise]}",),
         lambda doc: {**doc, "inference": {**doc["inference"],
                                           "twin": {"without": ["noise"]}}}),
    ],
    "A33": [
        # `init:` goes with the transform, and that is the half the page left
        # out until Plan 3B: `unit_mean_bandpass` takes the `(n_freq - 1,)`
        # free coordinates and RETURNS `(n_freq,)`, so `{ones: [n_freq]}`
        # produces a nine-channel bandpass for eight channels of data. The
        # pre-flight pass cannot see it -- it is a shape -- so only
        # `test_the_three_fixes_together_leave_a_document_that_LOADS` does.
        (("transform: unit_mean_bandpass", "`b`, whose free vector",
          "{ones: [7]}"),
         lambda doc: {**doc, "inference": {
             **doc["inference"],
             "parameters": {
                 **doc["inference"]["parameters"],
                 "b": {**doc["inference"]["parameters"]["b"],
                       "init": {"ones": [7]},
                       "transform": "unit_mean_bandpass"}}}}),
    ],
    "C18": [
        # Only the `model.noise.type:` escape: the page's OTHER escape
        # (`inference.noise.kind: homoscedastic`) also clears A27, which
        # would fail the "and only it" assertion below -- measured, it
        # leaves ['A30', 'A33', 'A49'], not ['A27', 'A30', 'A33'].
        (("model.noise.type: RadiometerNoiseOperator",),
         lambda doc: {**doc, "model": {
             **doc["model"],
             "noise": {"type": "RadiometerNoiseOperator",
                       "channel_width": {"value": 1.0, "unit": "MHz"},
                       "integration_time": {"value": 2.0, "unit": "s"}}}}),
    ],
}


#: Check ids a page REMEDY earns because the page's own advice is wrong, so
#: that the assertion below can still say "and nothing else" about every other
#: id.  **Each entry is a defect in ``docs/config-validation.md``, not in the
#: pass.**
#:
#: **EMPTY, and deliberately kept rather than deleted.**  Task 6 shipped it
#: holding ``A49``: the page's second A27 remedy told a reader to write
#: ``inference.noise.kind: radiometer_frozen`` and keep the rest, which left
#: ``include_logdet:`` on a kind that does not take it.  Plan 3B's Task 9
#: corrected the page -- the bullet now names all three edits -- so the
#: exemption has nothing left to exempt and the "and nothing else" assertion
#: below is unconditional again.
#:
#: The constant stays because the NEXT page defect wants a visible one-line
#: home, and because an empty frozenset subtracted from a set changes nothing:
#: measured, the assertion is identical with and without it, so keeping it
#: costs no strength.  **Subtracted rather than intersected with the page's
#: own ids**: ``listed`` does not contain A1, and A1 is exactly what the
#: ``width:`` clause in the test's docstring exists to catch, so an
#: intersection would delete that guard along with any exemption.
_BROKEN_ON_THE_PAGE: frozenset[str] = frozenset()


class TestTheValidationPageDocument:
    """``docs/config-validation.md``'s document is REFUSED here, by the pass.

    The 2B precedent (``TestTheWorkedDocumentOnThePage``) executes a page's
    document because a page that carries one is making a promise. This page's
    promise is the opposite shape -- that the document is wrong four ways,
    that all four come back from one call, and that each fix it offers really
    clears the finding it is offered for -- so the test is the same idea with
    the assertion inverted.

    It costs milliseconds, which is itself the claim: nothing is built. That
    is asserted below rather than left in this docstring, because a sentence
    about cost that no run checks is the class of defect this whole plan is
    about.
    """

    HEADING = "## A document that is wrong four ways"

    #: Every place the page states the SIZE of what its document earns, as
    #: ``(pattern, what it is)``. Anchored per sentence rather than a
    #: whole-word sweep for number words, because this section legitimately
    #: says "one call", "one template state", "one sigma up front" and "one
    #: exactly null direction" -- a blanket sweep would read four of those as
    #: counts and fail on a page that is entirely correct.
    #:
    #: The heading is scanned too, and it is the one that most needed it: it
    #: is also this class's locator, so a page corrected to four findings and
    #: a heading left saying "three" keeps every other assertion here passing.
    _COUNT_CLAIMS = (
        (r"wrong (\w+) ways", "the heading"),
        (r"and all (\w+) come back from one call", "the opening sentence"),
        (r"^(\w+) findings,", "the bullet list's lead-in"),
    )

    def _document(self):
        return _page_document(self.HEADING, "config-validation.md")

    def _body(self):
        return _section(_page("config-validation.md"), self.HEADING)

    def _ids_on_the_page(self):
        """The check ids the page's own bullet list names, read OFF THE PAGE.

        Derived rather than written, so the page and the assertion cannot
        disagree: a bullet added or removed moves this set.
        """
        return set(self._ordered_ids_on_the_page())

    def _ordered_ids_on_the_page(self):
        """The same ids IN THE ORDER THE PAGE WRITES THEM.

        The set-valued assertions below cannot see an order, and the page
        claims one ("in registry order") -- a claim nothing checked.

        ``[ABC]\\d+`` and not ``A\\d+``: C18 joined this page's document in
        Plan 3C Task 2, and the widened class is safe rather than merely
        convenient -- the three assertions this feeds are all EQUALITIES
        (``_ids_on_the_page() == preflight().checks()``, this list against
        the pass's own order, and ``listed - {check}``), so widening it can
        only ADD a subject it did not see before, never make one vacuous.
        """
        return re.findall(r"^- \*\*([ABC]\d+)\*\*", self._body(), re.MULTILINE)

    def test_the_page_lists_ids_at_all(self):
        """A regex that stopped matching would make the tests below vacuous."""
        assert len(self._ids_on_the_page()) >= 3

    def test_every_count_word_on_the_page_is_the_number_of_findings(self):
        """The bullet LIST is guarded both ways; the words beside it were not.

        Measured by review: registering one extra check that fires only on
        this document and adding its bullet in registry order left the
        heading ("wrong **three** ways"), the opening sentence ("all
        **three** come back from one call") and the lead-in ("**Three**
        findings") all stale, and the module exited 0. Every equality here is
        over a SET or a LIST, and no set knows its own size in words.

        ``len(report.findings)`` and not ``len(report.checks())``: the page
        counts findings, and a check that fired twice would make the two
        differ -- which is precisely the case a reader would want the page to
        be honest about.
        """
        from rheplicant.config import preflight

        report = preflight(self._document())
        text = f"{self.HEADING}\n{self._body()}"
        words = TestThePagesSayWhatTheLayerDoes._NUMBER_WORDS
        for pattern, what in self._COUNT_CLAIMS:
            found = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            assert found, (
                f"config-validation.md no longer counts its findings in "
                f"{what}, in the {pattern!r} form this guard reads. Reword the "
                "pattern or the page, but do not leave the count unread."
            )
            word = found.group(1).lower()
            assert word in words, f"{what}: unknown number word {word!r}"
            assert words[word] == len(report.findings), (
                f"{what} says {word} and the document earns "
                f"{len(report.findings)} findings "
                f"({[f.check for f in report.findings]})."
            )

    def test_the_kinds_a30_lets_keep_the_node_are_the_packages_own(self):
        """"`kind: forward` keeps the node" was true and INCOMPLETE.

        ``mmodes`` keeps it too -- §3.2 (e) 2 expressed A30's fitting
        condition as the complement ``{"forward", "mmodes"}`` precisely so
        that a kind added later defaults to *fitting*, and the page named one
        of the two. A reader running ``mmodes`` and reading this bullet is
        told their document is one A30 refuses, which it is not.

        Read off ``_A30_NOT_FITTING`` so the page moves when the complement
        does, and both directions: a kind on the page that A30 does refuse is
        worse than one it omits.
        """
        from rheplicant.config.preflight.model import _A30_NOT_FITTING

        [bullet] = re.findall(r"^- \*\*A30\*\*.*?(?=^- \*\*|\Z)", self._body(),
                              re.MULTILINE | re.DOTALL)
        named = set(re.findall(r"`kind: ([a-z][a-z._]*)`", bullet))
        assert named == set(_A30_NOT_FITTING), (
            f"the A30 bullet says {sorted(named)} keep the node; the package "
            f"exempts {sorted(_A30_NOT_FITTING)}."
        )

    def test_the_page_s_document_earns_exactly_the_findings_it_lists(self):
        """Both directions, and the second is the one that goes stale.

        A finding the page omits is a document the page half-explains; a
        finding the page claims and the document does not earn is advice about
        a document nobody has. Equality is deliberate: if a check landing in a
        later plan starts firing on this document, THE PAGE is what is wrong
        -- add the bullet, do not loosen this.
        """
        from rheplicant.config import preflight

        assert preflight(self._document()).checks() == self._ids_on_the_page()

    def test_the_page_lists_them_in_the_order_the_pass_produces(self):
        """The page says "in registry order" and nothing checked it.

        ``checks()`` is a frozenset and ``_ids_on_the_page`` is a set, so both
        assertions around this one pass with the bullets in any order at all
        -- and the order is a real claim: it is what a reader meets.

        **The reason this page used to give for that order is REFUTED**, and
        the page now gives the measured one. Alphabetical position in
        ``preflight/__init__.py``'s foot block decides nothing on its own: a
        foot-imported module's checks register after everything its own head
        imports transitively register, which is why ``beam_spill`` sorts first
        in that block and lands two-thirds of the way down ``CHECKS``.
        ``fitting`` does still register before ``model``, so A27 precedes A30
        and A33 -- but for a reason no reader could have derived from the
        sentence that used to be here.

        Kills a bullet list re-sorted for readability, and kills a foot import
        reordered without moving the page.
        """
        from rheplicant.config import preflight

        report = preflight(self._document())
        produced = list(dict.fromkeys(f.check for f in report.findings))
        assert produced == self._ordered_ids_on_the_page()

    def test_the_first_refusal_is_the_one_raise_if_refused_hands_back(self):
        """The collect-then-raise contract, on the page's own document.

        A Report that raised on the first finding would return one; this
        asserts that it collected several AND that the ConfigError a caller
        sees is the first of them verbatim, which is what keeps every existing
        ``pytest.raises(ConfigError, match=...)`` green. Index 0 is legitimate
        here precisely because the ORDERING is what is under test.
        """
        from rheplicant.config import ConfigError, preflight

        report = preflight(self._document())
        assert len(report.refusals()) >= 3
        with pytest.raises(ConfigError) as caught:
            report.raise_if_refused()
        assert report.refusals()[0].message in str(caught.value)

    def test_every_where_on_it_is_a_path_into_the_document(self):
        """``where`` is the line to EDIT. A ``src/`` path there is unactionable."""
        from rheplicant.config import preflight

        for finding in preflight(self._document()).findings:
            assert finding.where.split(".")[0].split("[")[0] in {
                "runtime", "observation", "resources", "model", "inference",
                "runs", "variants",
            }, finding.where

    @pytest.mark.parametrize("check", sorted(_PAGE_FIXES))
    def test_the_page_still_offers_the_fix_this_test_applies(self, check):
        """Half of the pair, and the half a patched-document test cannot see.

        Applying a patch that works proves nothing about the SENTENCE the
        reader is given. Task 6's mutation round put eight of nine survivors
        in refusal text; a documentation page is refusal text with a wider
        audience. This kills a bullet reworded to advise something else while
        the test below goes on applying the advice that used to be there.
        """
        bullet = re.search(rf"^- \*\*{check}\*\*.*?(?=^- \*\*|\Z)",
                           self._body(), re.MULTILINE | re.DOTALL)
        assert bullet, f"the page no longer carries a bullet for {check}"
        for phrases, _ in _PAGE_FIXES[check]:
            for phrase in phrases:
                assert phrase in bullet.group(0), (
                    f"{check}'s bullet no longer offers {phrase!r}"
                )

    @pytest.mark.parametrize(
        ("check", "index"),
        [(check, index)
         for check in sorted(_PAGE_FIXES)
         for index in range(len(_PAGE_FIXES[check]))],
    )
    def test_the_fix_the_page_names_clears_the_finding_and_only_it(self, check,
                                                                   index):
        """The other half: advice that does not work is worse than none.

        Each patch applies exactly what the bullet tells the reader to write,
        and the assertion is an EQUALITY against the page's other ids rather
        than ``check not in ...``. The weaker form would pass for advice that
        clears its own finding and silently earns a different one -- which is
        not hypothetical here: measured, changing `kind:` to `conjugate.gls`
        and leaving `width:` behind swaps A27 for A1, because `width` is
        `conjugate.wiener`'s key. The page names dropping it for exactly that
        reason, and this is what keeps the two in step.

        Kills a bullet that names the wrong key, the wrong value or the wrong
        latent: measured, ``transform: unit_mean_bandpass`` on ``g`` instead
        of ``b`` leaves A33 firing. A patch that does LESS than its bullet is
        killed too -- a typo'd key leaves the finding standing.

        **What it does not kill, measured by review:** a patch that does MORE
        than its bullet says. One that also rewrote ``runtime.seed`` exits 0,
        because the extra edit changes no finding. Closing that would mean
        asserting the patch's own shape rather than its effect, and the shape
        is what the phrase test above is for -- so the pair is asymmetric on
        purpose and this is the side it is open on.

        **The one id that used to be exempted, and why it no longer is.**
        Measured while Plan 3B hoisted A49: the page's second A27 remedy --
        *"inference.noise.kind: radiometer_frozen ... keeps the run as
        written"* -- left ``include_logdet: true`` behind on a kind whose key
        set does not carry it, so the document the page told a reader to write
        was refused with *"kind: radiometer_frozen does not take
        ['include_logdet']"*.  That was true before A49 was hoisted too; it
        arrived from ``build_noise`` at P2, where this test could not see it.
        **The PAGE was what was wrong, and Task 9 corrected it** rather than
        widening the exemption; :data:`_BROKEN_ON_THE_PAGE` is now empty.

        **A named subtraction and NOT an intersection with ``listed``**, which
        was this test's first repair and was wrong: ``listed`` is
        ``{A27, A30, A33}`` and **A1 is not in it**, so ``checks() & listed``
        cannot see A1 at all -- and A1 is the regression the ``width:``
        paragraph above names as this test's measured subject.  Verified by
        mutating ``_PAGE_FIXES["A27"][0]`` to keep ``width:``: the intersecting
        form exits 0 and both the original and this one exit 1.  Exempting one
        id keeps "and nothing else" for every other.
        """
        from rheplicant.config import preflight

        phrases, patch = _PAGE_FIXES[check][index]
        document = self._document()
        listed = self._ids_on_the_page()
        assert check in preflight(document).checks(), (
            f"{check} does not fire on the page's document at all, so this "
            "test cannot see whether the fix clears it"
        )
        left = preflight(patch(document)).checks() - _BROKEN_ON_THE_PAGE
        assert left == listed - {check}, (
            f"the page tells a reader to write {phrases!r} to clear {check}; "
            f"what that leaves is {sorted(left)} and the page's other faults "
            f"are {sorted(listed - {check})}"
        )

    @pytest.mark.parametrize("a27", range(len(_PAGE_FIXES["A27"])))
    def test_the_four_fixes_together_leave_a_document_that_LOADS(self, a27):
        """The assertion that was missing, and it found two broken remedies.

        Every test above stops at ``preflight``. A remedy can therefore clear
        its own finding, earn no other, and still leave a document the layer
        refuses one phase later -- and until Plan 3B **two of this page's
        original three A/A/A remedies did exactly that**, invisibly:

        * the A33 remedy named ``transform: unit_mean_bandpass`` and not the
          ``init:`` that goes with it, so the bind produced ``(9,)`` for an
          ``(8,)`` leaf and ``load_document`` refused;
        * the A27 ``radiometer_frozen`` remedy left ``include_logdet:``
          behind (A49 at P-1, ``build_noise`` before that) and never named
          ``source:``, which that kind has no default for.

        C18's remedy joined in Plan 3C Task 2 and is checked here too, on the
        same principle: clearing its own finding in isolation
        (``test_the_fix_the_page_names_clears_the_finding_and_only_it``) says
        nothing about whether the FOUR remedies compose into a document that
        actually loads.

        Parametrized over A27's two ways out, because the page offers two and
        a test that took only the first would have shipped the second broken
        for a second time.

        Costs one build per case on a document with no beam -- measured about
        0.7 s for the ``source: observed`` branch, which pays one forward
        evaluation through ``inference.observed: {from: simulation}``, and
        milliseconds for the other.
        """
        from rheplicant.config import load_document, preflight

        document = self._document()
        for check, index in (("A27", a27), ("A30", 0), ("A33", 0), ("C18", 0)):
            document = _PAGE_FIXES[check][index][1](document)
        remaining = [f.check for f in preflight(document).findings]
        assert remaining == [], (
            f"the four remedies together still leave {remaining}"
        )
        load_document(document)

    def test_the_pass_on_the_pages_document_is_free(self):
        """The page says the pass costs under 0.05 s. This is that sentence.

        ``test_config_preflight.py`` asserts the budget on the fixture
        document; this asserts it on the one the page shows a reader, which is
        the document the claim is made beside. Measured here: about 2 ms, so
        the margin is a factor of twenty-five and this is a guard against
        something being BUILT, not a benchmark.
        """
        import time

        from rheplicant.config import preflight

        document = self._document()
        start = time.perf_counter()
        preflight(document)
        assert time.perf_counter() - start < 0.05


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
