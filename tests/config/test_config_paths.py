"""The path grammar: what it compiles to, and the six things it refuses."""

import equinox as eqx
import jax.numpy as jnp
import pytest

from _rheplicant_bootstrap import path_syntax
from rheplicant.config import ConfigError
from rheplicant.config import paths as public_paths
from rheplicant.config.paths import compile_path, parse_path, resolve_path_on
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline


class Toy(AbstractOperator):
    gain: jnp.ndarray
    label: str = eqx.field(static=True, default="t")

    def __call__(self, state):
        return state


class TracedTag(AbstractOperator):
    """A field that is a genuine traced pytree leaf but not an array.

    Unlike ``Toy.label``, ``tag`` carries no ``eqx.field(static=True)``: it is
    a real leaf JAX will visit and tag with its own path. It exists to
    exercise the "reaches a leaf, but not an array" refusal, which is a
    different failure than landing on static configuration -- deleting that
    branch does not fail any other test in this file, which is exactly why it
    needs one of its own.
    """

    gain: jnp.ndarray
    tag: str

    def __call__(self, state):
        return state


class UnhashableStatic(AbstractOperator):
    """A static field whose value is a tuple holding something unhashable.

    Exists only to exercise the guard around ``key_path in leaves``: without
    it, resolving a path onto this field raises a bare ``TypeError`` instead
    of the module's own ``ConfigError``.
    """

    gain: jnp.ndarray
    weird: tuple = eqx.field(static=True, default=(1, [2, 3]))

    def __call__(self, state):
        return state


class StaticShapes(AbstractOperator):
    """Static fields shaped like the ones the tagged-subtree discriminator
    must not mistake for an operator with more structure below it.

    ``None`` and ``()`` both flatten to ZERO leaves, which defeated the
    earlier "is this a single bare leaf" check (it required exactly one);
    a multi-element static tuple flattens to leaves that are not key-path
    objects, which the discriminator must also recognise as "not a tagged
    subtree". All three are genuine static configuration, not an unfinished
    walk. Modelled on shipped fields: ``MomentRFIFlaggingOperator.kernel_shapes``
    (``static=True, default=()``) and several ``DriftScanProjector`` fields
    (``static=True, default=None``).
    """

    gain: jnp.ndarray
    opts: tuple = eqx.field(static=True, default=())
    ref: float | None = eqx.field(static=True, default=None)

    def __call__(self, state):
        return state


@pytest.fixture
def twin():
    return Pipeline(Toy(gain=jnp.asarray(1.0)), Toy(gain=jnp.asarray(2.0)),
                    names=["gain", "bandpass"])


class TestParsing:
    def test_the_public_parser_reuses_the_jax_free_step_grammar(self):
        """Catches a bootstrap/public path grammar drifting into two copies."""
        assert public_paths._STEP is path_syntax.PATH_STEP

    def test_public_and_bootstrap_paths_both_refuse_an_internal_newline(self):
        label = "root.a\n.leaf"
        assert path_syntax.is_legal_path(label) is False
        with pytest.raises(ConfigError):
            parse_path(label)

    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("variants.valid.runtime.seed", "variants.valid.runtime.seed"),
            ("variants.unity-gain.runtime.seed", "variants"),
            ("variants.slash/name.model", "variants"),
            ("inference.parameters.d-1.init", "inference.parameters"),
        ],
    )
    def test_longest_legal_prefix_keeps_the_deepest_spellable_path(
        self, label, expected
    ):
        assert path_syntax.longest_legal_prefix(label) == expected

    def test_longest_legal_prefix_visits_each_segment_at_most_once(
        self, monkeypatch
    ):
        """A long suffix after the first bad segment must not cause rescans."""
        real = path_syntax.PATH_STEP
        visited: list[str] = []

        class CountingPattern:
            def fullmatch(self, piece):
                visited.append(piece)
                return real.fullmatch(piece)

        monkeypatch.setattr(path_syntax, "PATH_STEP", CountingPattern())
        legal = ["root", *(f"step{index}" for index in range(256))]
        label = ".".join(
            [*legal, "bad-name", *(f"suffix{index}" for index in range(256))]
        )

        assert path_syntax.longest_legal_prefix(label) == ".".join(legal)
        assert visited == [*legal, "bad-name"]

    def test_a_head_alone(self):
        assert parse_path("gain") == ("gain",)

    def test_a_head_and_one_step(self):
        assert parse_path("gain.gain") == ("gain", "gain")

    def test_several_steps(self):
        assert parse_path("sky.sky_model.maps") == ("sky", "sky_model", "maps")

    def test_an_index_after_an_attribute(self):
        assert parse_path("filters.stages[0]") == ("filters", "stages", 0)

    def test_a_bare_index(self):
        assert parse_path("branches[1]") == ("branches", 1)

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            ".",
            "gain.",
            ".gain",
            "gain..gain",
            "gain[x]",
            "gain[-1]",
            "a[01]",  # leading zero -- only "0" itself or "[1-9][0-9]*" are indices
            "a[١]",  # Arabic-Indic digit one: not an ASCII digit
        ],
    )
    def test_malformed_paths_are_refused(self, bad):
        with pytest.raises(ConfigError):
            parse_path(bad)

    @pytest.mark.parametrize("padded", [" gain", "gain "])
    def test_a_whitespace_padded_path_names_whitespace_in_its_message(self, padded):
        """The whitespace guard can never fire as the SOLE cause of a refusal
        -- the per-segment regex would refuse a padded segment on its own,
        since a leading or trailing space is not a valid identifier
        character (``_STEP.match(" gain")`` already fails without the
        guard). The guard exists only to give a more specific message
        ('padded with whitespace') instead of the generic 'unusable
        segment' one; this pins THAT message, not merely that some
        ConfigError is raised, which the generic malformed-path
        parametrize above already covers for every other case."""
        with pytest.raises(ConfigError) as excinfo:
            parse_path(padded)
        assert "whitespace" in str(excinfo.value)

    def test_a_non_string_path_is_refused(self):
        """A YAML `into:` key written with no value (`into:` alone, nothing
        after the colon) parses to None, not to the empty string.
        `str(None)` would silently coerce it into the path "None" and search
        for a node literally named that, instead of refusing outright."""
        with pytest.raises(ConfigError):
            parse_path(None)


class TestCompilingToASelector:
    def test_it_produces_the_callable_Bind_actually_takes(self, twin):
        """Bind.into holds CALLABLES, not strings (inference/parameters.py:338
        -- `into: tuple[Callable, ...]`), and _resolve_targets INVOKES them
        against a tagged copy. A grammar that synthesised key paths directly
        would not survive Pipeline.__getitem__'s name-to-index translation:
        p['gain'] resolves through self.names.index('gain'), and the string
        'gain' never appears in the resulting path."""
        selector = compile_path("gain.gain")
        assert callable(selector)
        assert selector(twin) is twin["gain"].gain

    def test_the_selector_walks_the_objects_own_accessors(self, twin):
        import jax

        tagged = jax.tree_util.tree_map_with_path(lambda path, _: path, twin)
        path = compile_path("gain.gain")(tagged)
        assert jax.tree_util.keystr(path) == ".stages[0].gain"


class TestResolution:
    def test_it_returns_the_structural_path_and_the_leaf(self, twin):
        resolved = resolve_path_on("gain.gain", twin)
        assert resolved.keystr == ".stages[0].gain"
        assert float(resolved.leaf) == pytest.approx(1.0)
        assert resolved.declared == "gain.gain"

    def test_a_different_head_resolves_to_a_different_stage(self, twin):
        """Kills a mutant that always walks twin[0] regardless of the head:
        "bandpass" is stage 1, not stage 0. Until now, only the radio-gated
        AmbiguousNodeError test (behind importorskip) exercised more than
        one head, so a mutant hard-coding stage 0 would have passed
        whenever rheplicant.radio was unavailable."""
        resolved = resolve_path_on("bandpass.gain", twin)
        assert resolved.keystr == ".stages[1].gain"


class TestTheSixRefusals:
    def test_1_a_path_that_stops_on_an_operator_is_refused(self, twin):
        """The path stops one step short, on the operator subtree itself,
        rather than on a leaf. This used to share one message with the
        genuine-static-field case below and call the operator 'static
        configuration', which mis-diagnosed it; it now gets its own reason
        and names what was actually found."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_path_on("gain", twin)
        message = str(excinfo.value)
        assert "gain" in message  # the declared path
        assert "Toy" in message  # what was found: the operator's type
        assert "not a leaf" in message

    def test_a_static_field_is_refused_with_its_own_reason(self, twin):
        """A static field is not a pytree leaf at all -- measured on this
        fixture: `Toy.label` is declared `eqx.field(static=True, ...)`, so on
        a tagged twin `tagged['gain'].label` still holds the VALUE 't', not a
        path, because tree_map_with_path never visits a static field. So the
        walk cannot reach it, and the reason is not 'wrong path' but
        'inference cannot touch a treedef entry'."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_path_on("gain.label", twin)
        message = str(excinfo.value)
        assert "label" in message
        assert "static" in message
        assert "'t'" in message  # what was found: the field's own real value

    def test_static_fields_holding_none_empty_or_multi_element_are_all_STATIC(self):
        """The Important fix: the tagged-subtree discriminator must route
        EVERY static field to the static-configuration message regardless
        of what raw value it holds, not just ones that happen to look like
        a single bare leaf. The earlier check flattened the value and
        required exactly one leaf identical to itself; ``None`` and ``()``
        flatten to ZERO leaves, so both were misrouted to 'stops on an
        operator, go one step deeper' even though there is nothing to step
        into. Demonstrated on shipped MomentRFIFlaggingOperator.kernel_shapes
        (static=True, default=()) and DriftScanProjector's several
        static=True, default=None fields."""
        twin = Pipeline(StaticShapes(gain=jnp.asarray(1.0)), names=["op"])
        for field in ("opts", "ref"):
            with pytest.raises(ConfigError) as excinfo:
                resolve_path_on(f"op.{field}", twin)
            message = str(excinfo.value)
            assert "static" in message, f"{field}: {message}"
            assert "go one step deeper" not in message, f"{field}: {message}"

    def test_a_traced_non_array_leaf_is_refused_with_house_style_message(self):
        """The third resolve_path_on refusal: the walk reaches a REAL pytree
        leaf (TracedTag.tag has no static=True, so JAX does visit and tag
        it), but that leaf's value is not an array. Deleting this branch
        passes the rest of the suite, which is why it needs its own test;
        the message follows house style -- declared path, structural keystr,
        what was found, what to do."""
        twin = Pipeline(TracedTag(gain=jnp.asarray(1.0), tag="x"), names=["gain"])
        with pytest.raises(ConfigError) as excinfo:
            resolve_path_on("gain.tag", twin)
        message = str(excinfo.value)
        assert "gain.tag" in message  # the declared path
        assert ".stages[0].tag" in message  # the structural keystr
        assert "str" in message  # what was found
        assert "jnp.ndarray" in message or "array" in message  # what to do

    def test_an_unhashable_tagged_value_does_not_crash_membership_check(self):
        """Guard: a static field whose value happens to be a tuple containing
        something unhashable (e.g. a list) must not blow up the `in leaves`
        membership test with a raw TypeError -- the walk still cannot use it
        as a leaf, and should refuse as one, not crash."""
        twin = Pipeline(UnhashableStatic(gain=jnp.asarray(1.0)), names=["op"])
        with pytest.raises(ConfigError):
            resolve_path_on("op.weird", twin)

    def test_2_an_ambiguous_many_node_reproduces_the_packages_wording(self):
        """Refusal 2. AmbiguousNodeError is raised by Assembly.__getitem__
        (core/graph.py:458) DURING the walk, and _resolve_targets wraps it in
        its own template -- so the composite is what a user sees today, and it
        is the composite this layer reproduces."""
        pytest.importorskip("rheplicant.radio")
        from rheplicant.core.errors import AmbiguousNodeError
        from rheplicant.core.graph import At
        from rheplicant.radio import assemble

        assembly = assemble(
            At("foregrounds", Toy(gain=jnp.asarray(1.0))),
            At("foregrounds", Toy(gain=jnp.asarray(2.0))),
        )
        with pytest.raises((ConfigError, AmbiguousNodeError)) as excinfo:
            resolve_path_on("foregrounds.gain", assembly)
        message = str(excinfo.value)
        assert "foregrounds_1" in message  # the instance ids, as the package lists them
        assert "addresses none of them" in message

    def test_3_a_path_into_an_aliased_node_is_refused(self):
        """Refusal 3. Assembly.aliased is empty for every shipped graph and is
        the documented hazard for user-defined ones. The guard is path-based,
        not spelling-based: _aliased_leaf_paths returns every leaf path an
        aliased node owns, so naming the second copy by index is refused
        identically. Binding one branch and leaving the others would answer as
        if the latent were frozen everywhere but that branch -- finite,
        correctly shaped, wrong."""
        from rheplicant.config.paths import refuse_aliased_target

        # A stub, not a real forked Assembly: Assembly.aliased is empty for
        # every shipped graph, and building a real graph with an aliased node
        # is not cheap. refuse_aliased_target only ever reads `.aliased`, so a
        # stub carrying just that one attribute is a faithful, cheap double.
        class _Forked:
            aliased = ("x",)

        with pytest.raises(ConfigError) as excinfo:
            refuse_aliased_target("x.value", _Forked())
        message = str(excinfo.value)
        assert "x.value" in message  # the declared path
        assert "'x'" in message  # the node
        assert "several" in message  # why: it reaches the sink by several paths

    def test_a_path_into_an_unaliased_node_is_accepted(self, twin):
        from rheplicant.config.paths import refuse_aliased_target

        refuse_aliased_target("gain.gain", twin)

    def test_refuse_aliased_target_accepts_an_explicitly_empty_aliased(self):
        """Pins that a real, present `.aliased = ()` is accepted the same way
        an object with no `.aliased` attribute at all is (which getattr's
        default covers, exercised by the fixture-based accept test above).
        This does NOT pin the `if not aliased:` short-circuit itself:
        deleting that line is an equivalent mutant here, since `head in ()`
        is False regardless of whether the short-circuit runs first -- it
        only pins that presence-vs-absence of the attribute does not change
        the outcome."""
        from rheplicant.config.paths import refuse_aliased_target

        class _Unforked:
            aliased = ()

        refuse_aliased_target("x.value", _Unforked())

    def test_4_two_paths_reaching_one_leaf_are_refused_together(self, twin):
        """Refusal 4. The package refuses this in ParameterSpace.validate and
        names the leaf in keystr form; here both DECLARED spellings are named
        too, because two different strings reaching one leaf is the case where
        keystr alone tells the reader nothing. The two spellings must be
        genuinely different strings ("gain.gain" and "[0].gain", both proven
        to resolve to .stages[0].gain) -- passing the same string twice would
        let a mutant that prints the current `path` twice, and never reads
        `earlier`, pass unnoticed."""
        from rheplicant.config.paths import refuse_duplicate_targets

        with pytest.raises(ConfigError) as excinfo:
            refuse_duplicate_targets(["gain.gain", "[0].gain"], twin)
        message = str(excinfo.value)
        assert ".stages[0].gain" in message  # the structural path
        assert "gain.gain" in message  # the first spelling
        assert "[0].gain" in message  # the second, DIFFERENT spelling

    def test_6_a_region_key_that_is_not_the_last_covered_node_is_refused(self):
        """Refusal 6, check A47. At((a, b, c), op) is addressed by its LAST
        covered node id (core/graph.py:131-134, confirmed at fold.py:409-414
        and graph.py:1071); a config key naming any other node resolves to
        nothing, and the failure is a plain KeyError rather than the message
        the schema promised."""
        from rheplicant.config.paths import refuse_misaddressed_region

        with pytest.raises(ConfigError) as excinfo:
            refuse_misaddressed_region("my_stage", ["noise_wave", "cw_tone", "bandpass"])
        message = str(excinfo.value)
        assert "my_stage" in message
        assert "bandpass" in message  # what the key must be
        assert "LAST" in message or "last" in message

    def test_a_region_key_equal_to_the_last_node_is_accepted(self):
        from rheplicant.config.paths import refuse_misaddressed_region

        refuse_misaddressed_region("bandpass", ["noise_wave", "cw_tone", "bandpass"])

    def test_a_single_node_region_is_accepted_under_any_key(self):
        """Pins the `len(nodes) < 2` short-circuit: a one-node 'region' is not
        a region in the sense this check cares about (there is no other
        covered node a key could be confused with), so it is accepted no
        matter what the key is -- even one that does not name the node at
        all. Without this test, a mutant that deletes the short-circuit
        would only be caught if some OTHER test happened to exercise a
        single-node region, which none of the others do."""
        from rheplicant.config.paths import refuse_misaddressed_region

        refuse_misaddressed_region("a", ["a"])
        refuse_misaddressed_region("a", ["b"])


class TestTheRefusalNamesBothSpellings:
    def test_the_declared_path_and_the_structural_one_are_both_quoted(self, twin):
        from rheplicant.config.paths import refuse_duplicate_targets

        with pytest.raises(ConfigError) as excinfo:
            refuse_duplicate_targets(["gain.gain", "[0].gain"], twin)
        message = str(excinfo.value)
        assert "as written" in message or "declared" in message
        assert "as the twin sees it" in message or "structural" in message
        assert "gain.gain" in message  # the first spelling, verbatim
        assert "[0].gain" in message  # the second spelling, verbatim -- and different


class TestAgainstTheRealMachinery:
    def test_a_compiled_path_is_accepted_by_ParameterSpace_validate(self):
        """Step 5 of the plan, promoted to a test: the point of compile_path
        is that ParameterSpace accepts what it returns, not merely that this
        module's own tests do. Proved end to end against the real
        AntennaLossOperator and the real Bind/Latent/ParameterSpace machinery
        -- a script that runs once proves nothing next month."""
        pytest.importorskip("rheplicant.radio")
        from rheplicant.inference import Bind, Latent, ParameterSpace
        from rheplicant.radio.instrument.antenna_loss import AntennaLossOperator

        twin = Pipeline(
            AntennaLossOperator(efficiency=jnp.asarray(0.97), t_physical=jnp.asarray(293.0)),
            names=["antenna_loss"],
        )
        space = ParameterSpace(
            latents=[Latent("eta", init=jnp.asarray(0.97))],
            bindings=[Bind("eta", into=compile_path("antenna_loss.efficiency"))],
        )
        space.validate(twin)


# --- Plan 3B Task 9: the two schema rows with nothing left to check ---


#: A projector with no beam and no file, and a sky model with no file, so the
#: two reference sites in :class:`TestB7OneResourceIsOneObject` can coexist in
#: one loadable document.  ``engine: matrix`` returns before any beam is read
#: (``kinds/projectors.py``); the shape is written ``["n_time", 12]`` rather
#: than ``[16, 12]`` because a literal extent equal to ``n_time`` earns A41 and
#: the finding would be noise in a test that is not about it.
_SHARED_PROJECTOR = {"engine": "matrix", "matrix": {"zeros": ["n_time", 12]},
                     "provenance": {"built_by": "the test suite",
                                    "lat_deg": 0.0}}
_SHARED_SKY_MODEL = {"kind": "uniform", "n_pix": 12,
                     "amplitude": {"value": 200.0, "unit": "K"}}


@pytest.fixture(scope="module")
def shared_projector_run():
    """One ``load_document`` whose ``filters`` and ``observed_astro_sky``
    reference the SAME projector resource.

    Module-scoped because both classes below read the same built objects and
    a per-test build would pay the cold JAX trace several times.  Measured:
    about 1 s cold, 50 ms warm, and the document earns no finding at all.
    """
    from rheplicant.config import load_document
    from tests.config.preflight_helpers import (
        BASE_MODEL,
        BASE_OBSERVATION,
        preflight_document,
    )

    document = preflight_document(
        observation={**BASE_OBSERVATION,
                     "pointing": {"mode": "baked",
                                  "provenance": {"built_by": "a test"}}},
        resources={"projectors": {"p": _SHARED_PROJECTOR},
                   "sky_models": {"s": _SHARED_SKY_MODEL}},
        model={**BASE_MODEL,
               "observed_astro_sky": {
                   "sky_model": {"ref": "resources.sky_models.s"},
                   "projector": {"ref": "resources.projectors.p"}},
               "filters": [{"type": "SkySpaceFilter",
                            "projector": {"ref": "resources.projectors.p"},
                            "regularization": {"value": 1e-3,
                                               "unit": "dimensionless"}}]})
    return load_document(document)


def _out_degrees(edges) -> dict[str, int]:
    """How many edges leave each node, over a plain ``(source, target)`` list.

    A free function and not a method, so that
    ``TestB2TheShippedGraphIsATree.test_the_tree_reading_is_capable_of_failing``
    can hand it a FORKED edge list and watch it say so. A checker that is only
    ever run on the one graph it passes on is a checker nobody has seen fail.
    """
    degrees: dict[str, int] = {}
    for source, _target in edges:
        degrees[source] = degrees.get(source, 0) + 1
    return degrees


class TestB2TheShippedGraphIsATree:
    """Schema B2: "a path into an aliased node is refused".

    ``Assembly.aliased`` is empty for every shipped radio graph, which is why
    B2 is a **pinning test** in Plan 3B rather than a check --
    :meth:`TestTheSixRefusals.test_3_a_path_into_an_aliased_node_is_refused`
    above already says so in its own docstring and stubs the tuple, because
    no document can produce a non-empty one.

    **The half that is weak, measured rather than assumed -- and the plan's
    own statement of it is too strong.** Plan 3B's §0.3 E.5 says patching
    ``RADIO_GRAPH`` to the fork ``('uniform_sky', 't_ant_sum')`` leaves
    ``assemble(...).aliased == ()`` **on a non-tree graph**, because a
    document lights only 9 of the 33 nodes. Re-measured here with that exact
    fork applied to the real graph: this class's document earns
    ``aliased == ('uniform_sky',)`` and the pin below goes RED. So the
    weakness is **document-dependent**, not absolute: ``aliased == ()`` sees
    a fork the document happens to light and is blind to every fork it does
    not, which is a property of the fixture rather than of the graph. The
    assertion that fires the day someone adds an edge, whatever any document
    lights, is the STRUCTURAL one over the graph's own edge list -- and that
    is what this class leads with.

    Three legs, because the theorem needs all three:

    1. the shipped graph is a tree -- every node's out-degree is at most one,
       and ``|edges| == |nodes| - 1``;
    2. the config layer never chooses a graph: ``compose.py::_graph()`` takes
       no argument and hardcodes ``return RADIO_GRAPH``, which is what makes
       the public ``register_graph``/``get_graph`` pair harmless here;
    3. ``aliased`` really is ``()`` on a built twin -- kept, and labelled, as
       the weak half.
    """

    def test_the_shipped_graph_is_a_tree(self):
        """Out-degree <= 1 and one fewer edge than nodes: no node's
        contribution reaches the sink by two paths, so the fold can never
        embed one at two positions and ``aliased`` cannot be non-empty.

        Measured at this commit: 33 nodes, 32 edges, max out-degree 1, sink
        ``filters``, graph name ``single-antenna``. The counts are asserted
        as a RELATION and not as the two numbers, so adding a node with one
        edge -- which keeps the theorem -- is not a red test, while adding a
        second edge out of any node is.
        """
        from rheplicant.radio.graph import RADIO_GRAPH

        degrees = _out_degrees(RADIO_GRAPH.edges)
        forks = {node: out for node, out in degrees.items() if out > 1}
        assert forks == {}, (
            f"RADIO_GRAPH is no longer a tree: {forks} reach the sink by "
            "more than one path. Assembly.aliased can now be non-empty, and "
            "schema B2 -- a path into an aliased node -- stops being a "
            "structural theorem and becomes a check somebody has to write."
        )
        assert len(RADIO_GRAPH.edges) == len(RADIO_GRAPH.nodes) - 1, (
            f"{len(RADIO_GRAPH.nodes)} nodes and "
            f"{len(RADIO_GRAPH.edges)} edges: a tree with one sink has "
            "exactly one fewer edge than nodes, so this graph is either "
            "disconnected or has a cycle."
        )

    def test_the_tree_reading_is_capable_of_failing(self):
        """Anti-vacuity for the leg above, and it is not decoration.

        The whole reason B2 is a pinning test is that the OBVIOUS pin --
        ``aliased == ()`` -- passes on a graph that is not a tree. A tree
        check that had the same property would be the same defect one level
        up, so the checker is handed a fork built from the real edge list and
        must report it.
        """
        from rheplicant.radio.graph import RADIO_GRAPH

        forked = [*RADIO_GRAPH.edges, ("uniform_sky", "t_ant_sum")]
        assert _out_degrees(forked)["uniform_sky"] == 2

    def test_the_config_layer_never_chooses_a_graph(self):
        """``compose.py::_graph()`` hardcodes ``return RADIO_GRAPH``.

        ``core/graph.py`` publishes ``register_graph`` and ``get_graph``, so
        a second graph CAN exist in a process. What makes that harmless to
        every check keyed on ``RADIO_GRAPH`` -- B2 here, and A2/A3/A4 in
        ``preflight/model.py`` -- is that no document-reachable route selects
        one: the config layer's single accessor takes no parameter, and
        ``get_graph`` is called nowhere under ``config/`` outside two
        docstring citations.

        Asserted over ``ast`` rather than over the source text, so that a
        citation in a comment cannot make this red and a real call cannot
        hide behind formatting.

        **Both spellings, and that is measured.** A first version matched only
        ``ast.Name`` -- a bare ``get_graph(...)`` -- and review defeated it
        with one line: ``_core_graph.get_graph(RADIO_GRAPH.name)``, an
        ``ast.Attribute`` call that really does resolve through
        ``register_graph``'s mutable table (``get_graph("single-antenna") is
        RADIO_GRAPH`` is True) and **survived the whole of ``tests/config``**.
        The ``id``-or-``attr`` idiom below is the same one
        ``test_config_surface.py``'s walker floor uses, for the same reason.

        **Why leg 3 carries weight.** It is what makes leg 1 sufficient: if a
        document-reachable route could select a graph, out-degree <= 1 over
        ``RADIO_GRAPH`` would stop implying ``aliased == ()`` and B2 would
        stop being a theorem, which is this class's whole conclusion.
        """
        import ast
        import inspect
        from pathlib import Path

        from rheplicant.config.sections import compose
        from rheplicant.radio.graph import RADIO_GRAPH

        assert compose._graph() is RADIO_GRAPH
        assert not inspect.signature(compose._graph).parameters, (
            "compose._graph() grew a parameter; a document that can choose "
            "its graph makes every RADIO_GRAPH-keyed check conditional."
        )
        config = Path(compose.__file__).parent.parent
        modules = sorted(config.rglob("*.py"))
        # Anti-vacuity, the line this test's sibling floor in
        # `test_config_surface.py` already has: if `config` ever stops being
        # the package directory, `called == []` passes for the wrong reason
        # and this guard reports "no route selects a graph" about nothing.
        assert len(modules) > 20, (
            f"the walk found only {len(modules)} modules under {config}; it "
            "is not looking at config/ and every verdict below is vacuous"
        )
        called: list[str] = []
        for path in modules:
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.Call):
                    continue
                # `id` OR `attr`: `get_graph(...)` and `mod.get_graph(...)`
                # are the same call and only the first is an `ast.Name`.
                name = getattr(node.func, "id", None) or getattr(
                    node.func, "attr", None)
                if name == "get_graph":
                    called.append(str(path.relative_to(config)))
        assert called == [], f"config/ now selects a graph in {called}"

    def test_aliased_is_empty_on_a_built_twin_and_that_half_is_weak(
            self, shared_projector_run):
        """The half the plan named and then measured to be vacuous.

        Kept because it is the property ``refuse_aliased_target`` reads, and
        labelled because a reader finding only this test would conclude the
        tree property was checked. It is not: it is ``()`` on every fork this
        document does not light, and that is a fact about the document.
        Asserted on a real ``Assembly`` off ``load_document`` -- the module's
        ``twin`` fixture is a ``Pipeline``, which has no ``aliased`` at all.
        """
        assert shared_projector_run.twin.aliased == ()


class TestB7OneResourceIsOneObject:
    """Schema B7: two references to one resource name reach one object.

    B7 is "identity by construction" -- ``build_resources`` memoises by
    dotted name and ``sections/model.py`` resolves an object field to the
    built object rather than to a copy -- so Plan 3B ships a pinning test
    rather than a check. It lives beside the path grammar because that is
    the mechanism: ``{ref: resources.projectors.p}`` is a path, an inline
    projector at an object field is refused, and the identity is what the
    reference buys.

    **``is`` and never ``==``, measured.** ``MatrixProjector(m) ==
    MatrixProjector(m)`` and ``a == copy.deepcopy(a)`` are both truthy while
    ``is`` is False, so an equality here would pass on the very failure B7
    exists to exclude: two objects built twice from one spec. The weight
    used by ``filters`` and the sky average taken through
    ``observed_astro_sky`` have to come off ONE object, and only ``is`` says
    that.

    **No beam.** See :data:`_SHARED_PROJECTOR` and
    :func:`shared_projector_run` for the document and why it is shaped the
    way it is.
    """

    def test_both_references_reach_the_same_object(self,
                                                   shared_projector_run):
        """B7, stated: ``filters[].projector`` **is**
        ``observed_astro_sky.projector`` when they name one resource.

        And both are the object ``build_resources`` put in the table -- the
        three-way identity, because two nodes sharing a private copy would
        satisfy a two-way one and still not be the resource the rest of the
        layer resolves ``{ref:}`` to.
        """
        run = shared_projector_run
        built = run.resources.resources["resources.projectors.p"]
        assert run.twin["filters"].projector is built
        assert run.twin["observed_astro_sky"].projector is built

    def test_equality_would_not_have_said_that(self, shared_projector_run):
        """Why B7's pin is ``is``. §0.3 E.5 ruling 4, re-measured here.

        A deep copy of the one projector compares EQUAL to it and is a
        different object. An ``==`` pin above would therefore be green on a
        build that constructed the projector twice -- which is exactly the
        state B7 exists to exclude, because a weight and a sky average taken
        off two objects agree until one of them is rebuilt.
        """
        import copy

        run = shared_projector_run
        built = run.resources.resources["resources.projectors.p"]
        other = copy.deepcopy(built)
        assert other == built
        assert other is not built
