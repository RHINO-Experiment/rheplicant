"""The path grammar: what it compiles to, and the six things it refuses."""

import equinox as eqx
import jax.numpy as jnp
import pytest

from rheplicant.config import ConfigError
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
