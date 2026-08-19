"""Form 8: python:, its destination-addressed arguments, and the cost it states."""

import jax.numpy as jnp
import pytest

from _rheplicant_bootstrap.types import DestinationDescriptor
from rheplicant.config import ConfigError
from rheplicant.config import hatch as hatch_module
from rheplicant.config.context import ResolutionContext
from rheplicant.config.values import resolve_value


@pytest.fixture
def context():
    return ResolutionContext(
        freq=jnp.linspace(60e6, 85e6, 8), time=jnp.arange(4.0), dtype="float32"
    )


class TestTheHatch:
    def test_it_imports_and_calls(self, context):
        got = resolve_value(
            {"python": "jax.numpy:zeros", "literal": {"shape": [3]}}, context
        )
        assert got.value.shape == (3,)
        assert got.source == "python"

    def test_an_attribute_with_no_arguments_is_returned_as_is(self, context):
        got = resolve_value({"python": "math:pi"}, context)
        assert float(got.value) == pytest.approx(3.14159, abs=1e-4)

    def test_every_argument_value_is_itself_a_value_node(self, context):
        """v0 said 'forwarded as keyword arguments' and never said whether the
        value grammar applied inside them -- which mattered, because for three
        of the five stress ports python: was the ONLY route to the sky, and an
        unresolved {file: ...} would have been forwarded as a literal dict.

        Catches an `args` forwarded verbatim: fill_value would arrive as the
        dict {'value': 60.0, 'unit': 'MHz'} and jnp.full would refuse it. Also
        catches resolve_value(...) forwarded without .value, which arrives as
        a ResolvedValue tuple.
        """
        got = resolve_value(
            {
                "python": "jax.numpy:full",
                "literal": {"shape": [2]},
                "args": {"fill_value": {"value": 60.0, "unit": "MHz"}},
            },
            context,
        )
        assert float(got.value[0]) == pytest.approx(6e7)

    def test_an_args_value_resolves_against_the_run_s_own_context(self, context):
        """Catches an `args` resolved against a fresh ResolutionContext rather
        than the one being resolved in -- {from_grid: freq} would refuse,
        naming an axis the run does declare. Forwarding the resolved VALUE is
        not enough on its own; it has to be resolved in the caller's scope."""
        got = resolve_value(
            {"python": "jax.numpy:sum", "args": {"a": {"from_grid": "freq"}}}, context
        )
        assert float(got.value) == pytest.approx(float(jnp.sum(context.freq)), rel=1e-6)

    def test_literal_is_forwarded_untouched(self, context):
        """Catches a `literal` resolved through the value grammar: both values
        here are things resolve_value refuses outright -- a bare list is not a
        value node, and a string that is not '<number> <unit>' is not one
        either -- so a resolved literal cannot reach OrderedDict at all."""
        got = resolve_value(
            {
                "python": "collections:OrderedDict",
                "literal": {"a": [1, 2], "b": "left as written"},
            },
            context,
        )
        assert dict(got.value) == {"a": [1, 2], "b": "left as written"}

    def test_every_argument_target_is_validated_before_import(
        self, context, monkeypatch
    ):
        imported = []
        monkeypatch.setattr(
            hatch_module,
            "import_target",
            lambda target: imported.append(target) or (lambda **kwargs: kwargs),
        )
        destination = DestinationDescriptor(
            "model.global_signal.depth",
            "model_field",
            "rheplicant.radio.sky.global_signal.GlobalSignalOperator.depth",
        )
        with pytest.raises(ConfigError, match="unit"):
            resolve_value(
                {
                    "python": "probe.module:factory",
                    "args": {"bad": {"value": 1.0, "unit": 7}},
                },
                context,
                destination=destination,
            )
        assert imported == []


class TestWhetherItCalls:
    """The rule is syntactic: writing args:/literal: calls, writing neither
    hands the attribute over. These pin it against the three ways the line
    tends to be written instead."""

    def test_a_callable_named_alone_is_handed_over_uncalled(self, context):
        """THE separating case. Catches both 'always call' and 'call it if it
        turns out to be callable' -- under either, this returns a float
        timestamp instead of the function. core/operator.py:117 needs this
        one: LambdaOperator.fn is a static Callable[[State], State], and
        delivery.py records that this hatch is the only route to such a field,
        so an inferred call leaves it with no spelling at all."""
        got = resolve_value({"python": "time:monotonic"}, context)
        assert got.value is __import__("time").monotonic
        assert got.source == "python"

    def test_an_empty_args_mapping_is_how_a_zero_argument_call_is_spelled(self, context):
        """Catches the rule written against the truth of the keywords rather
        than the presence of the key -- `if keywords:` or `node.get('args')`
        both read `args: {}` as 'no arguments were written' and hand back the
        function, which leaves a zero-argument call with no spelling."""
        got = resolve_value({"python": "time:monotonic", "args": {}}, context)
        assert isinstance(got.value, float)

    def test_a_constant_asked_to_be_called_is_refused_against_the_target(self, context):
        """Catches 'never call' (which would return pi and pass silently) and
        an unwrapped TypeError escaping the config layer."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"python": "math:pi", "args": {}}, context)
        assert "math:pi" in str(excinfo.value)


class TestTheRefusals:
    def test_a_target_without_a_colon_is_refused_and_the_form_is_shown(self, context):
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"python": "jax.numpy.zeros"}, context)
        message = str(excinfo.value)
        assert "jax.numpy.zeros" in message
        assert "package.module:attribute" in message

    def test_an_unimportable_module_names_the_module_and_the_plugins_key(self, context):
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"python": "no_such_package_xyz:thing"}, context)
        message = str(excinfo.value)
        assert "no_such_package_xyz" in message
        assert "plugins:" in message
        # The INSTRUCTION, not just the word. The sibling message names
        # plugins: too -- to say it would not help -- so 'plugins: appears
        # somewhere' passes on an implementation that has collapsed the two
        # import failures into the wrong one of the two.
        assert "name that package under" in message

    def test_an_absent_parent_package_is_still_a_module_that_is_absent(self, context):
        """ModuleNotFoundError names the module the interpreter failed to
        find, not the one it was asked for: importing 'no_such_pkg_abc.sub'
        reports 'no_such_pkg_abc'. Catches the distinction written as equality
        against that name, which reads an absent parent as 'the module was
        found and its own import failed' -- the opposite of what happened, and
        it withholds the plugins: remedy that would have been the right one."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"python": "no_such_pkg_abc.sub:thing"}, context)
        assert "name that package under" in str(excinfo.value)

    def test_a_module_that_exists_but_fails_to_import_names_the_real_cause(
        self, context, tmp_path, monkeypatch
    ):
        """Catches the two import failures answered with one message. This
        module IS importable by name; its own body is what raised. Sending the
        reader to plugins: here is worse than vague -- naming an installed
        module there cannot fix its missing dependency, so the one key the
        advice points at is the one key that cannot help."""
        (tmp_path / "hatch_probe_broken.py").write_text("import definitely_not_installed_xyz\n")
        monkeypatch.syspath_prepend(str(tmp_path))
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"python": "hatch_probe_broken:thing"}, context)
        message = str(excinfo.value)
        assert "definitely_not_installed_xyz" in message
        assert "was found" in message
        # The word plugins: may appear -- this message uses it to say it would
        # not help. What must NOT appear is the other branch's instruction.
        assert "name that package under" not in message

    def test_a_missing_attribute_names_the_module(self, context):
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"python": "math:no_such_attribute"}, context)
        assert "no_such_attribute" in str(excinfo.value)

    def test_args_and_literal_may_not_name_the_same_argument(self, context):
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(
                {"python": "jax.numpy:zeros", "args": {"shape": {"list": [2]}},
                 "literal": {"shape": [2]}},
                context,
            )
        assert "shape" in str(excinfo.value)

    @pytest.mark.parametrize("given", [[], None, [1, 2], "shape"])
    def test_args_must_be_a_mapping(self, context, given):
        """Catches `node.get('args') or {}`. The two falsy cases are the ones
        that separate the two spellings -- a bare `args:` in YAML parses as
        None and `args: []` as a list, and `or {}` turns both into 'no
        arguments were written' and calls anyway. The refusal is asserted by
        its own words rather than by 'ConfigError was raised', because the
        mutant also raises one: it reaches the callee and fails there, so
        matching on the type alone reads the wrong refusal as the right one."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"python": "math:pi", "args": given}, context)
        message = str(excinfo.value)
        assert "mapping" in message
        assert type(given).__name__ in message

    def test_a_stray_sibling_is_refused_and_the_two_it_takes_are_named(self, context):
        """Catches register_form('python') with no arguments= (which refuses
        the legal args:/literal: siblings) and arguments=None with no check of
        its own (which admits every stray key silently)."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"python": "math:pi", "kwargs": {"x": 1}}, context)
        message = str(excinfo.value)
        assert "kwargs" in message
        assert "args" in message and "literal" in message

    def test_a_callee_that_refuses_its_keywords_is_reported_against_the_target(self, context):
        """Catches an unwrapped TypeError leaving this layer: a loader catches
        ConfigError to say 'this document is wrong', and a keyword the callee
        does not take is a document error however deep in jax it is noticed."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(
                {"python": "jax.numpy:zeros", "literal": {"shape": [2], "bogus": 1}}, context
            )
        message = str(excinfo.value)
        assert "jax.numpy:zeros" in message
        assert "bogus" in message


class TestTheStatedCost:
    def test_the_resolved_value_records_what_it_came_from(self, context):
        """The hash covers the STRING, not the code. Recording the target is
        what lets provenance.json say the run was not reproducible from the
        config alone, rather than implying it was."""
        got = resolve_value({"python": "math:pi"}, context)
        assert got.modifiers["_python"] == "math:pi"

    def test_recording_the_target_does_not_displace_the_written_modifiers(self, context):
        """Catches carried = {'_python': target} written in place of
        {**modifiers, '_python': target}. The record survives and every
        modifier the document wrote is dropped -- and dropping them is
        invisible in the modifiers dict alone, so the value is checked too:
        scale: is applied at the single exit point from exactly this dict."""
        got = resolve_value({"python": "math:pi", "scale": 2.0}, context)
        assert float(got.value) == pytest.approx(2 * 3.14159265, abs=1e-4)
        assert got.modifiers["_python"] == "math:pi"
        assert got.modifiers["scale"] == 2.0

    def test_a_declared_unit_converts_the_result(self, context):
        """Catches the unit: branch dropped -- pi would come back as 3.14 with
        a unit recorded but never applied, which every downstream check reads
        as a legal value."""
        got = resolve_value({"python": "math:pi", "unit": "MHz"}, context)
        assert float(got.value) == pytest.approx(3.14159265e6, rel=1e-6)
        assert got.unit is not None
