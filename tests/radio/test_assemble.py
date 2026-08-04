"""Tests for graph-guided assembly on the canonical single-antenna graph."""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import rheplicant.radio as radio
from rheplicant import AbstractOperator, Pipeline, SelectOperator, SumOperator
from rheplicant.core.errors import ParameterSpaceError
from rheplicant.core.graph import AmbiguousNodeError, AssemblyError, At
from rheplicant.inference.parameters import Bind, Latent, ParameterSpace
from rheplicant.radio import (
    RADIO_GRAPH,
    ADCOperator,
    BeamOperator,
    CalLoadOperator,
    ForegroundOperator,
    GainOperator,
    GlobalSignalOperator,
    GroundPickupOperator,
    IonosphereOperator,
    MatrixProjector,
    NoiseOperator,
    PointSourceOperator,
    PowerLawSkyModel,
    RFIOperator,
    SkyOperator,
    SkySourceOperator,
    assemble,
)

N_TIME, N_FREQ = 8, 4


def ops():
    """Fresh operator instances (deterministic parameters)."""
    return {
        "gs": GlobalSignalOperator(
            depth=jnp.array(0.2), centre=jnp.array(72e6), width=jnp.array(5e6)
        ),
        "fg": ForegroundOperator(
            amplitude=jnp.array(1e3), spectral_index=jnp.array(2.5), ref_freq=70e6
        ),
        "ps": PointSourceOperator(level=jnp.array(2.0)),
        "io": IonosphereOperator(delta=jnp.array(0.01), ref_freq=70e6),
        "rf": RFIOperator(amplitude=jnp.array(100.0), occupancy=0.05),
        "bm": BeamOperator(solid_angle=jnp.array(0.8)),
        "gd": GroundPickupOperator(coupling=jnp.array(0.01), t_ground=jnp.array(300.0)),
        "gn": GainOperator(gain=jnp.array(1.1)),
        "ns": NoiseOperator(sigma=jnp.array(0.5)),
        "ad": ADCOperator(scale=jnp.array(1.0), n_bits=14),
    }


class TestUserExamples:
    def test_sky_components_auto_sum(self, template_state):
        """User example 1: {21cm, foregrounds} -> the sky sum, nothing else."""
        o = ops()
        asm = assemble(o["gs"], o["fg"])
        assert isinstance(asm.operator, SumOperator)
        assert asm.operator.names == ("global_signal", "foregrounds")
        hand = SumOperator(o["gs"], o["fg"], names=("global_signal", "foregrounds"))
        assert jnp.array_equal(asm(template_state).data, hand(template_state).data)

    def test_beam_convolved_sky_part(self, template_state):
        """User example 2: {sky, ionosphere, beam} -> just that part of Tsys."""
        o = ops()
        sky = SkyOperator(amplitude=jnp.array(1e3))
        asm = assemble(sky, o["io"], o["bm"])
        hand = Pipeline(sky, o["io"], o["bm"], names=("uniform_sky", "ionosphere", "beam"))
        assert jnp.array_equal(asm(template_state).data, hand(template_state).data)
        assert asm.lit == ("uniform_sky", "ionosphere", "beam")


class TestCanonicalTopology:
    def test_rfi_field_passes_through_beam(self, template_state):
        """rfi_field is a pre-beam field: the beam factor applies to it."""
        o = ops()
        asm = assemble(o["rf"], o["bm"])
        direct = o["rf"](template_state).data
        assert jnp.allclose(asm(template_state).data, 0.8 * direct)

    def test_sky_source_enters_after_beam(self, template_state):
        """observed_astro_sky is post-beam: a standalone beam never touches it."""
        key = jax.random.key(5)
        source = SkySourceOperator(
            sky_model=PowerLawSkyModel(
                amplitude=jnp.ones(6), spectral_index=jnp.array(2.5),
                ref_freq=70e6, n_pix=6,
            ),
            projector=MatrixProjector(matrix=jax.random.normal(key, (N_TIME, 6))),
        )
        o = ops()
        asm = assemble(source, o["gd"])
        assert isinstance(asm.operator, SumOperator)
        assert asm.operator.names == ("observed_astro_sky", "ground_pickup")
        hand = SumOperator(source, o["gd"], names=("observed_astro_sky", "ground_pickup"))
        assert jnp.array_equal(asm(template_state).data, hand(template_state).data)

    def test_multi_component_foregrounds(self, template_state):
        """Two foreground components on one many=True node sum as siblings."""
        f1 = ForegroundOperator(
            amplitude=jnp.array(1e3), spectral_index=jnp.array(2.5), ref_freq=70e6
        )
        f2 = ForegroundOperator(
            amplitude=jnp.array(10.0), spectral_index=jnp.array(2.1), ref_freq=70e6
        )
        asm = assemble(f1, f2)
        expected = f1(template_state).data + f2(template_state).data
        assert jnp.allclose(asm(template_state).data, expected)

    def test_transform_rooted_branch_rejected(self):
        """{ionosphere, ground_pickup}: iono has no live source -> clear error."""
        o = ops()
        with pytest.raises(AssemblyError, match="no live source"):
            assemble(o["io"], o["gd"])

    def test_full_set_equals_hand_built_bitwise(self, template_state):
        o = ops()
        asm = assemble(*o.values())
        astro = Pipeline(
            SumOperator(o["gs"], o["fg"], o["ps"],
                        names=("global_signal", "foregrounds", "point_sources")),
            o["io"],
            names=("astro_sum", "ionosphere"),
        )
        field = SumOperator(astro, o["rf"], names=("astro_sum", "rfi_field"))
        upto_beam = Pipeline(field, o["bm"], names=("field_sum", "beam"))
        t_ant = SumOperator(upto_beam, o["gd"], names=("field_sum", "ground_pickup"))
        hand = Pipeline(
            t_ant, o["gn"], o["ns"], o["ad"],
            names=("t_ant_sum", "gain", "noise", "adc"),
        )
        assert eqx.tree_equal(asm.operator, hand)
        assert jnp.array_equal(asm(template_state).data, hand(template_state).data)

    def test_processing_segment_rides_along(self, template_state):
        from rheplicant.radio import FlaggingOperator, SiderealFilter

        o = ops()
        asm = assemble(
            SkyOperator(amplitude=jnp.array(1e3)), o["ns"],
            FlaggingOperator(threshold=2e3),
            SiderealFilter(n_days=2, mode="remove"),
        )
        out = asm(template_state)
        assert "flags" in out.aux
        assert out.data.shape == (N_TIME, N_FREQ)


class TestSwitchedCalibration:
    def test_selector_passes_through_without_loads(self, template_state):
        """Backward compatible: no cal_loads -> receiver_input is identity."""
        o = ops()
        asm = assemble(SkyOperator(amplitude=jnp.array(100.0)), o["gn"])
        assert "receiver_input" in asm.skipped
        assert jnp.allclose(asm(template_state).data, 110.0)

    def test_switching_cycle_selects_antenna_or_load(self, template_state):
        from rheplicant import SelectOperator
        from rheplicant.radio import CalLoadOperator

        switch = jnp.array([0, 1, 0, 0, 1, 0, 0, 1])
        state = template_state.replace(
            coords=template_state.coords.replace(
                extra={"receiver_input": switch}
            )
        )
        asm = assemble(
            SkyOperator(amplitude=jnp.array(100.0)),
            CalLoadOperator(t_load=jnp.array(300.0)),
        )
        assert isinstance(asm.operator, SelectOperator)
        # branch labels: first live node of each branch, edge order fixed
        assert asm.operator.names == ("uniform_sky", "cal_loads")
        assert asm.operator.switch_key == "receiver_input"
        out = asm(state)
        expected = jnp.where(switch[:, None] == 0, 100.0, 300.0)
        assert jnp.allclose(out.data, expected)

    def test_cal_load_shapes(self, template_state):
        """Regression: per-frequency t_load broadcasts along freq, with validation."""
        from rheplicant.core.errors import StateValidationError
        from rheplicant.radio import CalLoadOperator

        t_load = jnp.arange(1.0, N_FREQ + 1.0)
        out = CalLoadOperator(t_load=t_load)(template_state)
        assert jnp.array_equal(out.data[3], t_load)  # every time row = spectrum
        with pytest.raises(StateValidationError, match="channels"):
            CalLoadOperator(t_load=jnp.ones(N_FREQ + 1))(template_state)
        with pytest.raises(StateValidationError, match="ndim"):
            CalLoadOperator(t_load=jnp.ones((2, 2)))(template_state)

    def test_load_only_observation(self, template_state):
        """Only the load provided: selector passes it through (all samples load)."""
        from rheplicant.radio import CalLoadOperator

        asm = assemble(CalLoadOperator(t_load=jnp.array(300.0)))
        assert jnp.allclose(asm(template_state).data, 300.0)

    def test_switching_is_differentiable_wrt_both_branches(self, template_state):
        from rheplicant.radio import CalLoadOperator

        switch = jnp.array([0, 1] * 4)
        state = template_state.replace(
            coords=template_state.coords.replace(extra={"receiver_input": switch})
        )
        asm = assemble(
            SkyOperator(amplitude=jnp.array(100.0)),
            CalLoadOperator(t_load=jnp.array(300.0)),
        )

        def loss(a):
            return jnp.sum(a(state).data)

        g = eqx.filter_grad(loss)(asm)
        assert jnp.isfinite(g["uniform_sky"].amplitude) and g["uniform_sky"].amplitude != 0
        assert jnp.isfinite(g["cal_loads"].t_load) and g["cal_loads"].t_load != 0


class TestRegistryCompleteness:
    def test_every_concrete_operator_is_placeable(self):
        """Every exported radio operator class has a valid graph_node."""
        import inspect

        from rheplicant.core.operator import AbstractOperator

        # Surrogates deliberately require explicit At(...) placement.
        explicit_placement = {"NeuralOperator"}

        missing = []
        for name in radio.__all__:
            obj = getattr(radio, name)
            if not (inspect.isclass(obj) and issubclass(obj, AbstractOperator)):
                continue
            if inspect.isabstract(obj) or name.startswith("Abstract"):
                continue
            if name in explicit_placement:
                continue
            node = getattr(obj, "graph_node", None)
            if node is None or (isinstance(node, str) and node not in RADIO_GRAPH.nodes):
                missing.append(name)
        assert not missing, f"operators without a valid graph_node: {missing}"

    def test_reserved_leaves_exist(self):
        """The equivalent-entry placeholder leaves are part of the template."""
        for leaf in ("ground_field", "t_sys_extra"):
            assert RADIO_GRAPH.nodes[leaf].reserved

    def test_t_sys_extra_accepts_at_injection(self, template_state):
        asm = assemble(
            SkyOperator(amplitude=jnp.array(1e3)),
            BeamOperator(solid_angle=jnp.array(0.8)),
            At("t_sys_extra", GroundPickupOperator(
                coupling=jnp.array(0.02), t_ground=jnp.array(300.0)
            )),
        )
        assert isinstance(asm.operator, SumOperator)
        assert jnp.all(jnp.isfinite(asm(template_state).data))


class TestRendering:
    def test_mermaid_marks_segments_and_lit(self):
        o = ops()
        asm = assemble(o["gs"], o["gn"])
        mm = asm.to_mermaid()
        assert "class global_signal lit" in mm
        assert "class gain lit" in mm
        assert "class ionosphere wire" in mm  # traversed as identity
        assert "class flagging dim" in mm

    def test_html_render(self):
        o = ops()
        asm = assemble(o["gs"], o["gn"])
        html = asm.to_html()
        assert html.startswith("<!doctype html>")
        assert "<svg" in html and "global signal" in html
        assert 'class="lit"' in html and 'class="dim"' in html and 'class="wire"' in html
        assert "stroke-dasharray" in html  # reserved leaves drawn dashed
        # every node of the template appears
        for nid in RADIO_GRAPH.nodes:
            assert nid.replace("_", " ") in html or nid in html

    def test_html_escapes_title(self):
        html = RADIO_GRAPH.to_html(title='Run "A" & <B>')
        assert "<B>" not in html
        assert "&lt;B&gt;" in html and "&quot;A&quot;" in html or "&#x27;" in html

    def test_svg_render_is_self_contained(self):
        o = ops()
        asm = assemble(o["gs"], o["gn"])
        svg = asm.to_svg()
        assert svg.startswith("<svg") and svg.endswith("</svg>")
        assert "<body" not in svg and "<html" not in svg
        # opacity classes are styled INSIDE the svg (survives <img> embedding)
        assert "<style>" in svg and ".dim{opacity:.22}" in svg
        assert svg in asm.to_html()  # the page embeds the same figure


class TestManyInstancesComposeLikeTheirConsumer:
    """``many`` at a source folds as its CONSUMER folds: summed into a junction,
    switched at a selector."""

    def test_several_cal_loads_become_several_switch_positions(self, template_state):
        switch = jnp.arange(8) % 3
        twin = assemble(
            SkyOperator(amplitude=jnp.array(100.0)),
            CalLoadOperator(t_load=jnp.array(300.0)),
            CalLoadOperator(t_load=jnp.array(400.0)),
        )
        selector = twin["receiver_input"]
        assert isinstance(selector, SelectOperator)
        assert selector.names == ("uniform_sky", "cal_loads_1", "cal_loads_2")

        out = twin(template_state.replace(
            coords=template_state.coords.replace(extra={"receiver_input": switch})
        )).data
        assert jnp.allclose(out[switch == 0], 100.0)
        assert jnp.allclose(out[switch == 1], 300.0)
        assert jnp.allclose(out[switch == 2], 400.0)

    def test_the_loads_are_not_summed(self, template_state):
        """Summing them would be the junction rule, and would give 700 K on
        every load sample: finite, correctly shaped, a different instrument."""
        switch = jnp.ones(8, dtype=int)
        out = assemble(
            SkyOperator(amplitude=jnp.array(100.0)),
            CalLoadOperator(t_load=jnp.array(300.0)),
            CalLoadOperator(t_load=jnp.array(400.0)),
        )(template_state.replace(
            coords=template_state.coords.replace(extra={"receiver_input": switch})
        )).data
        assert jnp.allclose(out, 300.0)
        assert not jnp.allclose(out, 700.0)

    def test_a_many_source_into_a_junction_still_sums(self, template_state):
        """The unchanged path. ``foregrounds`` is ``many`` and feeds a junction,
        so its instances add -- and keep folding as a nested SumOperator, which
        is not cosmetic: SumOperator splits the PRNG key per branch, so a
        flatter tree would be a different seeded run."""
        twin = assemble(
            ForegroundOperator(amplitude=jnp.array(10.0),
                               spectral_index=jnp.array(-2.5),
                               ref_freq=jnp.array(70e6)),
            ForegroundOperator(amplitude=jnp.array(20.0),
                               spectral_index=jnp.array(-2.5),
                               ref_freq=jnp.array(70e6)),
        )
        assert isinstance(twin.operator, SumOperator)
        assert twin.operator.names == ("foregrounds_1", "foregrounds_2")
        out = twin(template_state).data
        one = assemble(
            ForegroundOperator(amplitude=jnp.array(30.0),
                               spectral_index=jnp.array(-2.5),
                               ref_freq=jnp.array(70e6))
        )(template_state).data
        assert jnp.allclose(out, one)

    def test_a_selector_with_one_live_branch_is_not_a_switch_at_all(
        self, template_state
    ):
        """The other half of the rule. A selector materializes only when it has
        something to choose between; with one live upstream it is traversed as
        identity, needs no switch array in coords, and leaves no SelectOperator
        behind. That is what makes the load-free antenna chain cost nothing."""
        twin = assemble(SkyOperator(amplitude=jnp.array(100.0)),
                        GainOperator(gain=jnp.array(2.0)))
        assert "receiver_input" in twin.skipped
        is_sel = lambda x: isinstance(x, SelectOperator)  # noqa: E731
        assert not any(map(is_sel, jax.tree.leaves(twin, is_leaf=is_sel)))
        out = twin(template_state).data          # no coords.extra["receiver_input"]
        assert jnp.allclose(out, 200.0)

    def test_one_cal_load_still_gives_a_two_position_switch(self, template_state):
        """And one load is a switch again -- the antenna and it."""
        switch = jnp.arange(8) % 2
        twin = assemble(
            SkyOperator(amplitude=jnp.array(100.0)),
            CalLoadOperator(t_load=jnp.array(300.0)),
        )
        assert twin["receiver_input"].names == ("uniform_sky", "cal_loads")
        out = twin(template_state.replace(
            coords=template_state.coords.replace(extra={"receiver_input": switch})
        )).data
        assert jnp.allclose(out[switch == 0], 100.0)
        assert jnp.allclose(out[switch == 1], 300.0)


class _TSysBasis(AbstractOperator):
    """A generic effective-T_sys contribution: ``basis @ coeff``, flat in time.

    ``t_sys_extra`` is the ``many=True`` node the graph docstring designates
    for exactly this, and it ships no operator — so the test brings its own,
    the way the next component to arrive there will.
    """

    graph_node: ClassVar[str] = "t_sys_extra"
    coeff: jax.Array
    basis: jax.Array

    def __call__(self, state):
        n_time = state.coords.time.shape[0]
        return state.with_data(
            jnp.broadcast_to(self.basis @ self.coeff, (n_time, self.basis.shape[0]))
        )


def _basis():
    return jnp.stack([jnp.ones(N_FREQ), jnp.linspace(-1.0, 1.0, N_FREQ)], axis=1)


def _coeff_space(node_id: str) -> ParameterSpace:
    return ParameterSpace(
        latents=(Latent("C", init=jnp.array([10.0, 1.0])),),
        bindings=(Bind("C", into=lambda p, n=node_id: p[n].coeff),),
    )


class TestManyNodeSurvivesASibling:
    """One ParameterSpace, two assemblies differing only by a sibling component.

    The design intent under test: *re-parameterizing never requires editing
    the instrument description*. The failure it replaces was the reverse —
    adding a component silently redirected an existing space onto the fold
    that sums the instances, so ``validate`` reported a missing ``coeff``
    attribute and named nothing the author could act on.
    """

    def test_the_space_validates_against_the_single_instance_assembly(self):
        one = assemble(
            _TSysBasis(coeff=jnp.array([10.0, 1.0]), basis=_basis()),
            GainOperator(gain=jnp.array(1.1)),
        )
        assert isinstance(one["t_sys_extra"], _TSysBasis)
        _coeff_space("t_sys_extra").validate(one)  # raises on failure

    def test_a_sibling_makes_the_bare_node_id_a_named_error(self):
        space = _coeff_space("t_sys_extra")
        two = assemble(
            _TSysBasis(coeff=jnp.array([10.0, 1.0]), basis=_basis()),
            _TSysBasis(coeff=jnp.array([2.0, 0.5]), basis=_basis()),
            GainOperator(gain=jnp.array(1.1)),
        )
        with pytest.raises(ParameterSpaceError) as excinfo:
            space.validate(two)
        message = str(excinfo.value)
        assert "t_sys_extra_1" in message and "t_sys_extra_2" in message

    def test_the_names_the_error_gives_are_the_ones_that_work(self):
        two = assemble(
            _TSysBasis(coeff=jnp.array([10.0, 1.0]), basis=_basis()),
            _TSysBasis(coeff=jnp.array([2.0, 0.5]), basis=_basis()),
            GainOperator(gain=jnp.array(1.1)),
        )
        _coeff_space("t_sys_extra_1").validate(two)
        _coeff_space("t_sys_extra_2").validate(two)
        bound = _coeff_space("t_sys_extra_2").bind(two, {"C": jnp.array([7.0, 0.0])})
        assert jnp.allclose(bound["t_sys_extra_2"].coeff, jnp.array([7.0, 0.0]))
        assert jnp.allclose(bound["t_sys_extra_1"].coeff, jnp.array([10.0, 1.0]))

    def test_the_sibling_is_still_in_the_forward_model(self, template_state):
        """The bug's signature was a component silently vanishing; check it did
        not, by the forward output rather than by shape."""
        a = _TSysBasis(coeff=jnp.array([10.0, 1.0]), basis=_basis())
        b = _TSysBasis(coeff=jnp.array([2.0, 0.5]), basis=_basis())
        two = assemble(a, b, GainOperator(gain=jnp.array(1.1)))
        expected = 1.1 * (a(template_state).data + b(template_state).data)
        assert jnp.allclose(two(template_state).data, expected)

    def test_replace_node_on_the_bare_id_refuses_instead_of_deleting(
        self, template_state
    ):
        a = _TSysBasis(coeff=jnp.array([10.0, 1.0]), basis=_basis())
        b = _TSysBasis(coeff=jnp.array([2.0, 0.5]), basis=_basis())
        two = assemble(a, b, GainOperator(gain=jnp.array(1.1)))
        before = two(template_state).data
        zeroed = _TSysBasis(coeff=jnp.zeros(2), basis=_basis())
        with pytest.raises(AmbiguousNodeError, match="t_sys_extra_1"):
            two.replace_node("t_sys_extra", zeroed)
        assert jnp.array_equal(two(template_state).data, before)
        # the named form drops instance 1 only, and says so in the output
        swapped = two.replace_node("t_sys_extra_1", zeroed)
        assert jnp.allclose(
            swapped(template_state).data, 1.1 * b(template_state).data
        )
