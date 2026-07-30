"""The canonical single-antenna signal-path graph and graph-guided assembly.

This is the flowchart that makes composition implicit: provide a set of
operators and :func:`assemble` lights up the connected sub-path they induce
and compiles it to the equivalent ``Pipeline``/``SumOperator`` nesting::

    from rheplicant.radio.graph import assemble

    twin = assemble(GlobalSignalOperator(...), ForegroundOperator(...),
                    BeamOperator(...), GainOperator(...))
    print(twin)                # lit nodes + skipped-as-identity nodes
    print(twin.to_mermaid())   # lit/dim signal-path rendering

Topology (v1.4; sum junctions marked ``(+)``)::

    global_signal | foregrounds | point_sources | uniform_sky
        -> (+) astro_sum -> ionosphere -> atmosphere_field* --\\
    ground_field* | rfi_field ----------------------> (+) field_sum -> beam --\\
    beam | observed_astro_sky -> (+) astro_ant_sum -> beam_spill --------------\\
    ground_pickup | t_sys_extra* | atmosphere ---------------------------------> (+) t_ant_sum
        -> antenna_loss -> (SW) receiver_input <- cal_loads
        -> noise_wave -> cw_tone -> bandpass -> gain
        -> noise -> emi -> adc
        -> flagging -> averaging -> apply_cal -> filters      [processing segment]

Equivalent-entry leaves (the ``*`` nodes are reserved placeholders with no
shipped operator yet): the same physical effect may enter at different
stages in different forms — ground spill either as a *field* before the beam
(``ground_field``, to be convolved) or as an *effective temperature* after it
(``ground_pickup`` / generic ``t_sys_extra``); the atmosphere either as
strict radiative transfer on the astro branch before the beam
(``atmosphere_field``, reserved — opacity acts on the astro sky alone, never
on ground pickup) or as a beam-averaged additive emission temperature in the
antenna-temperature sum (``atmosphere``); the whole astro path either as
component fields through the shared ``beam`` node or pre-convolved via
``observed_astro_sky`` (``SkySourceOperator``). Provide whichever form you
have; the graph keeps both entrances.

Switched calibration loads (elements taxonomy "calibration signals ...
switched in and out on a pre-defined cycle") enter through the
``receiver_input`` *selector* node: with only the antenna chain provided it
passes through; provide ``CalLoadOperator`` too and each time sample takes
the branch chosen by ``coords.extra["receiver_input"]``. ``cal_loads`` is
``many=True`` and feeds only the selector, so each instance becomes its OWN
switch position rather than being summed with its siblings: with two loads the
switch indexes 0 = antenna, 1 = first load, 2 = second load — the edge
declaration order, then the order the loads were provided. Three distinct
sources is what an identifiable per-channel noise-wave fit needs, and
``assemble()`` expresses it directly.

``beam_spill`` (v1.4) is the horizon split of a beam that does not stop at the
horizon: the part below it sees ground, not sky. It is the trunk stage of the
ASTRO branch — the two equivalent astro entrances (``beam``,
``observed_astro_sky``) meet at ``astro_ant_sum`` first — because the split
applies to the thing that genuinely is a beam integral over the celestial
sphere and to nothing else. The other ``t_ant_sum`` leaves are *effective*
temperatures by D13's construction, already carrying whatever beam weighting
their author intended, and ``ground_pickup`` in particular IS a below-horizon
share; running them through the split would weight them twice.

``antenna_loss`` (v1.3) is the antenna's own ohmic dissipation, on the trunk
between ``t_ant_sum`` and the switch: it acts on everything the beam
collected (unlike atmospheric opacity — see D13) and on nothing that connects
downstream of the antenna, which is why the calibration loads must enter after
it. Absent an ``AntennaLossOperator`` the node is skipped as identity, which
is the lossless-antenna assumption made explicit rather than hidden.

The forward physical chain ends at ``adc`` (the raw waterfall); the
processing segment (flagging/averaging/apply_cal/filters) is data-side and
applies identically to simulated and observed raw data.
"""

from rheplicant.core.graph import At, NodeSpec, SignalGraph, register_graph
from rheplicant.core.graph import assemble as _assemble
from rheplicant.core.operator import AbstractOperator

_S, _T, _J = "source", "transform", "junction"

RADIO_GRAPH = register_graph(
    SignalGraph(
        "single-antenna",
        {
            "global_signal": NodeSpec(_S, "21 cm global signal"),
            "foregrounds": NodeSpec(_S, "diffuse foregrounds", many=True),
            "point_sources": NodeSpec(_S, "beam-diluted point sources"),
            "uniform_sky": NodeSpec(_S, "uniform sky (simplest placeholder)"),
            "astro_sum": NodeSpec(_J, "astrophysical sum"),
            "ionosphere": NodeSpec(_T, "chromatic distortion of the astro sky"),
            "atmosphere_field": NodeSpec(
                _T, "atmospheric radiative transfer on the astro sky", reserved=True
            ),
            "ground_field": NodeSpec(_S, "ground as pre-beam field", reserved=True),
            "rfi_field": NodeSpec(_S, "RFI entering through sidelobes"),
            "field_sum": NodeSpec(_J, "pre-beam field sum"),
            "beam": NodeSpec(_T, "shared chromatic beam (the pain point)"),
            "observed_astro_sky": NodeSpec(_S, "pre-convolved astro sky (SkySource)"),
            "ground_pickup": NodeSpec(_S, "effective ground-spill temperature"),
            "t_sys_extra": NodeSpec(
                _S, "generic effective T_sys contribution", many=True, reserved=True
            ),
            "atmosphere": NodeSpec(_S, "beam-averaged atmospheric emission"),
            "astro_ant_sum": NodeSpec(_J, "beam-convolved astro sky, either entrance"),
            "beam_spill": NodeSpec(
                _T, "horizon split: f_sky * sky + (1-f_sky) * ground"
            ),
            "t_ant_sum": NodeSpec(_J, "antenna-temperature assembly"),
            "antenna_loss": NodeSpec(
                _T, "antenna ohmic loss: eta T + (1-eta) T_phys (before the switch)"
            ),
            "cal_loads": NodeSpec(
                _S,
                "switched calibration loads; one switch position per instance",
                many=True,
            ),
            "receiver_input": NodeSpec(
                "selector", "antenna/load switch (cycle in coords.extra)"
            ),
            "noise_wave": NodeSpec(_T, "reflection loss + noise-wave T terms"),
            "cw_tone": NodeSpec(_T, "CW calibration tone (before bandpass/gain)"),
            "bandpass": NodeSpec(_T, "receiver bandpass"),
            "gain": NodeSpec(_T, "time-dependent gain g(t)"),
            "noise": NodeSpec(_T, "post-gain thermal noise T_n"),
            "emi": NodeSpec(_T, "self-generated EMI comb"),
            "adc": NodeSpec(_T, "digitisation -> raw waterfall"),
            "flagging": NodeSpec(_T, "RFI flags -> aux", segment="processing"),
            "averaging": NodeSpec(_T, "time integration", segment="processing"),
            "apply_cal": NodeSpec(_T, "apply gain solution", segment="processing"),
            "filters": NodeSpec(
                _T, "sidereal / sky-space / Fourier filters",
                many=True, segment="processing",
            ),
        },
        [
            ("global_signal", "astro_sum"),
            ("foregrounds", "astro_sum"),
            ("point_sources", "astro_sum"),
            ("uniform_sky", "astro_sum"),
            ("astro_sum", "ionosphere"),
            ("ionosphere", "atmosphere_field"),
            ("atmosphere_field", "field_sum"),
            ("ground_field", "field_sum"),
            ("rfi_field", "field_sum"),
            ("field_sum", "beam"),
            ("beam", "astro_ant_sum"),
            ("observed_astro_sky", "astro_ant_sum"),
            ("astro_ant_sum", "beam_spill"),
            ("beam_spill", "t_ant_sum"),
            ("ground_pickup", "t_ant_sum"),
            ("t_sys_extra", "t_ant_sum"),
            ("atmosphere", "t_ant_sum"),
            ("t_ant_sum", "antenna_loss"),
            ("antenna_loss", "receiver_input"),
            ("cal_loads", "receiver_input"),
            ("receiver_input", "noise_wave"),
            ("noise_wave", "cw_tone"),
            ("cw_tone", "bandpass"),
            ("bandpass", "gain"),
            ("gain", "noise"),
            ("noise", "emi"),
            ("emi", "adc"),
            ("adc", "flagging"),
            ("flagging", "averaging"),
            ("averaging", "apply_cal"),
            ("apply_cal", "filters"),
        ],
    )
)


def assemble(*operators: AbstractOperator | At):
    """Assemble radio operators on the canonical single-antenna graph."""
    return _assemble(RADIO_GRAPH, *operators)


def _validate_registrations():
    """Import-time check: every radio operator's graph_node exists on the graph."""
    import rheplicant.radio as radio

    for name in radio.__all__:
        obj = getattr(radio, name)
        node = getattr(obj, "graph_node", None)
        if isinstance(node, str) and node not in RADIO_GRAPH.nodes:
            raise AssertionError(
                f"{name}.graph_node = {node!r} is not a node of RADIO_GRAPH."
            )
