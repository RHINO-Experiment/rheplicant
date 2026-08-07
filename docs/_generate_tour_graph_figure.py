"""Generate the tour's signal-path figure -- the tour's own twin, light and dark.

Run MANUALLY, and only when the tour's ``assemble(...)`` call changes::

    .venv/bin/python docs/_generate_tour_graph_figure.py

The tour tells the reader to call ``twin.to_svg()``, so the picture beside that
sentence has to BE ``twin.to_svg()``. It used to be a hand-written mermaid block
spelling the sum ``AS(("astro_sum"))`` and the switch ``SW{{"receiver_input"}}``
-- a third spelling of a convention the renderers already disagreed about, and a
drawing of a graph nobody had run.

Separate from ``_generate_tour_figures.py`` on purpose. That script measures the
tour's *results* and takes about a minute (GCR draws, 500 re-solves, four NUTS
chains); this one renders the tour's *structure*, which no number enters. Sharing
a module would mean paying for the sampler to redraw a flowchart, and would
rewrite eight committed figures whose only change would be their timestamp.

**What must not drift** is which nodes light up, and that is checked here rather
than trusted: the assembled twin's ``repr`` has to appear verbatim in ``tour.md``,
which is where the tour prints it. Parameter *values* never reach a rendering --
placeholders below stand in for the tour's measured ones -- so the operator SET
is the whole contract, and a change to it fails this script instead of quietly
shipping a figure of a twin the tour no longer builds.
"""

import re
from pathlib import Path

import jax.numpy as jnp

from rheplicant.radio import (
    ADCOperator,
    AntennaLossOperator,
    BeamSpillOperator,
    CalLoadOperator,
    ForegroundOperator,
    GainOperator,
    GlobalSignalOperator,
    NoiseOperator,
    NoiseWaveOperator,
    ReceiverOperator,
    assemble,
)

DOCS = Path(__file__).parent
STATIC = DOCS / "_static"
TOUR = DOCS / "tour.md"

N_FREQ = 8
_PER_CHANNEL = jnp.zeros((N_FREQ,))
_PER_SOURCE = jnp.zeros((4, N_FREQ))

#: The tour's Part 1 assembly, operator for operator. See "Graph assembly" in
#: ``tour.md``; only the set matters here, so the numbers are placeholders.
twin = assemble(
    GlobalSignalOperator(
        depth=jnp.array(0.5), centre=jnp.array(75e6), width=jnp.array(5e6)
    ),
    ForegroundOperator(
        amplitude=jnp.array(2500.0), spectral_index=jnp.array(2.55), ref_freq=70e6
    ),
    BeamSpillOperator(sky_fraction=jnp.array(0.97), t_ground=jnp.array(290.0)),
    AntennaLossOperator(efficiency=jnp.array(0.97), t_physical=jnp.array(293.0)),
    CalLoadOperator(t_load=jnp.array(300.0)),
    CalLoadOperator(t_load=jnp.array(400.0)),
    CalLoadOperator(t_load=jnp.array(1200.0)),
    NoiseWaveOperator(
        t_unc=_PER_CHANNEL, t_cos=_PER_CHANNEL, t_sin=_PER_CHANNEL, t_rx=_PER_CHANNEL,
        gamma_src_re=_PER_SOURCE, gamma_src_im=_PER_SOURCE,
        gamma_rec_re=_PER_CHANNEL, gamma_rec_im=_PER_CHANNEL,
    ),
    ReceiverOperator(bandpass=_PER_CHANNEL + 1.0),
    GainOperator(gain=jnp.array(1.0)),
    NoiseOperator(sigma=jnp.array(2.0)),
    ADCOperator(scale=jnp.array(0.25), n_bits=12),
)

TITLE = "The tour's twin on the single-antenna template: lit nodes are the operators it provided"


def _check_against_the_tour() -> None:
    """The figure and the tour must be drawings of the same assembly."""
    printed = re.sub(r"\s+", " ", TOUR.read_text())
    if re.sub(r"\s+", " ", repr(twin)) not in printed:
        raise SystemExit(
            "The assembly built here is not the one tour.md prints, so the figure "
            "would show a twin the tour does not build. Reconcile them:\n\n"
            f"  here: {twin!r}\n\n"
            f"Search tour.md for \"Assembly(graph='single-antenna'\"."
        )


def main() -> None:
    _check_against_the_tour()
    for theme in ("light", "dark"):
        out = STATIC / f"tour-graph-{theme}.svg"
        out.write_text(twin.to_svg(title=TITLE, theme=theme))
        print(f"wrote {out.relative_to(DOCS.parent)}")


if __name__ == "__main__":
    main()
