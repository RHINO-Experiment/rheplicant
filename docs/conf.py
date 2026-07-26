"""Sphinx configuration for the RHEPLICANT documentation (furo + MyST + autodoc)."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "RHEPLICANT"
author = "RHINO Experiment Collaboration"
copyright = "2026, RHINO Experiment Collaboration"

try:
    from rheplicant import __version__ as release
except ImportError:  # pragma: no cover - docs build without the package
    release = "0.0.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
]

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autosummary_generate = False
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
    "numpyro": ("https://num.pyro.ai/en/stable", None),
}

html_theme = "furo"
html_title = "RHEPLICANT — a differentiable replica of a radio antenna"
html_static_path = ["_static"]
html_logo = "_static/rheplicant-logo.png"
# The logo already carries the "rheplicant" wordmark, so don't repeat the name.
# source_repository adds per-page "view/edit source" links; footer_icons puts a
# GitHub mark in the footer so every page (incl. the landing page) links to the repo.
_GITHUB = "https://github.com/RHINO-Experiment/rheplicant"
html_theme_options = {
    "sidebar_hide_name": True,
    "source_repository": f"{_GITHUB}/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": _GITHUB,
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16" width="1em" height="1em"><path fill-rule="evenodd" '
                'd="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 '
                "0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13"
                "-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07"
                "-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2"
                "-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 "
                ".27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 "
                "0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 "
                '.21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}

exclude_patterns = ["_build"]

# The signal-path page is generated from the live template at build time,
# so the rendered graph can never drift from the code. Two example
# assemblies are rendered to SVG alongside it, showing the lit/dim view.
_DOCS_DIR = pathlib.Path(__file__).parent
_SIGNAL_PATH_PAGE = _DOCS_DIR / "signal-path.md"


def _example_svgs() -> None:
    import jax.numpy as jnp

    from rheplicant.radio import (
        AtmosphericEmissionOperator,
        BeamOperator,
        CalLoadOperator,
        ForegroundOperator,
        GainOperator,
        GlobalSignalOperator,
        GroundPickupOperator,
        IonosphereOperator,
        NoiseOperator,
        RFIOperator,
        assemble,
    )

    partial = assemble(
        GlobalSignalOperator(
            depth=jnp.array(0.2), centre=jnp.array(72e6), width=jnp.array(5e6)
        ),
        ForegroundOperator(
            amplitude=jnp.array(1e3), spectral_index=jnp.array(2.5), ref_freq=70e6
        ),
        IonosphereOperator(delta=jnp.array(0.01), ref_freq=70e6),
        BeamOperator(solid_angle=jnp.array(0.8)),
        GainOperator(gain=jnp.array(1.1)),
    )
    (_DOCS_DIR / "signal-path-partial.svg").write_text(
        partial.to_svg(title="Partial twin: beam-convolved sky through the gain")
    )

    fuller = assemble(
        GlobalSignalOperator(
            depth=jnp.array(0.2), centre=jnp.array(72e6), width=jnp.array(5e6)
        ),
        ForegroundOperator(
            amplitude=jnp.array(1e3), spectral_index=jnp.array(2.5), ref_freq=70e6
        ),
        RFIOperator(amplitude=jnp.array(2e3), occupancy=0.01),
        BeamOperator(solid_angle=jnp.array(0.8)),
        GroundPickupOperator(coupling=jnp.array(0.01), t_ground=jnp.array(300.0)),
        AtmosphericEmissionOperator(t_atm=jnp.array(150.0)),
        CalLoadOperator(t_load=jnp.array(400.0)),
        GainOperator(gain=jnp.array(1.1)),
        NoiseOperator(sigma=jnp.array(0.5)),
    )
    (_DOCS_DIR / "signal-path-fuller.svg").write_text(
        fuller.to_svg(title="Fuller twin: sky, RFI, ground, atmosphere, cal loads")
    )


try:
    from rheplicant.radio import RADIO_GRAPH

    _mermaid = RADIO_GRAPH.to_mermaid()
    _example_svgs()
    _SIGNAL_PATH_PAGE.write_text(
        "# The canonical signal path\n\n"
        "The single-antenna template every assembly lights up — generated "
        "from `rheplicant.radio.RADIO_GRAPH` at documentation build time. `(+)` "
        "nodes are sum junctions, `(sw)` the antenna/cal-load selector; see "
        "the [tour](tour.md#4-graph-assembly) for the assembly rules and "
        "[the operator catalog](operators.md) for what lives at each node.\n\n"
        "```{mermaid}\n" + _mermaid + "\n```\n\n"
        "## Lit and dim: what an assembly simulates\n\n"
        "`assemble(*operators)` never shows you only the piece you built — "
        "rendering always draws the *full* template, with the provided "
        "operators lit (colored, amber signal path), traversed-as-identity "
        "nodes half-lit, and everything else dimmed. Reserved placeholder "
        "leaves are dashed. Both examples below are real renders, generated "
        "at build time with `assembly.to_svg()` (`to_html()` produces the "
        "same figure as a standalone page).\n\n"
        "A partial twin — five operators, `assemble(global_signal, "
        "foregrounds, ionosphere, beam, gain)`; every junction on the way "
        "passes through and the rest of the receiver chain is identity:\n\n"
        "```{figure} signal-path-partial.svg\n"
        ":alt: Partial twin with five lit operators\n\n"
        "Partial twin: the beam-convolved sky through the gain.\n"
        "```\n\n"
        "A fuller twin — nine operators including RFI, ground pickup, "
        "atmospheric emission, and switched calibration loads (the `sw` "
        "selector node); note the two dashed reserved entrances "
        "(`atmosphere_field`, `t_sys_extra`) staying dim:\n\n"
        "```{figure} signal-path-fuller.svg\n"
        ":alt: Fuller twin with nine lit operators\n\n"
        "Fuller twin: sky, RFI, ground, atmosphere, and calibration loads.\n"
        "```\n"
    )
except ImportError:  # pragma: no cover
    _SIGNAL_PATH_PAGE.write_text(
        "# The canonical signal path\n\n(rheplicant is not importable in this "
        "build environment; graph rendering skipped.)\n"
    )
