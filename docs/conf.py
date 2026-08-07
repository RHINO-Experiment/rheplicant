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
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
]

# dollarmath is what makes `$...$` and `$$...$$` render as math rather than as
# literal dollar signs; amsmath admits `align`/`aligned` environments. Both were
# missing while the physics pages were already written in TeX, so every equation
# on them shipped as source text — a docs build cannot warn about that, only a
# reader can.
myst_enable_extensions = ["amsmath", "colon_fence", "deflist", "dollarmath"]
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

# Targets that a nitpicky build (`sphinx-build -n`) can never resolve, because
# they belong to packages with no published objects.inv. Listed one by one
# rather than silenced with a wildcard: a NEW unresolved target in this package
# is a documented promise pointing at nothing, and `-n` is how it gets caught.
#
# equinox/jaxlib entries come from autodoc rendering base classes and
# annotations, not from anything written by hand here. The limTOD ones are
# genuine cross-package references in docstrings, to an optional dependency.
nitpick_ignore = [
    ("py:class", "equinox._module._module.Module"),
    ("py:class", "equinox.nn._mlp.MLP"),
    ("py:class", "jaxlib._jax.pytree.PyTreeDef"),
    ("py:mod", "limTOD.cstbeam"),
    ("py:func", "limTOD.cstbeam.cst_beam_maps"),
    ("py:func", "limTOD.cstbeam.cst_frequency_table"),
    ("py:func", "limTOD.cstbeam.read_cst_farfield"),
    ("py:func", "limtod_jax.horizon_beam_fraction"),
    ("py:func", "limtod_jax.horizon_truncated_beam"),
]

html_theme = "furo"
html_title = "RHEPLICANT — a differentiable replica of a radio antenna"
html_static_path = ["_static"]
#: Caption typography: see the file. Nothing here overrides furo's own
#: variables, so it follows the reader's light/dark choice by construction.
html_css_files = ["custom.css"]
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

exclude_patterns = ["_build", "superpowers", "_generated"]

# The signal path's PROSE is a tracked source file, `docs/signal-path.md`. Only
# the artefacts it embeds are generated here: the mermaid diagram of the live
# template, and two example assembly renders.
#
# It used to be the other way round -- conf.py wrote the whole page as one string
# literal and `.gitignore` hid the result -- and that cost more than it looked
# like it did. The page carried the composition model, the node kinds, the
# "template, not the framework" argument and the custom-graph sketch; all of it
# was invisible to `git diff`, invisible to grep, unreachable on GitHub (the
# README's link to it 404'd), and impossible to edit where it is read. Worse, the
# ImportError fallback below replaced the entire page with a two-line stub, so a
# build where the package failed to import shipped documentation with no
# explanation of Cascade/Sum/Switch at all.
#
# Now a failed import costs a diagram, not an argument. Every artefact is written
# on both paths, so the page never references a file that is not there.
_DOCS_DIR = pathlib.Path(__file__).parent
_GENERATED = _DOCS_DIR / "_generated"

#: Shown in place of a render when `rheplicant` will not import. A visible
#: statement of what is missing beats a broken-image icon, and beats the silence
#: of a directive whose file is absent.
_PLACEHOLDER_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="60">'
    '<text x="8" y="34" font-family="sans-serif" font-size="14">'
    "rheplicant is not importable in this build environment; render skipped."
    "</text></svg>"
)
_PLACEHOLDER_MERMAID = (
    "flowchart LR\n"
    '    skipped["rheplicant is not importable in this build '
    'environment;\\ngraph rendering skipped"]\n'
)


def _example_svgs() -> dict[tuple[str, str], str]:
    """The two lit/dim example renders, each in both themes.

    Keyed ``(which, theme)``. Both themes are emitted because an SVG embedded
    through ``<img>`` cannot read the host page's theme class, so a page that
    switches has to ship a pair and show one of them -- the same arrangement the
    committed figures use.
    """
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
    titles = {
        "partial": "Partial twin: beam-convolved sky through the gain",
        "fuller": "Fuller twin: sky, RFI, ground, atmosphere, cal loads",
    }
    twins = {"partial": partial, "fuller": fuller}
    return {
        (which, theme): twins[which].to_svg(title=titles[which], theme=theme)
        for which in twins
        for theme in ("light", "dark")
    }


try:
    from rheplicant.radio import RADIO_GRAPH

    _mermaid = RADIO_GRAPH.to_mermaid()
    _examples = _example_svgs()
except ImportError:  # pragma: no cover - only when the package will not import
    _mermaid = _PLACEHOLDER_MERMAID
    _examples = {
        (which, theme): _PLACEHOLDER_SVG
        for which in ("partial", "fuller")
        for theme in ("light", "dark")
    }

_GENERATED.mkdir(exist_ok=True)
(_GENERATED / "radio-graph.mmd").write_text(_mermaid)
for (_which, _theme), _svg in _examples.items():
    (_DOCS_DIR / f"signal-path-{_which}-{_theme}.svg").write_text(_svg)
