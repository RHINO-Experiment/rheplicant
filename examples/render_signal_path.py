"""Render the canonical signal-path graph with an assembly's nodes lit.

Writes ``signal_path.html`` to the current directory — the full single-antenna
flowchart with the provided operators highlighted, traversed-as-identity
nodes half-lit, and everything else dimmed. Open it in a browser.

Run:  uv run python examples/render_signal_path.py
"""

import pathlib

import jax.numpy as jnp

from rheplicant.radio import (
    ForegroundOperator,
    GainOperator,
    GlobalSignalOperator,
    IonosphereOperator,
    assemble,
)

partial_twin = assemble(
    GlobalSignalOperator(depth=jnp.array(0.2), centre=jnp.array(72e6),
                         width=jnp.array(5e6)),
    ForegroundOperator(amplitude=jnp.array(1e3), spectral_index=jnp.array(2.5),
                       ref_freq=70e6),
    IonosphereOperator(delta=jnp.array(0.01), ref_freq=70e6),
    GainOperator(gain=jnp.array(1.1)),
)
print(partial_twin)

out = pathlib.Path("signal_path.html")
out.write_text(partial_twin.to_html())
print(f"wrote {out.resolve()} — open it in a browser")
