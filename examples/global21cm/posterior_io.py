"""Read the published run trees and turn 21 cm parameter draws into curves.

A run tree is what ``rheplicant run`` publishes under ``outputs.dir``: one
``runs/<encoded run name>/`` directory per run, with NPZ products whose keys
are ``mapping/<encoded latent name>``. Both encodings are ``n-`` followed by
the UTF-8 bytes in hex (``docs/config-cli.md``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from global21cm import signal21


def encode(name: str) -> str:
    """The path component the CLI writes for a run or latent name."""
    return "n-" + name.encode("utf-8").hex()


def run_dir(tree: Path, run: str) -> Path:
    path = Path(tree) / "runs" / encode(run)
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is missing: has `rheplicant run` produced {tree}?")
    return path


def load_mapping(tree: Path, run: str, product: str) -> dict[str, np.ndarray]:
    """``{latent: array}`` from ``runs/<run>/<product>.npz``."""
    with np.load(run_dir(tree, run) / f"{product}.npz") as data:
        out = {}
        for key in data.files:
            encoded = key.rsplit("/", 1)[-1]
            name = bytes.fromhex(encoded[2:]).decode("utf-8") if encoded.startswith("n-") else key
            out[name] = np.asarray(data[key])
        return out


def load_json(tree: Path, run: str, name: str) -> dict:
    return json.loads((run_dir(tree, run) / f"{name}.json").read_text())


def seconds(tree: Path, run: str) -> float:
    return float(load_json(tree, run, "timings")["seconds"])


def unit_draws(mapping: dict[str, np.ndarray]) -> np.ndarray:
    """``(S, 7)`` unit-normal latents in ``signal21.THETA_NAMES`` order."""
    return np.stack([np.asarray(mapping[n], dtype=np.float64) for n in signal21.THETA_NAMES], axis=1)


def theta_from_unit(u: np.ndarray) -> np.ndarray:
    """``(S, 7)`` log-parametrised theta."""
    low, high = (np.asarray(v) for v in signal21.prior_box())
    return np.asarray(signal21.box_from_unit_normal(u, low, high))


def curves(theta_log: np.ndarray, freqs_mhz) -> np.ndarray:
    """``(S, F)`` 21 cm curves in K, evaluated in batches."""
    theta_log = np.atleast_2d(theta_log)
    return np.concatenate([np.asarray(signal21.curve_kelvin(theta_log[i : i + 5000], freqs_mhz))
                           for i in range(0, theta_log.shape[0], 5000)])  # fmt: skip


def nuts_summary(tree: Path, run: str) -> dict:
    """Cross-chain R-hat and bulk ESS over the seven latents, and divergences."""
    diagnostics = load_json(tree, run, "run_diagnostics")
    rows = diagnostics["per_latent"].values()
    return {
        "chains": diagnostics["n_chain"],
        "draws": diagnostics["n_draw"],
        "r_hat_max": max(float(r["r_hat"]) for r in rows),
        "n_eff_min": min(float(r["n_eff"]) for r in rows),
        "divergences": diagnostics["divergences"],
        "seconds": seconds(tree, run),
    }
