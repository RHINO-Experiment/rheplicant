"""Write a memory to disk so that reading it back cannot lie about it.

``eqx.tree_serialise_leaves`` walks the *arrays* of a pytree and takes
everything else from the template it is given. Every static field of a stored
term -- whether it is exact, which estimator it encodes, what support it claims,
how many samples it saw -- therefore round-trips as whatever the template
happened to hold, with no error and no warning. Measured on this repo's
equinox 0.13.8: ``include_logdet=False`` comes back ``True``,
``noise_frozen_at="gls"`` comes back ``"none"``, ``n_observed=777`` comes back
``0``. A reloaded campaign would describe itself as a set of exact,
full-likelihood factors regardless of what was written, and the whole premise
of this layer is that the raw data is gone and cannot contradict it.

So the manifest is not provenance. **It is the reconstruction spec**: the
arrays come from the binary, and every static field, every dtype, and the
writer's x64 state come from the JSON beside it. ``load_memory`` builds the
template *from the manifest* and refuses -- not warns -- on any mismatch with
the running environment.
"""

import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.compressed import QuadraticLikelihood
from rheplicant.inference.factorize import Factorization
from rheplicant.inference.sqrtinfo import SqrtInfo

_FORMAT_VERSION = 1


def _manifest_path(path: Path) -> Path:
    return Path(path).with_suffix(".json")


def _describe(term: QuadraticLikelihood) -> dict[str, Any]:
    return {
        "epoch_id": term.epoch_id,
        "n_observed": term.n_observed,
        "exact": term.exact,
        "support": None if term.support is None else {
            name: list(bounds) for name, bounds in term.support.items()
        },
        "include_logdet": term.include_logdet,
        "noise_frozen_at": term.noise_frozen_at,
        "prior_share": list(term.prior_share),
        "rows": int(term.info.factor.shape[0]),
        "dtype": str(term.info.factor.dtype),
    }


def _reject_bad_archive(terms: list[dict[str, Any]]) -> None:
    """Re-run, on the manifest, the three refusals ``remember`` enforces.

    ``BayesMemory.remember`` refuses a repeated ``epoch_id``, a mixed
    estimator and a tempered term, and its docstring calls those three "worth
    the words they cost". None of them was checked here, and ``load_memory`` is
    a *second* way to build an archive -- it calls ``BayesMemory(archive=...)``
    directly, which validates nothing. Since this module's own premise is that
    the manifest is the reconstruction spec and therefore an editable text file,
    the bypass is reachable by the very mechanism the format documents.

    Measured before this check existed: a manifest with the same ``epoch_id``
    twice, one term ``include_logdet=True`` and one ``False``, loaded without
    complaint; ``audit()["estimator"]`` then reported ``("full", "none")`` for
    an archive holding both, because it reads ``archive[0]``; and ``remember``
    admitted further terms of whichever estimator happened to sit at index 0.

    Checked on the manifest rather than on the rebuilt terms so that a bad
    archive is refused before any array is read.
    """
    seen: set[str] = set()
    for entry in terms:
        epoch_id = entry["epoch_id"]
        if epoch_id in seen:
            raise StateValidationError(
                f"This archive holds epoch {epoch_id!r} more than once. Loading it "
                "would count that recording's data twice, narrowing the posterior "
                "with nothing to show for it. A memory built by `remember` cannot "
                "contain this unless duplicate=True was passed deliberately, so the "
                "manifest has most likely been edited or concatenated."
            )
        seen.add(epoch_id)

    estimators = {
        ("full" if entry["include_logdet"] else "gls", entry["noise_frozen_at"])
        for entry in terms
    }
    if len(estimators) > 1:
        raise StateValidationError(
            f"This archive mixes estimators {sorted(estimators)}. Generalized least "
            "squares and the full Gaussian likelihood (D21/D23) are different "
            "estimators; their sum is neither, and audit() would report only the "
            "first one it finds."
        )

    tempered = [
        entry["epoch_id"] for entry in terms if tuple(entry["prior_share"])[0] != 0
    ]
    if tempered:
        raise StateValidationError(
            f"Term(s) {tempered} carry a nonzero prior_share, but a streaming memory "
            "stores prior-free factors: log_posterior applies the prior exactly once, "
            "so a tempered term would apply it twice."
        )


def save_memory(memory, path: str | Path) -> None:
    """Write ``memory`` to ``path`` plus a manifest at ``path.json``.

    **The binary goes first, the manifest last, and that order is the commit.**
    The manifest is this format's reconstruction spec, so its presence is what
    says a readable archive exists. Written first -- as this did -- a failing
    ``tree_serialise_leaves`` left a manifest describing a file that was never
    created, and ``load_memory`` then died on a raw ``FileNotFoundError`` from
    equinox rather than on anything this module says. Written last, a crash
    mid-save leaves an orphan *binary*, which nothing reads, and the archive is
    simply absent rather than corrupt.

    Both writes are still separate operations, so this is crash-consistent, not
    atomic: a reader can catch the instant between them. That window is
    diagnosable -- the manifest is missing, which ``load_memory`` names -- where
    the reverse window was not.
    """
    path = Path(path)
    manifest = {
        "format_version": _FORMAT_VERSION,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "global_names": list(memory.factorization.global_names),
        "global_shapes": [list(shape) for shape in memory.factorization.global_shapes],
        "accumulated_rows": int(memory.accumulated.factor.shape[0]),
        "terms": [_describe(term) for term in memory.archive],
    }
    # Built before the write so a term this module cannot describe fails before
    # anything reaches disk.
    payload = json.dumps(manifest, indent=2)
    eqx.tree_serialise_leaves(path, memory)
    _manifest_path(path).write_text(payload)


def load_memory(path: str | Path, factorization: Factorization):
    """Read a memory back, refusing anything the manifest says it cannot be."""
    from rheplicant.inference.memory import BayesMemory

    path = Path(path)
    manifest_path = _manifest_path(path)
    if not manifest_path.exists():
        raise StateValidationError(
            f"No manifest at {manifest_path}. In this format the manifest is the "
            "reconstruction spec, not a sidecar: every static field, every dtype "
            "and the writer's x64 state live there, and equinox would otherwise "
            "silently take them from whatever template it was handed. A binary "
            "without its manifest is therefore unreadable rather than "
            "partially readable. save_memory writes the binary first and the "
            "manifest last, so this most likely means a save was interrupted."
        )
    manifest = json.loads(manifest_path.read_text())

    if manifest["format_version"] != _FORMAT_VERSION:
        raise StateValidationError(
            f"Archive format version {manifest['format_version']}, this rheplicant "
            f"writes {_FORMAT_VERSION}."
        )
    if not bool(jax.config.jax_enable_x64):
        raise StateValidationError(
            "This archive was written under jax_enable_x64=True but x64 is off here, "
            "so every leaf would be silently demoted to float32 -- which annihilates "
            'the quadratic form. Set jax.config.update("jax_enable_x64", True).'
        )
    for entry in manifest["terms"]:
        if entry["dtype"] != "float64":
            raise StateValidationError(
                f"Term {entry['epoch_id']!r} declares dtype {entry['dtype']}; the "
                "accumulation layer requires float64."
            )
    if list(factorization.global_names) != manifest["global_names"]:
        raise StateValidationError(
            f"This archive is over latents {manifest['global_names']}, but the "
            f"factorization supplied is over {list(factorization.global_names)}. The "
            "stored numbers are a quadratic form in a specific, ordered vector; "
            "reading them against a different one is not a rename, it is a different "
            "model."
        )
    if [list(s) for s in factorization.global_shapes] != manifest["global_shapes"]:
        raise StateValidationError(
            f"This archive's latent shapes {manifest['global_shapes']} do not match "
            f"the factorization's {[list(s) for s in factorization.global_shapes]}."
        )
    _reject_bad_archive(manifest["terms"])

    names = factorization.global_names
    shapes = factorization.global_shapes
    width = sum(int(jnp.zeros(shape).size) for shape in shapes)

    def _blank(rows: int) -> SqrtInfo:
        return SqrtInfo(
            factor=jnp.zeros((rows, width)),
            target=jnp.zeros(rows),
            offset=jnp.zeros(()),
            names=names,
            shapes=shapes,
        )

    template = BayesMemory(
        factorization=factorization,
        accumulated=_blank(manifest["accumulated_rows"]),
        archive=tuple(
            QuadraticLikelihood(
                info=_blank(entry["rows"]),
                epoch_id=entry["epoch_id"],
                n_observed=entry["n_observed"],
                exact=entry["exact"],
                support=None if entry["support"] is None else {
                    name: tuple(bounds) for name, bounds in entry["support"].items()
                },
                include_logdet=entry["include_logdet"],
                noise_frozen_at=entry["noise_frozen_at"],
                prior_share=tuple(entry["prior_share"]),
            )
            for entry in manifest["terms"]
        ),
    )
    return eqx.tree_deserialise_leaves(path, template)
