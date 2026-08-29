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

**The archive is written as its own pytree, beside the memory, and that is not
redundancy.** ``BayesMemory`` holds its terms behind an opaque leaf so that a
ten-thousand-epoch campaign costs the same to flatten as a one-epoch one (see
:mod:`rheplicant.inference.memory`). ``eqx.tree_serialise_leaves`` does not
object to a leaf it cannot serialise -- measured on equinox 0.13.8, it skips it,
writes a shorter file, and ``tree_deserialise_leaves`` returns the template's
arrays in its place. Every stored factor would come back as the zeros
``load_memory`` builds its template from: not an error, not a warning, just a
campaign that has forgotten its evidence and still reports the right epoch
count. So the pair ``(memory, tuple(memory.archive))`` is serialised
explicitly, and ``_FORMAT_VERSION`` moved to 2 because the byte layout changed
-- a version-1 file read by this code would deserialise the memory and then run
off the end of the file.

Version 3 adds section 9.3's per-epoch residual summary and section 9.5's input
provenance. Both are on the term because both must outlive the recording: the
summary is computed where the data still exists, and the provenance is what the
memory's conditional-independence refusal reads. ``residual_dof``,
``template_names`` and ``inputs`` are static and therefore go in the manifest --
``eqx.tree_serialise_leaves`` would take them from whatever template it was
handed, which for ``inputs`` means a reloaded campaign that has forgotten which
nights shared a calibration solution and will cheerfully sum them.

``template_projections`` needs one more field than its name suggests.
``template_names = ()`` with ``None`` and ``template_names = ()`` with a
length-zero array are the same claim to a reader and **different pytrees** to
equinox: one has a leaf at that position and the other has an empty subtree, so
a template built with the wrong one reads every later leaf from the wrong
offset. ``QuadraticLikelihood.__check_init__`` refuses the empty-array spelling
outright, and ``n_template_projections`` records ``None`` or a length so the
template is reconstructed from the file rather than from a convention.
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

#: 3 since section 9.3's residual summary and section 9.5's input provenance
#: joined the term -- see the module docstring. The byte layout changed with
#: them: `residual_chi2` and `template_projections` are dynamic, so a version-2
#: file read by this code would run off the end.
_FORMAT_VERSION = 3


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
        "residual_dof": int(term.residual_dof),
        "template_names": list(term.template_names),
        # Not derivable from `template_names`: see the module docstring on the
        # None-versus-empty-array pytree split.
        "n_template_projections": (
            None
            if term.template_projections is None
            else int(jnp.asarray(term.template_projections).shape[0])
        ),
        "inputs": [list(pair) for pair in term.inputs],
    }


def _reject_bad_archive(terms: list[dict[str, Any]], represents: Any) -> None:
    """Re-run, on the manifest, the four refusals ``remember`` enforces.

    ``BayesMemory.remember`` refuses a repeated ``epoch_id``, a mixed
    estimator, a tempered term and a shared input product. None of them was
    checked here, and ``load_memory`` is a *second* way to build an archive --
    it calls ``BayesMemory(archive=...)`` directly, which validates nothing.
    Since this module's own premise is that the manifest is the reconstruction
    spec and therefore an editable text file, the bypass is reachable by the
    very mechanism the format documents.

    Measured before the first three existed: a manifest with the same
    ``epoch_id`` twice, one term ``include_logdet=True`` and one ``False``,
    loaded without complaint; ``audit()["estimator"]`` then reported
    ``("full", "none")`` for an archive holding both, because it reads
    ``archive[0]``; and ``remember`` admitted further terms of whichever
    estimator happened to sit at index 0.

    **The fourth arrived after the other three and was missed.** Section 9.5's
    rule -- two epochs sharing an input-product hash are not conditionally
    independent unless the product is represented among the globals -- is the
    one this module's own docstring gives as the reason ``inputs`` is in the
    manifest at all: a reloaded campaign that has forgotten which nights shared
    a calibration solution "will cheerfully sum them". Measured with a
    one-character edit and no ``shared_inputs=`` anywhere: written,
    ``[[['beam_map', 'sha:abc']], [['beam_map', 'sha:def']]]``; edited,
    ``[[['beam_map', 'sha:abc']], [['beam_map', 'sha:abc']]]``; ``load_memory``
    ACCEPTED, while ``remember`` refused the identical pair and the *duplicate*
    rule fired on the very same edited file. Concatenating two runs' manifests
    reaches that state with no editing at all, which is why this is a
    correctness check and not a tampering one.

    ``represents`` comes from the caller's :class:`Factorization` rather than
    from the manifest, and it has to: whether a shared product is modelled is a
    property of the model being loaded *against*, not of the archive. The
    comparison is on the ``(product, hash)`` pair for the reason
    :func:`~rheplicant.inference.memory._reject_shared_inputs` gives -- a beam
    map re-measured between nights is a different beam map, and refusing on the
    name alone would refuse the normal campaign.

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

    # Last, for `reject_bad_term`'s reason: an epoch present twice trips this
    # rule as well as the duplicate one, and the duplicate message is the
    # actionable half of that pair.
    modelled = set(represents)
    first_use: dict[tuple[str, str], str] = {}
    for entry in terms:
        for pair in entry["inputs"]:
            product, digest = tuple(pair)
            if product in modelled:
                continue
            earlier = first_use.get((product, digest))
            if earlier is not None:
                raise StateValidationError(
                    f"This archive holds epochs {earlier!r} and "
                    f"{entry['epoch_id']!r} that share input product {product!r} "
                    f"(hash {digest!r}), and a memory sums its factors as though "
                    "the epochs were conditionally independent. They are not: one "
                    "calibration solution, one beam map or one flag table applied "
                    "to several nights is a shared error with no variance at all, "
                    "so per-epoch chi-square is right, split-half agrees, "
                    "leave-one-out agrees, and the answer is wrong -- measured at "
                    "52.6 sigma by N = 640 with every diagnostic clean. Model the "
                    "product as a global latent and load against "
                    f"Factorization(represents={{{product!r}: (...)}}). A memory "
                    "built by `remember` cannot contain this unless "
                    "shared_inputs=True was passed deliberately, so the manifest "
                    "has most likely been edited or concatenated -- and a "
                    "deliberately shared campaign has to declare itself again "
                    "here, because the archive is all that is left of it."
                )
            first_use[(product, digest)] = entry["epoch_id"]


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

    **The memory is checked before its terms are.** A foreign *term* has been
    refused by name since this module existed; a foreign *memory* was refused by
    ``AttributeError: 'ChainMemory' object has no attribute 'accumulated'``,
    from the manifest line that reads a bag's running QR. That names an
    implementation detail of the class it was not given, says nothing about what
    this format is, and offers no remedy.
    """
    from rheplicant.inference.memory import BayesMemory

    if not isinstance(memory, BayesMemory):
        raise StateValidationError(
            f"save_memory writes a BayesMemory and this is a "
            f"{type(memory).__name__}. The manifest is a reconstruction spec for "
            "a bag: it records `accumulated_rows`, the running QR a chain "
            "deliberately does not have, and `load_memory` returns a BayesMemory "
            "-- so there is no reading of this format under which it could "
            "return the other one. A chain also carries a live transition, and "
            "a HyperTransition's builder is a Python callable with no textual "
            "form for a manifest to record, so a reloaded chain would run a "
            "different model against the same numbers. Keep a chain in the "
            "process that built it."
        )
    path = Path(path)
    foreign = [
        term.epoch_id
        for term in memory.archive
        if not isinstance(term, QuadraticLikelihood)
    ]
    if foreign:
        raise StateValidationError(
            f"Term(s) {foreign} are not QuadraticLikelihood, and this format can "
            "only describe that one. The manifest is a reconstruction spec: it "
            "records every static field so that eqx.tree_serialise_leaves cannot "
            "take one from a template instead. A tier whose static half is a "
            "Python callable -- RawLikelihood's `predict`, "
            "ReducedBasisLikelihood's coefficient map -- has no textual form to "
            "record, so a reloaded term would evaluate a different model against "
            "the same numbers, with no error and no warning. Compress to a "
            "quadratic tier, or keep the memory in the process that built it."
        )
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
    # The terms travel beside the memory, not inside it: they hang off an opaque
    # leaf that equinox would skip in silence. See the module docstring.
    eqx.tree_serialise_leaves(path, (memory, tuple(memory.archive)))
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
        older = manifest["format_version"] < _FORMAT_VERSION
        raise StateValidationError(
            f"Archive format version {manifest['format_version']}, this rheplicant "
            f"writes {_FORMAT_VERSION}. "
            + (
                "This reader cannot be made tolerant of it: the versions differ "
                "in BYTE LAYOUT, not only in manifest fields, so reading an "
                "older binary through this template does not give a smaller "
                "answer -- it runs off the end or reads leaves at the wrong "
                "offset. Check out the rheplicant that wrote it, load the "
                "archive there, and re-archive under this version. "
                if older
                else "This archive was written by a NEWER rheplicant than the "
                "one reading it. Upgrade rather than converting: an older "
                "reader cannot know what the newer format added. "
            )
            + "There is no in-place conversion, and that is the constraint any "
            "future bump has to plan around -- this layer's premise is that "
            "the raw data is gone, so a version it cannot read is work it "
            "cannot recover. A bump therefore ships with a converter, or it "
            "does not ship."
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
    _reject_bad_archive(manifest["terms"], factorization.represents)

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
                # A zero of the right shape and dtype, for the same reason
                # `_blank` builds one: equinox reads the arrays out of the
                # binary and takes everything else from here.
                residual_chi2=jnp.zeros(()),
                template_projections=(
                    None
                    if entry["n_template_projections"] is None
                    else jnp.zeros(entry["n_template_projections"])
                ),
                residual_dof=entry["residual_dof"],
                template_names=tuple(entry["template_names"]),
                inputs=tuple(tuple(pair) for pair in entry["inputs"]),
            )
            for entry in manifest["terms"]
        ),
    )
    with path.open("rb") as handle:
        restored, terms = eqx.tree_deserialise_leaves(
            handle, (template, tuple(template.archive))
        )
        consumed = handle.tell()
    remaining = path.stat().st_size - consumed
    if remaining:
        raise StateValidationError(
            f"This manifest describes {consumed} bytes and the binary holds "
            f"{consumed + remaining}: {remaining} bytes were never read. The "
            "manifest is this format's reconstruction spec, so a manifest that "
            "stops early is not a smaller answer -- it is a different archive's "
            "spec, and equinox has just handed back a memory missing whatever "
            "those bytes held. Measured on this repo (ledger D39): pairing a "
            "with-templates binary against a without-templates manifest "
            "returned `template_projections=None` for every term, with every "
            "other field correct and no error. Pair each binary with the "
            "manifest written beside it."
        )
    # `restored` carries the template's own terms -- its archive is the leaf
    # equinox stepped over -- so the deserialised ones are put back by hand.
    # Rebuilding through the constructor rather than eqx.tree_at because the
    # archive is not a pytree node here, which is the entire point of it.
    return BayesMemory(
        factorization=restored.factorization,
        accumulated=restored.accumulated,
        archive=terms,
        coefficients=restored.coefficients,
        basis=restored.basis,
    )
