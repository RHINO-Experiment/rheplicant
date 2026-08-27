"""Write D12's read-back fixture, with TODAY's code, for Wave D to read.

Run it, and commit what it writes::

    JAX_ENABLE_X64=1 .venv/bin/python tests/evidence/fixtures/make_d12_archives.py

**Why the files are committed rather than built by a fixture.** D12 hands the
evidence containers' arithmetic to bayesmith and keeps rheplicant's own classes
(``__check_init__``'s exception identity is raised at construction, base class
first, so subclassing cannot translate it). Its stated PRECONDITION is that a
read-back regression exists whose input was written *before* that switch: a
fixture that rebuilds the archive at test time would be written by the switched
code and could only ever say that the new writer agrees with the new reader.
The whole question is whether the switched code still reads what the unswitched
code wrote, and that needs bytes from before.

The plan says it in as many words: "Wave D 若发现该 fixture 未提交,那一波不得
开工."

**Two forms, because ``template_projections`` is where the format is
sharpest.** ``template_names = ()`` with ``None`` and ``template_names = ()``
with a length-zero array are the same claim to a reader and DIFFERENT pytrees
to equinox -- one has a leaf at that position and the other an empty subtree,
so a template built with the wrong one reads every later leaf from the wrong
offset. ``archive.py``'s module docstring is where that is argued;
``n_template_projections`` in the manifest is what settles it. A fixture with
only one of the two forms would leave the offset arithmetic untested in
exactly the case it was written for.

**Written under x64**, and the manifest records that, because ``load_memory``
refuses a mismatch rather than warning. ``tests/evidence`` is an x64 session,
so the regression that reads these back is in the same arithmetic they were
written in.
"""

from __future__ import annotations

import pathlib
import sys

import jax
import jax.numpy as jnp

HERE = pathlib.Path(__file__).resolve().parent

#: The two files this writes, and the shape each one pins. Named here so the
#: regression that reads them can import the list rather than repeat it -- a
#: second spelling of a filename is a second thing to go stale.
ARCHIVES: tuple[tuple[str, str], ...] = (
    ("d12_with_templates.rhep", "template_projections is a length-2 array"),
    ("d12_without_templates.rhep", "template_projections is None"),
)

#: The manifest beside each one. ``archive._manifest_path`` SUBSTITUTES the
#: suffix rather than appending it, so ``x.rhep``'s manifest is ``x.json``.
#: Spelled here because a regression that guessed ``x.rhep.json`` would report
#: a missing fixture as a missing manifest.
MANIFESTS: tuple[str, ...] = tuple(
    pathlib.Path(name).with_suffix(".json").name for name, _ in ARCHIVES
)


def factorization():
    """The same two-latent space ``tests/evidence/test_memory.py`` uses.

    Reused rather than invented: a fixture whose model is written twice is the
    defect this codebase pays for most often, and Wave D's regression will want
    to build a ``Factorization`` of its own to load against.
    """
    import numpyro.distributions as dist

    from rheplicant.inference.factorize import Factorization
    from rheplicant.inference.parameters import Bind, Latent, ParameterSpace

    latents = (
        Latent("depth", init=-0.5, prior=dist.Normal(0.0, 1.0)),
        Latent("width", init=1.0, prior=dist.Normal(0.0, 2.0)),
    )
    space = ParameterSpace(
        latents=latents,
        bindings=tuple(
            Bind(latent.name, into=lambda p, n=latent.name: getattr(p, n))
            for latent in latents
        ),
    )
    return Factorization(space)


def _term(epoch_id, *, projections):
    """One stored term with EVERY static field off its default.

    A term whose statics are all defaults round-trips correctly under the very
    bug this format exists to prevent -- ``tree_serialise_leaves`` taking them
    from the template -- so every one of them is deliberately unguessable.
    """
    from rheplicant.inference.compressed import QuadraticLikelihood
    from rheplicant.inference.sqrtinfo import SqrtInfo

    return QuadraticLikelihood(
        info=SqrtInfo(
            factor=jnp.array([[1.5, 0.25], [0.0, 0.75]]),
            target=jnp.array([0.5, -0.25]),
            offset=jnp.array(-3.25),
            names=("depth", "width"),
            shapes=((), ()),
        ),
        epoch_id=epoch_id,
        n_observed=777,
        exact=False,
        support={"depth": (-2.0, 2.0), "width": (-1.0, 3.0)},
        include_logdet=False,
        noise_frozen_at="gls",
        residual_chi2=jnp.array(7.5),
        template_projections=projections,
        residual_dof=13,
        template_names=(
            ("gain_ripple", "ground_pickup") if projections is not None else ()
        ),
        inputs=(("beam_model", "sha256:b3ee"), ("cal_solution", "sha256:0f17")),
    )


def build(*, projections):
    from rheplicant.inference.memory import BayesMemory

    memory = BayesMemory(factorization())
    return memory.remember(_term("night-042", projections=projections))


def main() -> int:
    if not jax.config.jax_enable_x64:
        print(
            "Refusing to write: this must run under x64.\n"
            "The manifest records the writer's x64 state and `load_memory` "
            "refuses a mismatch, and `tests/evidence` is an x64 session -- so a "
            "float32 fixture would be committed and then unreadable by the only "
            "suite that reads it.\n"
            "    JAX_ENABLE_X64=1 .venv/bin/python "
            "tests/evidence/fixtures/make_d12_archives.py",
            file=sys.stderr,
        )
        return 2

    from rheplicant.inference.archive import save_memory

    forms = {
        "d12_with_templates.rhep": jnp.array([1.25, -0.5]),
        "d12_without_templates.rhep": None,
    }
    for name, projections in forms.items():
        save_memory(build(projections=projections), HERE / name)
        # `_manifest_path` SUBSTITUTES the suffix rather than appending it,
        # so the manifest of `x.rhep` is `x.json` and not `x.rhep.json`.
        print(f"wrote {name} and {pathlib.Path(name).with_suffix('.json').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
