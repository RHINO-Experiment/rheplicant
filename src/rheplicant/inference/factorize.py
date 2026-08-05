"""Which latents survive an epoch, and which are integrated out inside it.

A :class:`~rheplicant.inference.parameters.ParameterSpace` says what is
inferred and how it reaches the pipeline. A :class:`Factorization` adds the
one thing a streaming analysis needs on top: over what extent of data each
quantity is constant, which is what decides whether it is sampled against the
whole campaign or integrated away one epoch at a time.

It is derived, not declared twice. The user writes one space, tagging latents
with ``scope=``; this class partitions it and exposes the **global view** --
names, shapes and priors, and nothing that resolves against a pipeline. That
matters because :class:`~rheplicant.inference.memory.BayesMemory` has no
pipeline by construction: the whole point is that the raw data and the forward
evaluation are gone. Handing the memory a ``ParameterSpace`` would either
require dead bindings into a pipeline nobody will run, or open a sample site
for a per-epoch latent that compression already integrated -- D14's own named
failure, an unreached latent sampling happily and returning its prior, while a
marginalised copy of the same nuisance sits inside every stored term.
"""

from collections.abc import Callable, Mapping
from typing import Any

import equinox as eqx

from rheplicant.core.errors import ParameterSpaceError
from rheplicant.inference.parameters import Latent, ParameterSpace


class Factorization(eqx.Module):
    """A parameter space split by scope, plus what the non-global scopes need.

    Attributes:
        space: the single declaration everything is derived from.
        linked: ``{latent name: transition}`` for every ``scope="linked"``
            latent. The transition object is consumed by the chain filter; this
            class only checks that one exists for each linked latent and none
            for anything else.
        hyper: ``{per-epoch latent: (global latent names, builder)}``. The
            builder receives those globals' values and returns the per-epoch
            latent's prior, which is how a hierarchical ``pi(phi_e | psi)`` is
            expressed. Its presence means the epoch's nuisance cannot be
            integrated at compression time (the hyperparameter would be
            frozen); the joint block is stored instead and the integral is done
            at evaluation.
    """

    space: ParameterSpace
    linked: Mapping[str, Any] = eqx.field(static=True, default_factory=dict)
    hyper: Mapping[str, tuple[tuple[str, ...], Callable]] = eqx.field(
        static=True, default_factory=dict
    )

    def __check_init__(self):
        by_scope = {"global": [], "per_epoch": [], "linked": []}
        for latent in self.space.latents:
            by_scope[latent.scope].append(latent)

        if not by_scope["global"]:
            raise ParameterSpaceError(
                "A Factorization needs at least one global latent -- otherwise every "
                "quantity is integrated away inside its own epoch and there is "
                "nothing to accumulate across the campaign."
            )
        for latent in by_scope["global"]:
            if latent.prior is None:
                raise ParameterSpaceError(
                    f"Global latent {latent.name!r} has no prior. The accumulated "
                    "terms are prior-free by construction, so the space is where the "
                    "prior lives; a parameter with no prior has no place in a "
                    "posterior."
                )
        for latent in by_scope["per_epoch"]:
            if latent.prior is None and latent.name not in self.hyper:
                raise ParameterSpaceError(
                    f"Per-epoch latent {latent.name!r} has no prior. It is integrated "
                    "out once per epoch, so its prior is part of the model, not an "
                    "optional regulariser: without it the epoch's likelihood is not "
                    "defined. Give it a prior, or a `hyper` entry."
                )
        declared_linked = {latent.name for latent in by_scope["linked"]}
        missing = declared_linked - set(self.linked)
        if missing:
            raise ParameterSpaceError(
                f"Linked latent(s) {sorted(missing)} declare no transition. A linked "
                "latent is a Markov chain across epochs; without a transition there "
                "is no chain, and the accumulation would silently fall back to "
                "treating it as independent per epoch."
            )
        stray = set(self.linked) - declared_linked
        if stray:
            raise ParameterSpaceError(
                f"A transition was given for {sorted(stray)}, which is not declared "
                'scope="linked". Set the scope, or drop the transition.'
            )

        global_names = {latent.name for latent in by_scope["global"]}
        per_epoch_names = {latent.name for latent in by_scope["per_epoch"]}
        for target, (sources, _) in self.hyper.items():
            if target not in per_epoch_names:
                raise ParameterSpaceError(
                    f"hyper[{target!r}] describes a hierarchical prior, which only a "
                    'per-epoch latent has. Set scope="per_epoch", or drop the entry.'
                )
            outside = [name for name in sources if name not in global_names]
            if outside:
                raise ParameterSpaceError(
                    f"hyper[{target!r}] is built from {outside}, which are not global "
                    "latents. A hyperparameter shared across epochs must itself be "
                    "global, or every epoch would integrate against a different, "
                    "unrecorded prior."
                )

    # -------------------------------------------------------- the global view --

    def _of_scope(self, scope: str) -> tuple[Latent, ...]:
        return tuple(latent for latent in self.space.latents if latent.scope == scope)

    @property
    def global_names(self) -> tuple[str, ...]:
        """Names of the latents the memory accumulates over, in declared order."""
        return tuple(latent.name for latent in self._of_scope("global"))

    @property
    def global_shapes(self) -> tuple[tuple[int, ...], ...]:
        """Shapes of those latents, in the same order."""
        return tuple(latent.init.shape for latent in self._of_scope("global"))

    @property
    def global_priors(self) -> dict[str, Any]:
        """``{name: prior}`` for the global latents -- the campaign's only prior."""
        return {latent.name: latent.prior for latent in self._of_scope("global")}

    @property
    def per_epoch_names(self) -> tuple[str, ...]:
        """Names of the latents integrated out inside each epoch."""
        return tuple(latent.name for latent in self._of_scope("per_epoch"))

    @property
    def linked_names(self) -> tuple[str, ...]:
        """Names of the latents that form a Markov chain across epochs."""
        return tuple(latent.name for latent in self._of_scope("linked"))
