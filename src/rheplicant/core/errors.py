"""Exception hierarchy for rheplicant.

All framework errors derive from :class:`DirtError` so users can catch the
whole family with one ``except`` clause. Subclasses additionally derive from
the closest builtin (``ValueError`` / ``RuntimeError``) so generic handlers
keep working.
"""


class DirtError(Exception):
    """Base class for all rheplicant errors."""


class StateValidationError(DirtError, ValueError):
    """A State (or one of its containers) was constructed with invalid contents.

    Raised only for *structural* problems (wrong ndim, wrong dtype, bad key
    types) — never for traced array *values*, so validation stays jit-safe.
    """


class MissingKeyError(DirtError, RuntimeError):
    """An operator needed randomness but ``State.key`` is ``None``.

    Fix: construct the state with ``key=jax.random.key(seed)``.
    """


class PipelineError(DirtError, ValueError):
    """A Pipeline was misconfigured (empty, bad stage type, name collision...)."""


class ParameterSpaceError(DirtError, ValueError):
    """A parameter space was declared inconsistently.

    Covers both halves of the declaration: latents that nothing binds, a
    binding naming a latent that was never declared, two bindings writing the
    same leaf, a produced value whose shape does not fit its target leaf.
    Every one of these would otherwise yield a finite, correctly-shaped, wrong
    inference — so they are errors, not warnings.
    """
