"""Exception hierarchy for rheplicant.

All framework errors derive from :class:`DirtError` so users can catch the
whole family with one ``except`` clause. Subclasses additionally derive from
the closest builtin (``ValueError`` / ``RuntimeError``) so generic handlers
keep working.
"""

from _rheplicant_bootstrap.errors import DirtError


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


class DataIngestionError(DirtError, ValueError):
    """A data file could not be read, or its contents contradict what the
    caller declared about them.

    Distinct from :class:`StateValidationError`, which covers *structural*
    problems with an in-memory State — wrong ndim, wrong dtype, bad key types.
    A value given in a unit other than the one the caller declared, and a
    record that violates the file's own format rules, are neither: nothing is
    wrong with the shape of what was read, only with what it means. Both would
    otherwise propagate as a finite, correctly-shaped, wrong answer.
    """

class AssemblyError(DirtError, ValueError):
    """A provided operator set cannot be assembled on the signal graph."""


class AmbiguousNodeError(AssemblyError):
    """A node id was used as an address, but it holds more than one operator.

    Raised by :class:`~rheplicant.core.graph.Assembly`'s ``__getitem__`` and
    ``replace_node`` for a ``many=True`` node carrying several instances.
    (Those two are literals rather than ``:meth:`` roles: ``__getitem__`` is a
    dunder, which ``automodule`` emits no target for, so the role would be a
    nitpicky-build warning rather than a link. Inside ``graph.py`` the same
    text was an UNQUALIFIED role and was never checked -- moving the class
    here is what exposed it.) Answering with one of them (or with the
    fold over all of them) would silently pick a different operator than the
    caller means -- and, through ``replace_node``, silently delete the
    siblings. The message names every instance id instead.
    """
