"""Exception hierarchy for rheplicant.

All framework errors derive from :class:`DirtError` so users can catch the
whole family with one ``except`` clause. Subclasses additionally derive from
the closest builtin (``ValueError`` / ``RuntimeError``) so generic handlers
keep working.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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


class LinearityRefused(ParameterSpaceError):
    """``check_linearity`` measured a departure from linearity, and refused.

    A SUBCLASS and not a new member of the family, deliberately: every
    ``except ParameterSpaceError`` already written against
    :func:`~rheplicant.inference.linear.check_linearity` keeps catching this,
    and the refusal's message is unchanged, so the
    ``pytest.raises(ParameterSpaceError, match=...)`` sites keep matching too.
    Nothing about the refusal itself is new.

    What is new is that the per-probe departures the message *renders* are
    also carried as NUMBERS.  Before this class the failing branch put them in
    a sentence and dropped them, while the passing branch -- where every value
    is 0.0 -- returned them structured, so the only path with something to say
    was the only path with nothing to read.  A consumer that wanted the
    numbers had to parse the prose, which is a mapping this package's own
    source would not defend.

    Attributes:
        errors: ``{scale: relative departure}`` at every probed scale, PASSING
            probes included -- the trend across scales is the diagnostic, and
            "departs at 1x and 1000x but not at 0.001x" is a different fault
            from "departs everywhere".
        rtol: the tolerance the comparison actually used, which is derived
            from the prediction's dtype when the caller passes none.
        failed: the scales that exceeded it, ascending -- a subset of
            ``errors``' keys, and the same tuple the message names.
    """

    def __init__(
        self,
        *args: object,
        errors: Mapping[float, float],
        rtol: float,
        failed: Sequence[float],
    ) -> None:
        super().__init__(*args)
        # Copied rather than aliased: the caller's ``errors`` is the same dict
        # ``check_linearity`` returns on the passing branch, and an exception
        # that shares mutable state with its raiser is a trap for whoever
        # catches it.
        self.errors = dict(errors)
        self.rtol = float(rtol)
        self.failed = tuple(float(scale) for scale in failed)


class LogSpaceUnavailable(ParameterSpaceError):
    """A quantity ``log`` cannot be taken of, where log space was being asked for.

    A SUBCLASS for the same reason :class:`LinearityRefused` is one: every
    ``except ParameterSpaceError`` already written keeps catching it and the
    message is unchanged. What the subclass buys is a NARROW catch.

    Discovering whether a latent has a log-linear block means asking
    :func:`~rheplicant.inference.loglinear.check_log_linearity` and reading a
    refusal as "no". Two refusals mean that — a departure from affinity
    (:class:`LinearityRefused`) and a prediction that is negative, zero or not
    finite (this one) — while the rest of ``ParameterSpaceError``'s family, a
    latent of integer dtype or a name the space never declared, mean the
    question was never asked. Catching the base class to classify would file
    those as "not log-linear" and route a broken declaration to a gradient
    block with nothing said, which is the shape of failure this package spends
    its refusals avoiding.
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
