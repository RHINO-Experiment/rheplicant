"""resources.bases: a SeparableBasis whose sample count comes from the grid.

``n`` is never written. ``radio/t_sys.py`` states the failure it prevents --
"a basis built for another band would return a smooth, plausible, wrong
temperature" -- and taking ``n`` from ``observation.time.grid`` /
``observation.freq.grid`` makes that structurally impossible rather than
merely discouraged. For the same reason a ``file:`` route for a design matrix
is refused: schema §7 names it, and open question 11.8 recommends keeping the
refusal, because a copied ``built_for:`` provenance block is exactly what a
copied basis comes with.

One orientation hazard the package states and nothing can catch
(``core/basis.py:63-77``): ``T = time @ coeff @ freq.T``, so with
``n_time == n_freq`` and ``n_k == n_j`` a swapped pair of design matrices is
shape-legal and returns the transpose of the intended field. Writing the two
axes under their own keys is the only protection there is.
"""

from typing import Any

from rheplicant.config.context import ResolutionContext
from rheplicant.config.errors import ConfigError
from rheplicant.config.resources import register_kind
from rheplicant.config.units import canonical_unit
from rheplicant.config.values import ResolvedValue, register_form
from rheplicant.core.basis import BASIS_KINDS, SeparableBasis, basis_matrix


def _axis_matrix(axis: str, spec: Any, context: ResolutionContext):
    grid = {"time": context.time, "freq": context.freq}[axis]
    if not isinstance(spec, dict):
        raise ConfigError(f"bases: {axis} must be a mapping with kind and n_basis.")
    # Ordering invariant: the two specific refusals below (file, n) must run
    # BEFORE the generic unknown-keys sweep. Both "file" and "n" are unknown to
    # the {kind, n_basis} vocabulary, so if the sweep ran first it would catch
    # them as plain unknown keys and the caller would get "does not take
    # ['file']" instead of the reason a file route and a written n are each
    # refused for. Two tests defend this ordering by wording alone.
    if "file" in spec:
        raise ConfigError(
            f"bases: {axis} declares a file. A design matrix does not come from a "
            "file here: it is built from {kind, n_basis} with n taken from the run's "
            "own grid, because a basis built for another band returns a smooth, "
            "plausible, wrong temperature and nothing downstream can tell. The kinds "
            f"are {list(BASIS_KINDS)}. If you genuinely need a family that is not "
            "there, build the matrices through the python: hatch and hand them to "
            "SeparableBasis -- nothing below that line cares where the columns came "
            "from, and the hatch states its own cost."
        )
    if "n" in spec:
        raise ConfigError(
            f"bases: {axis} writes 'n'. n is never written -- it is len(observation."
            f"{axis}.grid), which is what makes a basis built for another grid "
            "impossible to declare. Remove it."
        )
    unknown = sorted(set(spec) - {"kind", "n_basis"})
    if unknown:
        raise ConfigError(f"bases: {axis} does not take {unknown}; it takes kind, n_basis.")
    for required in ("kind", "n_basis"):
        if required not in spec:
            raise ConfigError(f"bases: {axis} requires {required!r}.")
    if spec["kind"] not in BASIS_KINDS:
        raise ConfigError(
            f"bases: {axis} asks for kind={spec['kind']!r}; this package builds "
            f"{list(BASIS_KINDS)}. Each is a different claim about the quantity -- a "
            "Fourier basis says it is periodic on this axis, a polynomial one that it "
            "is smooth -- so the nearest-sounding name is not a safe guess. 'legendre' "
            "and 'polynomial' span exactly the same functions and differ only in "
            "conditioning (measured at n=32, n_basis=16: cond 7.86 against 2.81e+05)."
        )
    if grid is None:
        raise ConfigError(
            f"bases: {axis} needs observation.{axis}.grid, which this run does not "
            "declare -- that grid is where n comes from."
        )
    return basis_matrix(spec["kind"], n=int(grid.shape[0]), n_basis=int(spec["n_basis"]))


@register_kind("bases")
def build_basis(name: str, spec: dict, context: ResolutionContext) -> SeparableBasis:
    """Build one ``SeparableBasis`` from two per-axis declarations."""
    unknown = sorted(set(spec) - {"time", "freq"})
    if unknown:
        raise ConfigError(
            f"{name}: does not take {unknown}; a basis has a time axis and a freq axis."
        )
    for axis in ("time", "freq"):
        if axis not in spec:
            raise ConfigError(
                f"{name}: requires both 'time' and 'freq'. SeparableBasis takes one "
                "design matrix per axis and there is no default for either -- a basis "
                "that is constant on an axis is {kind: legendre, n_basis: 1}, which "
                "says so."
            )
    return SeparableBasis(
        time=_axis_matrix("time", spec["time"], context),
        freq=_axis_matrix("freq", spec["freq"], context),
    )


@register_form("basis_fit")
def _basis_fit(node: dict, context: ResolutionContext, modifiers: dict) -> ResolvedValue:
    """Least-squares coefficients of a field on a named basis.

    ``SeparableBasis.fit`` is the package's own solver (``core/basis.py:381``),
    so the coefficients this returns are the ones ``BasisTemperatureOperator``
    would have to be given to reproduce the field.
    """
    from rheplicant.config.refs import resolve_reference
    from rheplicant.config.values import resolve_value

    spec = node["basis_fit"]
    # The three checks below used to be one combined refusal; split so a
    # caller who mistyped one key sees which one rather than the whole
    # {basis, field} shape echoed back at them. Each keeps the expects-shape
    # sentence, because that is the one thing every variant still needs to say.
    expects = "basis_fit: expects {basis: {ref: resources.bases.<name>}, field: <value node>}"
    if not isinstance(spec, dict):
        raise ConfigError(f"{expects}; got {type(spec).__name__}: {spec!r}.")
    unknown = sorted(set(spec) - {"basis", "field"})
    if unknown:
        raise ConfigError(f"{expects}; does not take {unknown}.")
    missing = sorted({"basis", "field"} - set(spec))
    if missing:
        raise ConfigError(f"{expects}; missing {missing}.")
    reference = spec["basis"]
    if not isinstance(reference, dict) or "ref" not in reference:
        raise ConfigError(
            "basis_fit: 'basis' must be a {ref: resources.bases.<name>}. The basis is "
            "shared with whatever else fits on it, and a reference is what makes it "
            "the same object rather than a second one built for the same grid."
        )
    basis = resolve_reference(reference["ref"], context)
    if not isinstance(basis, SeparableBasis):
        raise ConfigError(
            f"basis_fit: {reference['ref']!r} is {type(basis).__name__}, not SeparableBasis."
        )
    field = resolve_value(spec["field"], context).value
    # canonical_unit(), not convert_to_canonical(): this only LABELS the fit's
    # result, it does not scale it. That is safe only because every token in
    # ACCEPTED_UNITS is today an identity conversion (factor 1, offset 0) --
    # refs._delivered() is where an actual conversion happens for other forms.
    # If a non-identity unit is ever added to the table, this line must switch
    # to converting the field (or the fitted coefficients) through it too.
    unit = canonical_unit(modifiers["unit"]) if "unit" in modifiers else None
    return ResolvedValue(basis.fit(field), unit, "basis_fit", modifiers)
