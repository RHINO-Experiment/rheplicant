"""resources.s_params: reflection coefficients, from a file or from rhino-cal.

``z0`` appears under ``kind: termination`` and nowhere else, and that is not an
oversight: ``Touchstone.z0`` is parsed and never read by any other module,
while ``termination_gamma(z0=)`` **is** read. A key exists where it is
consumed.

``flipped:`` is recorded even though the WIRING cannot be checked -- whether
the device really was connected to the VNA reversed is not something any
config layer can verify after the fact. The PLUMBING is a different claim
and this module's own tests pin it directly: ``read_touchstone`` applies the
port reversal *into* ``s`` before ``Touchstone`` is even constructed, so a
2-port fixture with distinct s11/s22 sweeps proves ``flipped:`` written here
actually reaches ``read_touchstone(flipped=)``.

One column-order hazard worth carrying: a Touchstone 2-port data row is
``freq S11 S21 S12 S22``. The second pair is **S21, not S12**. The module
docstring calls it "the single most likely thing to get wrong here"; this
layer never reorders, it only names a component and lets the reader answer.

``kind: touchstone`` reads its file through the PUBLIC ``file:`` value node
(``resolve_value({"file": file_spec}, context)``), not a private reach into
``files._READERS``. That is only possible because ``format: touchstone`` is
registered with ``array=False``: the reader returns a
:class:`~rheplicant.radio.touchstone.Touchstone`, not an array, and
``files.py``'s ``file:`` form hands an ``array=False`` reader's result back
unwrapped -- no ``jnp.asarray``, which would either mangle or refuse a
dataclass -- and refuses any modifiers on that node, because a modifier
describes what an array's numbers ARE and a Touchstone is not one.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np

from rheplicant.config.context import ResolutionContext
from rheplicant.config.derive import register_derivation
from rheplicant.config.errors import ConfigError
from rheplicant.config.files import register_reader
from rheplicant.config.refs import resolve_reference
from rheplicant.config.resources import check_unknown_keys, register_kind
from rheplicant.config.units import canonical_unit
from rheplicant.config.values import ResolvedValue, resolve_value

S_PARAM_KINDS: tuple[str, ...] = ("touchstone", "termination", "cable")
COMPONENTS: tuple[str, ...] = ("s11", "s12", "s21", "s22")
TERMINATIONS: tuple[str, ...] = ("open", "short", "matched", "resistive")

#: kind: termination -> every key its branch of build_s_param reads, derived
#: from its rhino_cal_jax.termination_gamma call. Checked by
#: :func:`check_unknown_keys`, AFTER the richer, kind-specific refusals
#: already in this module -- the termination-validity check, the
#: observation.freq.grid gate and the resistive-needs-impedance gate -- so
#: this sweep only ever reports the plain typo none of those messages is
#: about. The same ordering kinds/beams.py and kinds/projectors.py
#: established for their own gates.
_TERMINATION_KEYS: frozenset[str] = frozenset({"kind", "termination", "z0", "impedance", "n"})

#: kind: cable -> every key its branch reads, derived from its
#: rhino_cal_jax.cable_gamma call. Checked AFTER the richer refusals -- the
#: behind: shape gate and the observation.freq.grid requirement.
_CABLE_KEYS: frozenset[str] = frozenset({"kind", "behind", "length", "velocity_factor", "loss"})


@register_reader("touchstone", frozenset({"flipped"}), array=False)
def _read_touchstone_file(path, spec: dict):
    """``radio/touchstone.py:372`` ``read_touchstone``, into Plan 1A's table.

    ``array=False``: the return value is a
    :class:`~rheplicant.radio.touchstone.Touchstone`, not an array, so
    ``files.py``'s ``file:`` form must not ``jnp.asarray`` it and must refuse
    any modifiers written on the node that names it -- neither a unit, a
    part, nor a scale describes a dataclass of three fields.
    """
    from rheplicant.radio.touchstone import read_touchstone

    return read_touchstone(path, flipped=bool(spec.get("flipped", False)))


def _require_cal(name: str):
    try:
        import rhino_cal_jax
    except ImportError as exc:
        raise ConfigError(
            f"{name} needs rhino-cal-jax, the optional 'cal' extra. It is not on PyPI, "
            "so the extra names the requirement rather than resolving it, and the "
            "branch is not decoration -- the default branch carries the numpy pipeline "
            "and no pyproject.toml at all:\n"
            '  uv pip install "rhino-cal-jax @ '
            'git+https://github.com/RHINO-Experiment/rhino-cal@feat/rhino-cal-jax"\n'
            "Checked here rather than at first use, because the alternative is a "
            "40-minute run that fails at its first import."
        ) from exc
    return rhino_cal_jax


def _dimensioned(
    spec: dict, key: str, context: ResolutionContext, name: str, unit: str, what: str
) -> float:
    """Resolve ``spec[key]`` and require it to be in ``unit``, or unitless.

    Mirrors ``kinds/projectors.py``'s own ``_angle``: a value node that
    carries no unit token at all is trusted to already be in canonical
    units -- the same trust every bare-number value node gets elsewhere in
    this grammar -- but one that DOES carry a unit token must carry the
    right one, so that ``{value: 50, unit: "K"}`` on ``z0:`` is refused
    rather than silently treated as fifty ohms.
    """
    resolved = resolve_value(spec[key], context)
    if resolved.unit is not None and resolved.unit.canonical != unit:
        raise ConfigError(f"{name}: {key} must be {what}, got {resolved.unit.canonical!r}.")
    return float(resolved.value)


@register_kind("s_params")
def build_s_param(name: str, spec: dict, context: ResolutionContext) -> Any:
    """Build one reflection-coefficient array, discriminated on ``kind``."""
    kind = spec.get("kind")
    if kind not in S_PARAM_KINDS:
        raise ConfigError(
            f"{name}: kind={kind!r}; the s-parameter kinds are {list(S_PARAM_KINDS)}."
        )
    if kind == "touchstone":
        return _from_touchstone(name, spec, context)
    cal = _require_cal(name)
    if kind == "termination":
        return _termination(name, spec, context, cal)
    return _cable(name, spec, context, cal)


def _from_touchstone(name: str, spec: dict, context: ResolutionContext):
    from rheplicant.radio.touchstone import interpolate_onto

    unknown = sorted(set(spec) - {"kind", "file", "component", "flipped",
                                  "allow_extrapolation", "onto"})
    if unknown:
        extra = ""
        if "z0" in unknown:
            extra = (
                " z0 belongs to kind: termination and nowhere else: Touchstone.z0 is "
                "parsed and never read by any other module, while termination_gamma("
                "z0=) is read, so the key exists where it is consumed."
            )
        raise ConfigError(f"{name}: kind: touchstone does not take {unknown}.{extra}")
    component = spec.get("component", "s11")
    if component not in COMPONENTS:
        raise ConfigError(
            f"{name}: component={component!r}; a Touchstone file carries "
            f"{list(COMPONENTS)}. Note the data-row order for a 2-port file is "
            "freq S11 S21 S12 S22 -- the second pair is S21, not S12."
        )
    file_spec = dict(spec.get("file") or {})
    file_spec.setdefault("format", "touchstone")
    if "flipped" in spec:
        file_spec["flipped"] = spec["flipped"]
    if "path" not in file_spec:
        raise ConfigError(f"{name}: kind: touchstone requires file: {{path: ..., ...}}.")
    # The PUBLIC route: format: touchstone is registered with array=False
    # (files.py hands its result back unwrapped, with no jnp.asarray, and
    # refuses any modifiers on the node), so resolve_value gives back the
    # Touchstone object itself -- the same door every other file: reference
    # in this package goes through, rather than a private reach into
    # files._READERS that bypassed the hash check and the format's own
    # unknown-key sweep.
    touchstone = resolve_value({"file": file_spec}, context).value
    if spec.get("onto", "freq") != "freq":
        raise ConfigError(
            f"{name}: onto={spec.get('onto')!r}; the only grid to interpolate onto is "
            "'freq', the run's own frequency axis."
        )
    if context.freq is None:
        raise ConfigError(f"{name}: needs observation.freq.grid to interpolate onto.")
    return jnp.asarray(
        interpolate_onto(
            np.asarray(context.freq),
            touchstone,
            component=component,
            allow_extrapolation=bool(spec.get("allow_extrapolation", False)),
        )
    )


def _termination(name: str, spec: dict, context: ResolutionContext, cal):
    from rheplicant.config.symbols import resolve_extent

    kind = spec.get("termination")
    if kind not in TERMINATIONS:
        raise ConfigError(
            f"{name}: termination={kind!r}; they are {list(TERMINATIONS)}."
        )
    if context.freq is None:
        raise ConfigError(
            f"{name}: kind: termination needs observation.freq.grid. n: defaults to "
            "the shape symbol 'n_freq', which resolves to 0 with no frequency grid "
            "declared -- termination_gamma(n_freq=0) is a silently empty array, not "
            "a diagnostic, so this is refused up front rather than built short."
        )
    if kind == "resistive" and "impedance" not in spec:
        raise ConfigError(
            f"{name}: termination: resistive requires an impedance. open, short and "
            "matched are defined by their standard, a resistive load is defined by its "
            "value, and there is no default resistance to fall back on."
        )
    check_unknown_keys(name, spec, _TERMINATION_KEYS, label="kind: termination")
    # `n_freq`, not `n`: rhino_cal_jax.termination_gamma's positional-or-
    # keyword parameter is spelled n_freq. The config's own key stays the
    # short `n` (it is a shape-symbol position, resolved the same way every
    # other shape symbol in this package is), so the two are deliberately
    # different names bridged right here.
    arguments = {
        "kind": kind,
        "n_freq": resolve_extent(spec.get("n", "n_freq"), context.shape_scope),
    }
    if "z0" in spec:
        arguments["z0"] = _dimensioned(spec, "z0", context, name, "ohm", "an impedance (ohm)")
    if "impedance" in spec:
        arguments["impedance"] = _dimensioned(
            spec, "impedance", context, name, "ohm", "an impedance (ohm)"
        )
    return jnp.asarray(cal.termination_gamma(**arguments))


def _cable(name: str, spec: dict, context: ResolutionContext, cal):
    behind = spec.get("behind")
    if not isinstance(behind, dict) or "ref" not in behind:
        raise ConfigError(
            f"{name}: kind: cable requires behind: {{ref: resources.s_params.<name>}} "
            "-- the termination the cable sits in front of."
        )
    if context.freq is None:
        raise ConfigError(f"{name}: kind: cable needs observation.freq.grid.")
    check_unknown_keys(name, spec, _CABLE_KEYS, label="kind: cable")
    # velocity_factor and loss go through the same resolve_value + dimension
    # check as length -- three arguments, one convention -- but both keep
    # their existing 1.0 default rather than requiring a value node be
    # written for every cable.
    velocity_factor = (
        _dimensioned(spec, "velocity_factor", context, name, "dimensionless", "dimensionless")
        if "velocity_factor" in spec else 1.0
    )
    loss = (
        _dimensioned(spec, "loss", context, name, "dimensionless", "dimensionless")
        if "loss" in spec else 1.0
    )
    return jnp.asarray(
        cal.cable_gamma(
            gamma_termination=np.asarray(resolve_reference(behind["ref"], context)),
            freq=np.asarray(context.freq),
            length=_dimensioned(spec, "length", context, name, "m", "a length (m)"),
            velocity_factor=velocity_factor,
            loss=loss,
        )
    )


@register_derivation(
    "interpolate_onto", frozenset({"of", "onto", "component", "allow_extrapolation"})
)
def _interpolate_onto(node, context, modifiers) -> ResolvedValue:
    """``radio/touchstone.py:580`` ``interpolate_onto``, into Plan 1A's table.

    ``of:`` is resolved through the value grammar itself -- ``resolve_value``
    -- rather than read out as a bare ``{ref: ...}``, so a Touchstone can be
    named either by reading one straight from disk
    (``{file: {path: ..., format: touchstone}}``) or by referencing one back
    (``{ref: resources.s_params.<name>}``, which still works exactly as
    before: a ``ref`` node resolves through the same dispatcher). No
    ``resources.s_params`` entry holds a raw Touchstone today -- ``kind:
    touchstone`` resolves straight to the already-interpolated array -- so
    the isinstance check below is what turns a reference to the wrong thing
    into a named refusal instead of an uncaught ``AttributeError`` the first
    time this derivation touches ``.freq_hz``.
    """
    from rheplicant.radio.touchstone import Touchstone, interpolate_onto

    if "of" not in node:
        raise ConfigError(
            "interpolate_onto: 'of' is required -- a value node naming the "
            "Touchstone to interpolate. {file: {path: ..., format: touchstone}} "
            "reads one from disk; {ref: resources.s_params.<name>} reads one back "
            "if that resource holds one."
        )
    source = resolve_value(node["of"], context).value
    if not isinstance(source, Touchstone):
        raise ConfigError(
            f"interpolate_onto: 'of' resolved to a {type(source).__name__}, not a "
            "Touchstone. interpolate_onto reads the raw S-parameter sweep (freq_hz, "
            "s, z0), so 'of' must name one directly -- {file: {path: ..., format: "
            "touchstone}} reads a Touchstone straight from disk. {ref: "
            "resources.s_params.<name>} only works if that entry itself held a "
            "Touchstone, and none of kind: touchstone/termination/cable do: kind: "
            "s_params always resolves to the already-interpolated array, not the "
            "raw sweep it came from."
        )
    if node.get("onto", "freq") != "freq":
        raise ConfigError("interpolate_onto: 'onto' is 'freq', the run's own axis.")
    return ResolvedValue(
        jnp.asarray(
            interpolate_onto(
                np.asarray(context.freq),
                source,
                component=node.get("component", "s11"),
                allow_extrapolation=bool(node.get("allow_extrapolation", False)),
            )
        ),
        canonical_unit("dimensionless"),
        "from",
        modifiers,
    )
