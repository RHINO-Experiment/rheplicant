"""Touchstone .s1p / .s2p files as complex S-parameter arrays.

A thin adapter, in the same sense as :mod:`rheplicant.radio.beams`: it turns
bytes on disk into numpy, in the units this package carries at its seams (Hz),
and adds nothing else. What an S-parameter *means* -- how a reflection
coefficient becomes a coupling spectrum -- is ``rhino_cal_jax``'s subject.

Ported from ``rhino-cal``'s ``utils/utils.py::read_s2p``, with its silent
failure modes turned into errors. The one that matters most: that function
skips any data row whose column count is not exactly nine, so a trailing ``!``
comment or a truncated line removes a frequency point without a word, and the
caller gets a shorter sweep that still interpolates cleanly.

**Column order.** A Touchstone 2-port data row is ``freq S11 S21 S12 S22``.
The second pair is S21, not S12. This is the single most likely thing to get
wrong here, because every other 2x2 convention in this package is row-major and
because a test that only checks ``s11`` cannot see the error.

**No unstated frequency unit.** Touchstone v1 defaults an omitted frequency
unit to GHz; this reader raises instead of applying that default silently. A
wrong frequency axis does not crash downstream -- interpolation onto an
observing band still returns a finite, correctly-shaped array, just one built
from the wrong slice of the measurement. This package's RHINO observation-HDF5
reader applies the same refusal, for the identical reason: its ``freq_unit``
argument has no default. The stakes here are higher, since a missed unit token
is a 10⁹ error rather than 10⁶, and Touchstone files that omit the token are
rare enough among VNA exports that raising costs less than a silently rescaled
calibration sweep would.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from rheplicant.core.errors import DataIngestionError

_FREQ_MULTIPLIER = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}
_FORMATS = ("RI", "MA", "DB")
#: Touchstone network-parameter type tokens; only "S" (scattering) is
#: supported -- see _parse_option_line.
_NETWORK_PARAMETERS = ("S", "Y", "Z", "H", "G")
#: Port count, keyed by the number of columns in a data row: 1 + 2 * n_port**2.
_PORTS_BY_COLUMN_COUNT = {3: 1, 9: 2}


@dataclasses.dataclass(frozen=True)
class Touchstone:
    """Parsed Touchstone contents.

    Attributes:
        freq_hz: ``(n,)`` strictly ascending frequencies [Hz].
        s: ``(n, p, p)`` complex S-parameters, ``p`` in ``{1, 2}``.
        z0: reference impedance [ohm] from the option line.
    """

    freq_hz: np.ndarray
    s: np.ndarray
    z0: float

    @property
    def n_port(self) -> int:
        return int(self.s.shape[1])

    def _entry(self, row: int, col: int, name: str) -> np.ndarray:
        if row >= self.n_port or col >= self.n_port:
            raise DataIngestionError(
                f"{name} was requested from a {self.n_port}-port file. A 1-port "
                "measurement carries only s11; there is no transmission term to "
                "return and a zero would read as a perfectly isolated port."
            )
        return self.s[:, row, col]

    @property
    def s11(self) -> np.ndarray:
        return self._entry(0, 0, "s11")

    @property
    def s12(self) -> np.ndarray:
        return self._entry(0, 1, "s12")

    @property
    def s21(self) -> np.ndarray:
        return self._entry(1, 0, "s21")

    @property
    def s22(self) -> np.ndarray:
        return self._entry(1, 1, "s22")


def _to_complex(fmt: str, a: float, b: float) -> complex:
    if fmt == "RI":
        return complex(a, b)
    if fmt == "MA":
        return a * np.exp(1j * np.deg2rad(b))
    return 10 ** (a / 20.0) * np.exp(1j * np.deg2rad(b))


def _parse_impedance(parts: list[str], path: Path, lineno: int, line: str) -> float:
    """The option line's ``R <z0>`` clause; 50 ohm (the v1 default) if absent.

    Unlike the frequency unit, an implicit 50 ohm reference is the near-
    universal case, not a plausible order-of-magnitude error, so this one
    default is applied rather than refused. ``R`` present but malformed --
    trailing with no value, or a non-numeric value -- still raises; those
    used to leak a bare ``IndexError`` / ``ValueError`` with no path or line.
    """
    if "R" not in parts:
        return 50.0
    index = parts.index("R") + 1
    if index >= len(parts):
        raise DataIngestionError(
            f"{path}:{lineno}: the option line ends with 'R' but names no impedance "
            f"value. Line: {line!r}"
        )
    try:
        return float(parts[index])
    except ValueError as exc:
        raise DataIngestionError(
            f"{path}:{lineno}: the impedance after 'R' is not a number: "
            f"{parts[index]!r}. Line: {line!r}"
        ) from exc


def _parse_option_line(line: str, path: Path, lineno: int) -> tuple[float, str, float]:
    """Parse one Touchstone ``#`` option line into ``(multiplier, fmt, z0)``.

    Raises rather than guessing at anything the line does not state outright:
    the frequency unit (see the module docstring for why the v1 spec's GHz
    default is refused), the data format (``RI`` / ``MA`` / ``DB``), and the
    network-parameter type -- only ``S`` is supported, since a Y/Z/H/G-
    parameter file would otherwise parse without complaint and land in
    :attr:`Touchstone.s` mislabelled as S-parameters.
    """
    parts = line[1:].upper().split()

    multiplier = None
    fmt = None
    for token in parts:
        if token in _FREQ_MULTIPLIER:
            multiplier = _FREQ_MULTIPLIER[token]
        elif token in _FORMATS:
            fmt = token

    if multiplier is None:
        raise DataIngestionError(
            f"{path}:{lineno}: the option line names no frequency unit. Touchstone "
            f"v1 defaults an omitted unit to GHz, but this reader will not apply an "
            f"unstated 10⁹ rescaling -- name one of {tuple(_FREQ_MULTIPLIER)} "
            f"explicitly. Line: {line!r}"
        )
    if fmt is None:
        raise DataIngestionError(
            f"{path}:{lineno}: the option line names no data format; one of "
            f"{_FORMATS} is required. Line: {line!r}"
        )
    if "S" not in parts:
        other = next((token for token in parts if token in _NETWORK_PARAMETERS), None)
        found = f" (found {other!r} instead)" if other else ""
        raise DataIngestionError(
            f"{path}:{lineno}: the option line does not specify S-parameters{found}; "
            f"Y/Z/H/G-parameter files are not supported. Line: {line!r}"
        )

    return multiplier, fmt, _parse_impedance(parts, path, lineno, line)


def _parse_data_row(
    line: str, path: Path, lineno: int, fmt: str, n_column: int | None
) -> tuple[float, list[complex], int]:
    """Validate and parse one Touchstone data row.

    ``n_column`` threads the file's column count across calls: ``None`` on the
    first row, after which every later row is checked against what that row
    established, not just against the format spec's ``{3, 9}`` -- so a row
    that is merely inconsistent with the rest of its own file, not invalid on
    its own terms, still raises.

    Returns:
        ``(freq, row, n_column)``: the row's raw frequency value (not yet
        unit-converted), its S-parameter entries in file order, and the
        (possibly newly established) column count.
    """
    values = line.split()
    if n_column is None:
        if len(values) not in _PORTS_BY_COLUMN_COUNT:
            raise DataIngestionError(
                f"{path}:{lineno}: {len(values)} columns; a Touchstone data "
                f"row has 3 (1-port) or 9 (2-port). Line: {line!r}"
            )
        n_column = len(values)
    elif len(values) != n_column:
        raise DataIngestionError(
            f"{path}:{lineno}: {len(values)} columns, but this file's first "
            f"data row had {n_column}. Line: {line!r}"
        )

    numbers = [float(v) for v in values]
    pairs = list(zip(numbers[1::2], numbers[2::2], strict=True))
    row = [_to_complex(fmt, a, b) for a, b in pairs]
    return numbers[0], row, n_column


def _assemble_s(flat: np.ndarray, n_port: int) -> np.ndarray:
    """Lay ``(n, n_port**2)`` file-order entries out as ``(n, n_port, n_port)``.

    The only place the Touchstone-vs-row-major permutation from the module
    docstring's "Column order" note is actually applied.
    """
    s = np.empty((flat.shape[0], n_port, n_port), dtype=complex)
    if n_port == 1:
        s[:, 0, 0] = flat[:, 0]
    else:
        # Touchstone 2-port order: S11 S21 S12 S22.
        s[:, 0, 0], s[:, 1, 0], s[:, 0, 1], s[:, 1, 1] = (
            flat[:, 0], flat[:, 1], flat[:, 2], flat[:, 3]
        )
    return s


def _check_strictly_ascending(freq_hz: np.ndarray, path: Path) -> None:
    """Raise unless ``freq_hz`` is finite and strictly ascending.

    Two checks, deliberately kept separate, run in this order:

    NaN first, checked directly against ``freq_hz`` rather than through
    ``np.diff``. A NaN frequency spreads into the diff cells on *both* sides
    of it, and the ordering check below always blames the *later* element of
    a diff pair -- right for a genuine ordering violation, but wrong for a
    leading NaN, which has no earlier row to blame and would be misattributed
    to the row after it. Checking ``freq_hz`` directly sidesteps that
    asymmetry and reports the NaN's own row. It also gets its own message:
    "not strictly ascending" describes a disordered measurement, not an
    invalid one, and the fix differs too -- correct that row's value, not
    reorder the file.

    Ascending order second, now that NaN is ruled out: interpolating this
    sweep onto the observing band (a later module) goes through ``np.interp``,
    which *assumes* ascending order and does not check it -- unsorted input
    comes back as a finite, silently wrong answer rather than an error.

    :func:`_interp_strict`, later in this module, mirrors this same
    NaN-first-then-ascending structure over a general ``x`` axis rather than
    a file's own ``freq_hz``. It is a second, independent implementation
    rather than a shared one -- the two report different things (1-based
    file rows against a ``path`` here, 0-based array indices against a
    ``what`` label there) -- so if this function's check order or reasoning
    changes, that one needs the same change made by hand.
    """
    nan_rows = np.flatnonzero(np.isnan(freq_hz))
    if nan_rows.size:
        row = int(nan_rows[0]) + 1
        raise DataIngestionError(
            f"{path}: frequency at data row {row} is NaN (counting only data "
            "rows -- comments and blank lines are not counted). That is a "
            "different problem from rows being out of order: fix that row's "
            "value; np.interp cannot use it either way."
        )

    # NaN is already ruled out above, so the plain form is safe here: with no
    # NaN in play, `np.any(diffs <= 0)` and `not np.all(diffs > 0)` agree. They
    # would NOT agree with a NaN present (NaN compares False to everything,
    # including "<= 0" -- see the docstring above for why that matters) --
    # do not "simplify" this without keeping the NaN check ahead of it.
    diffs = np.diff(freq_hz)
    if np.any(diffs <= 0):
        bad_row = int(np.argmin(diffs > 0)) + 2
        raise DataIngestionError(
            f"{path}: frequencies are not strictly ascending (first offender "
            f"at data row {bad_row}, counting only data rows -- comments and "
            "blank lines are not counted). Interpolation onto this sweep "
            "assumes they are, and np.interp does not check."
        )


def _check_suffix_matches_port_count(path: Path, n_column: int, n_port: int) -> None:
    """Cross-check a ``.s1p``/``.s2p`` suffix against the port count the rows carry.

    Nothing in a Touchstone file states its own port count independently of
    the column count, so the filename is the only second opinion available --
    and disagreement matters: a mislabelled file would otherwise put a
    transmission term where a reflection coefficient belongs, or vice versa,
    with nothing to catch it. A file with no suffix, or one this reader does
    not otherwise recognise, has no second opinion to check against and
    passes through untouched.
    """
    suffix = path.suffix.lower()
    if suffix not in (".s1p", ".s2p"):
        return
    declared = int(suffix[2])
    if declared != n_port:
        raise DataIngestionError(
            f"{path}: the name says {suffix} but the rows carry {n_column} "
            f"columns, i.e. {n_port}-port data. One of the two is wrong and "
            "guessing which would silently mislabel a port."
        )


def _build_touchstone(
    path: Path,
    freq: list[float],
    rows: list[list[complex]],
    n_column: int,
    z0: float,
    flipped: bool,
) -> Touchstone:
    """Validate the fully-parsed rows and assemble them into a ``Touchstone``.

    Both cross-file checks live here rather than in the parse loop, because
    both need every row before they mean anything: ascending order is a
    property of the whole sequence, and the suffix check needs the column
    count the *first* row established, which is only final once parsing ends.
    """
    freq_hz = np.asarray(freq, dtype=float)
    _check_strictly_ascending(freq_hz, path)

    n_port = _PORTS_BY_COLUMN_COUNT[n_column]
    _check_suffix_matches_port_count(path, n_column, n_port)
    s = _assemble_s(np.asarray(rows, dtype=complex), n_port)

    if flipped:
        if n_port == 1:
            raise DataIngestionError(
                f"{path}: flipped=True on a 1-port file. Port reversal exchanges "
                "two ports; there is only one."
            )
        s = s[:, ::-1, ::-1]

    return Touchstone(freq_hz=freq_hz, s=s, z0=z0)


def _read_option_line(
    line: str, path: Path, lineno: int, already_seen: bool
) -> tuple[float, str, float]:
    """Parse a ``#`` option line, first checking it is the file's only one.

    Touchstone allows exactly one; a second would silently re-specify the
    unit, format or impedance for the rows that follow it, leaving one
    ``Touchstone.s`` array built from two different readings.
    """
    if already_seen:
        raise DataIngestionError(
            f"{path}:{lineno}: a second '#' option line. Touchstone allows exactly "
            f"one option line per file. Line: {line!r}"
        )
    return _parse_option_line(line, path, lineno)


def _read_data_line(
    line: str,
    path: Path,
    lineno: int,
    fmt: str | None,
    n_column: int | None,
    option_line_seen: bool,
) -> tuple[float, list[complex], int]:
    """Parse a data row, first checking an option line has set ``fmt``.

    Without one, the frequency unit and data format are unknown -- see
    :func:`_parse_option_line`.
    """
    if not option_line_seen:
        raise DataIngestionError(
            f"{path}:{lineno}: data before any '#' option line, so the frequency "
            f"unit and data format are unknown. First data line: {line!r}"
        )
    return _parse_data_row(line, path, lineno, fmt, n_column)


def read_touchstone(path: str | Path, *, flipped: bool = False) -> Touchstone:
    """Read a Touchstone v1 ``.s1p`` or ``.s2p`` file.

    Args:
        path: the file.
        flipped: swap the two ports (``s11``<->``s22``, ``s12``<->``s21``) as a
            genuine reversal of the parsed matrix, not a relabelling of the
            return values. Set it when the device under test was wired to the
            VNA with this codebase's port 1 and port 2 reversed.

    Raises:
        DataIngestionError: on any malformed content. Nothing is skipped.
    """
    path = Path(path)
    multiplier, fmt, z0, option_line_seen = 1.0, None, 50.0, False
    freq: list[float] = []
    rows: list[list[complex]] = []
    n_column = None

    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue

        if line.startswith("#"):
            multiplier, fmt, z0 = _read_option_line(line, path, lineno, option_line_seen)
            option_line_seen = True
            continue
        raw_freq, row, n_column = _read_data_line(
            line, path, lineno, fmt, n_column, option_line_seen
        )
        freq.append(raw_freq * multiplier)
        rows.append(row)

    if n_column is None:
        # Equivalent to `if not freq:` -- every loop iteration that appends to
        # `freq` also assigns `n_column` (see above) -- but phrased this way,
        # the check documents and enforces, rather than merely implies, the
        # invariant _build_touchstone depends on: its `n_column` parameter is
        # typed `int`, not `int | None`.
        raise DataIngestionError(f"{path}: no data rows.")

    return _build_touchstone(path, freq, rows, n_column, z0, flipped)


#: A "few ULPs" of slack for _interp_strict's tolerance band, at the sampled
#: axis's own magnitude (np.spacing() of it). One ULP would cover a value
#: compared exactly as read; the 4x margin budgets headroom for a short chain
#: of ordinary float64 ops between a raw reading and the comparison here (a
#: unit-multiplier conversion, an epoch offset, a subtraction for elapsed
#: time), each of which can add roughly another ULP of drift -- without
#: approaching a magnitude that could mask a genuine range violation.
_INTERP_ULP_SLACK = 4
#: Relative tolerance on the interpolation span in _interp_strict, applied to
#: (hi - lo) so a target grid whose endpoint agrees with the source only to
#: within a unit conversion's rounding still passes. Deliberately tiny: this
#: exists to swallow float rounding, not to make extrapolation lenient.
_INTERP_SPAN_TOL = 1e-9


def _interp_strict(
    x_new: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    allow_extrapolation: bool = False,
    what: str = "value",
) -> np.ndarray:
    """Linear interpolation that refuses to leave the sampled range.

    ``np.interp`` *clamps* outside ``x`` rather than raising, so a reflection
    coefficient measured over a narrower band than the one being observed comes
    back as a constant at the edges, with no diagnostic. That is the failure
    this guard exists for; the reference implementation
    (``rhino-cal/utils/utils.py::interp_vals_to_new_freq``) has it.

    ``x`` must be non-empty, finite, and strictly ascending -- checked here,
    not assumed, in that order and for the same reason
    :func:`_check_strictly_ascending` checks them in that order (see that
    function's docstring for the cross-reference back to this one).
    ``Touchstone.freq_hz`` arrives already checked, but this is a general
    helper, shared with a caller (a thermistor time axis) that carries no
    such guarantee at all. NaN gets its own check and its own message ahead
    of the ordering check: ``np.diff(x) <= 0`` compares ``False`` against
    NaN, so a NaN axis would otherwise sail straight through the ordering
    check and ``np.interp`` would return NaN silently rather than this
    function raising. Once NaN is ruled out, ``np.interp`` assumes ascending
    order without checking it either, so unsorted input would otherwise come
    back as a finite, silently wrong answer instead of raising.

    Args:
        x_new: the target grid to interpolate onto.
        x: the sampled grid ``y`` was measured on. Must be 1-D, non-empty,
            finite, strictly ascending, and the same length as ``y`` (see
            above and Raises below).
        y: the sampled values, real or complex, co-indexed with ``x``.
        allow_extrapolation: if ``False`` (the default), raise when any of
            ``x_new`` falls outside ``[x[0], x[-1]]`` by more than the
            tolerance band (see Raises below). If ``True``, skip that check
            entirely and let ``np.interp`` clamp to the edge values as it
            normally would.
        what: a short label identifying the call site, folded into every
            error message this function raises (e.g. ``"s11 interpolation"``
            or ``"thermistor column 1 for 'heated_load'"``).

    Returns:
        ``y`` interpolated onto ``x_new``. Complex if ``y`` is complex, with
        the real and imaginary parts interpolated separately -- not
        magnitude/phase, since across a mismatch resonance the phase wraps,
        and interpolating a wrapped angle is worse than interpolating
        Cartesian components, not better. Real otherwise: this is Task 9's
        actual case (a thermistor temperature series), and that branch
        returns a plain float array, not a complex one with a zero
        imaginary part.

    Raises:
        DataIngestionError: if ``x`` and ``y`` disagree in length; if ``x``
            is empty or contains NaN; if ``x`` is not strictly ascending; or
            if ``allow_extrapolation`` is ``False`` and any of ``x_new``
            falls outside ``x``'s range by more than a tolerance band of
            roughly ``_INTERP_SPAN_TOL * span(x)`` plus
            ``_INTERP_ULP_SLACK`` ULPs at ``x``'s own magnitude -- see the
            body comment above the tolerance computation for why both terms
            are needed. That band, not point equality, is what
            "outside the sampled range" means here; it is the most
            carefully-tuned, and previously buggy, part of this function.
    """
    x_new = np.asarray(x_new, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    if x.size == 0:
        raise DataIngestionError(f"{what}: the sampled range is empty; nothing to interpolate.")
    if x.shape != y.shape:
        raise DataIngestionError(
            f"{what}: x has {x.shape[0]} samples but y has {y.shape[0]}; they "
            "must be co-indexed. np.interp raises its own unbranded "
            "ValueError here ('fp and xp are not of the same length'), with "
            "no indication of which call site or which arrays were involved."
        )
    nan_indices = np.flatnonzero(np.isnan(x))
    if nan_indices.size:
        raise DataIngestionError(
            f"{what}: the sampled x-axis has NaN at index {int(nan_indices[0])} "
            "(0-based). That is a different problem from being out of order: "
            "fix that value; np.interp cannot use it either way."
        )
    diffs = np.diff(x)
    if np.any(diffs <= 0):
        bad_index = int(np.argmin(diffs > 0)) + 1
        raise DataIngestionError(
            f"{what}: the sampled x-axis is not strictly ascending (first "
            f"offender at index {bad_index}, 0-based). np.interp assumes "
            "ascending order and does not check it, so unsorted input would "
            "come back as a finite, silently wrong answer instead."
        )
    lo, hi = float(x[0]), float(x[-1])
    if not allow_extrapolation and x_new.size:
        # Two terms, added. A relative tolerance on the span, so a target
        # grid whose endpoint agrees with the source to within a unit
        # conversion's rounding still passes -- plus a few ULPs of slack at
        # the axis's own magnitude, for float64 representation noise that
        # does not shrink along with a short span.
        #
        # Both terms are needed. On a large-magnitude, short-span axis (a
        # unix-epoch time axis sampled a few seconds apart, ~1.75e9 in
        # magnitude), the span term alone is smaller than float64's own
        # spacing there -- ~1e-8 s of span-tolerance against ~2e-7 s of
        # representation noise -- which would reject an endpoint that is
        # genuinely equal up to roundoff. The single-term `abs(hi)`-only form
        # this replaced overcorrected the other way: on that same axis it
        # gave ~1.75 s of slack *regardless of window length*, so a target
        # 1.5 s past the sampled range was silently accepted and clamped --
        # the exact failure this function exists to catch.
        #
        # x_new.size guards .min()/.max(): both raise on a zero-size array,
        # and an empty target grid is vacuously within any range anyway.
        tol = _INTERP_SPAN_TOL * (hi - lo) + _INTERP_ULP_SLACK * np.spacing(max(abs(lo), abs(hi)))
        if x_new.min() < lo - tol or x_new.max() > hi + tol:
            raise DataIngestionError(
                f"{what}: the target range [{x_new.min():.6g}, {x_new.max():.6g}] "
                f"lies outside the sampled range [{lo:.6g}, {hi:.6g}]. "
                "np.interp would clamp to the edge values and report nothing; "
                "pass allow_extrapolation=True only if that is what you want."
            )
    if np.iscomplexobj(y):
        return np.interp(x_new, x, np.real(y)) + 1j * np.interp(x_new, x, np.imag(y))
    return np.interp(x_new, x, np.asarray(y, dtype=float))


def interpolate_onto(
    freq_hz: np.ndarray,
    source: Touchstone,
    *,
    component: str = "s11",
    allow_extrapolation: bool = False,
) -> np.ndarray:
    """Interpolate one S-parameter of ``source`` onto ``freq_hz`` [Hz].

    ``component`` is checked here only against the four canonical S-parameter
    names. Whether ``source`` actually carries that component -- a 1-port file
    has no s12/s21/s22 -- is ``Touchstone``'s own concern: its property
    getters already raise a precise, reason-naming ``DataIngestionError`` (see
    ``Touchstone._entry``), and duplicating that check here would only be a
    second place for the message to drift out of sync with the first.
    """
    if component not in ("s11", "s12", "s21", "s22"):
        raise DataIngestionError(f"component must be one of s11, s12, s21, s22; got {component!r}.")
    values = getattr(source, component)
    return _interp_strict(
        freq_hz,
        source.freq_hz,
        values,
        allow_extrapolation=allow_extrapolation,
        what=f"{component} interpolation",
    )
