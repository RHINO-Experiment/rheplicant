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
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from rheplicant.core.errors import DataIngestionError

_FREQ_MULTIPLIER = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}
_FORMATS = ("RI", "MA", "DB")
#: Columns in a data row, keyed by port count: 1 + 2 * n_port**2.
_COLUMNS = {1: 3, 2: 9}


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


def read_touchstone(path, *, flipped: bool = False) -> Touchstone:
    """Read a Touchstone v1 ``.s1p`` or ``.s2p`` file.

    Args:
        path: the file.
        flipped: treat the measurement as port-reversed (see below).

    Raises:
        DataIngestionError: on any malformed content. Nothing is skipped.
    """
    path = Path(path)
    multiplier = 1.0
    fmt = None
    z0 = 50.0
    freq: list[float] = []
    rows: list[list[complex]] = []
    n_column = None

    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue

        if line.startswith("#"):
            parts = line[1:].upper().split()
            for token in parts:
                if token in _FREQ_MULTIPLIER:
                    multiplier = _FREQ_MULTIPLIER[token]
                elif token in _FORMATS:
                    fmt = token
            if "R" in parts:
                z0 = float(parts[parts.index("R") + 1])
            if fmt is None:
                raise DataIngestionError(
                    f"{path}:{lineno}: the option line names no data format; one "
                    f"of {_FORMATS} is required."
                )
            continue

        values = line.split()
        if fmt is None:
            raise DataIngestionError(
                f"{path}:{lineno}: data before any '#' option line, so the "
                f"frequency unit and data format are unknown. First data line: "
                f"{line!r}"
            )
        if n_column is None:
            if len(values) not in _COLUMNS.values():
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
        freq.append(numbers[0] * multiplier)
        pairs = list(zip(numbers[1::2], numbers[2::2], strict=True))
        rows.append([_to_complex(fmt, a, b) for a, b in pairs])

    if not freq:
        raise DataIngestionError(f"{path}: no data rows.")

    n_port = 1 if n_column == 3 else 2
    freq_hz = np.asarray(freq, dtype=float)
    flat = np.asarray(rows, dtype=complex)

    s = np.empty((len(freq_hz), n_port, n_port), dtype=complex)
    if n_port == 1:
        s[:, 0, 0] = flat[:, 0]
    else:
        # Touchstone 2-port order: S11 S21 S12 S22.
        s[:, 0, 0], s[:, 1, 0], s[:, 0, 1], s[:, 1, 1] = (
            flat[:, 0], flat[:, 1], flat[:, 2], flat[:, 3]
        )

    if flipped:
        if n_port == 1:
            raise DataIngestionError(
                f"{path}: flipped=True on a 1-port file. Port reversal exchanges "
                "two ports; there is only one."
            )
        s = s[:, ::-1, ::-1]

    return Touchstone(freq_hz=freq_hz, s=s, z0=z0)
