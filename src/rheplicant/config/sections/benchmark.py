"""Variant-aware wall-time and Python-traced-memory benchmarks."""

from __future__ import annotations

import statistics
import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from rheplicant.config.errors import ConfigError
from rheplicant.config.sections.exit_support import (
    ParsedRun,
    _number,
    _sweep,
    parsed_options,
    register,
)

_BENCHMARK_KEYS = frozenset({"variants", "repeats", "warmup", "metrics"})

#: What this exit defaults each of its optional keys to.
#:
#: **Written as a map because a literal at the ``use_default`` call site is
#: invisible to the one thing whose job is to notice a disagreement.**
#: ``gui/form_catalog.py::_contested`` decides whether a widget may publish a
#: default by comparing every exit that owns the key; it reads five maps and
#: nothing else. ``warmup`` was 1 here and ``None`` in
#: ``exits._SAMPLE_DEFAULTS``, so the census saw ONE owner, concluded the key
#: was uncontested, and published ``None`` -- a plausible wrong figure for the
#: exit that actually defaults it to 1, which is the one kind of error that
#: survives being looked at. Measured: it was the only such disagreement in
#: the layer, and the only one the census could not have seen.
#:
#: The map is read for that comparison and is also what the parser reads, so
#: the two cannot drift: there is one number per key and it lives here.
_BENCHMARK_DEFAULTS = {
    "repeats": 5,
    "warmup": 1,
    "metrics": ("wall_time",),
}
_METRICS = ("wall_time", "peak_memory")
_DEFERRED = ("benchmark.variants_available",)


@dataclass(frozen=True, slots=True)
class BenchmarkMetric:
    samples: tuple[int, ...]
    minimum: int
    median: float
    mean: float
    unit: str


@dataclass(frozen=True, slots=True)
class BenchmarkVariant:
    name: str
    metrics: Mapping[str, BenchmarkMetric]


@dataclass(frozen=True, slots=True)
class BenchmarkProduct:
    variants: tuple[BenchmarkVariant, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkContext:
    configured: object
    layers: Sequence[object]


def _block_until_ready(value: object) -> None:
    """Block every JAX leaf before a measurement is stopped."""
    import jax

    leaves = jax.tree_util.tree_leaves(value)
    for leaf in leaves:
        block = getattr(leaf, "block_until_ready", None)
        if callable(block):
            block()


def _metric(samples: list[int], *, unit: str) -> BenchmarkMetric:
    if not samples:
        raise ConfigError("benchmark produced no measured samples.")
    return BenchmarkMetric(
        tuple(samples),
        min(samples),
        float(statistics.median(samples)),
        float(statistics.fmean(samples)),
        unit,
    )


def benchmark_callables(
    targets: Mapping[str, Callable[[], object]],
    *,
    repeats: int,
    warmup: int,
    metrics: Sequence[str],
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> BenchmarkProduct:
    """Benchmark ordered callables, retaining every measured raw sample."""
    if type(repeats) is not int or repeats <= 0:
        raise ConfigError("benchmark repeats must be a positive integer.")
    if type(warmup) is not int or warmup < 0:
        raise ConfigError("benchmark warmup must be a non-negative integer.")
    chosen = tuple(metrics)
    if not chosen or len(chosen) != len(set(chosen)) or any(
        metric not in _METRICS for metric in chosen
    ):
        raise ConfigError(f"benchmark metrics must be a non-empty subset of {list(_METRICS)}.")
    if not targets:
        raise ConfigError("benchmark needs at least one target.")
    rows: list[BenchmarkVariant] = []
    for name, target in targets.items():
        for _ in range(warmup):
            _block_until_ready(target())
        wall_samples: list[int] = []
        memory_samples: list[int] = []
        for _ in range(repeats):
            tracking = "peak_memory" in chosen
            if tracking:
                tracemalloc.start()
            try:
                started = clock_ns() if "wall_time" in chosen else None
                value = target()
                _block_until_ready(value)
                if started is not None:
                    wall_samples.append(clock_ns() - started)
                if tracking:
                    _current, peak = tracemalloc.get_traced_memory()
                    memory_samples.append(peak)
            finally:
                if tracking:
                    tracemalloc.stop()
        measured: dict[str, BenchmarkMetric] = {}
        for metric in chosen:
            if metric == "wall_time":
                measured[metric] = _metric(wall_samples, unit="ns")
            else:
                measured[metric] = _metric(
                    memory_samples,
                    unit="python_traced_bytes",
                )
        rows.append(BenchmarkVariant(name, MappingProxyType(dict(measured))))
    return BenchmarkProduct(tuple(rows))


def _unique_names(value: object, *, where: str) -> tuple[str, ...]:
    if (
        type(value) is not list
        or not value
        or any(type(name) is not str or not name for name in value)
        or len(value) != len(set(value))
    ):
        raise ConfigError(f"{where}: is a non-empty list of unique variant names.")
    return tuple(value)


def _parse_benchmark(options: Mapping[str, object], context: object):
    spec = context.spec
    where = f"runs[{spec.name!r}]"
    _sweep(spec, _BENCHMARK_KEYS)
    if "variants" not in options:
        raise ConfigError(f"{where}: variants: is required.")
    variants = _unique_names(options["variants"], where=f"{where}: variants")
    repeats_node = (
        options["repeats"]
        if "repeats" in options
        else context.configured_run.context.use_default(
            "runs[].options.repeats", _BENCHMARK_DEFAULTS["repeats"]
        )
    )
    warmup_node = (
        options["warmup"]
        if "warmup" in options
        else context.configured_run.context.use_default(
            "runs[].options.warmup", _BENCHMARK_DEFAULTS["warmup"]
        )
    )
    repeats = _number(spec, "repeats", repeats_node, kind=int, minimum=1)
    warmup = _number(spec, "warmup", warmup_node, kind=int, minimum=0)
    metrics_node = (
        options["metrics"]
        if "metrics" in options
        else context.configured_run.context.use_default(
            "runs[].options.metrics", list(_BENCHMARK_DEFAULTS["metrics"])
        )
    )
    metrics = _unique_names(metrics_node, where=f"{where}: metrics")
    for index, metric in enumerate(metrics):
        if metric not in _METRICS:
            raise ConfigError(
                f"{where}: metrics[{index}]: must be one of {list(_METRICS)}."
            )
    normalized = {
        "variants": variants,
        "repeats": repeats,
        "warmup": warmup,
        "metrics": metrics,
    }
    return parsed_options(
        normalized,
        resolved={**normalized, "variants": list(variants), "metrics": list(metrics)},
    )


def _layer_table(context: BenchmarkContext, *, run_name: str) -> dict[str, object]:
    if type(context) is not BenchmarkContext:
        raise ConfigError(
            f"runs[{run_name!r}]: benchmark requires the prepared document's "
            "complete layer set; use run_document or the config CLI."
        )
    table: dict[str, object] = {}
    for layer in context.layers:
        name = "base" if layer.layer.kind == "base" else layer.layer.name
        if type(name) is not str or not name or name in table:
            raise ConfigError("benchmark received an invalid prepared layer table.")
        table[name] = layer
    return table


def _benchmark_pre_execute(
    parsed: ParsedRun,
    context: BenchmarkContext,
    _previous: Mapping[str, object],
) -> None:
    table = _layer_table(context, run_name=parsed.name)
    missing = [name for name in parsed.options["variants"] if name not in table]
    if missing:
        raise ConfigError(
            f"runs[{parsed.name!r}]: variants: {missing} are not prepared; "
            f"available layers are {list(table)}."
        )


@register(
    "benchmark",
    parse=_parse_benchmark,
    pre_execute=_benchmark_pre_execute,
    deferred_checks=_DEFERRED,
)
def _run_benchmark(
    run: ParsedRun,
    context: BenchmarkContext,
    _previous: Mapping[str, object],
) -> BenchmarkProduct:
    _sweep(run, _BENCHMARK_KEYS)
    table = _layer_table(context, run_name=run.name)
    targets: dict[str, Callable[[], object]] = {}
    for name in run.options["variants"]:
        configured = table[name].configured

        def target(configured=configured):
            return configured.twin(configured.state)

        targets[name] = target
    return benchmark_callables(
        targets,
        repeats=run.options["repeats"],
        warmup=run.options["warmup"],
        metrics=run.options["metrics"],
    )


__all__ = [
    "BenchmarkContext",
    "BenchmarkMetric",
    "BenchmarkProduct",
    "BenchmarkVariant",
    "benchmark_callables",
]
