from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest

from _rheplicant_bootstrap.audit import AuditTrace
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.types import LayerIdentity
from rheplicant.config.context import ResolutionContext, using_resolution_audit
from rheplicant.config.resolution_audit import ResolutionAudit, to_json_value

CONFIG = Path(__file__).parents[2] / "src" / "rheplicant" / "config"
PRODUCER_FILES = (
    "arrays.py",
    "delivery.py",
    "derive.py",
    "draws.py",
    "files.py",
    "gating.py",
    "hatch.py",
    "modifiers.py",
    "resources.py",
    "values.py",
    "kinds/bases.py",
    "kinds/beams.py",
    "kinds/projectors.py",
    "kinds/s_params.py",
    "kinds/sky_models.py",
    "sections/benchmark.py",
    "sections/compose.py",
    "sections/conjugate.py",
    "sections/conjugate_support.py",
    "sections/diagnostics.py",
    "sections/exit_support.py",
    "sections/exits.py",
    "sections/inference.py",
    "sections/ingest.py",
    "sections/model.py",
    "sections/noise.py",
    "sections/npe.py",
    "sections/nuts.py",
    "sections/observation.py",
    "sections/observed.py",
    "sections/parameters.py",
    "sections/pointing.py",
    "sections/posterior_support.py",
    "sections/runs.py",
    "sections/runtime.py",
    "sections/switching.py",
    "sections/transforms.py",
    "sections/twin.py",
)
EXPECTED_ROUTES = {
    "arrays.py": {"use_default"},
    "delivery.py": {"use_default", "record_delivery"},
    "derive.py": {"use_default"},
    "draws.py": {"use_default"},
    "files.py": {"use_default", "consume_captured_file"},
    "gating.py": {"use_default", "gate"},
    "hatch.py": {"python_target"},
    "modifiers.py": {"use_default"},
    "resources.py": {"resource"},
    "values.py": {"apply_modifiers"},
    "kinds/bases.py": {"use_default", "record_resolved_delivery"},
    "kinds/beams.py": {"use_default", "consume_captured_file"},
    "kinds/projectors.py": {"use_default"},
    "kinds/s_params.py": {"use_default"},
    "kinds/sky_models.py": {"use_default", "python_target"},
    "sections/benchmark.py": {"use_default"},
    "sections/compose.py": {"use_default"},
    "sections/conjugate.py": {"use_default"},
    "sections/conjugate_support.py": {"use_default"},
    "sections/diagnostics.py": {"use_default"},
    "sections/exit_support.py": {"record_parsed_run"},
    "sections/exits.py": {"use_default"},
    "sections/inference.py": {"use_default"},
    "sections/ingest.py": {"use_default"},
    "sections/model.py": {"use_default", "record_delivery"},
    "sections/noise.py": {"use_default"},
    "sections/npe.py": {"use_default"},
    "sections/nuts.py": {"use_default"},
    "sections/observation.py": {"use_default", "record_resolved_delivery"},
    "sections/observed.py": {"use_default", "seed_for"},
    "sections/parameters.py": {"use_default"},
    "sections/pointing.py": {"use_default"},
    "sections/posterior_support.py": {"seed_for"},
    "sections/runs.py": {"use_default"},
    "sections/runtime.py": {"use_default", "seed"},
    "sections/switching.py": {"use_default"},
    "sections/transforms.py": {"use_default", "python_target"},
    "sections/twin.py": {"use_default"},
}
EXPECTED_DEFAULT_PATHS = {
    "arrays.py": ("f'{target.destination.document_path}.unit'",),
    "delivery.py": ("f'{destination.document_path}.as'",),
    "derive.py": ("'value.from.channel_spacing.times'", "'value.from.sample_cadence.times'"),
    "draws.py": (
        "'value.normal.loc'",
        "'value.normal.scale'",
        "'value.uniform.low'",
        "'value.uniform.high'",
    ),
    "files.py": (
        "f'{destination.document_path}.file.skiprows'",
        "f'{destination.document_path}.unit'",
        "f'{destination.document_path}.file.delimiter'",
        "f'{destination.document_path}.file.columns'",
    ),
    "gating.py": (
        "'inference.checks.identifiability.rtol'",
        "'inference.checks.linearity.mode'",
        "'inference.checks.linearity.report'",
        "'inference.checks.prior_sensitivity.mode'",
        "'inference.checks.identifiability.mode'",
        "'inference.checks.identifiability.report'",
        "'inference.checks.prior_sensitivity.report'",
    ),
    "hatch.py": (),
    "modifiers.py": ("'value.scale'", "'value.offset'"),
    "resources.py": (),
    "values.py": (),
    "kinds/bases.py": ("f'{target.destination.document_path}.unit'",),
    "kinds/beams.py": (
        "'resources.beams[].horizon.mode'",
        "'resources.beams[].horizon'",
        "'resources.beams[].suffix'",
        "'resources.beams[].horizon.el_deg'",
        "'resources.beams[].horizon.apod_deg'",
    ),
    "kinds/projectors.py": (
        "'resources.projectors[].acknowledge_float32_sky'",
        "'resources.projectors[].optimizations'",
        "'resources.projectors[].beam_iterations'",
        "'resources.projectors[].beam_iterations'",
    ),
    "kinds/s_params.py": (
        "'resources.s_params[].component'",
        "'resources.s_params[].file.format'",
        "'resources.s_params[].onto'",
        "'resources.s_params[].velocity_factor'",
        "'resources.s_params[].loss'",
        "'value.from.interpolate_onto.onto'",
        "'resources.s_params[].n'",
        "'resources.s_params[].allow_extrapolation'",
        "'value.from.interpolate_onto.component'",
        "'value.from.interpolate_onto.allow_extrapolation'",
    ),
    "kinds/sky_models.py": (
        "'resources.sky_models[].order'",
        "'resources.sky_models[].maps'",
        "'resources.sky_models[].freq'",
    ),
    "sections/benchmark.py": (
        "'runs[].options.repeats'",
        "'runs[].options.warmup'",
        "'runs[].options.metrics'",
    ),
    "sections/compose.py": ("'model.kind'",),
    "sections/conjugate.py": (
        "'runs[].options.noise_from'",
        "'runs[].options.n_draws'",
        "'runs[].options.acknowledge_unconverged_covariance'",
    ),
    "sections/conjugate_support.py": ("'runs[].options.check'", "f'runs[].options.{key}'"),
    "sections/diagnostics.py": ("'runs[].options.rtol'",),
    "sections/exit_support.py": (),
    "sections/exits.py": (
        "'runs[].options.space'",
        "'runs[].options.jitter'",
        "'runs[].options.loss'",
        "'runs[].options.max_iter'",
        "'runs[].options.tol'",
        "'runs[].options.min_sweeps'",
        "'runs[].options.check_identifiability'",
        "'runs[].options.solve_tol'",
        "'runs[].options.solve_guard'",
        "'runs[].options.warmup'",
        "'runs[].options.rhat_max'",
        "'runs[].options.beta1'",
        "'runs[].options.beta2'",
        "'runs[].options.eps'",
    ),
    "sections/inference.py": ("'inference'", "'inference.twin'"),
    "sections/ingest.py": (
        "'observation.from_file.thermistor_columns'",
        "'observation.from_file.settle_seconds'",
        "'observation.from_file.thermistor_unit'",
    ),
    "sections/model.py": ("f'model.{node_id}.{name}'",),
    "sections/noise.py": (
        "'inference.noise.channel_width'",
        "'inference.noise.integration_time'",
        "'inference.noise.floor'",
        "'inference.noise'",
        "'inference.noise.axis'",
    ),
    "sections/npe.py": (
        "'inference.npe.embed'",
        "f'inference.npe.create.{key}'",
        "f'inference.npe.train.{key}'",
    ),
    "sections/nuts.py": (
        "'runs[].options.init'",
        "'runs[].options.num_chains'",
        "'runs[].options.chain_method'",
        "'runs[].options.thinning'",
        "'runs[].options.progress_bar'",
        "'runs[].options.target_accept_prob'",
    ),
    "sections/observation.py": (
        "'observation.time.epoch'",
        "'observation.time.integration_time'",
        "'observation.time.channel_width'",
        "'observation.site'",
        "'observation.environment'",
        "'observation.environment.temperature'",
        "'observation.environment.humidity'",
        "'observation.environment.extra'",
        "'observation.extra'",
        "'observation.aux'",
        "'observation.meta'",
        "f'observation.site.{key}'",
        "'observation.switching'",
        "'observation.data'",
    ),
    "sections/observed.py": ("'inference.observed[].twin'",),
    "sections/parameters.py": (
        "'inference.parameters[].scope'",
        "'inference.parameters[].linear'",
    ),
    "sections/pointing.py": (
        "'observation.pointing'",
        "'observation.pointing.mode'",
        "'observation.pointing.lst.lst0_deg'",
        "'observation.pointing.az_deg'",
        "'observation.pointing.el_deg'",
        "'observation.pointing.lst.n_time'",
        "'observation.pointing.selfrot_deg'",
    ),
    "sections/posterior_support.py": (),
    "sections/runs.py": (
        "'runs[].name'",
        "'runs[].reuse'",
        "'runs[].variant'",
        "'runs[].on'",
        "'runs[].expect'",
    ),
    "sections/runtime.py": (
        "'runtime.jax_enable_x64'",
        "'runtime.platform'",
        "'runtime.seed'",
        "'runtime.seeds'",
    ),
    "sections/switching.py": (
        "'observation.switching'",
        "'observation.switching.mode'",
        "'observation.switching.cycle'",
        "'observation.switching.dwell'",
    ),
    "sections/transforms.py": (
        "'inference.transforms[].affine.scale'",
        "'inference.transforms[].affine.offset'",
    ),
    "sections/twin.py": ("'inference.twin.without'",),
}


def _call_name(node):
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _literal_path(node):
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    if not isinstance(node, ast.JoinedStr):
        return False
    return all(
        isinstance(part, ast.Constant)
        or (
            isinstance(part, ast.FormattedValue)
            and isinstance(part.value, (ast.Name, ast.Attribute))
        )
        for part in node.values
    )


def test_producer_census_is_exact_and_default_paths_are_source_literals():
    assert tuple(EXPECTED_ROUTES) == PRODUCER_FILES
    assert tuple(EXPECTED_DEFAULT_PATHS) == PRODUCER_FILES
    failures = []
    for relative in PRODUCER_FILES:
        path = CONFIG / relative
        assert path.is_file(), relative
        tree = ast.parse(path.read_text())
        found = {_call_name(node) for node in ast.walk(tree) if isinstance(node, ast.Call)}
        missing = EXPECTED_ROUTES[relative] - found
        failures.extend((relative, "<module>", 0, route) for route in sorted(missing))
        found_defaults = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) != "use_default":
                continue
            if not node.args or not _literal_path(node.args[0]):
                failures.append((relative, "<call>", node.lineno, "literal path"))
                continue
            found_defaults.append(ast.unparse(node.args[0]))
        expected = Counter(EXPECTED_DEFAULT_PATHS[relative])
        actual = Counter(found_defaults)
        for path, count in sorted((expected - actual).items()):
            failures.append((relative, "<default>", 0, f"missing {count} x {path}"))
        for path, count in sorted((actual - expected).items()):
            failures.append((relative, "<default>", 0, f"unexpected {count} x {path}"))
    assert failures == []

    allowed = set(PRODUCER_FILES) | {"context.py", "resolution_audit.py"}
    unexpected = []
    for path in CONFIG.rglob("*.py"):
        relative = str(path.relative_to(CONFIG))
        tree = ast.parse(path.read_text())
        if any(
            isinstance(node, ast.Call) and _call_name(node) == "use_default"
            for node in ast.walk(tree)
        ) and relative not in allowed:
            unexpected.append(relative)
    assert unexpected == []


def test_resolution_audit_returns_exact_consumed_default_and_binds_layer():
    trace = AuditTrace()
    layer = LayerIdentity("variant", "v")
    audit = ResolutionAudit(layer, trace)
    value = {"x": [1]}
    assert audit.use_default("model.x", value) is value
    value["x"][0] = 9
    row = trace.snapshot().defaults[0]
    assert row.layer == layer
    assert row.path == "model.x"
    assert row.value["x"] == (1,)


def test_document_context_gets_one_layer_bound_audit():
    trace = AuditTrace()
    layer = LayerIdentity("base", None)
    with using_resolution_audit(layer, trace, None):
        context = ResolutionContext()
    assert context.audit is not None
    context.use_default("runtime.platform", "auto")
    assert trace.snapshot().defaults[0].layer == layer


@pytest.mark.parametrize("value", [object(), b"bytes", lambda: None])
def test_resolved_values_refuse_live_or_binary_objects(value):
    with pytest.raises(ConfigError, match="resolved audit"):
        to_json_value(value)
