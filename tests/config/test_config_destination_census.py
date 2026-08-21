"""The value-resolution graph names every concrete destination."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from rheplicant.config import values as values_module
from rheplicant.config.dimension_catalog import FORMULA_REGISTRATIONS

ROOT = Path(__file__).parents[2]
CONFIG_SOURCES = tuple(sorted((ROOT / "src/rheplicant/config").rglob("*.py")))

EXPECTED_ROUTES = {
    ("derive.py", "_unit_mean_free", "resolve_operand", None, "'unit_mean_free'", "'bandpass'"),
    ("draws.py", "_resolve_operand", "resolve_operand", None, "formula", "role"),
    ("hatch.py", "_python", "resolve_operand", "'python.args.*'", "'python'", "'args.*'"),
    ("inflight/grids.py", "_static_number", "resolve_value", "destination", None, None),
    (
        "kinds/arrays.py",
        "build_array",
        "resolve_value",
        "'rheplicant.config.kinds.arrays.build_array.value'",
        None,
        None,
    ),
    ("kinds/bases.py", "_basis_fit", "resolve_operand", None, "'basis_fit'", "'field'"),
    (
        "kinds/beams.py",
        "_maps_for",
        "resolve_value",
        "'rheplicant.config.kinds.beams.build_beam.cst.phi0_deg'",
        None,
        None,
    ),
    (
        "kinds/beams.py",
        "_maps_for",
        "resolve_value",
        "f'rheplicant.config.kinds.beams.build_beam.{fmt}.maps'",
        None,
        None,
    ),
    (
        "kinds/beams.py",
        "_maps_for",
        "resolve_value",
        "'rheplicant.config.kinds.beams.build_beam.inline.maps'",
        None,
        None,
    ),
    (
        "kinds/beams.py",
        "_maps_for",
        "resolve_value",
        "'rheplicant.config.kinds.beams.build_beam.python.args.*'",
        None,
        None,
    ),
    (
        "kinds/beams.py",
        "_healpix_maps",
        "resolve_value",
        "'rheplicant.config.kinds.beams.build_beam.healpix.freq'",
        None,
        None,
    ),
    (
        "kinds/beams.py",
        "_gaussian",
        "resolve_value",
        "f'rheplicant.config.kinds.beams.build_beam.gaussian.{width_key}'",
        None,
        None,
    ),
    (
        "kinds/beams.py",
        "_horizon_angle",
        "resolve_value",
        "f'rheplicant.config.kinds.beams.build_beam.horizon.{key}'",
        None,
        None,
    ),
    (
        "kinds/projectors.py",
        "_angle",
        "resolve_value",
        "f'rheplicant.config.kinds.projectors.build_projector.{engine}.{key}'",
        None,
        None,
    ),
    (
        "kinds/projectors.py",
        "build_projector",
        "resolve_value",
        "'rheplicant.config.kinds.projectors.build_projector.matrix.matrix'",
        None,
        None,
    ),
    (
        "kinds/projectors.py",
        "build_projector",
        "resolve_value",
        "'rheplicant.config.kinds.projectors.build_projector.general_pointing.beam_alms'",
        None,
        None,
    ),
    (
        "kinds/s_params.py",
        "_dimensioned",
        "resolve_value",
        "f'rheplicant.config.kinds.s_params.build_s_param.{kind}.{key}'",
        None,
        None,
    ),
    (
        "kinds/s_params.py",
        "_from_touchstone",
        "resolve_value",
        "'rheplicant.config.kinds.s_params.build_s_param.touchstone.file'",
        None,
        None,
    ),
    (
        "kinds/s_params.py",
        "_interpolate_onto",
        "resolve_operand",
        "'rheplicant.config.kinds.s_params.build_s_param.touchstone.file'",
        "'interpolate_onto'",
        "'of'",
    ),
    (
        "kinds/sky_models.py",
        "_traced",
        "resolve_value",
        "f'rheplicant.config.kinds.sky_models.build_sky_model.{kind}.{key}'",
        None,
        None,
    ),
    (
        "kinds/sky_models.py",
        "_static_float",
        "resolve_value",
        "f'rheplicant.config.kinds.sky_models.build_sky_model.{kind}.{key}'",
        None,
        None,
    ),
    (
        "kinds/sky_models.py",
        "build_sky_model",
        "resolve_value",
        "'rheplicant.config.kinds.sky_models.build_sky_model.python.args.*'",
        None,
        None,
    ),
    (
        "kinds/sky_models.py",
        "_build_maps",
        "resolve_value",
        "'rheplicant.config.kinds.sky_models.build_sky_model.maps.maps'",
        None,
        None,
    ),
    (
        "kinds/sky_models.py",
        "_build_maps",
        "resolve_value",
        "'rheplicant.config.kinds.sky_models.build_sky_model.maps.freq'",
        None,
        None,
    ),
    ("refs.py", "_stack", "resolve_operand", None, "'stack'", "'entry[]'"),
    ("sections/diagnostics.py", "_at_values", "resolve_value", "'runs[].at.*'", None, None),
    (
        "sections/inference.py",
        "build_inference",
        "resolve_value",
        "'inference.truth.*'",
        None,
        None,
    ),
    (
        "sections/ingest.py",
        "parse_from_file",
        "resolve_value",
        "'observation.from_file'",
        None,
        None,
    ),
    (
        "sections/model.py",
        "_field_value",
        "resolve_value",
        "f'{cls.__module__}.{cls.__qualname__}.{name}'",
        None,
        None,
    ),
    (
        "sections/model.py",
        "_c7_beam_spill",
        "resolve_value",
        "'rheplicant.radio.instrument.beam_spill.BeamSpillOperator.sky_fraction'",
        None,
        None,
    ),
    ("sections/model.py", "_apply_eqx_leaves", "resolve_value", "'model.*.eqx_leaves'", None, None),
    ("sections/noise.py", "_noise_value", "resolve_value", "f'inference.noise.{key}'", None, None),
    ("sections/observation.py", "_dimensioned", "resolve_value", "where", None, None),
    (
        "sections/observation.py",
        "_environment",
        "resolve_value",
        "'observation.environment.extra.*'",
        None,
        None,
    ),
    ("sections/observation.py", "_extra", "resolve_value", "'observation.extra.*'", None, None),
    ("sections/observation.py", "_aux", "resolve_value", "'observation.aux.flags'", None, None),
    ("sections/observation.py", "_data", "resolve_value", "'observation.data'", None, None),
    ("sections/observed.py", "_realise", "resolve_value", "f'{selector}.sigma'", None, None),
    ("sections/observed.py", "_one", "resolve_value", "f'{selector}.file'", None, None),
    ("sections/observed.py", "_one", "resolve_value", "f'{selector}.at.*'", None, None),
    ("sections/parameters.py", "_operand", "resolve_value", "selector", None, None),
    (
        "sections/parameters.py",
        "_parse_prior",
        "resolve_value",
        "'inference.parameters.*.prior.python'",
        None,
        None,
    ),
    (
        "sections/parameters.py",
        "parse_latents",
        "resolve_value",
        "'inference.parameters.*.init'",
        None,
        None,
    ),
    (
        "sections/parameters.py",
        "parse_latents",
        "resolve_value",
        "'inference.parameters.*.ref'",
        None,
        None,
    ),
    (
        "sections/pointing.py",
        "_lst",
        "resolve_value",
        "'observation.pointing.lst.from_file'",
        None,
        None,
    ),
    (
        "sections/switching.py",
        "compile_switching",
        "resolve_value",
        "'observation.switching.index'",
        None,
        None,
    ),
    (
        "sections/switching.py",
        "compile_switching",
        "resolve_value",
        "'observation.switching.dwell'",
        None,
        None,
    ),
    ("sections/transforms.py", "_operand", "resolve_operand", None, "formula", "role"),
    ("sections/transforms.py", "parse_transform", "resolve_operand", None, "'matmul'", "'design'"),
}

EXPECTED_FORMULA_ROLES = {
    ("unit_mean_free", "bandpass"),
    ("normal", "loc"),
    ("normal", "scale"),
    ("uniform", "low"),
    ("uniform", "high"),
    ("python", "args.*"),
    ("basis_fit", "field"),
    ("interpolate_onto", "of"),
    ("stack", "entry[]"),
    ("transform_affine", "scale"),
    ("transform_affine", "offset"),
    ("matmul", "design"),
}

EXPECTED_AUDIT_OWNERS = {
    ("derive.py", "_unit_mean_free"),
    ("draws.py", "_resolve_operand"),
    ("hatch.py", "_python"),
    ("inflight/grids.py", "_static_number"),
    ("kinds/arrays.py", "build_array"),
    ("kinds/bases.py", "_basis_fit"),
    ("kinds/beams.py", "_maps_for"),
    ("kinds/beams.py", "_healpix_maps"),
    ("kinds/beams.py", "_gaussian"),
    ("kinds/beams.py", "_horizon_angle"),
    ("kinds/projectors.py", "_angle"),
    ("kinds/projectors.py", "build_projector"),
    ("kinds/s_params.py", "_dimensioned"),
    ("kinds/sky_models.py", "_traced"),
    ("kinds/sky_models.py", "_static_float"),
    ("kinds/sky_models.py", "build_sky_model"),
    ("kinds/sky_models.py", "_build_maps"),
    ("refs.py", "_stack"),
    ("sections/diagnostics.py", "_at_values"),
    ("sections/inference.py", "build_inference"),
    ("sections/model.py", "_field_value"),
    ("sections/noise.py", "_noise_value"),
    ("sections/observation.py", "_dimensioned"),
    ("sections/observation.py", "_environment"),
    ("sections/observation.py", "_extra"),
    ("sections/observation.py", "_aux"),
    ("sections/observation.py", "_data"),
    ("sections/observed.py", "_realise"),
    ("sections/observed.py", "_one"),
    ("sections/parameters.py", "_operand"),
    ("sections/parameters.py", "parse_latents"),
    ("sections/pointing.py", "_lst"),
    ("sections/switching.py", "compile_switching"),
    ("sections/transforms.py", "_operand"),
    ("sections/transforms.py", "parse_transform"),
}


def _called_name(call: ast.Call) -> str | None:
    return call.func.id if isinstance(call.func, ast.Name) else None


def _keyword_text(call: ast.Call, name: str) -> str | None:
    value = next((item.value for item in call.keywords if item.arg == name), None)
    return None if value is None else ast.unparse(value)


def _destination_text(call: ast.Call, function: ast.FunctionDef) -> str | None:
    value = next((item.value for item in call.keywords if item.arg == "destination"), None)
    if value is None:
        return None
    if isinstance(value, ast.Name):
        candidates = []
        for assignment in ast.walk(function):
            if not isinstance(assignment, ast.Assign) or assignment.lineno >= call.lineno:
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == value.id
                for target in assignment.targets
            ):
                continue
            assigned = assignment.value
            if (
                isinstance(assigned, ast.Call)
                and _called_name(assigned) == "DestinationDescriptor"
                and len(assigned.args) >= 3
            ):
                candidates.append(assignment)
        if candidates:
            value = max(candidates, key=lambda item: item.lineno).value
    if isinstance(value, ast.Call) and len(value.args) >= 3:
        return ast.unparse(value.args[2])
    return ast.unparse(value)


def _functions(tree: ast.AST):
    found = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.stack = []

        def visit_FunctionDef(self, node):
            self.stack.append(node.name)
            found.append((node, node.name))
            self.generic_visit(node)
            self.stack.pop()

    Visitor().visit(tree)
    return found


def test_every_production_value_resolution_names_its_destination():
    """A builder cannot accidentally resolve a value outside A9's catalog."""
    missing = []
    routes = []
    for path in CONFIG_SOURCES:
        tree = ast.parse(path.read_text())
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call) or _called_name(call) not in {
                "resolve_value",
                "resolve_operand",
            }:
                continue
            if path.name == "values.py":
                continue
            called = _called_name(call)
            routes.append((path, call.lineno, called))
            keywords = {keyword.arg for keyword in call.keywords}
            if called == "resolve_value" and "destination" not in keywords:
                missing.append(f"{path.relative_to(ROOT)}:{call.lineno}")
            if (
                called == "resolve_operand"
                and not {"parent", "segment", "formula", "role"} <= keywords
            ):
                missing.append(f"{path.relative_to(ROOT)}:{call.lineno}")
    assert missing == []
    assert len(routes) == 49
    assert len({path for path, _, _ in routes}) == 22


def test_route_ledger_pins_every_producer_identity():
    found = set()
    for path in CONFIG_SOURCES:
        tree = ast.parse(path.read_text())
        relative = str(path.relative_to(ROOT / "src/rheplicant/config"))
        for function, name in _functions(tree):
            for call in ast.walk(function):
                called = _called_name(call) if isinstance(call, ast.Call) else None
                if called not in {"resolve_value", "resolve_operand"}:
                    continue
                found.add(
                    (
                        relative,
                        name,
                        called,
                        _destination_text(call, function),
                        _keyword_text(call, "formula"),
                        _keyword_text(call, "role"),
                    )
                )
    assert found == EXPECTED_ROUTES


def test_every_registered_form_obeys_the_four_argument_protocol():
    assert len(values_module._RESOLVERS) == 17
    for name, resolver in values_module._RESOLVERS.items():
        assert len(inspect.signature(resolver).parameters) == 4, name


def test_recursive_formula_routes_name_registered_roles():
    by_name = {formula.name: formula for formula in FORMULA_REGISTRATIONS}
    found = []
    for path in CONFIG_SOURCES:
        tree = ast.parse(path.read_text())
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call) or _called_name(call) != "resolve_operand":
                continue
            keywords = {keyword.arg: keyword.value for keyword in call.keywords}
            formula = keywords.get("formula")
            role = keywords.get("role")
            if not (
                isinstance(formula, ast.Constant)
                and isinstance(formula.value, str)
                and isinstance(role, ast.Constant)
                and isinstance(role.value, str)
            ):
                continue
            registration = by_name[formula.value]
            operand = next(item for item in registration.operands if item.role == role.value)
            if operand.spec.disposition in ("open", "structural"):
                assert "destination" in keywords
            found.append((formula.value, role.value))
    assert set(found) <= EXPECTED_FORMULA_ROLES


def test_complete_formula_role_ledger_is_exercised_by_recursive_calls():
    found = set()
    for path in CONFIG_SOURCES:
        tree = ast.parse(path.read_text())
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call) or _called_name(call) not in {
                "resolve_operand",
                "_resolve_operand",
                "_operand",
            }:
                continue
            formula = _keyword_text(call, "formula")
            role = _keyword_text(call, "role")
            if formula and role and formula.startswith(("'", '"')):
                found.add((ast.literal_eval(formula), ast.literal_eval(role)))
    assert found == EXPECTED_FORMULA_ROLES


def test_numeric_audit_owner_ledger_is_exact():
    found = set()
    for path in CONFIG_SOURCES:
        tree = ast.parse(path.read_text())
        relative = str(path.relative_to(ROOT / "src/rheplicant/config"))
        for function, name in _functions(tree):
            if (relative, name) == ("delivery.py", "record_resolved_delivery"):
                continue
            calls = [call for call in ast.walk(function) if isinstance(call, ast.Call)]
            if any(
                _called_name(call) == "record_resolved_delivery"
                or (isinstance(call.func, ast.Attribute) and call.func.attr == "record_delivery")
                for call in calls
            ):
                found.add((relative, name))
    assert found == EXPECTED_AUDIT_OWNERS


def test_every_production_delivery_is_immediately_audited():
    path = ROOT / "src/rheplicant/config/sections/model.py"
    tree = ast.parse(path.read_text())
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and _called_name(node.value) == "deliver"
    ]
    assert len(assignments) == 1
    assignment = assignments[0]
    parent_body = next(
        node.body
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.If)) and assignment in node.body
    )
    following = parent_body[parent_body.index(assignment) + 1]
    assert isinstance(following, ast.If)
    assert any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "record_delivery"
        for call in ast.walk(following)
    )
