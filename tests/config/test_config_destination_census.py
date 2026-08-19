"""The value-resolution graph names every concrete destination."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from rheplicant.config import values as values_module
from rheplicant.config.dimension_catalog import FORMULA_REGISTRATIONS

ROOT = Path(__file__).parents[2]
CONFIG_SOURCES = tuple(sorted((ROOT / "src/rheplicant/config").rglob("*.py")))


def _called_name(call: ast.Call) -> str | None:
    return call.func.id if isinstance(call.func, ast.Name) else None


def test_every_production_value_resolution_names_its_destination():
    """A builder cannot accidentally resolve a value outside A9's catalog."""
    missing = []
    routes = []
    for path in CONFIG_SOURCES:
        tree = ast.parse(path.read_text())
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call) or _called_name(call) not in {
                "resolve_value", "resolve_operand"
            }:
                continue
            if path.name == "values.py":
                continue
            called = _called_name(call)
            routes.append((path, call.lineno, called))
            keywords = {keyword.arg for keyword in call.keywords}
            if called == "resolve_value" and "destination" not in keywords:
                missing.append(f"{path.relative_to(ROOT)}:{call.lineno}")
            if called == "resolve_operand" and not {
                "parent", "segment", "formula", "role"
            } <= keywords:
                missing.append(f"{path.relative_to(ROOT)}:{call.lineno}")
    assert missing == []
    assert len(routes) == 49
    assert len({path for path, _, _ in routes}) == 22


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
            operand = next(
                item for item in registration.operands if item.role == role.value
            )
            if operand.spec.disposition in ("open", "structural"):
                assert "destination" in keywords
            found.append((formula.value, role.value))
    assert found


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
