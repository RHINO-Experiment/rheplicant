"""Process-global JAX runtime bootstrap for Config Plan 4A."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import traceback
from dataclasses import fields
from pathlib import Path

import pytest

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.process import RuntimeSpec
from _rheplicant_bootstrap.runtime import (
    RUNTIME_ACTUAL_ROW_KEYS,
    RUNTIME_PRIOR_ENVIRONMENT_ROW_KEYS,
    RUNTIME_REQUESTED_ROW_KEYS,
    RUNTIME_ROW_KEYS,
    PriorEnvironment,
    RuntimeActual,
    RuntimeSession,
    establish_runtime,
    runtime_audit_row,
)

SRC = Path(__file__).parents[2] / "src"


class _InstanceClassTrap:
    def __init__(self, *, forged_type: type | None) -> None:
        self.forged_type = forged_type
        self.calls = 0

    @property
    def __class__(self):
        self.calls += 1
        if self.forged_type is None:
            raise RuntimeError("descriptor secret")
        return self.forged_type


@pytest.mark.parametrize("forged_type", (None, str))
def test_runtime_actual_never_reads_instance_class_descriptors(forged_type):
    value = _InstanceClassTrap(forged_type=forged_type)
    with pytest.raises(ConfigError) as caught:
        RuntimeActual(False, value)
    assert value.calls == 0
    assert "descriptor secret" not in str(caught.value)


def _clean_probe(program: str, *, environment: dict[str, str | None] | None = None):
    env = os.environ.copy()
    env.pop("JAX_ENABLE_X64", None)
    env.pop("JAX_PLATFORMS", None)
    for key, value in (environment or {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    source = f"import sys; sys.path.insert(0, {str(SRC)!r})\n" + textwrap.dedent(
        program
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", source],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


@pytest.mark.parametrize(
    ("requested", "prior", "expected_env"),
    [
        (
            {"jax_enable_x64": True, "platform": "cpu"},
            {},
            {"JAX_ENABLE_X64": "true", "JAX_PLATFORMS": "cpu"},
        ),
        (
            {"jax_enable_x64": False, "platform": "auto"},
            {"JAX_PLATFORMS": "cpu"},
            {"JAX_ENABLE_X64": "false", "JAX_PLATFORMS": "cpu"},
        ),
        (
            {"jax_enable_x64": False, "platform": "auto"},
            {},
            {"JAX_ENABLE_X64": "false"},
        ),
    ],
)
def test_environment_is_set_before_the_import(requested, prior, expected_env):
    result = _clean_probe(
        f"""
        import json, os, sys
        from _rheplicant_bootstrap.process import RuntimeSpec
        from _rheplicant_bootstrap.runtime import establish_runtime

        requested = RuntimeSpec(
            {requested['jax_enable_x64']!r}, {requested['platform']!r}, None, {{}}
        )
        seen = {{}}
        pid = os.getpid()
        def import_main():
            seen["environment_seen_by_import"] = {{
                key: os.environ[key]
                for key in ("JAX_ENABLE_X64", "JAX_PLATFORMS")
                if key in os.environ
            }}
            seen["imported_before_callback"] = "jax" in sys.modules
            import jax
            return "main"

        session, main = establish_runtime(requested, import_main=import_main)
        seen["same_process"] = os.getpid() == pid
        seen["main"] = main
        seen["actual"] = [session.actual.jax_enable_x64, session.actual.backend]
        seen["prior"] = [
            session.prior_environment.jax_enable_x64,
            session.prior_environment.jax_platforms,
        ]
        print(json.dumps(seen, sort_keys=True))
        """,
        environment=prior,
    )
    assert result["environment_seen_by_import"] == expected_env
    assert result["imported_before_callback"] is False
    assert result["same_process"] is True
    assert result["main"] == "main"
    assert result["actual"] == [requested["jax_enable_x64"], "cpu"]
    assert result["prior"] == [
        prior.get("JAX_ENABLE_X64"),
        prior.get("JAX_PLATFORMS"),
    ]


def test_platform_auto_preserves_caller_environment():
    result = _clean_probe(
        """
        import json, os, sys, types
        from _rheplicant_bootstrap.process import RuntimeSpec
        from _rheplicant_bootstrap.runtime import establish_runtime

        class Config:
            def read(self, name):
                return False

        seen = {}
        def import_main():
            seen["platform"] = os.environ.get("JAX_PLATFORMS")
            fake = types.ModuleType("jax")
            fake.config = Config()
            fake.default_backend = lambda: "gpu"
            sys.modules["jax"] = fake
            return object()

        establish_runtime(
            RuntimeSpec(False, "auto", None, {}), import_main=import_main
        )
        print(json.dumps(seen))
        """,
        environment={"JAX_PLATFORMS": "gpu"},
    )
    assert result == {"platform": "gpu"}


@pytest.mark.parametrize(
    ("existing", "requested"), [(False, True), (True, False)]
)
def test_existing_jax_refuses_both_x64_mismatch_directions(existing, requested):
    result = _clean_probe(
        f"""
        import json, os
        os.environ["JAX_ENABLE_X64"] = {('true' if existing else 'false')!r}
        os.environ["JAX_PLATFORMS"] = "cpu"
        import jax
        from _rheplicant_bootstrap.errors import ConfigError
        from _rheplicant_bootstrap.process import RuntimeSpec
        from _rheplicant_bootstrap.runtime import establish_runtime

        called = False
        def import_main():
            global called
            called = True
        try:
            establish_runtime(
                RuntimeSpec({requested!r}, "cpu", None, {{}}),
                import_main=import_main,
            )
        except ConfigError as error:
            print(json.dumps({{"message": str(error), "called": called}}))
        else:
            raise AssertionError("mismatch was accepted")
        """
    )
    assert result["called"] is False
    assert result["message"] == (
        f"runtime.jax_enable_x64: requested {requested!r}, but existing process "
        f"has {existing!r}."
    )


def test_existing_jax_refuses_an_explicit_platform_mismatch():
    result = _clean_probe(
        """
        import json, os
        os.environ["JAX_ENABLE_X64"] = "false"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import jax
        from _rheplicant_bootstrap.errors import ConfigError
        from _rheplicant_bootstrap.process import RuntimeSpec
        from _rheplicant_bootstrap.runtime import establish_runtime
        try:
            establish_runtime(
                RuntimeSpec(False, "gpu", None, {}), import_main=lambda: None
            )
        except ConfigError as error:
            print(json.dumps({"message": str(error)}))
        else:
            raise AssertionError("mismatch was accepted")
        """
    )
    assert result["message"] == (
        "runtime.platform: requested 'gpu', but existing process selected 'cpu'."
    )


def test_unavailable_backend_is_wrapped_at_the_main_import_boundary():
    result = _clean_probe(
        """
        import json, sys, types
        from _rheplicant_bootstrap.errors import ConfigError
        from _rheplicant_bootstrap.process import RuntimeSpec
        from _rheplicant_bootstrap.runtime import establish_runtime

        def import_main():
            class Config:
                def read(self, name):
                    return False
            def unavailable():
                raise RuntimeError("backend is unavailable")
            fake = types.ModuleType("jax")
            fake.config = Config()
            fake.default_backend = unavailable
            sys.modules["jax"] = fake
            return fake
        try:
            establish_runtime(
                RuntimeSpec(False, "gpu", None, {}),
                import_main=import_main,
            )
        except ConfigError as error:
            print(json.dumps({"message": str(error)}))
        else:
            raise AssertionError("unavailable backend was accepted")
        """
    )
    assert result["message"].startswith(
        "runtime: main import could not select/inspect the requested JAX backend "
    )


def test_already_imported_matching_jax_is_not_mutated_and_main_import_runs_once():
    result = _clean_probe(
        """
        import json, os
        os.environ["JAX_ENABLE_X64"] = "false"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import jax
        from _rheplicant_bootstrap.process import RuntimeSpec
        from _rheplicant_bootstrap.runtime import establish_runtime

        before = dict(os.environ)
        calls = []
        session, returned = establish_runtime(
            RuntimeSpec(False, "cpu", None, {}),
            import_main=lambda: calls.append("main") or "loaded",
        )
        print(json.dumps({
            "calls": calls,
            "returned": returned,
            "unchanged": dict(os.environ) == before,
            "actual": [session.actual.jax_enable_x64, session.actual.backend],
        }))
        """
    )
    assert result == {
        "calls": ["main"],
        "returned": "loaded",
        "unchanged": True,
        "actual": [False, "cpu"],
    }


def test_runtime_records_and_projection_have_the_closed_field_order():
    assert tuple(field.name for field in fields(PriorEnvironment)) == (
        "jax_enable_x64",
        "jax_platforms",
    )
    assert tuple(field.name for field in fields(RuntimeActual)) == (
        "jax_enable_x64",
        "backend",
    )
    assert tuple(field.name for field in fields(RuntimeSession)) == (
        "requested",
        "prior_environment",
        "actual",
    )
    assert RUNTIME_ROW_KEYS == ("requested", "actual", "prior_environment")
    assert RUNTIME_REQUESTED_ROW_KEYS == ("jax_enable_x64", "platform")
    assert RUNTIME_ACTUAL_ROW_KEYS == ("jax_enable_x64", "backend")
    assert RUNTIME_PRIOR_ENVIRONMENT_ROW_KEYS == (
        "jax_enable_x64",
        "jax_platforms",
    )

    session = RuntimeSession(
        RuntimeSpec(True, "cpu", 17, {"worker": 19}),
        PriorEnvironment("false", "gpu"),
        RuntimeActual(True, "cpu"),
    )
    row = runtime_audit_row(session)
    assert tuple(row) == RUNTIME_ROW_KEYS
    assert tuple(row["requested"]) == RUNTIME_REQUESTED_ROW_KEYS
    assert tuple(row["actual"]) == RUNTIME_ACTUAL_ROW_KEYS
    assert tuple(row["prior_environment"]) == RUNTIME_PRIOR_ENVIRONMENT_ROW_KEYS
    assert row == {
        "requested": {"jax_enable_x64": True, "platform": "cpu"},
        "actual": {"jax_enable_x64": True, "backend": "cpu"},
        "prior_environment": {"jax_enable_x64": "false", "jax_platforms": "gpu"},
    }
    assert "seed" not in row["requested"]
    assert "seeds" not in row["requested"]
    with pytest.raises(TypeError):
        row["actual"]["backend"] = "gpu"


def test_runtime_projection_refuses_an_unverified_or_malformed_session():
    requested = RuntimeSpec(False, "auto", None, {})
    with pytest.raises(ConfigError, match="has not been verified"):
        runtime_audit_row(RuntimeSession(requested, PriorEnvironment(None, None)))

    malformed_actual = RuntimeActual(False, "cpu")
    malformed = RuntimeSession(
        requested, PriorEnvironment(None, None), malformed_actual
    )
    object.__setattr__(malformed.actual, "backend", "")
    with pytest.raises(ConfigError, match="backend"):
        runtime_audit_row(malformed)


def test_runtime_records_validate_and_canonicalize_direct_construction():
    class Text(str):
        def __str__(self):
            raise AssertionError("foreign str must not run")

    prior = PriorEnvironment(Text("true"), Text("cpu"))
    actual = RuntimeActual(True, Text("cpu"))
    assert type(prior.jax_enable_x64) is str
    assert type(prior.jax_platforms) is str
    assert type(actual.backend) is str

    spec = RuntimeSpec(True, "cpu", None, {})
    for constructor, args in (
        (PriorEnvironment, (1, None)),
        (RuntimeActual, (1, "cpu")),
        (RuntimeActual, (False, "")),
        (RuntimeSession, (object(), prior)),
        (RuntimeSession, (spec, object())),
        (RuntimeSession, (spec, prior, object())),
    ):
        with pytest.raises(ConfigError):
            constructor(*args)


def test_runtime_session_detaches_all_nested_input_records():
    requested = RuntimeSpec(False, "auto", 7, {"worker": 9})
    prior = PriorEnvironment("false", "cpu")
    actual = RuntimeActual(False, "cpu")
    session = RuntimeSession(requested, prior, actual)

    object.__setattr__(requested, "platform", "gpu")
    object.__setattr__(prior, "jax_platforms", "gpu")
    object.__setattr__(actual, "backend", "gpu")

    assert session.requested is not requested
    assert session.requested.platform == "auto"
    assert session.requested.seeds == {"worker": 9}
    assert session.prior_environment is not prior
    assert session.prior_environment.jax_platforms == "cpu"
    assert session.actual is not actual
    assert session.actual.backend == "cpu"


def test_verify_uses_one_requested_snapshot_across_jax_callbacks(monkeypatch):
    import types

    session = RuntimeSession(
        RuntimeSpec(False, "auto", None, {}),
        PriorEnvironment(None, None),
    )

    class Config:
        def read(self, name):
            object.__setattr__(session.requested, "platform", "cpu")
            return False

    fake_jax = types.ModuleType("jax")
    fake_jax.config = Config()
    fake_jax.default_backend = lambda: "cpu"
    monkeypatch.setitem(sys.modules, "jax", fake_jax)

    actual = session.verify(boundary="probe")
    assert actual.backend == "cpu"
    assert session.requested.platform == "auto"


def test_establish_runtime_detaches_caller_requested_before_main_import():
    result = _clean_probe(
        """
        import json, sys, types
        from _rheplicant_bootstrap.process import RuntimeSpec
        from _rheplicant_bootstrap.runtime import establish_runtime

        requested = RuntimeSpec(False, "auto", None, {})
        def import_main():
            object.__setattr__(requested, "platform", "gpu")
            fake = types.ModuleType("jax")
            class Config:
                def read(self, name):
                    return False
            fake.config = Config()
            fake.default_backend = lambda: "cpu"
            sys.modules["jax"] = fake
            return "main"

        session, main = establish_runtime(requested, import_main=import_main)
        print(json.dumps({
            "caller": requested.platform,
            "session": session.requested.platform,
            "main": main,
        }))
        """
    )
    assert result == {"caller": "gpu", "session": "auto", "main": "main"}


@pytest.mark.parametrize(
    ("constructor", "args"),
    (
        (PriorEnvironment, ("\ud800", None)),
        (PriorEnvironment, (None, "\ud800")),
        (RuntimeActual, (False, "\ud800")),
    ),
)
def test_runtime_records_reject_non_utf8_retained_text(constructor, args):
    with pytest.raises(ConfigError, match="UTF-8"):
        constructor(*args)


@pytest.mark.parametrize(
    ("target", "field_name"),
    (
        ("prior", "jax_enable_x64"),
        ("prior", "jax_platforms"),
        ("actual", "backend"),
    ),
)
def test_runtime_projection_rejects_forged_non_utf8_retained_text(
    target,
    field_name,
):
    requested = RuntimeSpec(False, "auto", None, {})
    prior = PriorEnvironment(None, None)
    actual = RuntimeActual(False, "cpu")
    session = RuntimeSession(requested, prior, actual)
    retained = session.prior_environment if target == "prior" else session.actual
    object.__setattr__(retained, field_name, "\ud800")
    with pytest.raises(ConfigError, match="UTF-8"):
        runtime_audit_row(session)


def test_runtime_session_revalidates_forged_nested_records_at_construction():
    spec = RuntimeSpec(False, "cpu", None, {})
    prior = PriorEnvironment(None, None)
    actual = RuntimeActual(False, "cpu")

    object.__setattr__(spec, "platform", object())
    with pytest.raises(ConfigError, match="requested platform"):
        RuntimeSession(spec, prior, actual)

    spec = RuntimeSpec(False, "cpu", None, {})
    object.__setattr__(prior, "jax_platforms", object())
    with pytest.raises(ConfigError, match="prior_environment"):
        RuntimeSession(spec, prior, actual)

    prior = PriorEnvironment(None, None)
    object.__setattr__(actual, "backend", "")
    with pytest.raises(ConfigError, match="backend"):
        RuntimeSession(spec, prior, actual)


def test_runtime_projection_rechecks_forged_requested_and_actual_state():
    spec = RuntimeSpec(False, "cpu", None, {})
    session = RuntimeSession(
        spec, PriorEnvironment(None, None), RuntimeActual(False, "cpu")
    )
    object.__setattr__(session.actual, "jax_enable_x64", True)
    with pytest.raises(ConfigError, match="contradicts"):
        runtime_audit_row(session)

    object.__setattr__(session.actual, "jax_enable_x64", False)
    object.__setattr__(session.actual, "backend", "gpu")
    with pytest.raises(ConfigError, match="contradicts"):
        runtime_audit_row(session)

    object.__setattr__(session.requested, "platform", object())
    with pytest.raises(ConfigError, match="requested platform"):
        runtime_audit_row(session)


@pytest.mark.parametrize(
    "error",
    [ValueError("ordinary"), ConfigError("configured"), KeyboardInterrupt(), SystemExit()],
)
def test_import_main_failures_are_called_once_and_propagate_unchanged(error):
    import jax

    requested = RuntimeSpec(
        bool(jax.config.read("jax_enable_x64")), "auto", None, {}
    )
    calls = []

    def fail():
        calls.append("main")
        raise error

    with pytest.raises(BaseException) as caught:
        establish_runtime(requested, import_main=fail)
    assert caught.value is error
    assert calls == ["main"]


def test_noncallable_import_main_retains_direct_python_type_error():
    import jax

    requested = RuntimeSpec(
        bool(jax.config.read("jax_enable_x64")), "auto", None, {}
    )
    with pytest.raises(TypeError):
        establish_runtime(requested, import_main=None)  # type: ignore[arg-type]


def test_main_import_mutation_is_refused_by_the_post_callback_verification():
    result = _clean_probe(
        """
        import json, sys, types
        from _rheplicant_bootstrap.errors import ConfigError
        from _rheplicant_bootstrap.process import RuntimeSpec
        from _rheplicant_bootstrap.runtime import establish_runtime

        class Config:
            value = False
            def read(self, name):
                return self.value
        config = Config()
        calls = []
        def import_main():
            calls.append("main")
            fake = types.ModuleType("jax")
            fake.config = config
            fake.default_backend = lambda: "cpu"
            sys.modules["jax"] = fake
            config.value = True
            return fake
        try:
            establish_runtime(
                RuntimeSpec(False, "cpu", None, {}), import_main=import_main
            )
        except ConfigError as error:
            print(json.dumps({"calls": calls, "message": str(error)}))
        else:
            raise AssertionError("post-import mutation was accepted")
        """
    )
    assert result == {
        "calls": ["main"],
        "message": (
            "runtime.jax_enable_x64: requested False, but main import has True."
        ),
    }


def test_verifier_wraps_ordinary_inspection_errors_without_foreign_render_hooks(
    monkeypatch,
):
    import jax

    descriptor_calls = 0
    render_calls = 0

    class HostileMeta(type):
        @property
        def __name__(cls):
            nonlocal descriptor_calls
            descriptor_calls += 1
            return "ForgedName"

    class HostileError(RuntimeError, metaclass=HostileMeta):
        @property
        def args(self):
            raise AssertionError("foreign args descriptor must not run")

        def __str__(self):
            nonlocal render_calls
            render_calls += 1
            raise AssertionError("foreign str must not run")

        def __repr__(self):
            nonlocal render_calls
            render_calls += 1
            raise AssertionError("foreign repr must not run")

    def fail():
        raise HostileError("safe detail")

    monkeypatch.setattr(jax, "default_backend", fail)
    session = RuntimeSession(
        RuntimeSpec(bool(jax.config.read("jax_enable_x64")), "auto", None, {}),
        PriorEnvironment(None, None),
    )
    with pytest.raises(ConfigError) as caught:
        session.verify(boundary="probe")
    expected = (
        "runtime: probe could not select/inspect the requested JAX backend "
        "(HostileError: safe detail)."
    )
    if str(caught.value) != expected:
        pytest.fail(f"unexpected static diagnostic: {caught.value.args!r}")
    if descriptor_calls != 0:
        pytest.fail(f"metaclass __name__ descriptor ran {descriptor_calls} times")
    rendered = "".join(traceback.format_exception(caught.value))
    assert "HostileError: safe detail" in rendered
    assert descriptor_calls == 0
    assert render_calls == 0


def test_verifier_does_not_swallow_baseexception_from_jax_inspection(monkeypatch):
    import jax

    def stop():
        raise KeyboardInterrupt

    monkeypatch.setattr(jax, "default_backend", stop)
    session = RuntimeSession(
        RuntimeSpec(bool(jax.config.read("jax_enable_x64")), "auto", None, {}),
        PriorEnvironment(None, None),
    )
    with pytest.raises(KeyboardInterrupt):
        session.verify(boundary="probe")


def test_verifier_statically_summarizes_an_oversized_integer_exception(
    monkeypatch,
):
    import jax

    def fail():
        raise ValueError(10**5000)

    monkeypatch.setattr(jax, "default_backend", fail)
    session = RuntimeSession(
        RuntimeSpec(bool(jax.config.read("jax_enable_x64")), "auto", None, {}),
        PriorEnvironment(None, None),
    )
    with pytest.raises(ConfigError) as caught:
        session.verify(boundary="probe")
    assert str(caught.value) == (
        "runtime: probe could not select/inspect the requested JAX backend "
        "(ValueError: details unavailable)."
    )


def test_verifier_statically_summarizes_an_oversized_string_exception(
    monkeypatch,
):
    import jax

    def fail():
        raise ValueError("x" * (1024 * 1024))

    monkeypatch.setattr(jax, "default_backend", fail)
    session = RuntimeSession(
        RuntimeSpec(bool(jax.config.read("jax_enable_x64")), "auto", None, {}),
        PriorEnvironment(None, None),
    )
    with pytest.raises(ConfigError) as caught:
        session.verify(boundary="probe")
    assert str(caught.value) == (
        "runtime: probe could not select/inspect the requested JAX backend "
        "(ValueError: details unavailable)."
    )


def test_verifier_statically_summarizes_a_non_utf8_string_exception(
    monkeypatch,
):
    import jax

    def fail():
        raise ValueError("\ud800")

    monkeypatch.setattr(jax, "default_backend", fail)
    session = RuntimeSession(
        RuntimeSpec(bool(jax.config.read("jax_enable_x64")), "auto", None, {}),
        PriorEnvironment(None, None),
    )
    with pytest.raises(ConfigError) as caught:
        session.verify(boundary="probe")
    message = str(caught.value)
    assert message == (
        "runtime: probe could not select/inspect the requested JAX backend "
        "(ValueError: details unavailable)."
    )
    assert message.encode("utf-8", "strict")


def test_verifier_rejects_a_non_utf8_boundary_before_jax_inspection(
    monkeypatch,
):
    from types import ModuleType

    calls = []

    class UntouchedJax(ModuleType):
        def __getattribute__(self, name):
            if name in ("config", "default_backend"):
                calls.append(name)
                pytest.fail("JAX was inspected before boundary validation")
            return ModuleType.__getattribute__(self, name)

    monkeypatch.setitem(sys.modules, "jax", UntouchedJax("jax"))
    session = RuntimeSession(
        RuntimeSpec(False, "auto", None, {}),
        PriorEnvironment(None, None),
    )
    with pytest.raises(ConfigError, match="UTF-8"):
        session.verify(boundary="\ud800")
    assert calls == []


def test_establishment_never_calls_an_exec_primitive(monkeypatch):
    import jax

    from _rheplicant_bootstrap import runtime as runtime_module

    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("exec is forbidden")

    for name in dir(runtime_module.os):
        if name.startswith("exec"):
            monkeypatch.setattr(runtime_module.os, name, forbidden)
    requested = RuntimeSpec(
        bool(jax.config.read("jax_enable_x64")), "auto", None, {}
    )
    establish_runtime(requested, import_main=lambda: object())
    assert calls == []


def test_runtime_modules_remain_jax_free_until_verification_is_called():
    result = _clean_probe(
        """
        import json, sys
        import _rheplicant_bootstrap.runtime
        print(json.dumps(sorted({name.split('.')[0] for name in sys.modules})))
        """
    )
    assert set(result).isdisjoint(
        {"jax", "jaxlib", "equinox", "numpy", "numpyro", "rheplicant"}
    )
