"""Audited, ordered plugin imports for Config Plan 4A."""

from __future__ import annotations

import hashlib
import importlib.machinery
import itertools
import json
import math
import os
import subprocess
import sys
import traceback
import tracemalloc
from collections.abc import Mapping
from dataclasses import fields
from inspect import signature
from pathlib import Path, PurePosixPath
from types import MappingProxyType, ModuleType

import pytest

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.plugins import (
    PLUGIN_DISTRIBUTION_ROW_KEYS,
    PLUGIN_ROW_KEYS,
    PluginDistributionRecord,
    PluginRecord,
    import_plugin,
    plugin_audit_row,
)
from _rheplicant_bootstrap.types import UNAVAILABLE_REASONS

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


class _FakeDistribution:
    def __init__(
        self,
        name: str,
        root: Path,
        *,
        files: tuple[str, ...] = (),
        version: str | BaseException = "1.0",
        direct_url: str | None | BaseException = None,
    ) -> None:
        self.metadata = {"Name": name}
        self._root = root
        self.files = tuple(PurePosixPath(item) for item in files)
        self._version = version
        self._direct_url = direct_url

    @property
    def version(self):
        if isinstance(self._version, BaseException):
            raise self._version
        return self._version

    def locate_file(self, entry):
        return self._root / Path(*PurePosixPath(entry).parts)

    def read_text(self, name):
        assert name == "direct_url.json"
        if isinstance(self._direct_url, BaseException):
            raise self._direct_url
        return self._direct_url


def _module(name: str, *, origin: str = "<generated>", loader=None) -> ModuleType:
    module = ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(
        name, loader, origin=origin
    )
    return module


def _direct_url_metadata_row(monkeypatch, module_name, raw_direct_url):
    from _rheplicant_bootstrap import plugins as plugin_module

    module = _module(module_name)
    distribution = _FakeDistribution(
        f"{module_name}-dist",
        Path("/tmp"),
        direct_url=raw_direct_url,
    )
    monkeypatch.setattr(
        plugin_module.importlib,
        "import_module",
        lambda name: module,
    )
    monkeypatch.setattr(
        plugin_module.metadata,
        "packages_distributions",
        lambda: {module_name: [f"{module_name}-dist"]},
    )
    monkeypatch.setattr(plugin_module.metadata, "distributions", lambda: ())
    monkeypatch.setattr(
        plugin_module.metadata,
        "distribution",
        lambda name: distribution,
    )
    return import_plugin(module_name).distributions[0]


def _valid_distribution(**changes) -> PluginDistributionRecord:
    values = {
        "name": "example-dist",
        "version": "1.2.3",
        "version_reason": None,
        "direct_url": {"url": "file:///tmp/example", "dir_info": {}},
        "direct_url_reason": None,
    }
    values.update(changes)
    return PluginDistributionRecord(**values)


def _valid_record(**changes) -> PluginRecord:
    values = {
        "name": "example.plugin",
        "already_imported": False,
        "origin": "/tmp/example.py",
        "origin_reason": None,
        "loader_type": "_frozen_importlib_external.SourceFileLoader",
        "loader_type_reason": None,
        "resolved_path": "/tmp/example.py",
        "resolved_path_reason": None,
        "distributions": (_valid_distribution(),),
        "distributions_reason": None,
        "code_hash": "a" * 64,
        "code_hash_reason": None,
        "unobserved_io": True,
    }
    values.update(changes)
    return PluginRecord(**values)


@pytest.mark.parametrize("forged_type", (None, str))
@pytest.mark.parametrize("target", ("import", "distribution"))
def test_plugin_string_boundaries_never_read_instance_class_descriptors(
    forged_type,
    target,
):
    value = _InstanceClassTrap(forged_type=forged_type)
    with pytest.raises(ConfigError) as caught:
        if target == "import":
            import_plugin(value)
        else:
            _valid_distribution(name=value)
    assert value.calls == 0
    assert "descriptor secret" not in str(caught.value)


def test_plugin_record_fields_and_projection_keys_have_exact_order():
    assert tuple(field.name for field in fields(PluginDistributionRecord)) == (
        "name",
        "version",
        "version_reason",
        "direct_url",
        "direct_url_reason",
    )
    assert tuple(field.name for field in fields(PluginRecord)) == (
        "name",
        "already_imported",
        "origin",
        "origin_reason",
        "loader_type",
        "loader_type_reason",
        "resolved_path",
        "resolved_path_reason",
        "distributions",
        "distributions_reason",
        "code_hash",
        "code_hash_reason",
        "unobserved_io",
    )
    assert tuple(signature(PluginRecord).parameters) == (
        "name",
        "already_imported",
        "origin",
        "origin_reason",
        "loader_type",
        "loader_type_reason",
        "resolved_path",
        "resolved_path_reason",
        "distributions",
        "distributions_reason",
        "code_hash",
        "code_hash_reason",
        "unobserved_io",
    )
    assert PLUGIN_DISTRIBUTION_ROW_KEYS == (
        "name",
        "version",
        "version_reason",
        "direct_url",
        "direct_url_reason",
    )
    assert PLUGIN_ROW_KEYS == (
        "name",
        "already_imported",
        "origin",
        "origin_reason",
        "loader_type",
        "loader_type_reason",
        "resolved_path",
        "resolved_path_reason",
        "distributions",
        "distributions_reason",
        "code_hash",
        "code_hash_reason",
        "unobserved_io",
    )
    assert UNAVAILABLE_REASONS == (
        "not_installed",
        "not_a_git_checkout",
        "command_failed",
        "timeout",
        "no_origin",
        "not_regular_file",
        "unreadable",
        "namespace_package",
        "extension_module",
        "generated_module",
        "no_distribution",
        "missing_direct_url",
        "unobserved_executable_io",
    )


@pytest.mark.parametrize(
    ("constructor", "changes"),
    (
        (_valid_distribution, {"name": "\ud800"}),
        (_valid_distribution, {"version": "\ud800"}),
        (_valid_record, {"origin": "/tmp/\ud800.py"}),
        (_valid_record, {"loader_type": "module.\ud800"}),
        (_valid_record, {"resolved_path": "/tmp/\ud800.py"}),
    ),
)
def test_plugin_records_reject_non_utf8_retained_text(constructor, changes):
    with pytest.raises(ConfigError, match="UTF-8"):
        constructor(**changes)


@pytest.mark.parametrize(
    ("target", "field_name"),
    (
        ("distribution", "name"),
        ("distribution", "version"),
        ("record", "origin"),
        ("record", "loader_type"),
        ("record", "resolved_path"),
    ),
)
def test_plugin_projection_rejects_forged_non_utf8_retained_text(
    target,
    field_name,
):
    record = _valid_record()
    if target == "distribution":
        object.__setattr__(record.distributions[0], field_name, "\ud800")
    else:
        object.__setattr__(record, field_name, "\ud800")
    with pytest.raises(ConfigError, match="UTF-8"):
        plugin_audit_row(record)


def test_regular_plugin_facts_hash_the_resolved_artifact(tmp_path, monkeypatch):
    package = tmp_path / "fact_plugin"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("VALUE = 7\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(plugin_module.metadata, "distributions", lambda: ())

    record = import_plugin("fact_plugin")
    assert record.name == "fact_plugin"
    assert record.already_imported is False
    assert record.origin == str(source)
    assert record.origin_reason is None
    assert record.loader_type.endswith(".SourceFileLoader")
    assert record.loader_type_reason is None
    assert record.resolved_path == str(source.resolve())
    assert record.resolved_path_reason is None
    assert record.code_hash == hashlib.sha256(source.read_bytes()).hexdigest()
    assert record.code_hash_reason is None
    assert record.distributions == ()
    assert record.distributions_reason == "no_distribution"
    assert record.unobserved_io is True


def test_plugin_cannot_hide_source_hash_by_mutating_extension_suffixes(
    tmp_path, monkeypatch
):
    source = tmp_path / "suffix_mutator_plugin.py"
    source.write_text(
        "import importlib.machinery\n"
        "importlib.machinery.EXTENSION_SUFFIXES[:] = ['.py']\n"
    )
    expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.syspath_prepend(str(tmp_path))

    from _rheplicant_bootstrap import plugins as plugin_module

    suffixes = list(importlib.machinery.EXTENSION_SUFFIXES)
    monkeypatch.setattr(importlib.machinery, "EXTENSION_SUFFIXES", suffixes)
    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(plugin_module.metadata, "distributions", lambda: ())

    record = import_plugin("suffix_mutator_plugin")
    assert importlib.machinery.EXTENSION_SUFFIXES == [".py"]
    assert record.resolved_path == str(source.resolve())
    assert record.code_hash == expected_hash
    assert record.code_hash_reason is None


def test_loader_cannot_forge_extension_identity_through_class_descriptor(
    tmp_path, monkeypatch
):
    source = tmp_path / "class_forgery_plugin.py"
    source.write_text("VALUE = 1\n")
    expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    descriptor_calls = 0

    class Loader:
        @property
        def __class__(self):
            nonlocal descriptor_calls
            descriptor_calls += 1
            return importlib.machinery.ExtensionFileLoader

    module = _module(
        "class_forgery_plugin", origin=str(source), loader=Loader()
    )

    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(
        plugin_module.importlib, "import_module", lambda name: module
    )
    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(plugin_module.metadata, "distributions", lambda: ())

    record = import_plugin("class_forgery_plugin")
    if descriptor_calls != 0:
        pytest.fail(f"loader __class__ descriptor ran {descriptor_calls} times")
    assert record.code_hash == expected_hash
    assert record.code_hash_reason is None


def test_plugins_import_once_in_order_and_each_is_verified(monkeypatch):
    from _rheplicant_bootstrap import plugins as plugin_module

    events: list[tuple[str, str]] = []
    modules = {
        name: _module(name) for name in ("pkg.a", "pkg.b")
    }
    monkeypatch.setattr(
        plugin_module.importlib,
        "import_module",
        lambda name: events.append(("import", name)) or modules[name],
    )
    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(plugin_module.metadata, "distributions", lambda: ())

    class Session:
        def verify(self, *, boundary):
            events.append(("verify", boundary))

    session = Session()
    records = []
    for name in ("pkg.a", "pkg.b"):
        row = import_plugin(name)
        events.append(("record", name))
        records.append(row)
        session.verify(boundary=f"plugin {name!r}")

    assert events == [
        ("import", "pkg.a"),
        ("record", "pkg.a"),
        ("verify", "plugin 'pkg.a'"),
        ("import", "pkg.b"),
        ("record", "pkg.b"),
        ("verify", "plugin 'pkg.b'"),
    ]
    assert tuple(record.name for record in records) == ("pkg.a", "pkg.b")


def test_already_imported_is_recorded_without_reexecuting_the_module(monkeypatch):
    from _rheplicant_bootstrap import plugins as plugin_module

    name = "fixture.already_loaded"
    loaded = _module(name)
    monkeypatch.setitem(sys.modules, name, loaded)
    calls = []
    monkeypatch.setattr(
        plugin_module.importlib,
        "import_module",
        lambda requested: calls.append(requested) or sys.modules[requested],
    )
    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(plugin_module.metadata, "distributions", lambda: ())

    record = import_plugin(name)
    assert calls == [name]
    assert record.already_imported is True


def test_module_subclass_cannot_forge_a_missing_spec_with_a_descriptor(
    monkeypatch,
):
    from _rheplicant_bootstrap import plugins as plugin_module

    descriptor_calls = 0

    class HostileModule(ModuleType):
        @property
        def __spec__(self):
            nonlocal descriptor_calls
            descriptor_calls += 1
            raise AttributeError("forged missing spec")

    module = HostileModule("hostile_spec_plugin")
    monkeypatch.setattr(
        plugin_module.importlib, "import_module", lambda name: module
    )
    with pytest.raises(ConfigError, match="module specification"):
        import_plugin("hostile_spec_plugin")
    assert descriptor_calls == 0


def test_nonmodule_cannot_forge_module_identity_through_class_descriptor(
    monkeypatch,
):
    from _rheplicant_bootstrap import plugins as plugin_module

    descriptor_calls = 0

    class ForgedModule:
        @property
        def __class__(self):
            nonlocal descriptor_calls
            descriptor_calls += 1
            return ModuleType

    monkeypatch.setattr(
        plugin_module.importlib,
        "import_module",
        lambda name: ForgedModule(),
    )
    with pytest.raises(ConfigError, match="did not return a module"):
        import_plugin("forged_module_identity")
    if descriptor_calls != 0:
        pytest.fail(f"module __class__ descriptor ran {descriptor_calls} times")


@pytest.mark.parametrize(
    ("field", "forged"),
    (
        ("origin", "<generated>"),
        ("loader", None),
        ("submodule_search_locations", None),
    ),
)
def test_module_spec_data_descriptors_are_statically_refused_without_binding(
    monkeypatch, field, forged
):
    from _rheplicant_bootstrap import plugins as plugin_module

    descriptor_calls = 0

    class Spec:
        pass

    spec = Spec()
    spec.origin = "<generated>"
    spec.loader = None
    spec.submodule_search_locations = None

    def hostile_getter(_self):
        nonlocal descriptor_calls
        descriptor_calls += 1
        return forged

    setattr(Spec, field, property(hostile_getter))
    module = ModuleType(f"hostile_spec_{field}")
    module.__spec__ = spec
    monkeypatch.setattr(
        plugin_module.importlib, "import_module", lambda name: module
    )
    with pytest.raises(ConfigError, match="module specification"):
        import_plugin(f"hostile_spec_{field}")
    if descriptor_calls != 0:
        pytest.fail(f"spec descriptor ran {descriptor_calls} times for {field}")


def test_loader_class_namespace_is_read_without_metaclass_descriptors(
    monkeypatch,
):
    from _rheplicant_bootstrap import plugins as plugin_module

    descriptor_calls = 0

    class ForgedModule:
        def __get__(self, instance, owner=None):
            nonlocal descriptor_calls
            descriptor_calls += 1
            return "forged.loader"

        def __set__(self, instance, value):
            raise AssertionError("loader module descriptor must not be assigned")

    class LoaderMeta(type):
        pass

    LoaderMeta.__module__ = ForgedModule()

    class HonestLoader(metaclass=LoaderMeta):
        pass

    name = "static_loader_plugin"
    module = _module(name, loader=HonestLoader())
    monkeypatch.setattr(
        plugin_module.importlib, "import_module", lambda requested: module
    )
    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(plugin_module.metadata, "distributions", lambda: ())

    record = import_plugin(name)
    if descriptor_calls != 0:
        pytest.fail(f"loader metaclass descriptors ran {descriptor_calls} times")
    if "HonestLoader" not in record.loader_type:
        pytest.fail(f"loader type was not read statically: {record.loader_type!r}")
    if record.loader_type.startswith("forged.loader"):
        pytest.fail(f"forged loader namespace accepted: {record.loader_type!r}")


@pytest.mark.parametrize(
    "name", ["", ".plug", "plug.", "two..dots", "not-a-module", "for"]
)
def test_import_plugin_revalidates_and_canonicalizes_module_names(name):
    with pytest.raises(ConfigError, match="dot-separated Python module name"):
        import_plugin(name)

    class Text(str):
        pass

    with pytest.raises(ConfigError) as caught:
        import_plugin(Text(name))
    assert type(caught.value.args[0]) is str


def test_import_plugin_canonicalizes_a_valid_string_subclass(monkeypatch):
    from _rheplicant_bootstrap import plugins as plugin_module

    class Text(str):
        def __str__(self):
            raise AssertionError("foreign str must not run")

    module = _module("valid_plugin")
    monkeypatch.setattr(
        plugin_module.importlib, "import_module", lambda name: module
    )
    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(plugin_module.metadata, "distributions", lambda: ())

    record = import_plugin(Text("valid_plugin"))
    assert type(record.name) is str
    assert record.name == "valid_plugin"


def test_foreign_exception_is_wrapped_without_swallowing_base_exception(monkeypatch):
    from _rheplicant_bootstrap import plugins as plugin_module

    def fail_value_error(name):
        raise ValueError("foreign message")

    monkeypatch.setattr(plugin_module.importlib, "import_module", fail_value_error)
    with pytest.raises(ConfigError) as caught:
        import_plugin("fixture.raises_value_error")
    assert str(caught.value) == (
        "plugins: importing 'fixture.raises_value_error' raised "
        "ValueError: foreign message."
    )

    def fail_keyboard_interrupt(name):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        plugin_module.importlib, "import_module", fail_keyboard_interrupt
    )
    with pytest.raises(KeyboardInterrupt):
        import_plugin("fixture.raises_keyboard_interrupt")


def test_foreign_exception_message_never_calls_exception_render_hooks(monkeypatch):
    from _rheplicant_bootstrap import plugins as plugin_module

    descriptor_calls = 0
    render_calls = 0

    class HostileMeta(type):
        @property
        def __name__(cls):
            nonlocal descriptor_calls
            descriptor_calls += 1
            return "ForgedName"

    class HostileError(ValueError, metaclass=HostileMeta):
        def __str__(self):
            nonlocal render_calls
            render_calls += 1
            raise AssertionError("foreign str must not run")

        def __repr__(self):
            nonlocal render_calls
            render_calls += 1
            raise AssertionError("foreign repr must not run")

    def fail(name):
        raise HostileError("safe detail")

    monkeypatch.setattr(plugin_module.importlib, "import_module", fail)
    with pytest.raises(ConfigError) as caught:
        import_plugin("fixture.hostile_error")
    expected = (
        "plugins: importing 'fixture.hostile_error' raised "
        "HostileError: safe detail."
    )
    if str(caught.value) != expected:
        pytest.fail(f"unexpected static diagnostic: {caught.value.args!r}")
    if descriptor_calls != 0:
        pytest.fail(f"metaclass __name__ descriptor ran {descriptor_calls} times")
    rendered = "".join(traceback.format_exception(caught.value))
    assert "HostileError: safe detail" in rendered
    assert descriptor_calls == 0
    assert render_calls == 0


def test_foreign_exception_with_an_oversized_integer_has_a_static_summary(
    monkeypatch,
):
    from _rheplicant_bootstrap import plugins as plugin_module

    def fail(name):
        raise ValueError(10**5000)

    monkeypatch.setattr(plugin_module.importlib, "import_module", fail)
    with pytest.raises(ConfigError) as caught:
        import_plugin("fixture.oversized_error")
    assert str(caught.value) == (
        "plugins: importing 'fixture.oversized_error' raised "
        "ValueError: details unavailable."
    )


def test_foreign_exception_renders_only_an_exact_string_detail(monkeypatch):
    from _rheplicant_bootstrap import plugins as plugin_module

    hooks = 0

    class HostileText(str):
        def __str__(self):
            nonlocal hooks
            hooks += 1
            return "forged"

        def __repr__(self):
            nonlocal hooks
            hooks += 1
            return "forged"

    def fail(_name):
        raise ValueError(HostileText("secret"))

    monkeypatch.setattr(plugin_module.importlib, "import_module", fail)
    with pytest.raises(ConfigError) as caught:
        import_plugin("fixture.foreign_string_detail")
    assert str(caught.value) == (
        "plugins: importing 'fixture.foreign_string_detail' raised "
        "ValueError: details unavailable."
    )
    if hooks != 0:
        pytest.fail(f"foreign string detail hooks ran {hooks} times")


def test_foreign_exception_statically_summarizes_oversized_string_detail(
    monkeypatch,
):
    from _rheplicant_bootstrap import plugins as plugin_module

    def fail(_name):
        raise ValueError("x" * (1024 * 1024))

    monkeypatch.setattr(plugin_module.importlib, "import_module", fail)
    with pytest.raises(ConfigError) as caught:
        import_plugin("fixture.oversized_string_detail")
    assert str(caught.value) == (
        "plugins: importing 'fixture.oversized_string_detail' raised "
        "ValueError: details unavailable."
    )


def test_foreign_exception_statically_summarizes_non_utf8_string_detail(
    monkeypatch,
):
    from _rheplicant_bootstrap import plugins as plugin_module

    def fail(_name):
        raise ValueError("\ud800")

    monkeypatch.setattr(plugin_module.importlib, "import_module", fail)
    with pytest.raises(ConfigError) as caught:
        import_plugin("fixture.non_utf8_string_detail")
    message = str(caught.value)
    assert message == (
        "plugins: importing 'fixture.non_utf8_string_detail' raised "
        "ValueError: details unavailable."
    )
    assert message.encode("utf-8", "strict")


def test_namespace_generated_builtin_and_extension_reasons(tmp_path, monkeypatch):
    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(plugin_module.metadata, "distributions", lambda: ())

    namespace = tmp_path / "namespace_fact"
    namespace.mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    namespace_record = import_plugin("namespace_fact")
    assert namespace_record.origin is None
    assert namespace_record.origin_reason == "namespace_package"
    assert namespace_record.resolved_path_reason == "namespace_package"
    assert namespace_record.code_hash_reason == "namespace_package"

    generated_name = "generated_fact"
    generated = _module(generated_name)
    monkeypatch.setitem(sys.modules, generated_name, generated)
    generated_record = import_plugin(generated_name)
    assert generated_record.origin == "<generated>"
    assert generated_record.origin_reason is None
    assert generated_record.loader_type is None
    assert generated_record.loader_type_reason == "generated_module"
    assert generated_record.resolved_path_reason == "generated_module"
    assert generated_record.code_hash_reason == "generated_module"

    builtin_record = import_plugin("sys")
    assert builtin_record.origin == "built-in"
    assert builtin_record.loader_type.endswith(".BuiltinImporter")
    assert builtin_record.resolved_path_reason == "generated_module"
    assert builtin_record.code_hash_reason == "generated_module"

    # DERIVED, not hard-coded. `math` stood here, and it is an extension
    # module on this developer's interpreter and a BUILT-IN one on the CI
    # runner's: CPython builds differ in which stdlib C modules they compile
    # in, and a built-in has no file at all, so `resolved_path is not None`
    # failed against a record that was entirely correct. What this block is
    # about is the extension path, so it asks the interpreter which module
    # actually takes it.
    import importlib.machinery
    import importlib.util

    extension_name = next(
        (
            name
            for name in ("math", "_ctypes", "zlib", "_json", "binascii", "select")
            if (spec := importlib.util.find_spec(name)) is not None
            and isinstance(spec.loader, importlib.machinery.ExtensionFileLoader)
        ),
        None,
    )
    if extension_name is None:  # pragma: no cover - no such build met yet
        pytest.skip(
            "this interpreter compiles every stdlib C module in, so there is "
            "no extension module to take the extension path with. That is a "
            "property of the build, not a missing dependency."
        )
    extension_record = import_plugin(extension_name)
    assert extension_record.origin is not None
    assert extension_record.resolved_path is not None
    assert extension_record.resolved_path_reason is None
    assert extension_record.code_hash is None
    assert extension_record.code_hash_reason == "extension_module"


def test_extension_loader_identity_accepts_class_instance_and_subclass(
    tmp_path, monkeypatch
):
    from _rheplicant_bootstrap import plugins as plugin_module

    artifact = tmp_path / "extension_identity.bin"
    artifact.write_bytes(b"extension identity fixture\n")

    class ExtensionSubclass(importlib.machinery.ExtensionFileLoader):
        pass

    loaders = (
        (importlib.machinery.ExtensionFileLoader, True),
        (ExtensionSubclass, True),
        (
            importlib.machinery.ExtensionFileLoader(
                "extension_instance", str(artifact)
            ),
            True,
        ),
        (ExtensionSubclass("extension_subclass", str(artifact)), True),
        (importlib.machinery.SourceFileLoader, False),
        (object(), False),
    )
    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(plugin_module.metadata, "distributions", lambda: ())

    for index, (loader, expected_extension) in enumerate(loaders):
        name = f"extension_identity_{index}"
        module = _module(name, origin=str(artifact), loader=loader)
        monkeypatch.setattr(
            plugin_module.importlib,
            "import_module",
            lambda requested, module=module: module,
        )
        record = import_plugin(name)
        if expected_extension:
            assert record.code_hash is None
            assert record.code_hash_reason == "extension_module"
        else:
            assert record.code_hash == hashlib.sha256(
                artifact.read_bytes()
            ).hexdigest()
            assert record.code_hash_reason is None


@pytest.mark.parametrize("length", (256, 257))
def test_loader_class_text_has_an_exact_bounded_diagnostic_limit(
    monkeypatch, length
):
    from _rheplicant_bootstrap import plugins as plugin_module

    loader = type("Q" * length, (), {})
    loader.__module__ = "m" * length
    module = _module("bounded_loader_text", loader=loader)
    monkeypatch.setattr(
        plugin_module.importlib, "import_module", lambda name: module
    )
    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(plugin_module.metadata, "distributions", lambda: ())

    record = import_plugin("bounded_loader_text")
    if length == 256:
        assert record.loader_type == f"{'m' * length}.{'Q' * length}"
    else:
        assert record.loader_type == "builtins.unknown"


def test_non_regular_plugin_origin_has_closed_path_and_hash_reasons(
    tmp_path, monkeypatch
):
    from _rheplicant_bootstrap import plugins as plugin_module

    name = "directory_origin"
    module = _module(
        name,
        origin=str(tmp_path),
        loader=importlib.machinery.SourceFileLoader(name, str(tmp_path)),
    )
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(plugin_module.metadata, "distributions", lambda: ())

    record = import_plugin(name)
    assert record.origin == str(tmp_path)
    assert record.resolved_path is None
    assert record.resolved_path_reason == "not_regular_file"
    assert record.code_hash is None
    assert record.code_hash_reason == "not_regular_file"


def test_artifact_replaced_between_path_inspection_and_hash_open_is_unreadable(
    tmp_path, monkeypatch
):
    package = tmp_path / "replace_before_hash"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("VALUE = 'loaded'\n")
    replacement = tmp_path / "replacement.py"
    replacement.write_text("VALUE = 'replacement'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    __import__("replace_before_hash")

    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(plugin_module.metadata, "distributions", lambda: ())
    real_open = plugin_module.os.open
    swapped = False

    def replace_then_open(path, flags):
        nonlocal swapped
        if not swapped and os.fspath(path) == str(source.resolve()):
            replacement.replace(source)
            swapped = True
        return real_open(path, flags)

    monkeypatch.setattr(plugin_module.os, "open", replace_then_open)
    record = import_plugin("replace_before_hash")
    assert swapped is True
    assert record.resolved_path == str(source.resolve())
    assert record.code_hash is None
    assert record.code_hash_reason == "unreadable"


def test_artifact_truncated_during_descriptor_read_is_unreadable(
    tmp_path, monkeypatch
):
    package = tmp_path / "truncate_during_hash"
    package.mkdir()
    source = package / "__init__.py"
    source.write_bytes(b"VALUE = 1\n" + b"#" * (2 * 1024 * 1024))
    monkeypatch.syspath_prepend(str(tmp_path))
    __import__("truncate_during_hash")

    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(plugin_module.metadata, "distributions", lambda: ())
    real_read = plugin_module.os.read
    truncated = False

    def read_then_truncate(descriptor, size):
        nonlocal truncated
        chunk = real_read(descriptor, size)
        if chunk and not truncated:
            source.write_bytes(b"")
            truncated = True
        return chunk

    monkeypatch.setattr(plugin_module.os, "read", read_then_truncate)
    record = import_plugin("truncate_during_hash")
    assert truncated is True
    assert record.resolved_path == str(source.resolve())
    assert record.code_hash is None
    assert record.code_hash_reason == "unreadable"


def test_artifact_close_failure_is_reported_as_unreadable(tmp_path, monkeypatch):
    package = tmp_path / "close_failure_plugin"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    __import__("close_failure_plugin")

    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(plugin_module.metadata, "distributions", lambda: ())
    real_open = plugin_module.os.open
    real_close = plugin_module.os.close
    artifact_descriptor = None

    def remember_open(path, flags):
        nonlocal artifact_descriptor
        descriptor = real_open(path, flags)
        if os.fspath(path) == str(source.resolve()):
            artifact_descriptor = descriptor
        return descriptor

    def close_then_fail(descriptor):
        real_close(descriptor)
        if descriptor == artifact_descriptor:
            raise RuntimeError("close failed")

    monkeypatch.setattr(plugin_module.os, "open", remember_open)
    monkeypatch.setattr(plugin_module.os, "close", close_then_fail)
    record = import_plugin("close_failure_plugin")
    assert artifact_descriptor is not None
    assert record.code_hash is None
    assert record.code_hash_reason == "unreadable"


def test_artifact_read_and_close_failures_still_report_unreadable(
    tmp_path, monkeypatch
):
    package = tmp_path / "read_close_failure_plugin"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    __import__("read_close_failure_plugin")

    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(plugin_module.metadata, "distributions", lambda: ())
    real_open = plugin_module.os.open
    real_read = plugin_module.os.read
    real_close = plugin_module.os.close
    artifact_descriptor = None

    def remember_open(path, flags):
        nonlocal artifact_descriptor
        descriptor = real_open(path, flags)
        if os.fspath(path) == str(source.resolve()):
            artifact_descriptor = descriptor
        return descriptor

    def fail_read(descriptor, size):
        if descriptor == artifact_descriptor:
            raise RuntimeError("read failed")
        return real_read(descriptor, size)

    def close_then_fail(descriptor):
        real_close(descriptor)
        if descriptor == artifact_descriptor:
            raise ConfigError("forged close error")

    monkeypatch.setattr(plugin_module.os, "open", remember_open)
    monkeypatch.setattr(plugin_module.os, "read", fail_read)
    monkeypatch.setattr(plugin_module.os, "close", close_then_fail)
    record = import_plugin("read_close_failure_plugin")
    assert artifact_descriptor is not None
    assert record.code_hash is None
    assert record.code_hash_reason == "unreadable"


def test_artifact_close_baseexception_propagates(tmp_path, monkeypatch):
    package = tmp_path / "close_interrupt_plugin"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    __import__("close_interrupt_plugin")

    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(plugin_module.metadata, "distributions", lambda: ())
    real_open = plugin_module.os.open
    real_close = plugin_module.os.close
    artifact_descriptor = None

    def remember_open(path, flags):
        nonlocal artifact_descriptor
        descriptor = real_open(path, flags)
        if os.fspath(path) == str(source.resolve()):
            artifact_descriptor = descriptor
        return descriptor

    def close_then_interrupt(descriptor):
        real_close(descriptor)
        if descriptor == artifact_descriptor:
            raise KeyboardInterrupt

    monkeypatch.setattr(plugin_module.os, "open", remember_open)
    monkeypatch.setattr(plugin_module.os, "close", close_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        import_plugin("close_interrupt_plugin")


def test_growing_artifact_is_read_only_through_initial_size_plus_sentinel(
    tmp_path, monkeypatch
):
    package = tmp_path / "growing_hash_plugin"
    package.mkdir()
    source = package / "__init__.py"
    source.write_bytes(b"VALUE = 1\n")
    initial_size = source.stat().st_size
    monkeypatch.syspath_prepend(str(tmp_path))
    __import__("growing_hash_plugin")

    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(plugin_module.metadata, "distributions", lambda: ())
    real_open = plugin_module.os.open
    real_read = plugin_module.os.read
    artifact_descriptor = None
    requested_sizes: list[int] = []

    def remember_open(path, flags):
        nonlocal artifact_descriptor
        descriptor = real_open(path, flags)
        if os.fspath(path) == str(source.resolve()):
            artifact_descriptor = descriptor
        return descriptor

    def grow_then_read(descriptor, size):
        if descriptor != artifact_descriptor:
            return real_read(descriptor, size)
        requested_sizes.append(size)
        if len(requested_sizes) <= 3:
            with source.open("ab") as stream:
                stream.write(b"x")
            return real_read(descriptor, size)
        return b""

    monkeypatch.setattr(plugin_module.os, "open", remember_open)
    monkeypatch.setattr(plugin_module.os, "read", grow_then_read)
    record = import_plugin("growing_hash_plugin")
    if sum(requested_sizes) > initial_size + 1:
        pytest.fail(
            "artifact reader requested beyond the initial descriptor size "
            f"plus sentinel: {requested_sizes!r}"
        )
    if requested_sizes[-1:] != [1]:
        pytest.fail(f"artifact reader omitted the growth sentinel: {requested_sizes!r}")
    assert record.code_hash is None
    assert record.code_hash_reason == "unreadable"


@pytest.mark.parametrize("mode", ("over_return", "non_bytes", "early_eof"))
def test_hostile_artifact_read_results_are_unreadable_and_bounded(
    tmp_path, monkeypatch, mode
):
    package = tmp_path / f"hostile_read_{mode}"
    package.mkdir()
    source = package / "__init__.py"
    source.write_bytes(b"VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    __import__(f"hostile_read_{mode}")

    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(plugin_module.metadata, "distributions", lambda: ())
    real_open = plugin_module.os.open
    real_read = plugin_module.os.read
    artifact_descriptor = None
    calls = 0

    def remember_open(path, flags):
        nonlocal artifact_descriptor
        descriptor = real_open(path, flags)
        if os.fspath(path) == str(source.resolve()):
            artifact_descriptor = descriptor
        return descriptor

    def hostile_read(descriptor, size):
        nonlocal calls
        if descriptor != artifact_descriptor:
            return real_read(descriptor, size)
        calls += 1
        if mode == "over_return":
            return b"x" * (size + 1) if calls == 1 else b""
        if mode == "non_bytes":
            return "not bytes"
        if calls == 1:
            return real_read(descriptor, max(1, size // 2))
        return b""

    monkeypatch.setattr(plugin_module.os, "open", remember_open)
    monkeypatch.setattr(plugin_module.os, "read", hostile_read)
    record = import_plugin(f"hostile_read_{mode}")
    if mode in ("over_return", "non_bytes") and calls != 1:
        pytest.fail(f"invalid read result was consumed again: {calls} calls")
    assert record.code_hash is None
    assert record.code_hash_reason == "unreadable"


def test_distribution_candidates_are_top_level_plus_record_root_union(
    tmp_path, monkeypatch
):
    package = tmp_path / "candidate_plugin"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("VALUE = 1\n")
    unrelated = tmp_path / "unrelated_package"
    unrelated.mkdir()
    (unrelated / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))

    top_a = _FakeDistribution(
        "A_Name",
        tmp_path,
        version="2.0",
        direct_url='{"url":"https://example.invalid/a","archive_info":{}}',
    )
    record_dist = _FakeDistribution(
        "record_dist",
        tmp_path,
        files=("candidate_plugin/__init__.py",),
        version=LookupError("version unavailable"),
        direct_url="{}",
    )
    shared_site_bystander = _FakeDistribution(
        "bystander",
        tmp_path,
        files=("unrelated_package/__init__.py",),
        direct_url="{}",
    )
    by_name = {
        "a-name": top_a,
        "record-dist": record_dist,
    }

    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(
        plugin_module.metadata,
        "packages_distributions",
        lambda: {"candidate_plugin": ["zed.name", "A_Name", "a-name"]},
    )
    monkeypatch.setattr(
        plugin_module.metadata,
        "distributions",
        lambda: (record_dist, shared_site_bystander),
    )

    def lookup(name):
        if name == "zed-name":
            raise plugin_module.metadata.PackageNotFoundError(name)
        return by_name[name]

    monkeypatch.setattr(plugin_module.metadata, "distribution", lookup)

    record = import_plugin("candidate_plugin")
    assert tuple(item.name for item in record.distributions) == (
        "a-name",
        "record-dist",
        "zed-name",
    )
    assert "bystander" not in {item.name for item in record.distributions}

    a_name, record_root, missing = record.distributions
    assert (a_name.version, a_name.version_reason) == ("2.0", None)
    assert a_name.direct_url == {
        "url": "https://example.invalid/a",
        "archive_info": {},
    }
    assert a_name.direct_url_reason is None
    assert (record_root.version, record_root.version_reason) == (
        None,
        "unreadable",
    )
    assert record_root.direct_url == {}
    assert record_root.direct_url_reason is None
    assert (missing.version, missing.version_reason) == (None, "not_installed")
    assert missing.direct_url is None
    assert missing.direct_url_reason == "not_installed"


def test_no_candidate_never_guesses_a_distribution_from_import_name(
    tmp_path, monkeypatch
):
    package = tmp_path / "guess_me"
    package.mkdir()
    (package / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))

    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(plugin_module.metadata, "distributions", lambda: ())
    calls = []
    monkeypatch.setattr(
        plugin_module.metadata,
        "distribution",
        lambda name: calls.append(name) or pytest.fail("identity was guessed"),
    )

    record = import_plugin("guess_me")
    assert calls == []
    assert record.distributions == ()
    assert record.distributions_reason == "no_distribution"


def test_recorded_top_level_artifact_root_claims_a_package_submodule(
    tmp_path, monkeypatch
):
    package = tmp_path / "root_claim"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "submodule.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    distribution = _FakeDistribution(
        "root-claim-dist",
        tmp_path,
        files=("root_claim/__init__.py",),
        direct_url="{}",
    )

    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(
        plugin_module.metadata, "distributions", lambda: (distribution,)
    )
    monkeypatch.setattr(
        plugin_module.metadata,
        "distribution",
        lambda name: distribution,
    )

    record = import_plugin("root_claim.submodule")
    assert tuple(item.name for item in record.distributions) == (
        "root-claim-dist",
    )


def test_recorded_symlink_root_cannot_escape_the_distribution_base(
    tmp_path, monkeypatch
):
    metadata_base = tmp_path / "site"
    metadata_base.mkdir()
    shared = tmp_path / "shared"
    shared.mkdir()
    source = shared / "symlink_escape_plugin.py"
    source.write_text("VALUE = 1\n")
    (metadata_base / "claimed").symlink_to(shared, target_is_directory=True)
    monkeypatch.syspath_prepend(str(shared))
    distribution = _FakeDistribution(
        "symlink-escape-dist",
        metadata_base,
        files=("claimed/recorded.py",),
        direct_url="{}",
    )

    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(
        plugin_module.metadata, "distributions", lambda: (distribution,)
    )
    monkeypatch.setattr(
        plugin_module.metadata,
        "distribution",
        lambda name: pytest.fail("escaped symlink guessed a distribution"),
    )

    record = import_plugin("symlink_escape_plugin")
    assert record.distributions == ()
    assert record.distributions_reason == "no_distribution"


def test_record_locator_snapshots_base_and_repeated_top_level_only_once(
    tmp_path, monkeypatch
):
    source = tmp_path / "locator_cache_plugin.py"
    source.write_text("VALUE = 1\n")
    unrelated = tmp_path / "other"
    unrelated.mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    calls: dict[str, int] = {}

    class Distribution(_FakeDistribution):
        def locate_file(self, entry):
            parts = tuple(PurePosixPath(entry).parts)
            key = "<base>" if not parts else parts[0]
            calls[key] = calls.get(key, 0) + 1
            if calls[key] > 1:
                raise ConfigError(f"forged repeated locator call for {key}")
            return self._root if not parts else self._root / parts[0]

    distribution = Distribution(
        "locator-cache-dist",
        tmp_path,
        files=("other/first.py", "other/second.py"),
        direct_url="{}",
    )

    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(
        plugin_module.metadata, "distributions", lambda: (distribution,)
    )
    record = import_plugin("locator_cache_plugin")
    assert record.distributions == ()
    assert record.distributions_reason == "no_distribution"
    assert calls == {"<base>": 1, "other": 1}


def test_unsafe_record_paths_are_ignored_without_claiming_the_plugin(
    tmp_path, monkeypatch
):
    source = tmp_path / "unsafe_record_plugin.py"
    source.write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    distribution = _FakeDistribution(
        "unsafe-record-dist",
        tmp_path,
        files=(
            "/unsafe_record_plugin.py",
            "../unsafe_record_plugin.py",
            ".",
            "bad\0root/file.py",
            "back\\slash/file.py",
            "C:/unsafe_record_plugin.py",
        ),
        direct_url="{}",
    )

    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(
        plugin_module.metadata,
        "distributions",
        lambda: (distribution,),
    )
    monkeypatch.setattr(
        plugin_module.metadata,
        "distribution",
        lambda name: pytest.fail("unsafe RECORD guessed a distribution"),
    )

    record = import_plugin("unsafe_record_plugin")
    assert record.distributions == ()
    assert record.distributions_reason == "no_distribution"


def test_record_path_components_are_budgeted_before_collection(
    tmp_path, monkeypatch
):
    source = tmp_path / "record_component_budget.py"
    source.write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    class Parts:
        def __getitem__(self, index):
            if index >= 4:
                raise IndexError(index)
            return "component"

        def __len__(self):
            return 4

        def __iter__(self):
            return itertools.repeat("component", 4)

    class Entry:
        @property
        def parts(self):
            return Parts()

        def is_absolute(self):
            return False

    distribution = _FakeDistribution("component-budget-dist", tmp_path)
    distribution.files = (Entry(),)

    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(plugin_module, "_METADATA_EVIDENCE_LIMIT", 4)
    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(
        plugin_module.metadata, "distributions", lambda: (distribution,)
    )
    with pytest.raises(ConfigError, match="metadata.*budget"):
        import_plugin("record_component_budget")


def test_exact_string_record_path_is_budgeted_before_component_splitting(
    tmp_path, monkeypatch
):
    source = tmp_path / "record_text_budget.py"
    source.write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    distribution = _FakeDistribution("record-text-budget-dist", tmp_path)
    distribution.files = (("part/" * 200_000) + "module.py",)

    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(plugin_module, "_METADATA_EVIDENCE_LIMIT", 4)
    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(
        plugin_module.metadata, "distributions", lambda: (distribution,)
    )

    tracemalloc.start()
    try:
        with pytest.raises(ConfigError, match="metadata.*budget"):
            import_plugin("record_text_budget")
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    if peak >= 4 * 1024 * 1024:
        pytest.fail(f"RECORD text was pre-split before its budget: peak={peak}")


@pytest.mark.parametrize(
    "entry",
    (
        ("./" * 1_000) + "record_skipped_segments.py",
        "record_skipped_segments/" + ("/" * 1_000) + "module.py",
    ),
)
def test_exact_record_text_charges_dot_and_empty_segments(
    entry,
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "record_skipped_segments.py"
    source.write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    distribution = _FakeDistribution("skipped-segment-dist", tmp_path)
    distribution.files = (entry,)

    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(plugin_module, "_METADATA_EVIDENCE_LIMIT", 4)
    monkeypatch.setattr(
        plugin_module.metadata,
        "packages_distributions",
        lambda: {},
    )
    monkeypatch.setattr(
        plugin_module.metadata,
        "distributions",
        lambda: (distribution,),
    )
    with pytest.raises(ConfigError, match="metadata.*budget"):
        import_plugin("record_skipped_segments")


def test_long_single_record_component_uses_bounded_base_string_scans(
    tmp_path, monkeypatch
):
    source = tmp_path / "record_single_component.py"
    source.write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    distribution = _FakeDistribution("single-component-dist", tmp_path)
    distribution.files = ("x" * 1_000_000,)

    from _rheplicant_bootstrap import plugins as plugin_module

    range_calls = 0

    def forbidden_range(*_args):
        nonlocal range_calls
        range_calls += 1
        pytest.fail("RECORD component used a Python per-character range")

    monkeypatch.setattr(plugin_module, "_METADATA_EVIDENCE_LIMIT", 2)
    monkeypatch.setattr(plugin_module, "range", forbidden_range, raising=False)
    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(
        plugin_module.metadata, "distributions", lambda: (distribution,)
    )
    with pytest.raises(ConfigError, match="metadata.*budget"):
        import_plugin("record_single_component")
    if range_calls != 0:
        pytest.fail(f"RECORD component range ran {range_calls} times")


def test_exact_string_record_paths_preserve_posix_normalization(
    tmp_path, monkeypatch
):
    package = tmp_path / "record_posix_semantics"
    package.mkdir()
    source = package / "module.py"
    source.write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    distributions = []
    for index, entry in enumerate(
        (
            "record_posix_semantics//module.py",
            "./record_posix_semantics/./module.py",
            "record_posix_semantics/module.py/",
        )
    ):
        distribution = _FakeDistribution(f"posix-dist-{index}", tmp_path)
        distribution.files = (entry,)
        distributions.append(distribution)

    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(
        plugin_module.metadata,
        "distributions",
        lambda: tuple(distributions),
    )
    record = import_plugin("record_posix_semantics.module")
    assert tuple(row.name for row in record.distributions) == (
        "posix-dist-0",
        "posix-dist-1",
        "posix-dist-2",
    )


def test_record_path_absolute_protocol_requires_an_exact_bool(
    tmp_path, monkeypatch
):
    source = tmp_path / "record_absolute_protocol.py"
    source.write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    class Entry:
        parts = ("record_absolute_protocol.py",)

        def is_absolute(self):
            return 1

    distribution = _FakeDistribution("absolute-protocol-dist", tmp_path)
    distribution.files = (Entry(),)

    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(
        plugin_module.metadata, "distributions", lambda: (distribution,)
    )
    with pytest.raises(ConfigError, match="RECORD path inspection"):
        import_plugin("record_absolute_protocol")


def test_record_component_protocol_failures_are_static_and_baseexception_passes(
    tmp_path, monkeypatch
):
    source = tmp_path / "record_component_protocol.py"
    source.write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    hooks = 0

    class HostileError(RuntimeError):
        def __str__(self):
            nonlocal hooks
            hooks += 1
            return "forged"

        def __repr__(self):
            nonlocal hooks
            hooks += 1
            return "forged"

    class Parts:
        def __init__(self, error):
            self.error = error

        def __iter__(self):
            return self

        def __getitem__(self, index):
            raise IndexError(index)

        def __len__(self):
            return 1

        def __next__(self):
            raise self.error

    class Entry:
        def __init__(self, error):
            self.parts = Parts(error)

        def is_absolute(self):
            return False

    distribution = _FakeDistribution("component-protocol-dist", tmp_path)

    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(
        plugin_module.metadata, "distributions", lambda: (distribution,)
    )

    distribution.files = (Entry(HostileError("secret")),)
    with pytest.raises(ConfigError) as caught:
        import_plugin("record_component_protocol")
    if str(caught.value) != (
        "plugins: distribution metadata RECORD path component traversal failed."
    ):
        pytest.fail(f"unexpected RECORD protocol failure: {caught.value.args!r}")
    if hooks != 0:
        pytest.fail(f"RECORD exception rendering hooks ran {hooks} times")

    distribution.files = (Entry(KeyboardInterrupt()),)
    with pytest.raises(KeyboardInterrupt):
        import_plugin("record_component_protocol")


def test_hostile_record_locator_cannot_claim_the_filesystem_anchor(
    tmp_path, monkeypatch
):
    source = tmp_path / "anchor_claim_plugin.py"
    source.write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    class Distribution(_FakeDistribution):
        def locate_file(self, entry):
            if str(entry) in ("", "."):
                return self._root
            return Path("/")

    distribution = Distribution(
        "anchor-claim-dist",
        tmp_path,
        files=("claimed/__init__.py",),
        direct_url="{}",
    )

    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(
        plugin_module.metadata,
        "distributions",
        lambda: (distribution,),
    )

    record = import_plugin("anchor_claim_plugin")
    assert record.distributions == ()


def test_filesystem_anchor_cannot_be_used_as_distribution_metadata_base(
    tmp_path, monkeypatch
):
    source = tmp_path / "anchor_base_plugin.py"
    source.write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    first_component = source.resolve().parts[1]
    claimed_root = Path(source.resolve().anchor) / first_component

    class Distribution(_FakeDistribution):
        def locate_file(self, entry):
            if str(entry) in ("", "."):
                return Path(source.resolve().anchor)
            return claimed_root

    distribution = Distribution(
        "anchor-base-dist",
        tmp_path,
        files=(f"{first_component}/anything.py",),
        direct_url="{}",
    )
    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(
        plugin_module.metadata, "distributions", lambda: (distribution,)
    )

    record = import_plugin("anchor_base_plugin")
    assert record.distributions == ()


def test_distribution_with_no_record_files_is_a_non_candidate(
    tmp_path, monkeypatch
):
    package = tmp_path / "no_record_files"
    package.mkdir()
    (package / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    distribution = _FakeDistribution("no-record-dist", tmp_path)
    distribution.files = None

    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(
        plugin_module.metadata, "distributions", lambda: (distribution,)
    )
    record = import_plugin("no_record_files")
    assert record.distributions == ()


def test_origin_without_a_file_is_not_mislabeled_as_unreadable(
    tmp_path, monkeypatch
):
    from _rheplicant_bootstrap import plugins as plugin_module

    name = "missing_origin_plugin"
    missing = tmp_path / "missing.py"
    module = _module(
        name,
        origin=str(missing),
        loader=importlib.machinery.SourceFileLoader(name, str(missing)),
    )
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(plugin_module.metadata, "distributions", lambda: ())

    record = import_plugin(name)
    assert record.resolved_path is None
    assert record.resolved_path_reason == "not_regular_file"
    assert record.code_hash_reason == "not_regular_file"


def test_extension_suffix_is_unhashable_even_with_a_nonextension_loader(
    tmp_path, monkeypatch
):
    from _rheplicant_bootstrap import plugins as plugin_module

    name = "suffix_extension_plugin"
    artifact = tmp_path / f"plugin{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    artifact.write_bytes(b"not a real extension")
    module = _module(
        name,
        origin=str(artifact),
        loader=importlib.machinery.SourceFileLoader(name, str(artifact)),
    )
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(plugin_module.metadata, "distributions", lambda: ())

    record = import_plugin(name)
    assert record.resolved_path == str(artifact.resolve())
    assert record.code_hash is None
    assert record.code_hash_reason == "extension_module"


def test_namespace_without_a_regular_artifact_skips_record_enumeration(
    tmp_path, monkeypatch
):
    namespace = tmp_path / "namespace_no_scan"
    namespace.mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))

    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(
        plugin_module.metadata,
        "distributions",
        lambda: pytest.fail("namespace plugin scanned installed RECORD files"),
    )
    record = import_plugin("namespace_no_scan")
    assert record.origin_reason == "namespace_package"
    assert record.distributions == ()


@pytest.mark.parametrize(
    "direct_url",
    [
        "",
        "not json",
        "[]",
        '{"duplicate":1,"duplicate":2}',
        '{"value": NaN}',
        '{"value": Infinity}',
        '{"\\ud800":"value"}',
        '{"value":"\\ud800"}',
        '{"value":' + "[" * 101 + "0" + "]" * 101 + "}",
    ],
)
def test_malformed_direct_url_metadata_is_marked_unreadable(
    tmp_path, monkeypatch, direct_url
):
    package = tmp_path / "direct_url_fact"
    package.mkdir()
    (package / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    distribution = _FakeDistribution(
        "direct-url-dist",
        tmp_path,
        direct_url=direct_url,
    )

    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(
        plugin_module.metadata,
        "packages_distributions",
        lambda: {"direct_url_fact": ["direct-url-dist"]},
    )
    monkeypatch.setattr(plugin_module.metadata, "distributions", lambda: ())
    monkeypatch.setattr(
        plugin_module.metadata, "distribution", lambda name: distribution
    )

    row = import_plugin("direct_url_fact").distributions[0]
    assert row.direct_url is None
    assert row.direct_url_reason == "unreadable"


@pytest.mark.parametrize("extra", (0, 1))
def test_direct_url_raw_text_has_an_exact_one_mebibyte_boundary(
    tmp_path, monkeypatch, extra
):
    package = tmp_path / f"direct_url_size_{extra}"
    package.mkdir()
    (package / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    raw = "{}" + " " * ((1024 * 1024) - 2 + extra)
    distribution = _FakeDistribution(
        "direct-url-size-dist", tmp_path, direct_url=raw
    )

    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(
        plugin_module.metadata,
        "packages_distributions",
        lambda: {f"direct_url_size_{extra}": ["direct-url-size-dist"]},
    )
    monkeypatch.setattr(plugin_module.metadata, "distributions", lambda: ())
    monkeypatch.setattr(
        plugin_module.metadata, "distribution", lambda name: distribution
    )

    row = import_plugin(f"direct_url_size_{extra}").distributions[0]
    if extra == 0:
        assert row.direct_url == {}
        assert row.direct_url_reason is None
    else:
        assert row.direct_url is None
        assert row.direct_url_reason == "unreadable"


@pytest.mark.parametrize("extra", (0, 1))
def test_direct_url_metadata_string_subclass_has_an_exact_character_boundary(
    monkeypatch, extra
):
    hook_calls = 0

    class Text(str):
        def __len__(self):
            nonlocal hook_calls
            hook_calls += 1
            raise AssertionError("string subclass length hook must not run")

        def __str__(self):
            nonlocal hook_calls
            hook_calls += 1
            raise AssertionError("string subclass render hook must not run")

        def encode(self, *args, **kwargs):
            nonlocal hook_calls
            hook_calls += 1
            raise AssertionError("string subclass encode hook must not run")

    raw = Text("{}" + " " * ((1024 * 1024) - 2 + extra))
    row = _direct_url_metadata_row(
        monkeypatch,
        f"direct_url_subclass_boundary_{extra}",
        raw,
    )

    if extra == 0:
        assert row.direct_url == {}
        assert row.direct_url_reason is None
    else:
        assert row.direct_url is None
        assert row.direct_url_reason == "unreadable"
    assert hook_calls == 0


@pytest.mark.parametrize("extra", (0, 1))
def test_direct_url_metadata_string_subclass_enforces_the_utf8_byte_boundary(
    monkeypatch, extra
):
    hook_calls = 0

    class Text(str):
        def __len__(self):
            nonlocal hook_calls
            hook_calls += 1
            raise AssertionError("string subclass length hook must not run")

        def __str__(self):
            nonlocal hook_calls
            hook_calls += 1
            raise AssertionError("string subclass render hook must not run")

        def encode(self, *args, **kwargs):
            nonlocal hook_calls
            hook_calls += 1
            raise AssertionError("string subclass encode hook must not run")

    raw = Text('{"value":"' + "é" * (524_282 + extra) + '"}')
    row = _direct_url_metadata_row(
        monkeypatch,
        f"direct_url_multibyte_boundary_{extra}",
        raw,
    )

    if extra == 0:
        assert row.direct_url == {"value": "é" * 524_282}
        assert row.direct_url_reason is None
    else:
        assert row.direct_url is None
        assert row.direct_url_reason == "unreadable"
    assert hook_calls == 0


def test_oversized_direct_url_metadata_subclass_is_rejected_before_copy(
    monkeypatch,
):
    hook_calls = 0

    class Text(str):
        def __len__(self):
            nonlocal hook_calls
            hook_calls += 1
            raise AssertionError("string subclass length hook must not run")

        def __str__(self):
            nonlocal hook_calls
            hook_calls += 1
            raise AssertionError("string subclass render hook must not run")

        def __repr__(self):
            nonlocal hook_calls
            hook_calls += 1
            raise AssertionError("string subclass repr hook must not run")

        def encode(self, *args, **kwargs):
            nonlocal hook_calls
            hook_calls += 1
            raise AssertionError("string subclass encode hook must not run")

    oversized = Text("{}" + " " * (4 * 1024 * 1024))

    tracemalloc.start()
    try:
        row = _direct_url_metadata_row(
            monkeypatch,
            "direct_url_oversized_subclass",
            oversized,
        )
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert row.direct_url is None
    assert row.direct_url_reason == "unreadable"
    assert hook_calls == 0
    if peak >= 2 * 1024 * 1024:
        pytest.fail(f"oversized metadata text was copied before its cap: {peak=}")


def test_direct_url_scalar_text_is_utf8_and_limited_at_direct_construction():
    exact = "x" * (1024 * 1024)
    distribution = _valid_distribution(direct_url={"value": exact})
    assert distribution.direct_url == {"value": exact}

    with pytest.raises(ConfigError, match="direct_url.*scalar|direct_url.*UTF-8"):
        _valid_distribution(direct_url={"value": exact + "x"})
    with pytest.raises(ConfigError, match="direct_url.*UTF-8"):
        _valid_distribution(direct_url={"\ud800": "value"})
    with pytest.raises(ConfigError, match="direct_url.*UTF-8"):
        _valid_distribution(direct_url={"value": "\ud800"})


@pytest.mark.parametrize("projection", (False, True))
def test_oversized_direct_url_scalar_is_rejected_before_utf8_allocation(
    projection,
):
    oversized = "x" * 8_000_000
    if projection:
        distribution = _valid_distribution(
            direct_url=None,
            direct_url_reason="missing_direct_url",
        )
        object.__setattr__(
            distribution,
            "direct_url",
            MappingProxyType({"value": oversized}),
        )
        object.__setattr__(distribution, "direct_url_reason", None)
        forged = _valid_record()
        object.__setattr__(forged, "distributions", (distribution,))

        def operation():
            return plugin_audit_row(forged)

    else:

        def operation():
            return _valid_distribution(direct_url={"value": oversized})

    tracemalloc.start()
    try:
        with pytest.raises(ConfigError, match="direct_url.*scalar"):
            operation()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    if peak >= 4 * 1024 * 1024:
        pytest.fail(f"oversized direct_url text was encoded before its cap: {peak=}")


@pytest.mark.parametrize(
    "binary_factory",
    (
        pytest.param(lambda: bytearray(8_000_000), id="bytearray"),
        pytest.param(
            lambda: memoryview(bytearray(8_000_000)),
            id="memoryview",
        ),
    ),
)
@pytest.mark.parametrize("projection", (False, True))
def test_non_json_binary_is_rejected_before_direct_url_copy(
    binary_factory,
    projection,
):
    binary = binary_factory()
    if projection:
        distribution = _valid_distribution(
            direct_url=None,
            direct_url_reason="missing_direct_url",
        )
        object.__setattr__(
            distribution,
            "direct_url",
            MappingProxyType({"value": binary}),
        )
        object.__setattr__(distribution, "direct_url_reason", None)
        forged = _valid_record()
        object.__setattr__(forged, "distributions", (distribution,))

        def operation():
            return plugin_audit_row(forged)

    else:

        def operation():
            return _valid_distribution(direct_url={"value": binary})

    tracemalloc.start()
    try:
        with pytest.raises(ConfigError, match="not JSON"):
            operation()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    if peak >= 4 * 1024 * 1024:
        pytest.fail(f"non-JSON binary was copied before rejection: {peak=}")


@pytest.mark.parametrize(
    "value",
    (
        pytest.param({"alpha", "beta"}, id="set"),
        pytest.param(iter(("alpha", "beta")), id="iterator"),
    ),
)
def test_unordered_or_streaming_values_are_not_json_arrays(value):
    with pytest.raises(ConfigError, match="not JSON|unsupported evidence"):
        _valid_distribution(direct_url={"value": value})


class _HugeJsonInt(int):
    pass


@pytest.mark.parametrize("integer_type", (int, _HugeJsonInt))
@pytest.mark.parametrize("projection", (False, True))
def test_oversized_json_integer_is_rejected_before_exact_copy(
    integer_type,
    projection,
):
    oversized = integer_type(1 << 64_000_000)
    if projection:
        distribution = _valid_distribution(
            direct_url=None,
            direct_url_reason="missing_direct_url",
        )
        object.__setattr__(
            distribution,
            "direct_url",
            MappingProxyType({"value": oversized}),
        )
        object.__setattr__(distribution, "direct_url_reason", None)
        forged = _valid_record()
        object.__setattr__(forged, "distributions", (distribution,))

        def operation():
            return plugin_audit_row(forged)

    else:

        def operation():
            return _valid_distribution(direct_url={"value": oversized})

    tracemalloc.start()
    try:
        with pytest.raises(ConfigError, match="direct_url.*integer"):
            operation()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    if peak >= 4 * 1024 * 1024:
        pytest.fail(f"oversized direct_url integer was copied first: {peak=}")


def test_direct_url_integer_limit_is_derived_from_the_one_mibibyte_budget():
    from _rheplicant_bootstrap import plugins as plugin_module

    expected = math.ceil((1024 * 1024) * math.log2(10))
    assert plugin_module._DIRECT_URL_INTEGER_BIT_LIMIT == expected
    exact = 1 << (expected - 1)
    assert _valid_distribution(direct_url={"value": exact}).direct_url == {
        "value": exact
    }
    with pytest.raises(ConfigError, match="direct_url.*integer"):
        _valid_distribution(direct_url={"value": 1 << expected})


def test_direct_url_nodes_consume_the_shared_metadata_budget(monkeypatch):
    from _rheplicant_bootstrap import plugins as plugin_module

    module = _module("direct_url_node_budget")
    distribution = _FakeDistribution(
        "direct-url-node-dist", Path("/tmp"), direct_url='{"a":[]}'
    )
    monkeypatch.setattr(plugin_module, "_METADATA_EVIDENCE_LIMIT", 4, raising=False)
    monkeypatch.setattr(
        plugin_module.importlib, "import_module", lambda name: module
    )
    monkeypatch.setattr(
        plugin_module.metadata,
        "packages_distributions",
        lambda: {"direct_url_node_budget": ["direct-url-node-dist"]},
    )
    monkeypatch.setattr(plugin_module.metadata, "distributions", lambda: ())
    monkeypatch.setattr(
        plugin_module.metadata, "distribution", lambda name: distribution
    )
    with pytest.raises(ConfigError, match="metadata.*budget"):
        import_plugin("direct_url_node_budget")


def test_direct_url_freeze_consumes_the_shared_budget_before_nested_copy(
    monkeypatch,
):
    from _rheplicant_bootstrap import plugins as plugin_module

    class CountingSequence:
        emissions = 0

        def __len__(self):
            return 20

        def __getitem__(self, index):
            if index >= 20:
                raise IndexError(index)
            type(self).emissions += 1
            return index

    monkeypatch.setattr(plugin_module, "_METADATA_EVIDENCE_LIMIT", 1)
    budget = plugin_module._MetadataBudget()
    with pytest.raises(ConfigError, match="metadata.*budget"):
        plugin_module._freeze_direct_url(
            {"items": CountingSequence()},
            where="plugin distribution direct_url",
            budget=budget,
        )
    assert CountingSequence.emissions == 0


def test_plugin_record_direct_url_copy_uses_the_remaining_aggregate_budget(
    monkeypatch,
):
    from _rheplicant_bootstrap import plugins as plugin_module

    class CountingSequence:
        emissions = 0

        def __len__(self):
            return 20

        def __getitem__(self, index):
            if index >= 20:
                raise IndexError(index)
            type(self).emissions += 1
            return index

    distribution = _valid_distribution(
        direct_url=None,
        direct_url_reason="missing_direct_url",
    )
    object.__setattr__(
        distribution,
        "direct_url",
        MappingProxyType({"items": CountingSequence()}),
    )
    object.__setattr__(distribution, "direct_url_reason", None)
    monkeypatch.setattr(plugin_module, "_METADATA_EVIDENCE_LIMIT", 2)

    with pytest.raises(ConfigError, match="metadata.*budget"):
        _valid_record(distributions=(distribution,))
    assert CountingSequence.emissions == 0


def test_shared_direct_url_budget_precedes_mapping_pair_unpack(monkeypatch):
    from _rheplicant_bootstrap import plugins as plugin_module

    class BrokenPair:
        unpack_calls = 0

        def __iter__(self):
            type(self).unpack_calls += 1
            raise AssertionError("over-budget JSON pair was unpacked")

    class PairMapping:
        def items(self):
            return (BrokenPair(),)

    monkeypatch.setattr(plugin_module, "_METADATA_EVIDENCE_LIMIT", 1)
    budget = plugin_module._MetadataBudget()
    with pytest.raises(ConfigError, match="metadata.*budget"):
        plugin_module._freeze_direct_url(
            PairMapping(),
            where="plugin distribution direct_url",
            budget=budget,
        )
    assert BrokenPair.unpack_calls == 0


class _StreamingMetadataSequence:
    iter_calls = 0

    def __iter__(self):
        type(self).iter_calls += 1
        raise AssertionError("streaming metadata field was consumed")


@pytest.mark.parametrize("route", ("candidates", "files", "parts"))
@pytest.mark.parametrize("value_kind", ("stream", "set", "mapping"))
def test_metadata_sequence_fields_require_a_static_ordered_sequence(
    route,
    value_kind,
    tmp_path,
    monkeypatch,
):
    from _rheplicant_bootstrap import plugins as plugin_module

    _StreamingMetadataSequence.iter_calls = 0
    values = {
        "stream": _StreamingMetadataSequence(),
        "set": {"alpha", "beta"},
        "mapping": {"alpha": 1, "beta": 2},
    }
    value = values[value_kind]
    name = f"ordered_metadata_{route}_{value_kind}"
    source = tmp_path / f"{name}.py"
    source.write_text("VALUE = 1\n")
    module = _module(
        name,
        origin=str(source),
        loader=importlib.machinery.SourceFileLoader(name, str(source)),
    )
    distribution = _FakeDistribution("ordered-metadata-dist", tmp_path)
    if route == "candidates":
        top_map = {name: value}
        installed = ()
    elif route == "files":
        top_map = {}
        distribution.files = value
        installed = (distribution,)
    else:
        class Entry:
            parts = value

            def is_absolute(self):
                return False

        top_map = {}
        distribution.files = (Entry(),)
        installed = (distribution,)
    monkeypatch.setattr(
        plugin_module.importlib, "import_module", lambda raw_name: module
    )
    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: top_map
    )
    monkeypatch.setattr(
        plugin_module.metadata, "distributions", lambda: installed
    )

    with pytest.raises(ConfigError, match="ordered sequence"):
        import_plugin(name)
    assert _StreamingMetadataSequence.iter_calls == 0


def test_distribution_enumeration_failures_are_not_reported_as_no_distribution(
    monkeypatch,
):
    from _rheplicant_bootstrap import plugins as plugin_module

    module = _module("metadata_failure")
    monkeypatch.setattr(
        plugin_module.importlib, "import_module", lambda name: module
    )

    def fail():
        raise RuntimeError("metadata failed")

    monkeypatch.setattr(plugin_module.metadata, "packages_distributions", fail)
    with pytest.raises(ConfigError, match="metadata"):
        import_plugin("metadata_failure")

    def stop():
        raise KeyboardInterrupt

    monkeypatch.setattr(plugin_module.metadata, "packages_distributions", stop)
    with pytest.raises(KeyboardInterrupt):
        import_plugin("metadata_failure")


def test_packages_distribution_items_are_snapshotted_without_getitem(
    monkeypatch,
):
    from _rheplicant_bootstrap import plugins as plugin_module

    module = _module("items_only_top_map")
    distribution = _FakeDistribution("items-only-dist", Path("/tmp"))

    class ItemsOnly:
        getitem_calls = 0

        def items(self):
            return (("items_only_top_map", ("items-only-dist",)),)

        def __getitem__(self, key):
            type(self).getitem_calls += 1
            raise KeyboardInterrupt("top-map getitem must not run")

    monkeypatch.setattr(
        plugin_module.importlib, "import_module", lambda name: module
    )
    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", ItemsOnly
    )
    monkeypatch.setattr(plugin_module.metadata, "distributions", lambda: ())
    monkeypatch.setattr(
        plugin_module.metadata, "distribution", lambda name: distribution
    )

    record = import_plugin("items_only_top_map")
    assert tuple(row.name for row in record.distributions) == (
        "items-only-dist",
    )
    assert ItemsOnly.getitem_calls == 0


@pytest.mark.parametrize(
    "error_type",
    (ValueError, KeyboardInterrupt),
    ids=("ordinary", "base_exception"),
)
def test_packages_distribution_items_failure_has_the_static_boundary(
    monkeypatch,
    error_type,
):
    from _rheplicant_bootstrap import plugins as plugin_module

    class FailingItems:
        def items(self):
            raise error_type("private top-map failure")

    monkeypatch.setattr(
        plugin_module.importlib,
        "import_module",
        lambda name: _module("top_map_items_failure"),
    )
    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", FailingItems
    )
    if error_type is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            import_plugin("top_map_items_failure")
    else:
        with pytest.raises(ConfigError, match="metadata") as caught:
            import_plugin("top_map_items_failure")
        assert "private" not in str(caught.value)


@pytest.mark.parametrize(
    "error_type",
    (ValueError, KeyboardInterrupt),
    ids=("ordinary", "base_exception"),
)
def test_packages_distribution_items_binding_has_the_static_boundary(
    monkeypatch,
    error_type,
):
    from _rheplicant_bootstrap import plugins as plugin_module

    class FailingBinding:
        def __getattribute__(self, name):
            if name == "items":
                raise error_type("private items binding failure")
            return object.__getattribute__(self, name)

        def items(self):
            return ()

    monkeypatch.setattr(
        plugin_module.importlib,
        "import_module",
        lambda name: _module("top_map_items_binding_failure"),
    )
    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", FailingBinding
    )
    if error_type is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            import_plugin("top_map_items_binding_failure")
    else:
        with pytest.raises(ConfigError, match="metadata") as caught:
            import_plugin("top_map_items_binding_failure")
        assert "private" not in str(caught.value)


def test_packages_distribution_keys_collide_after_canonicalization(
    monkeypatch,
):
    from _rheplicant_bootstrap import plugins as plugin_module

    class Key(str):
        pass

    class DuplicateItems:
        def items(self):
            return (
                (Key("duplicate_top_map"), ("first",)),
                ("duplicate_top_map", ("second",)),
            )

    monkeypatch.setattr(
        plugin_module.importlib,
        "import_module",
        lambda name: _module("duplicate_top_map"),
    )
    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", DuplicateItems
    )
    with pytest.raises(ConfigError, match="collide after canonicalization"):
        import_plugin("duplicate_top_map")


def test_packages_distribution_pair_budget_precedes_limit_plus_one_unpack(
    monkeypatch,
):
    from _rheplicant_bootstrap import plugins as plugin_module

    class BrokenPair:
        unpack_calls = 0

        def __iter__(self):
            type(self).unpack_calls += 1
            raise AssertionError("over-budget pair was unpacked")

    class PairStream:
        def items(self):
            return iter((("unrelated", ()), BrokenPair()))

    monkeypatch.setattr(plugin_module, "_METADATA_EVIDENCE_LIMIT", 1)
    monkeypatch.setattr(
        plugin_module.importlib,
        "import_module",
        lambda name: _module("top_map_pair_budget"),
    )
    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", PairStream
    )
    with pytest.raises(ConfigError, match="metadata.*budget"):
        import_plugin("top_map_pair_budget")
    assert BrokenPair.unpack_calls == 0


@pytest.mark.parametrize("count", (250_000, 250_001))
def test_installed_distribution_emission_budget_has_an_exact_public_boundary(
    tmp_path, monkeypatch, count
):
    source = tmp_path / f"metadata_budget_{count}.py"
    source.write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    distribution = _FakeDistribution("non-candidate", tmp_path)
    distribution.files = None

    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(
        plugin_module.metadata,
        "distributions",
        lambda: itertools.repeat(distribution, count),
    )
    name = f"metadata_budget_{count}"
    if count == 250_000:
        record = import_plugin(name)
        assert record.distributions == ()
    else:
        with pytest.raises(ConfigError, match="metadata.*budget"):
            import_plugin(name)


def test_metadata_budget_is_shared_across_candidate_arms(tmp_path, monkeypatch):
    from _rheplicant_bootstrap import plugins as plugin_module

    source = tmp_path / "shared_metadata_budget.py"
    source.write_text("VALUE = 1\n")
    module = _module(
        "shared_metadata_budget",
        origin=str(source),
        loader=importlib.machinery.SourceFileLoader(
            "shared_metadata_budget", str(source)
        ),
    )
    distribution = _FakeDistribution("not-a-candidate", tmp_path)
    distribution.files = None
    monkeypatch.setattr(plugin_module, "_METADATA_EVIDENCE_LIMIT", 3)
    monkeypatch.setattr(
        plugin_module.importlib, "import_module", lambda name: module
    )
    monkeypatch.setattr(
        plugin_module.metadata,
        "packages_distributions",
        lambda: {"shared_metadata_budget": ("same", "same")},
    )
    monkeypatch.setattr(
        plugin_module.metadata,
        "distributions",
        lambda: (distribution,),
    )
    with pytest.raises(ConfigError, match="metadata.*budget"):
        import_plugin("shared_metadata_budget")


def test_record_entries_and_final_candidates_share_the_metadata_budget(
    tmp_path, monkeypatch
):
    source = tmp_path / "record_budget_plugin.py"
    source.write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    distribution = _FakeDistribution(
        "record-budget-dist",
        tmp_path,
        files=(".", ".", ".", "."),
    )

    from _rheplicant_bootstrap import plugins as plugin_module

    monkeypatch.setattr(plugin_module, "_METADATA_EVIDENCE_LIMIT", 4, raising=False)
    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(
        plugin_module.metadata, "distributions", lambda: (distribution,)
    )
    with pytest.raises(ConfigError, match="metadata.*budget"):
        import_plugin("record_budget_plugin")

    module = _module("candidate_budget_plugin")
    monkeypatch.setattr(
        plugin_module.importlib, "import_module", lambda name: module
    )
    monkeypatch.setattr(plugin_module, "_METADATA_EVIDENCE_LIMIT", 3, raising=False)
    monkeypatch.setattr(
        plugin_module.metadata,
        "packages_distributions",
        lambda: {"candidate_budget_plugin": ("one", "two")},
    )
    monkeypatch.setattr(plugin_module.metadata, "distributions", lambda: ())
    with pytest.raises(ConfigError, match="metadata.*budget"):
        import_plugin("candidate_budget_plugin")


def test_metadata_iteration_failures_are_static_and_baseexception_propagates(
    tmp_path, monkeypatch
):
    source = tmp_path / "metadata_iteration_failure.py"
    source.write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    from _rheplicant_bootstrap import plugins as plugin_module

    hooks = 0

    class HostileError(RuntimeError):
        def __str__(self):
            nonlocal hooks
            hooks += 1
            return "forged"

        def __repr__(self):
            nonlocal hooks
            hooks += 1
            return "forged"

    class Broken:
        def __iter__(self):
            return self

        def __next__(self):
            raise HostileError("metadata secret")

    monkeypatch.setattr(
        plugin_module.metadata, "packages_distributions", lambda: {}
    )
    monkeypatch.setattr(plugin_module.metadata, "distributions", Broken)
    with pytest.raises(ConfigError) as caught:
        import_plugin("metadata_iteration_failure")
    if str(caught.value) != (
        "plugins: distribution metadata installed-distribution traversal failed."
    ):
        pytest.fail(f"unexpected metadata failure: {caught.value.args!r}")
    if hooks != 0:
        pytest.fail(f"metadata exception rendering hooks ran {hooks} times")

    class Stop:
        def __iter__(self):
            return self

        def __next__(self):
            raise KeyboardInterrupt

    monkeypatch.setattr(plugin_module.metadata, "distributions", Stop)
    with pytest.raises(KeyboardInterrupt):
        import_plugin("metadata_iteration_failure")


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"origin": None, "origin_reason": None}, "origin"),
        ({"origin": "/x", "origin_reason": "no_origin"}, "origin"),
        ({"loader_type": None, "loader_type_reason": None}, "loader_type"),
        (
            {"resolved_path": None, "resolved_path_reason": None},
            "resolved_path",
        ),
        ({"distributions": (), "distributions_reason": None}, "distributions"),
        (
            {"distributions_reason": "no_distribution"},
            "distributions",
        ),
        ({"code_hash": None, "code_hash_reason": None}, "code_hash"),
        ({"code_hash": "A" * 64}, "code_hash"),
        ({"code_hash": "a" * 63}, "code_hash"),
        ({"unobserved_io": 1}, "unobserved_io"),
    ],
)
def test_plugin_projection_refuses_every_malformed_value_reason_pair(
    changes, match
):
    record = _valid_record()
    for name, value in changes.items():
        object.__setattr__(record, name, value)
    with pytest.raises(ConfigError, match=match):
        plugin_audit_row(record)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"name": "Not_Normalized"}, "distribution name"),
        ({"version": None, "version_reason": None}, "version"),
        ({"version": "1", "version_reason": "unreadable"}, "version"),
        ({"direct_url": None, "direct_url_reason": None}, "direct_url"),
        (
            {"direct_url": {}, "direct_url_reason": "missing_direct_url"},
            "direct_url",
        ),
        ({"direct_url": []}, "recursively frozen"),
        (
            {"direct_url": MappingProxyType({"value": float("nan")})},
            "finite",
        ),
        (
            {"direct_url": MappingProxyType({"value": b"bytes"})},
            "JSON",
        ),
    ],
)
def test_distribution_projection_refuses_malformed_rows(changes, match):
    malformed = _valid_distribution()
    for name, value in changes.items():
        object.__setattr__(malformed, name, value)
    record = _valid_record()
    object.__setattr__(record, "distributions", (malformed,))
    with pytest.raises(ConfigError, match=match):
        plugin_audit_row(record)


def test_plugin_records_validate_malformed_direct_construction():
    with pytest.raises(ConfigError, match="distribution name"):
        _valid_distribution(name="Not_Normalized")
    with pytest.raises(ConfigError, match="version"):
        _valid_distribution(version=None, version_reason=None)
    with pytest.raises(ConfigError, match="direct_url"):
        _valid_distribution(direct_url=None, direct_url_reason=None)
    with pytest.raises(ConfigError, match="finite"):
        _valid_distribution(direct_url={"value": float("nan")})

    with pytest.raises(ConfigError, match="origin"):
        _valid_record(origin=None, origin_reason=None)
    with pytest.raises(ConfigError, match="distributions"):
        _valid_record(distributions=(), distributions_reason=None)
    with pytest.raises(ConfigError, match="code_hash"):
        _valid_record(code_hash="A" * 64)
    with pytest.raises(ConfigError, match="unobserved_io"):
        _valid_record(unobserved_io=1)


def test_plugin_record_distribution_limit_is_checked_before_any_copy(
    monkeypatch,
):
    from _rheplicant_bootstrap import plugins as plugin_module

    def distribution(index):
        return _valid_distribution(
            name=f"dist-{index}",
            direct_url=None,
            direct_url_reason="missing_direct_url",
        )

    monkeypatch.setattr(plugin_module, "_METADATA_EVIDENCE_LIMIT", 4)
    exact = tuple(distribution(index) for index in range(4))
    assert len(_valid_record(distributions=exact).distributions) == 4

    oversized = (*exact, distribution(4))
    forged = _valid_record(distributions=exact)
    object.__setattr__(forged, "distributions", oversized)
    copy_calls = 0

    def forbidden_copy(_value, **_kwargs):
        nonlocal copy_calls
        copy_calls += 1
        pytest.fail("oversized distributions were copied before the limit")

    monkeypatch.setattr(
        plugin_module, "_copy_distribution_record", forbidden_copy
    )
    for operation in (
        lambda: _valid_record(distributions=oversized),
        lambda: plugin_audit_row(forged),
    ):
        with pytest.raises(ConfigError, match="distribution.*budget"):
            operation()
    if copy_calls != 0:
        pytest.fail(f"oversized distribution copy ran {copy_calls} times")


def test_plugin_record_uses_one_shared_budget_for_nested_direct_urls(
    monkeypatch,
):
    from _rheplicant_bootstrap import plugins as plugin_module

    distribution = _valid_distribution(
        name="budgeted-direct-url", direct_url={"a": 1}
    )
    monkeypatch.setattr(plugin_module, "_METADATA_EVIDENCE_LIMIT", 5)
    assert _valid_record(distributions=(distribution,)).distributions

    monkeypatch.setattr(plugin_module, "_METADATA_EVIDENCE_LIMIT", 4)
    with pytest.raises(ConfigError, match="distribution.*budget"):
        _valid_record(distributions=(distribution,))


def test_plugin_record_charges_each_repeated_tuple_edge(monkeypatch):
    from _rheplicant_bootstrap import plugins as plugin_module

    shared = "shared scalar"
    distribution = _valid_distribution(
        name="tuple-edge-budget",
        direct_url={"items": [shared, shared, shared, shared]},
    )
    monkeypatch.setattr(plugin_module, "_METADATA_EVIDENCE_LIMIT", 10)
    assert _valid_record(distributions=(distribution,)).distributions

    monkeypatch.setattr(plugin_module, "_METADATA_EVIDENCE_LIMIT", 9)
    with pytest.raises(ConfigError, match="distribution.*budget"):
        _valid_record(distributions=(distribution,))


def test_aggregate_direct_url_copy_preserves_each_roots_logical_depth():
    exact_limit = True
    for _ in range(99):
        exact_limit = {"child": exact_limit}
    distribution = _valid_distribution(direct_url=exact_limit)
    record = _valid_record(distributions=(distribution,))
    assert record.distributions[0].direct_url is not distribution.direct_url

    over_limit = {"child": exact_limit}
    with pytest.raises(ConfigError, match="depth 101 exceeds limit 100"):
        _valid_distribution(direct_url=over_limit)


def test_plugin_record_reuses_one_prevalidated_shared_direct_url(monkeypatch):
    from _rheplicant_bootstrap import plugins as plugin_module

    direct_url = {"leaf": True}
    for _ in range(20):
        direct_url = {"left": direct_url, "right": direct_url}
    shared = _valid_distribution(direct_url=direct_url).direct_url
    distributions = []
    for index in range(8):
        distribution = _valid_distribution(
            name=f"shared-{index}",
            direct_url=None,
            direct_url_reason="missing_direct_url",
        )
        object.__setattr__(distribution, "direct_url", shared)
        object.__setattr__(distribution, "direct_url_reason", None)
        distributions.append(distribution)

    template = _valid_record()
    real_freeze = plugin_module.freeze_evidence
    real_freeze_roots = plugin_module._freeze_evidence_roots
    freeze_calls = 0

    def count_freeze(*args, **kwargs):
        nonlocal freeze_calls
        freeze_calls += 1
        return real_freeze(*args, **kwargs)

    def count_freeze_roots(*args, **kwargs):
        nonlocal freeze_calls
        freeze_calls += 1
        return real_freeze_roots(*args, **kwargs)

    monkeypatch.setattr(plugin_module, "_METADATA_EVIDENCE_LIMIT", 256)
    monkeypatch.setattr(plugin_module, "freeze_evidence", count_freeze)
    monkeypatch.setattr(
        plugin_module,
        "_freeze_evidence_roots",
        count_freeze_roots,
    )
    record = PluginRecord(
        name=template.name,
        already_imported=template.already_imported,
        origin=template.origin,
        origin_reason=template.origin_reason,
        loader_type=template.loader_type,
        loader_type_reason=template.loader_type_reason,
        resolved_path=template.resolved_path,
        resolved_path_reason=template.resolved_path_reason,
        distributions=tuple(distributions),
        distributions_reason=None,
        code_hash=template.code_hash,
        code_hash_reason=template.code_hash_reason,
        unobserved_io=True,
    )
    if freeze_calls != 1:
        pytest.fail(
            "shared direct_url was not copied exactly once: "
            f"{freeze_calls} freezes"
        )
    private_copy = record.distributions[0].direct_url
    if private_copy is shared:
        pytest.fail("PluginRecord retained a caller-owned frozen mapping")
    if any(
        item.direct_url is not private_copy for item in record.distributions
    ):
        pytest.fail("private shared direct_url identity was not reused")

    plugin_audit_row(record)
    if freeze_calls != 3:
        pytest.fail(
            "projection did not snapshot once before its final freeze: "
            f"{freeze_calls} calls"
        )


def test_plugin_record_aggregate_copy_preserves_cross_root_shared_children(
    monkeypatch,
):
    from _rheplicant_bootstrap import plugins as plugin_module

    deep = {"leaf": True}
    for _ in range(20):
        deep = {"left": deep, "right": deep}
    shared_child = _valid_distribution(direct_url=deep).direct_url
    roots = []
    backings = []
    distributions = []
    for index in range(4):
        backing = {"index": index, "shared": shared_child}
        root = MappingProxyType(backing)
        distribution = _valid_distribution(
            name=f"cross-root-{index}",
            direct_url=None,
            direct_url_reason="missing_direct_url",
        )
        object.__setattr__(distribution, "direct_url", root)
        object.__setattr__(distribution, "direct_url_reason", None)
        roots.append(root)
        backings.append(backing)
        distributions.append(distribution)

    template = _valid_record()
    real_freeze = plugin_module.freeze_evidence
    real_freeze_roots = plugin_module._freeze_evidence_roots
    freeze_calls = []

    def count_freeze(*args, **kwargs):
        freeze_calls.append(kwargs.get("where"))
        return real_freeze(*args, **kwargs)

    def count_freeze_roots(*args, **kwargs):
        freeze_calls.append(kwargs.get("where"))
        return real_freeze_roots(*args, **kwargs)

    monkeypatch.setattr(plugin_module, "_METADATA_EVIDENCE_LIMIT", 256)
    monkeypatch.setattr(plugin_module, "freeze_evidence", count_freeze)
    monkeypatch.setattr(
        plugin_module,
        "_freeze_evidence_roots",
        count_freeze_roots,
    )
    record = PluginRecord(
        name=template.name,
        already_imported=template.already_imported,
        origin=template.origin,
        origin_reason=template.origin_reason,
        loader_type=template.loader_type,
        loader_type_reason=template.loader_type_reason,
        resolved_path=template.resolved_path,
        resolved_path_reason=template.resolved_path_reason,
        distributions=tuple(distributions),
        distributions_reason=None,
        code_hash=template.code_hash,
        code_hash_reason=template.code_hash_reason,
        unobserved_io=True,
    )
    if freeze_calls != ["plugin distribution direct_urls"]:
        pytest.fail(f"direct_url roots were not aggregate-copied: {freeze_calls}")
    copied_roots = tuple(item.direct_url for item in record.distributions)
    if any(copied is raw for copied, raw in zip(copied_roots, roots, strict=True)):
        pytest.fail("PluginRecord retained a caller-owned direct_url root")
    copied_child = copied_roots[0]["shared"]
    if any(root["shared"] is not copied_child for root in copied_roots):
        pytest.fail("aggregate copy expanded a child shared across distinct roots")
    backings[0]["injected"] = "after construction"
    if "injected" in copied_roots[0]:
        pytest.fail("caller mutation reached the detached PluginRecord snapshot")

    row = plugin_audit_row(record)
    if freeze_calls != [
        "plugin distribution direct_urls",
        "plugin distribution direct_urls",
        "plugin audit row",
    ]:
        pytest.fail(f"projection did not use one snapshot and one close: {freeze_calls}")
    row_roots = tuple(item["direct_url"] for item in row["distributions"])
    row_child = row_roots[0]["shared"]
    if any(root["shared"] is not row_child for root in row_roots):
        pytest.fail("projection expanded a child shared across distinct roots")


class _ReentrantDistributionMapping(Mapping):
    def __init__(self) -> None:
        self.distribution = None
        self.calls = 0

    def __getitem__(self, key):
        if key == "ok":
            return True
        raise KeyError(key)

    def __iter__(self):
        return iter(("ok",))

    def __len__(self):
        return 1

    def items(self):
        self.calls += 1
        distribution = self.distribution
        assert distribution is not None
        object.__setattr__(distribution, "name", "after-dist")
        object.__setattr__(distribution, "version", None)
        object.__setattr__(distribution, "version_reason", "unreadable")
        object.__setattr__(distribution, "direct_url", None)
        object.__setattr__(distribution, "direct_url_reason", "unreadable")
        return {"ok": True}.items()


@pytest.mark.parametrize("projection", (False, True))
def test_plugin_record_snapshots_each_complete_distribution_before_protocols(
    projection,
):
    backing = _ReentrantDistributionMapping()
    distribution = _valid_distribution(
        name="before-dist",
        version="before",
        direct_url=None,
        direct_url_reason="missing_direct_url",
    )
    root = MappingProxyType(backing)
    object.__setattr__(distribution, "direct_url", root)
    object.__setattr__(distribution, "direct_url_reason", None)
    backing.distribution = distribution

    if projection:
        forged = _valid_record()
        object.__setattr__(forged, "distributions", (distribution,))
        retained = plugin_audit_row(forged)["distributions"][0]
    else:
        retained = _valid_record(distributions=(distribution,)).distributions[0]

    assert backing.calls == 1
    assert distribution.name == "after-dist"
    if projection:
        assert retained == {
            "name": "before-dist",
            "version": "before",
            "version_reason": None,
            "direct_url": {"ok": True},
            "direct_url_reason": None,
        }
    else:
        assert retained.name == "before-dist"
        assert retained.version == "before"
        assert retained.version_reason is None
        assert retained.direct_url == {"ok": True}
        assert retained.direct_url_reason is None


class _ReentrantPluginRecordMapping(Mapping):
    def __init__(self) -> None:
        self.record = None
        self.calls = 0

    def __getitem__(self, key):
        if key == "ok":
            return True
        raise KeyError(key)

    def __iter__(self):
        return iter(("ok",))

    def __len__(self):
        return 1

    def items(self):
        self.calls += 1
        record = self.record
        assert record is not None
        object.__setattr__(record, "already_imported", 1)
        object.__setattr__(record, "unobserved_io", False)
        return {"ok": True}.items()


@pytest.mark.parametrize("projection", (False, True))
def test_plugin_record_restores_snapshotted_boolean_fields_after_protocols(
    projection,
):
    backing = _ReentrantPluginRecordMapping()
    distribution = _valid_distribution(
        direct_url=None,
        direct_url_reason="missing_direct_url",
    )
    object.__setattr__(distribution, "direct_url", MappingProxyType(backing))
    object.__setattr__(distribution, "direct_url_reason", None)

    record = object.__new__(PluginRecord)
    backing.record = record
    PluginRecord.__init__(
        record,
        name="example.plugin",
        already_imported=False,
        origin="/tmp/example.py",
        origin_reason=None,
        loader_type="_frozen_importlib_external.SourceFileLoader",
        loader_type_reason=None,
        resolved_path="/tmp/example.py",
        resolved_path_reason=None,
        distributions=(distribution,),
        distributions_reason=None,
        code_hash="a" * 64,
        code_hash_reason=None,
        unobserved_io=True,
    )

    if projection:
        retained = plugin_audit_row(record)
        assert retained["already_imported"] is False
        assert retained["unobserved_io"] is True
        assert backing.calls == 1
    else:
        assert record.already_imported is False
        assert record.unobserved_io is True
        assert backing.calls == 1


def test_plugin_record_has_no_public_direct_url_copy_bypass():
    from _rheplicant_bootstrap import plugins as plugin_module

    token = getattr(plugin_module, "_REUSE_VALIDATED_DIRECT_URLS", object())
    with pytest.raises(TypeError, match="unexpected keyword"):
        _valid_record(_direct_url_copy_policy=token)


@pytest.mark.parametrize(
    ("injected", "message"),
    (
        (float("nan"), "finite JSON numbers"),
        ("\ud800", "valid UTF-8"),
    ),
)
def test_plugin_projection_validates_the_detached_snapshot_after_copy(
    injected,
    message,
    monkeypatch,
):
    from _rheplicant_bootstrap import plugins as plugin_module

    backing = {"stable": True}
    forged_url = MappingProxyType(backing)
    distribution = _valid_distribution(
        direct_url=None,
        direct_url_reason="missing_direct_url",
    )
    object.__setattr__(distribution, "direct_url", forged_url)
    object.__setattr__(distribution, "direct_url_reason", None)
    forged = _valid_record()
    object.__setattr__(forged, "distributions", (distribution,))

    real_freeze_roots = plugin_module._freeze_evidence_roots

    def mutate_at_snapshot(value, *, where, **kwargs):
        if where == "plugin distribution direct_urls":
            backing["injected"] = injected
        return real_freeze_roots(value, where=where, **kwargs)

    monkeypatch.setattr(
        plugin_module,
        "_freeze_evidence_roots",
        mutate_at_snapshot,
    )
    with pytest.raises(ConfigError, match=message):
        plugin_audit_row(forged)


def test_plugin_record_rebudgets_a_direct_url_that_grows_at_snapshot(
    monkeypatch,
):
    from _rheplicant_bootstrap import plugins as plugin_module

    backing = {"stable": True}
    distribution = _valid_distribution(
        direct_url=None,
        direct_url_reason="missing_direct_url",
    )
    object.__setattr__(distribution, "direct_url", MappingProxyType(backing))
    object.__setattr__(distribution, "direct_url_reason", None)
    real_freeze_roots = plugin_module._freeze_evidence_roots

    def grow_at_snapshot(value, *, where, **kwargs):
        backing.update({f"late-{index}": index for index in range(20)})
        return real_freeze_roots(value, where=where, **kwargs)

    monkeypatch.setattr(plugin_module, "_METADATA_EVIDENCE_LIMIT", 8)
    monkeypatch.setattr(
        plugin_module,
        "_freeze_evidence_roots",
        grow_at_snapshot,
    )
    with pytest.raises(ConfigError, match="distribution.*budget"):
        _valid_record(distributions=(distribution,))


def test_plugin_record_rejects_a_forged_frozen_direct_url_cycle():
    backing = {}
    cyclic = MappingProxyType(backing)
    backing["self"] = cyclic
    distribution = _valid_distribution(
        direct_url=None, direct_url_reason="missing_direct_url"
    )
    object.__setattr__(distribution, "direct_url", cyclic)
    object.__setattr__(distribution, "direct_url_reason", None)

    with pytest.raises(ConfigError, match="direct_url.*cyclic"):
        _valid_record(distributions=(distribution,))


def test_plugin_record_rejects_a_cross_scheduled_frozen_cycle():
    left_backing = {}
    right_backing = {}
    left = MappingProxyType(left_backing)
    right = MappingProxyType(right_backing)
    left_backing["right"] = right
    right_backing["left"] = left
    root = MappingProxyType({"left": left, "right": right})
    distribution = _valid_distribution(
        direct_url=None, direct_url_reason="missing_direct_url"
    )
    object.__setattr__(distribution, "direct_url", root)
    object.__setattr__(distribution, "direct_url_reason", None)

    with pytest.raises(ConfigError, match="direct_url.*cyclic"):
        _valid_record(distributions=(distribution,))


class _ForgedMutableDirectUrl(Mapping):
    def __init__(self, *, interrupt: bool) -> None:
        self.calls = 0
        self.emissions = 0
        self.interrupt = interrupt

    def __getitem__(self, key):
        raise KeyError(key)

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def items(self):
        self.calls += 1
        if self.interrupt:
            raise KeyboardInterrupt
        for _ in range(100_000):
            self.emissions += 1
            yield f"key-{self.emissions}", None


@pytest.mark.parametrize("interrupt", (False, True))
def test_plugin_record_statically_rejects_a_forged_mutable_direct_url(
    interrupt,
):
    direct_url = _ForgedMutableDirectUrl(interrupt=interrupt)
    distribution = _valid_distribution(
        direct_url=None,
        direct_url_reason="missing_direct_url",
    )
    object.__setattr__(distribution, "direct_url", direct_url)
    object.__setattr__(distribution, "direct_url_reason", None)

    with pytest.raises(ConfigError, match="recursively frozen"):
        _valid_record(distributions=(distribution,))
    assert direct_url.calls == 0
    assert direct_url.emissions == 0


def test_plugin_record_bounds_a_hostile_exact_mapping_proxy(monkeypatch):
    from _rheplicant_bootstrap import frozen as frozen_module

    backing = _ForgedMutableDirectUrl(interrupt=False)
    direct_url = MappingProxyType(backing)
    distribution = _valid_distribution(
        direct_url=None,
        direct_url_reason="missing_direct_url",
    )
    object.__setattr__(distribution, "direct_url", direct_url)
    object.__setattr__(distribution, "direct_url_reason", None)
    monkeypatch.setattr(frozen_module, "_EVIDENCE_EDGE_LIMIT", 8)

    with pytest.raises(ConfigError, match="emission count"):
        _valid_record(distributions=(distribution,))
    assert backing.calls == 1
    assert 0 < backing.emissions <= 9


def test_plugin_record_propagates_baseexception_from_an_exact_mapping_proxy():
    backing = _ForgedMutableDirectUrl(interrupt=True)
    direct_url = MappingProxyType(backing)
    distribution = _valid_distribution(
        direct_url=None,
        direct_url_reason="missing_direct_url",
    )
    object.__setattr__(distribution, "direct_url", direct_url)
    object.__setattr__(distribution, "direct_url_reason", None)

    with pytest.raises(KeyboardInterrupt):
        _valid_record(distributions=(distribution,))
    assert backing.calls == 1


class _BrokenJsonPair:
    def __init__(self, error: BaseException | None = None) -> None:
        self.calls = 0
        self.error = error

    def __iter__(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return iter(("key", None))


class _PairMapping(Mapping):
    def __init__(self, pair: object) -> None:
        self.pair = pair

    def __getitem__(self, key):
        raise KeyError(key)

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 1

    def items(self):
        return iter((self.pair,))


def _validate_forged_proxy_pair(pair: object) -> None:
    from _rheplicant_bootstrap import plugins as plugin_module

    plugin_module._validate_frozen_json(
        MappingProxyType(_PairMapping(pair)),
        where="plugin distribution direct_url",
        budget=plugin_module._MetadataBudget(),
        require_frozen=True,
    )


def test_direct_url_budget_stops_before_unpacking_the_limit_plus_one_pair(
    monkeypatch,
):
    from _rheplicant_bootstrap import plugins as plugin_module

    pair = _BrokenJsonPair()
    monkeypatch.setattr(plugin_module, "_METADATA_EVIDENCE_LIMIT", 1)
    with pytest.raises(ConfigError, match="distribution.*budget"):
        _validate_forged_proxy_pair(pair)
    assert pair.calls == 0


def test_direct_url_pair_unpack_failures_are_static_and_baseexception_passes():
    hooks = 0

    class HostileError(ValueError):
        def __str__(self):
            nonlocal hooks
            hooks += 1
            return "forged"

        def __repr__(self):
            nonlocal hooks
            hooks += 1
            return "forged"

    with pytest.raises(ConfigError, match="JSON mapping traversal failed"):
        _validate_forged_proxy_pair(_BrokenJsonPair(HostileError("secret")))
    assert hooks == 0

    with pytest.raises(KeyboardInterrupt):
        _validate_forged_proxy_pair(_BrokenJsonPair(KeyboardInterrupt()))


def test_plugin_record_detaches_a_forged_mappingproxy_backing():
    backing = {"value": 1}
    caller_owned = MappingProxyType(backing)
    distribution = _valid_distribution(
        direct_url=None, direct_url_reason="missing_direct_url"
    )
    object.__setattr__(distribution, "direct_url", caller_owned)
    object.__setattr__(distribution, "direct_url_reason", None)

    record = _valid_record(distributions=(distribution,))
    backing["value"] = 2
    assert record.distributions[0].direct_url == {"value": 1}


def test_plugin_record_copy_does_not_swallow_baseexception(monkeypatch):
    from _rheplicant_bootstrap import plugins as plugin_module

    distribution = _valid_distribution(
        direct_url=None, direct_url_reason="missing_direct_url"
    )

    def stop(_value, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(plugin_module, "_copy_distribution_record", stop)
    with pytest.raises(KeyboardInterrupt):
        _valid_record(distributions=(distribution,))


def test_plugin_projection_is_recursively_frozen_and_detached():
    direct_url = {"url": "https://example.invalid", "nested": [1, {"ok": True}]}
    distribution = _valid_distribution(direct_url=direct_url)
    direct_url["nested"].append("after-construction")
    record = _valid_record(distributions=(distribution,))
    row = plugin_audit_row(record)
    direct_url["nested"].append(2)

    assert tuple(row) == PLUGIN_ROW_KEYS
    assert tuple(row["distributions"][0]) == PLUGIN_DISTRIBUTION_ROW_KEYS
    assert row["distributions"][0]["direct_url"] == {
        "url": "https://example.invalid",
        "nested": (1, {"ok": True}),
    }
    with pytest.raises(TypeError):
        row["distributions"][0]["direct_url"]["late"] = True


def test_shared_direct_url_dag_is_validated_once_per_unique_container():
    direct_url = {"leaf": True}
    for _ in range(40):
        direct_url = {"left": direct_url, "right": direct_url}

    distribution = _valid_distribution(direct_url=direct_url)
    record = _valid_record(distributions=(distribution,))
    frozen = plugin_audit_row(record)["distributions"][0]["direct_url"]

    for _ in range(40):
        assert frozen["left"] is frozen["right"]
        frozen = frozen["left"]
    assert frozen == {"leaf": True}


def test_plugin_modules_remain_jax_free_on_import():
    source = (
        f"import sys; sys.path.insert(0, {str(SRC)!r}); "
        "import _rheplicant_bootstrap.plugins; "
        "import json, sys; "
        "print(json.dumps(sorted({name.split('.')[0] for name in sys.modules})))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", source],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert set(json.loads(completed.stdout)).isdisjoint(
        {"jax", "jaxlib", "equinox", "numpy", "numpyro", "rheplicant"}
    )


def test_plugin_row_precedes_reverification_and_mutation_leaves_boundary_open(
    tmp_path,
):
    plugin_name = "fixture_mutates_jax"
    (tmp_path / f"{plugin_name}.py").write_text(
        "from jax import config\n"
        "config.update('jax_enable_x64', True)\n"
    )
    program = f"""
import json, sys
sys.path.insert(0, {str(SRC)!r})
sys.path.insert(0, {str(tmp_path)!r})
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap import plugins as plugin_module
from _rheplicant_bootstrap.plugins import import_plugin, plugin_audit_row
from _rheplicant_bootstrap.process import RuntimeSpec
from _rheplicant_bootstrap.runtime import establish_runtime

main_calls = []
def import_main():
    main_calls.append("main")
    import jax
    return jax

session, _ = establish_runtime(
    RuntimeSpec(False, "cpu", None, {{}}), import_main=import_main
)
boundaries = ["runtime"]
rows = []
plugin_module.metadata.packages_distributions = lambda: {{}}
plugin_module.metadata.distributions = lambda: ()
try:
    record = import_plugin({plugin_name!r})
    rows.append(plugin_audit_row(record))
    session.verify(boundary="plugin {plugin_name!r}")
    boundaries.append("plugins")
except ConfigError as error:
    print(json.dumps({{
        "main_calls": main_calls,
        "records": [row["name"] for row in rows],
        "boundaries": boundaries,
        "message": str(error),
    }}))
else:
    raise AssertionError("plugin mutation was accepted")
"""
    environment = os.environ.copy()
    environment.pop("JAX_ENABLE_X64", None)
    environment.pop("JAX_PLATFORMS", None)
    completed = subprocess.run(
        [sys.executable, "-I", "-c", program],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "main_calls": ["main"],
        "records": [plugin_name],
        "boundaries": ["runtime"],
        "message": (
            "runtime.jax_enable_x64: requested False, but plugin "
            f"'{plugin_name}' has True."
        ),
    }
