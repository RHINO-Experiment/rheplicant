"""What rheplicant.config exports, and the boundary it keeps."""

import pathlib

import rheplicant
import rheplicant.config


class TestTheSurface:
    def test_everything_in_all_is_importable_from_the_package(self):
        for name in rheplicant.config.__all__:
            assert hasattr(rheplicant.config, name), name

    def test_the_config_layer_is_not_re_exported_from_the_top_level(self):
        """rheplicant.__all__ is the package's advertised surface and every
        name in it is a modelling object. A config layer that leaked into it
        would make `from rheplicant import *` import a document parser."""
        assert not set(rheplicant.config.__all__) & set(rheplicant.__all__)

    def test_the_registry_views_are_live_rather_than_snapshots(self):
        """FILE_FORMATS and DERIVATIONS are re-exported because a caller asking
        "what can this document say?" should not have to import the private
        module that happens to hold the table. They must stay the LiveNames
        views -- a tuple() taken here would freeze at import time and go short
        the moment Plan 1B registers into the same tables."""
        from rheplicant.config.derive import _DERIVATIONS
        from rheplicant.config.files import _READERS

        assert set(rheplicant.config.FILE_FORMATS) == set(_READERS)
        assert set(rheplicant.config.DERIVATIONS) == set(_DERIVATIONS)


class TestTheLayerBoundaryIsMechanical:
    def test_no_config_module_is_imported_by_core_radio_or_inference(self):
        """The other direction is guarded by tests/core/test_layering.py for
        core. This is the whole-package half: nothing below config may reach
        up into it, or config stops being removable."""
        src = pathlib.Path(rheplicant.__file__).parent
        offenders = [
            str(path.relative_to(src))
            for path in src.rglob("*.py")
            if "config" not in path.parts
            and (
                "from rheplicant.config" in path.read_text()
                or "import rheplicant.config" in path.read_text()
            )
        ]
        assert not offenders, offenders


class TestEveryFormHasAResolver:
    def test_the_registry_covers_the_declared_grammar(self):
        """VALUE_FORMS is the grammar; _RESOLVERS is what is implemented. The
        two must not drift apart silently -- a form declared and unregistered
        is a key a user can write that does nothing recognisable."""
        from rheplicant.config.values import _RESOLVERS, VALUE_FORMS

        declared = set(VALUE_FORMS) - {"value"}  # form 1 is handled inline
        implemented = set(_RESOLVERS)
        deferred = set()  # nothing deferred: every declared form has a resolver
        assert declared - implemented == deferred, declared - implemented
        assert implemented - declared == set()


class TestPlan1BOnTheSurface:
    def test_the_path_and_resource_entry_points_are_exported(self):
        import rheplicant.config as config

        for name in ("compile_path", "resolve_path_on", "build_resources", "RESOURCE_KINDS"):
            assert name in config.__all__, name

    def test_every_registry_is_reachable_from_the_package(self):
        """Four registries, and a reader who wants to know what is available
        should not have to import four private modules to find out."""
        import rheplicant.config as config

        for name in ("VALUE_FORMS", "FILE_FORMATS", "DERIVATIONS", "RESOURCE_KINDS"):
            assert name in config.__all__, name


class TestThePlan2ASurface:
    def test_the_document_layer_is_exported(self):
        import rheplicant.config as config

        for name in ("ConfiguredRun", "apply_variant", "load_document",
                     "recursive_update", "run_forward"):
            assert name in config.__all__
            assert getattr(config, name) is not None

    def test_importing_the_package_registers_the_object_readers(self):
        import rheplicant.config as config

        assert "rhino_hdf5" in config.FILE_FORMATS
        assert "eqx_leaves" in config.FILE_FORMATS
