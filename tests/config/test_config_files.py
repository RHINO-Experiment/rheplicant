"""Form 4: where a path resolves from, what reads it, and what it hashes to."""

import hashlib

import jax
import numpy as np
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.context import ResolutionContext
from rheplicant.config.files import FILE_FORMATS, register_reader, resolve_file_path
from rheplicant.config.values import resolve_value


@pytest.fixture
def workspace(tmp_path):
    np.save(tmp_path / "gain.npy", np.array([1.0, 2.0, 3.0]))
    np.savez(tmp_path / "bundle.npz", gain=np.array([4.0, 5.0]))
    np.savez(
        tmp_path / "pair.npz",
        first=np.array([1.0, 2.0, 3.0, 4.0]),
        second=np.array([7.0, 8.0]),
    )
    (tmp_path / "grid.txt").write_text("1.0 10.0\n2.0 20.0\n3.0 30.0\n")
    (tmp_path / "skip.txt").write_text("999.0 999.0\n1.0 10.0\n2.0 20.0\n")
    (tmp_path / "table.csv").write_text("az_deg,el_deg\n0.0,90.0\n1.0,89.0\n")
    return tmp_path


@pytest.fixture
def context(workspace):
    return ResolutionContext(dtype="float32", base_dir=str(workspace))


class TestTheArrayReaders:
    def test_npy(self, context):
        got = resolve_value(
            {"file": {"path": "gain.npy", "format": "npy"}, "unit": "dimensionless"}, context
        )
        assert got.value.shape == (3,)
        assert got.source == "file"

    def test_npz_requires_a_key(self, context):
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"file": {"path": "bundle.npz", "format": "npz"}}, context)
        message = str(excinfo.value)
        assert "key" in message
        assert "gain" in message  # the archive's own keys, listed

    def test_npz_with_a_key(self, context):
        got = resolve_value(
            {"file": {"path": "bundle.npz", "format": "npz", "key": "gain"}}, context
        )
        assert got.value.shape == (2,)

    def test_npz_reads_the_named_array_and_not_the_first(self, context):
        """Catches a reader that ignores `key` and hands back archive.files[0].
        bundle.npz holds one array, so only a two-array archive can tell the
        difference -- and the two here differ in length, not only in content."""
        got = resolve_value(
            {"file": {"path": "pair.npz", "format": "npz", "key": "second"}}, context
        )
        assert [float(v) for v in got.value] == pytest.approx([7.0, 8.0])

    def test_npz_refuses_a_key_the_archive_does_not_hold(self, context):
        """Catches `archive[key]` reached without the membership check: numpy
        raises a bare KeyError naming neither the file nor the keys it does
        hold, which is the whole content of the refusal."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"file": {"path": "pair.npz", "format": "npz", "key": "third"}}, context)
        message = str(excinfo.value)
        assert "third" in message
        assert "first" in message and "second" in message

    def test_txt_with_a_column(self, context):
        got = resolve_value({"file": {"path": "grid.txt", "format": "txt", "column": 1}}, context)
        assert [float(v) for v in got.value] == pytest.approx([10.0, 20.0, 30.0])

    def test_txt_column_is_one_dimensional(self, context):
        """Catches `column` handled as `columns` -- data[:, [1]] has shape
        (3, 1), and float() on each row of it still yields 10.0, 20.0, 30.0, so
        the value check above passes a reader that returns the wrong rank. (n,)
        against (n, 1) is precisely the distinction modifiers.column: exists to
        make, so a silent (n, 1) here is not cosmetic."""
        got = resolve_value({"file": {"path": "grid.txt", "format": "txt", "column": 1}}, context)
        assert got.value.shape == (3,)

    def test_txt_columns_takes_several(self, context):
        """Catches `columns` handled as `column` -- data[:, [0, 1]] is the plural
        reading and int(spec['columns']) on a list raises instead."""
        got = resolve_value(
            {"file": {"path": "grid.txt", "format": "txt", "columns": [0, 1]}}, context
        )
        assert got.value.shape == (3, 2)

    def test_txt_skiprows_is_applied(self, context):
        """Catches skiprows read but not passed to loadtxt: the dropped first
        row is numeric, so the file still parses and the column simply comes
        back one element longer with a plausible number at the front."""
        got = resolve_value(
            {"file": {"path": "skip.txt", "format": "txt", "skiprows": 1, "column": 1}}, context
        )
        assert [float(v) for v in got.value] == pytest.approx([10.0, 20.0])

    def test_txt_with_neither_column_nor_columns_reads_the_whole_table(self, context):
        got = resolve_value({"file": {"path": "grid.txt", "format": "txt"}}, context)
        assert got.value.shape == (3, 2)

    def test_csv_by_column_name(self, context):
        got = resolve_value(
            {"file": {"path": "table.csv", "format": "csv", "columns": ["el_deg"]}, "unit": "deg"},
            context,
        )
        assert [float(v) for v in got.value] == pytest.approx([90.0, 89.0])

    def test_csv_names_the_column_rather_than_taking_the_first(self, context):
        """Catches a reader that ignores `columns` and returns data[names[0]].
        az_deg is 0.0, 1.0 and el_deg is 90.0, 89.0, so the wrong column is a
        perfectly plausible pair of angles."""
        got = resolve_value(
            {"file": {"path": "table.csv", "format": "csv", "columns": ["az_deg"]}}, context
        )
        assert [float(v) for v in got.value] == pytest.approx([0.0, 1.0])

    def test_csv_without_columns_stacks_every_one(self, context):
        got = resolve_value({"file": {"path": "table.csv", "format": "csv"}}, context)
        assert got.value.shape == (2, 2)

    def test_csv_with_two_columns_stacks_those(self, context):
        """Catches the len(columns) == 1 branch applied to every length, which
        would return only the first named column and drop the rest."""
        got = resolve_value(
            {"file": {"path": "table.csv", "format": "csv", "columns": ["el_deg", "az_deg"]}},
            context,
        )
        assert got.value.shape == (2, 2)
        assert [float(v) for v in got.value[:, 0]] == pytest.approx([90.0, 89.0])

    def test_csv_refuses_a_column_name_the_header_does_not_have(self, context):
        """Catches the `missing` check dropped: numpy raises KeyError naming
        neither the file nor its header."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(
                {"file": {"path": "table.csv", "format": "csv", "columns": ["ra_deg"]}}, context
            )
        message = str(excinfo.value)
        assert "ra_deg" in message
        assert "el_deg" in message

    def test_csv_takes_a_declared_delimiter(self, workspace, context):
        """Catches `delimiter` accepted as a key and then not passed on: the
        default comma finds no separator, so every row becomes one field and the
        header names collapse to a single column."""
        (workspace / "semi.csv").write_text("az_deg;el_deg\n0.0;90.0\n1.0;89.0\n")
        got = resolve_value(
            {
                "file": {
                    "path": "semi.csv",
                    "format": "csv",
                    "delimiter": ";",
                    "columns": ["el_deg"],
                }
            },
            context,
        )
        assert [float(v) for v in got.value] == pytest.approx([90.0, 89.0])

    def test_the_unit_converts_a_file_s_contents_too(self, context):
        got = resolve_value({"file": {"path": "gain.npy", "format": "npy"}, "unit": "MHz"}, context)
        assert float(got.value[0]) == pytest.approx(1e6)

    def test_the_form_is_reported_as_file_on_both_branches(self, context):
        """Catches the source reported as anything else on the no-unit return.
        test_npy declares a unit, so it exercises only the converted branch --
        and the two branches build their ResolvedValue at two separate call
        sites. The source is what check A40 dispatches on, so a wrong one there
        does not fail loudly, it lets an array onto a static field."""
        with_unit = resolve_value(
            {"file": {"path": "gain.npy", "format": "npy"}, "unit": "MHz"}, context
        )
        without = resolve_value({"file": {"path": "gain.npy", "format": "npy"}}, context)
        assert with_unit.source == "file"
        assert without.source == "file"

    def test_the_value_is_a_jax_array(self, context):
        """Catches the reader's numpy result returned unwrapped. Every other
        form in this grammar hands back a jnp array, and delivery passes a value
        through untouched on a static_other field, so a numpy array here would
        reach a treedef by a route ARRAY_FORMS does not cover."""
        got = resolve_value({"file": {"path": "gain.npy", "format": "npy"}}, context)
        assert isinstance(got.value, jax.Array)


class TestPathResolution:
    def test_a_relative_path_resolves_against_the_documents_directory(self, workspace, context):
        assert resolve_file_path("gain.npy", context) == workspace / "gain.npy"

    def test_then_against_the_declared_roots(self, workspace, tmp_path):
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "beam.npy").write_bytes(b"x")
        context = ResolutionContext(base_dir=str(workspace), roots=(str(elsewhere),))
        assert resolve_file_path("beam.npy", context) == elsewhere / "beam.npy"

    def test_the_documents_directory_is_tried_before_the_roots(self, workspace, tmp_path):
        """Catches the roots being searched ahead of the document's own
        directory. The test above cannot: beam.npy exists in exactly one of the
        two places, so either order finds the same file. Only a name present in
        BOTH tells the order apart -- and that is the case that matters, since a
        stale copy on a shared root is what the ordering exists to lose to."""
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "gain.npy").write_bytes(b"a different copy")
        context = ResolutionContext(base_dir=str(workspace), roots=(str(elsewhere),))
        assert resolve_file_path("gain.npy", context) == workspace / "gain.npy"

    def test_the_roots_are_tried_in_the_order_declared(self, workspace, tmp_path):
        """Catches roots iterated in any order but the document's -- sorted(),
        reversed(), or a set -- which a single root cannot detect."""
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        (first / "beam.npy").write_bytes(b"a")
        (second / "beam.npy").write_bytes(b"b")
        context = ResolutionContext(base_dir=str(workspace), roots=(str(second), str(first)))
        assert resolve_file_path("beam.npy", context) == second / "beam.npy"

    def test_an_absolute_path_is_taken_as_written(self, workspace, context):
        assert resolve_file_path(str(workspace / "gain.npy"), context) == workspace / "gain.npy"

    def test_a_tilde_expands(self, context):
        assert "~" not in str(resolve_file_path("~/nowhere.npy", context, must_exist=False))

    def test_an_environment_variable_expands(self, monkeypatch, workspace, context):
        monkeypatch.setenv("RHEPLICANT_TEST_DIR", str(workspace))
        assert (
            resolve_file_path("${RHEPLICANT_TEST_DIR}/gain.npy", context) == workspace / "gain.npy"
        )

    def test_must_exist_false_returns_a_candidate_rather_than_refusing(self, workspace, context):
        """Catches must_exist=False falling through into the existence loop,
        which would refuse the one case the flag exists for."""
        assert (
            resolve_file_path("absent.npy", context, must_exist=False) == workspace / "absent.npy"
        )

    def test_a_relative_path_falls_back_to_the_working_directory(self, monkeypatch, workspace):
        """Catches the final 'as written' candidate dropped. A document read
        from stdin or built in memory declares no base_dir and may declare no
        roots, and with both absent every relative path would be refused with an
        empty 'looked for at:' list."""
        monkeypatch.chdir(workspace)
        context = ResolutionContext()
        assert resolve_file_path("gain.npy", context) == workspace.resolve() / "gain.npy"

    def test_a_missing_file_names_every_place_it_was_looked_for(self, workspace):
        context = ResolutionContext(base_dir=str(workspace), roots=("/opt/beams",))
        with pytest.raises(ConfigError) as excinfo:
            resolve_file_path("absent.npy", context)
        message = str(excinfo.value)
        assert "absent.npy" in message
        assert str(workspace) in message
        assert "/opt/beams" in message


class TestTheHash:
    def test_a_file_reference_records_its_sha256(self, workspace, context):
        got = resolve_value({"file": {"path": "gain.npy", "format": "npy"}}, context)
        expected = hashlib.sha256((workspace / "gain.npy").read_bytes()).hexdigest()
        assert got.modifiers["_sha256"] == expected

    def test_the_hash_is_over_the_bytes_and_not_the_path(self, workspace, context):
        """Catches sha256(str(path)) -- which is stable, 64 hex characters, and
        records nothing about the file. Two copies of the same bytes under
        different names must hash alike; the path digest cannot."""
        (workspace / "copy.npy").write_bytes((workspace / "gain.npy").read_bytes())
        one = resolve_value({"file": {"path": "gain.npy", "format": "npy"}}, context)
        other = resolve_value({"file": {"path": "copy.npy", "format": "npy"}}, context)
        assert one.modifiers["_sha256"] == other.modifiers["_sha256"]
        assert (
            one.modifiers["_sha256"]
            != hashlib.sha256(str(workspace / "gain.npy").encode()).hexdigest()
        )

    def test_the_resolved_path_is_recorded_alongside_the_hash(self, workspace, context):
        """Catches _path dropped or recorded as the raw declared string: a hash
        with no path says which bytes were read but not which copy of the file
        the search order actually landed on."""
        got = resolve_value({"file": {"path": "gain.npy", "format": "npy"}}, context)
        assert got.modifiers["_path"] == str(workspace / "gain.npy")

    def test_a_declared_sha256_that_disagrees_is_refused(self, context):
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(
                {"file": {"path": "gain.npy", "format": "npy", "sha256": "0" * 64}}, context
            )
        message = str(excinfo.value)
        assert "0000" in message  # what was declared
        assert "gain.npy" in message

    def test_a_declared_sha256_that_agrees_is_accepted(self, workspace, context):
        """The complement, and the one the inverted comparison fails: with
        `declared == digest` as the refusal condition, every honest declaration
        in every document is refused and every wrong one passes."""
        digest = hashlib.sha256((workspace / "gain.npy").read_bytes()).hexdigest()
        got = resolve_value(
            {"file": {"path": "gain.npy", "format": "npy", "sha256": digest}}, context
        )
        assert got.value.shape == (3,)

    def test_a_declared_sha256_differing_only_at_its_end_is_refused(self, workspace, context):
        """Catches a comparison over a prefix -- declared[:8], startswith, or
        anything short of the whole digest. A truncated compare is the shape
        this check most plausibly degrades into, and it accepts a digest that
        agrees for eight characters and disagrees about the file."""
        digest = hashlib.sha256((workspace / "gain.npy").read_bytes()).hexdigest()
        near = digest[:-1] + ("0" if digest[-1] != "0" else "1")
        with pytest.raises(ConfigError):
            resolve_value({"file": {"path": "gain.npy", "format": "npy", "sha256": near}}, context)


class TestWhenAReaderFails:
    def test_a_serialised_object_array_is_refused(self, workspace, context):
        """The guard between an untrusted .npy and arbitrary code execution at
        config-load time. An object-dtype .npy is a serialised Python object
        graph, and reconstructing one runs whatever code it names; numpy
        declines to do that by default, and files.py passes no argument to
        change it. This test is what makes that a decision rather than an
        accident -- the obvious "fix" for a contributor who legitimately wants
        such a file loaded is to pass the argument that turns it on, and
        nothing else in the package would push back. Asserting on the refusal's
        own wording rather than merely on ConfigError is the point: were the
        default ever to flip, np.load would succeed and the failure would move
        to jnp.asarray, which raises too -- so a bare `raises(ConfigError)`
        would stay green while the guarantee was gone."""
        np.save(workspace / "objects.npy", np.array([{"a": 1}], dtype=object))
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"file": {"path": "objects.npy", "format": "npy"}}, context)
        assert "Object arrays cannot be loaded" in str(excinfo.value)

    def test_a_malformed_file_is_refused_with_the_documents_own_context(self, workspace, context):
        """Catches the reader's exception escaping unwrapped. np.loadtxt raises
        a bare ValueError naming a line number and nothing else -- not the
        resolved path this layer chose out of several candidates, not the
        format that was declared, not what to check."""
        (workspace / "ragged.txt").write_text("1.0 2.0\n3.0\n")
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"file": {"path": "ragged.txt", "format": "txt"}}, context)
        message = str(excinfo.value)
        assert str(workspace / "ragged.txt") in message
        assert "txt" in message
        assert "ValueError" in message  # the library's own type, named not hidden
        assert "skiprows" in message  # and what to check

    def test_a_malformed_csv_is_refused_the_same_way(self, workspace, context):
        """genfromtxt is a different numpy entry point from loadtxt and fails
        differently; the wrapper is shared, so this catches it being applied to
        only one of the two readers."""
        (workspace / "ragged.csv").write_text("az_deg,el_deg\n0.0,90.0,7.0\nx,y\n")
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(
                {"file": {"path": "ragged.csv", "format": "csv", "columns": ["el_deg"]}}, context
            )
        assert "delimiter" in str(excinfo.value)

    def test_a_column_index_past_the_end_is_refused(self, context):
        """IndexError, not ValueError -- the enumerate-the-types version of this
        guard catches loadtxt's failure and lets the indexing failure past."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"file": {"path": "grid.txt", "format": "txt", "column": 9}}, context)
        assert "IndexError" in str(excinfo.value)

    def test_the_librarys_exception_is_chained(self, workspace, context):
        """Catches `raise ConfigError(...)` written without `from exc`. Without
        the chain the traceback stops at this layer and the library's own frame
        -- the only thing that says where in the file parsing failed -- is
        gone. __context__ is set implicitly inside an except block, so only
        __cause__ tells a chained raise from an unchained one."""
        (workspace / "ragged.txt").write_text("1.0 2.0\n3.0\n")
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"file": {"path": "ragged.txt", "format": "txt"}}, context)
        assert excinfo.value.__cause__ is not None
        assert not isinstance(excinfo.value.__cause__, ConfigError)

    def test_a_readers_own_refusal_is_not_rewrapped(self, context):
        """Catches the `except ConfigError: raise` clause dropped. ConfigError
        subclasses ValueError, so a broad catch swallows this layer's own
        refusals and buries a message that named the archive's keys under
        advice about delimiters and header rows. The message check cannot see
        it -- the original text is quoted inside the wrapper -- but a
        re-wrapped refusal carries a __cause__ and a deliberate one does not."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"file": {"path": "bundle.npz", "format": "npz"}}, context)
        assert excinfo.value.__cause__ is None


class TestTheRegistry:
    def test_an_unknown_format_is_refused_and_the_registered_ones_are_listed(self, context):
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"file": {"path": "gain.npy", "format": "fits"}}, context)
        message = str(excinfo.value)
        assert "fits" in message
        for name in FILE_FORMATS:
            assert name in message, name

    def test_the_listed_formats_are_read_live_rather_than_snapshotted(self, context):
        """Catches FILE_FORMATS = tuple(sorted(_READERS)) taken at import. Plan
        1B registers four more formats from its own module, which imports after
        this one, so a snapshot would leave every refusal naming a set the
        loader no longer has -- and the test above cannot see it, because the
        snapshot and the table agree for as long as nothing else registers."""

        @register_reader("probe_format_for_the_test")
        def _probe(path, spec):  # pragma: no cover - never read, only registered
            raise AssertionError

        try:
            assert "probe_format_for_the_test" in FILE_FORMATS
            assert "probe_format_for_the_test" in list(FILE_FORMATS)
            with pytest.raises(ConfigError) as excinfo:
                resolve_value({"file": {"path": "gain.npy", "format": "fits"}}, context)
            assert "probe_format_for_the_test" in str(excinfo.value)
        finally:
            from rheplicant.config.files import _READERS

            _READERS.pop("probe_format_for_the_test")
        assert "probe_format_for_the_test" not in FILE_FORMATS

    def test_healpix_is_refused_by_name_and_names_its_route(self, context):
        """format: healpix is real -- it lives at resources.beams (D-C7) -- but
        a bare value node has nowhere to put order:, the declared frequency
        grid or frame:, so it is not read through one."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"file": {"path": "sky.fits", "format": "healpix"}}, context)
        message = str(excinfo.value)
        assert "healpix" in message
        assert "resources.beams" in message
        assert "order" in message

    def test_cst_dir_is_refused_and_names_its_route(self, context):
        """format: cst_dir is real too -- it lives at resources.beams, format:
        cst -- and the _ELSEWHERE branch in _file is generic, not a second
        healpix-only special case, so this exercises that generality."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"file": {"path": "cst/", "format": "cst_dir"}}, context)
        message = str(excinfo.value)
        assert "resources.beams" in message
        assert "format: cst" in message

    def test_rhino_hdf5_is_refused_and_names_its_route(self, context):
        """format: rhino_hdf5 lives at observation.from_file, which does not
        exist until Plan 2 -- named anyway, so the refusal points somewhere
        real rather than at a section that is not there yet."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"file": {"path": "obs.hdf5", "format": "rhino_hdf5"}}, context)
        assert "observation.from_file" in str(excinfo.value)

    def test_the_healpix_refusal_is_reached_before_the_unknown_format_one(self, context):
        """healpix is not in the reader table, so the generic 'unknown format'
        refusal would answer it first if the dedicated branch were moved or
        removed -- and that message names neither remedy. Naming what the
        generic one says, and asserting it is NOT what came back, is what keeps
        the ordering rather than merely the wording."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"file": {"path": "sky.fits", "format": "healpix"}}, context)
        assert "unknown format" not in str(excinfo.value)

    def test_a_misspelled_reader_key_is_refused(self, context):
        """Catches the unknown-key check removed. 'colum: 1' would otherwise be
        dropped in silence and the whole two-column table delivered where one
        column was meant -- finite, correctly shaped, and a different quantity."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"file": {"path": "grid.txt", "format": "txt", "colum": 1}}, context)
        message = str(excinfo.value)
        assert "colum" in message
        assert "column" in message  # what this format does take

    def test_a_key_belonging_to_another_format_is_refused(self, context):
        """Catches one shared key set for every reader: 'key' is npz's and npy
        has no use for it, so accepting it here means a document can name an
        array inside a file that holds exactly one, and be ignored."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"file": {"path": "gain.npy", "format": "npy", "key": "gain"}}, context)
        assert "key" in str(excinfo.value)

    def test_path_is_required(self, context):
        with pytest.raises(ConfigError, match="path"):
            resolve_value({"file": {"format": "npy"}}, context)

    def test_format_is_required(self, context):
        """Catches the format inferred from the extension: gain.npy would then
        read, and the refusal exists because two producers of one extension
        disagree often enough that guessing reads the wrong thing quietly."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"file": {"path": "gain.npy"}}, context)
        assert "format" in str(excinfo.value)

    def test_file_expects_a_mapping(self, context):
        with pytest.raises(ConfigError, match="mapping"):
            resolve_value({"file": "gain.npy"}, context)

    def test_a_static_field_cannot_take_a_file(self, context):
        """Check A40 through the real path: a file produces an array, and an
        array in the treedef is the jit-cache corruption delivery refuses."""
        from rheplicant.config.delivery import deliver, field_specs
        from rheplicant.radio.sky.foregrounds import ForegroundOperator

        got = resolve_value({"file": {"path": "gain.npy", "format": "npy"}}, context)
        with pytest.raises(ConfigError, match="static") as excinfo:
            deliver(
                got.value,
                field_specs(ForegroundOperator)["ref_freq"],
                dtype="float32",
                source=got.source,
            )
        # Which refusal, not merely that one came. A ref_freq handed an array
        # under any source at all is refused by _as_static_float, whose message
        # also says "static" -- so match="static" alone passes even when A40
        # never fired and the form was misreported. The treedef clause is A40's
        # and only A40's.
        message = str(excinfo.value)
        assert "treedef" in message
        assert "'file' form" in message
