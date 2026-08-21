"""The typecheck over the React component suite is closed rather than absent.

``tests/gui/react/`` was in no tsconfig program at all: vitest transpiles it
with esbuild, which strips types without checking them, so nineteen suites and
several hundred tests compiled every fixture no matter how far it had drifted
from the type it claimed to be. Turning the check on found six real defects on
the first run, two of them fixtures that no longer matched ``EditorSnapshot``
and ``SessionTransport``.
"""

import json
import pathlib
import subprocess

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SUITE = (_ROOT / "tests/gui/react").resolve()
_REACT = _ROOT / "tools/config_gui_spike/react"
_SCRIPT = "tsc --noEmit -p tsconfig.tests.json"
_SETUP = "src/test-setup.ts"
_GLOBS = [
    "../../../tests/gui/react/**/*.ts",
    "../../../tests/gui/react/**/*.tsx",
]
#: The seven flags the sibling e2e project carries, plus the two this suite
#: needs: ``jsx`` because every file is a component test, and ``paths`` because
#: the suite lives outside this directory and bare specifiers resolve from the
#: importing file, not from the tsconfig.
_FLAGS = {
    "noEmit": True,
    "strict": True,
    "skipLibCheck": True,
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "types": ["node"],
    "jsx": "react-jsx",
}


def _manifest() -> dict:
    return json.loads((_REACT / "package.json").read_text())


def _project() -> dict:
    return json.loads((_REACT / "tsconfig.tests.json").read_text())


def _on_disk() -> set[pathlib.Path]:
    """Every TypeScript file the component suite is made of.

    Both extensions, because every file here is ``.tsx`` today and a plain
    ``.ts`` helper is the obvious next addition; a guard matching only one of
    them would bless exactly the escape it exists to refuse. ``rglob`` rather
    than ``glob`` for the same reason a subdirectory gives: a suite in a folder
    is still a suite. ``.mts``/``.cts`` are matched by neither side, here or in
    the project's own globs -- the suite uses neither, and the two sides must
    keep agreeing by construction rather than by luck.
    """
    return {
        path.resolve()
        for suffix in ("*.ts", "*.tsx")
        for path in _SUITE.rglob(suffix)
        if "node_modules" not in path.parts
    }


class TestTheReactTestTypecheckIsClosed:
    def test_the_script_names_a_project_rather_than_a_list_of_files(self):
        """A hardcoded list is open by construction: the twentieth suite is
        typechecked only if whoever wrote it remembered to extend the command,
        and nothing goes red when they do not. A project reference cannot be
        forgotten because it names no file at all."""
        script = _manifest()["scripts"]["check:tests"]
        assert script == _SCRIPT
        assert ".test.tsx" not in script
        assert "tests/gui/react" not in script

    def test_the_project_states_its_flags_rather_than_inheriting_them(self):
        """Written out rather than extended from ``./tsconfig.json``: that file
        sets an explicit ``lib``, ``isolatedModules``, ``allowJs`` and five more
        this check never asked for, so extending it would quietly retune the
        check while looking like a tidy-up. The same argument the e2e project
        records, for the same reason."""
        project = _project()
        assert "extends" not in project, (
            "the flags below would stop being the whole truth"
        )
        assert project["compilerOptions"] | _FLAGS == project["compilerOptions"]
        assert "lib" not in project["compilerOptions"], (
            "the command omits it, so the target's default lib is the baseline"
        )

    def test_the_include_carries_the_setup_file_and_both_recursive_globs(self):
        """``src/test-setup.ts`` is not decoration. It carries
        ``import "@testing-library/jest-dom/vitest"``, whose module augmentation
        is what teaches ``expect`` about ``toBeVisible`` and its forty siblings.
        Without it in the program the check reports 893 phantom errors and is
        useless; with it, six real ones. An ``exclude`` would let a suite be
        carved back out, so it is refused outright."""
        project = _project()
        assert project["include"] == [_SETUP, *_GLOBS]
        assert "exclude" not in project
        assert "files" not in project
        for glob in _GLOBS:
            prefix, marker, tail = glob.partition("**/")
            assert marker and tail in {"*.ts", "*.tsx"}
            assert (_REACT / prefix).resolve() == _SUITE

    def test_every_typescript_file_in_the_suite_is_in_the_program(self):
        """The empirical half. ``--listFiles`` reports the program tsc actually
        built, so a file that is neither included nor imported by something
        included -- which is exactly what a newly added, not-yet-imported suite
        is -- simply does not appear.

        Compared as an equality, so a file that should NOT be checked is a
        failure too; that half is a scope belt rather than a defect belt. The
        return code is deliberately not asserted: ``check:tests`` is the
        typecheck gate, and a guard with two jobs reports neither clearly.
        """
        completed = subprocess.run(
            [
                "node_modules/.bin/tsc",
                "--noEmit",
                "-p",
                "tsconfig.tests.json",
                "--listFiles",
            ],
            cwd=_REACT,
            check=False,
            capture_output=True,
            text=True,
        )
        listed = {
            pathlib.Path(line).resolve()
            for line in completed.stdout.splitlines()
            if line.endswith((".ts", ".tsx"))
        }
        covered = {path for path in listed if _SUITE in path.parents}
        missing = sorted(str(path) for path in _on_disk() - covered)
        extra = sorted(str(path) for path in covered - _on_disk())
        assert not missing and not extra, (
            f"missing={missing} extra={extra}\n"
            f"{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
        )
