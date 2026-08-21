"""The typecheck over the Playwright suite is closed rather than enumerated."""

import json
import pathlib
import subprocess

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_E2E = (_ROOT / "tests/gui/e2e").resolve()
_REACT = _ROOT / "tools/config_gui_spike/react"
_SCRIPT = "tsc --noEmit -p tsconfig.e2e.json"
_GLOB = "../../../tests/gui/e2e/**/*.ts"


def _manifest() -> dict:
    return json.loads((_REACT / "package.json").read_text())


def _project() -> dict:
    return json.loads((_REACT / "tsconfig.e2e.json").read_text())


def _on_disk() -> set[pathlib.Path]:
    """Every TypeScript file the Playwright suite is made of.

    ``rglob`` rather than ``glob``: a spec in a subdirectory is still a spec,
    and a guard that only looks at the top level would bless exactly the escape
    it exists to refuse. ``.mts``/``.cts`` are not matched, here or by the
    project's own glob -- the suite uses neither today, and the two sides must
    keep agreeing by construction rather than by luck.
    """
    return {
        path.resolve()
        for path in _E2E.rglob("*.ts")
        if "node_modules" not in path.parts
    }


class TestTheE2ETypecheckIsClosed:
    def test_the_script_names_a_project_rather_than_a_list_of_files(self):
        """A hardcoded list is open by construction: the eighth spec is
        typechecked only if whoever wrote it remembered to extend the command,
        and nothing goes red when they do not. A project reference cannot be
        forgotten because it names no file at all."""
        script = _manifest()["scripts"]["check:e2e"]
        assert script == _SCRIPT
        assert ".spec.ts" not in script
        assert "tests/gui/e2e" not in script

    def test_the_project_reproduces_the_strictness_the_command_carried(self):
        """The seven flags the enumerated command passed, one for one.

        Written out rather than inherited from ``./tsconfig.json``: that file
        sets an explicit ``lib``, plus ``jsx``, ``isolatedModules``,
        ``esModuleInterop``, ``useDefineForClassFields``, ``allowJs`` and
        ``resolveJsonModule``, none of which the command ever passed. Extending
        it would quietly retune the check while reading as a tidy-up.
        """
        project = _project()
        assert "extends" not in project, (
            "the flags below would stop being the whole truth"
        )
        assert project["compilerOptions"] == {
            "noEmit": True,
            "strict": True,
            "skipLibCheck": True,
            "target": "ES2022",
            "module": "ESNext",
            "moduleResolution": "Bundler",
            "types": ["node"],
        }
        assert "lib" not in project["compilerOptions"], (
            "the command omitted it, so the target's default lib is the baseline"
        )

    def test_the_include_is_a_recursive_glob_rooted_at_the_suite(self):
        """``*.ts`` would leave a spec in a subdirectory outside the program and
        an ``exclude`` would let one be carved back out, so both are pinned. The
        prefix is RESOLVED rather than string-matched: the glob is relative to
        the tsconfig's own directory and has to land on ``tests/gui/e2e``."""
        project = _project()
        assert project["include"] == ["playwright.config.ts", _GLOB]
        assert "exclude" not in project
        assert "files" not in project
        prefix, marker, tail = _GLOB.partition("**/")
        assert marker and tail == "*.ts"
        assert (_REACT / prefix).resolve() == _E2E

    def test_every_typescript_file_in_the_suite_is_in_the_program(self):
        """The empirical half. ``--listFiles`` reports the program tsc actually
        built, so a file that is neither included nor imported by something
        included -- which is exactly what a newly added, not-yet-imported spec
        is -- simply does not appear.

        Compared as an equality. ``missing`` is the defect belt: a spec that
        escaped the typecheck. ``extra`` is only a scope belt -- it says the
        program grew a file under this directory that is not on disk as one,
        which cannot happen today and would mean the glob had been re-rooted.
        Note that a file EXCLUDED from ``include`` but imported by an included
        one is still in the program, and still covered; being in the program is
        what "covered" means, and it is the orphan this catches.

        This does NOT assert tsc's return code. ``check:e2e`` is the typecheck
        gate; this is the coverage gate, and a guard with two jobs reports
        neither clearly.
        """
        completed = subprocess.run(
            [
                "node_modules/.bin/tsc",
                "--noEmit",
                "-p",
                "tsconfig.e2e.json",
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
            if line.endswith(".ts")
        }
        covered = {path for path in listed if _E2E in path.parents}
        on_disk = _on_disk()
        missing = sorted(str(path) for path in on_disk - covered)
        extra = sorted(str(path) for path in covered - on_disk)
        assert not missing and not extra, (
            f"missing={missing} extra={extra}\n"
            f"{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
        )
