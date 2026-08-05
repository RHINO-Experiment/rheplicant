"""List every `raise` in src/ and say whether the suite executes it.

Reads a coverage JSON (`--cov-report=json`). Without one it still reports the
raise inventory, which is the half that needs no test run.
"""
import argparse
import ast
import json
import sys
from pathlib import Path


def raise_lines(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc is not None:
            exc = node.exc
            name = getattr(getattr(exc, "func", exc), "id", None) or getattr(
                getattr(exc, "func", exc), "attr", "?"
            )
            out.append((node.lineno, name))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="src/rheplicant")
    ap.add_argument("--coverage", help="coverage.json from --cov-report=json")
    ap.add_argument("--fail-on-uncovered", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(root.rglob("*.py"))
    missing_by_file: dict[str, set[int]] = {}
    excluded_by_file: dict[str, set[int]] = {}
    if args.coverage:
        data = json.loads(Path(args.coverage).read_text())
        for name, entry in data["files"].items():
            key = str(Path(name).resolve())
            missing_by_file[key] = set(entry["missing_lines"])
            # `# pragma: no cover` puts a line in excluded_lines and NEVER in
            # missing_lines. Reading only missing_lines is how the first
            # version of this tool reported "0 deliberately marked" while two
            # pragmas were in place -- a counter that is structurally always
            # zero, and worse, a way to silence the whole audit by adding
            # pragmas and still print "0 uncovered".
            excluded_by_file[key] = set(entry.get("excluded_lines", []))

    total = uncovered = 0
    rows: list[str] = []
    excluded_rows: list[str] = []
    for path in files:
        key = str(path.resolve())
        missing = missing_by_file.get(key)
        for lineno, exc in raise_lines(path):
            total += 1
            if missing is None:
                continue
            if lineno in excluded_by_file.get(key, ()):
                excluded_rows.append(f"{path}:{lineno}: {exc}")
            elif lineno in missing:
                uncovered += 1
                rows.append(f"{path}:{lineno}: {exc}")

    for row in rows:
        print(row)
    print(f"\n{total} raise statements in {args.root}")
    if args.coverage:
        print(f"{uncovered} never executed and not marked `pragma: no cover`")
        print(f"{len(excluded_rows)} excluded by `pragma: no cover`:")
        for row in excluded_rows:
            # Listed, never merely counted: an exclusion is a claim that a
            # branch cannot be reached, and a claim nobody reads is how a
            # pragma outlives the argument it was added with.
            print(f"  {row}")
    return 1 if (args.fail_on_uncovered and uncovered) else 0


if __name__ == "__main__":
    sys.exit(main())
