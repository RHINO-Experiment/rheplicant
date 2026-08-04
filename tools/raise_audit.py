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


def has_pragma(path: Path, lineno: int) -> bool:
    lines = path.read_text(encoding="utf-8").splitlines()
    window = lines[max(0, lineno - 3): lineno]
    return any("pragma: no cover" in ln for ln in window)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="src/rheplicant")
    ap.add_argument("--coverage", help="coverage.json from --cov-report=json")
    ap.add_argument("--fail-on-uncovered", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(root.rglob("*.py"))
    missing_by_file = {}
    if args.coverage:
        data = json.loads(Path(args.coverage).read_text())
        for name, entry in data["files"].items():
            missing_by_file[str(Path(name).resolve())] = set(entry["missing_lines"])

    total = uncovered = pragma = 0
    rows = []
    for path in files:
        key = str(path.resolve())
        missing = missing_by_file.get(key)
        for lineno, exc in raise_lines(path):
            total += 1
            if missing is None:
                continue
            if lineno in missing:
                if has_pragma(path, lineno):
                    pragma += 1
                else:
                    uncovered += 1
                    rows.append(f"{path}:{lineno}: {exc}")

    for row in rows:
        print(row)
    print(f"\n{total} raise statements in {args.root}")
    if args.coverage:
        print(f"{uncovered} never executed and not marked `pragma: no cover`")
        print(f"{pragma} never executed but deliberately marked")
    return 1 if (args.fail_on_uncovered and uncovered) else 0


if __name__ == "__main__":
    sys.exit(main())
