from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from rheplicant.config.schemas import SCHEMA_NAMES
from tests.config.test_config_cli import document
from tests.config.test_config_document import synthetic_document
from tests.config.wheel_support import (
    BAYESMITH_VARIABLE,
    PROJECT_ROOT,
    BayesmithCheckout,
    Install,
    _run,
    build_distributions,
    fresh_install_factory,
    locate_bayesmith,
    running_gui,
    verified_release,
)

PRESET = PROJECT_ROOT / "src/rheplicant/config/presets/rhino_v1.yaml"
#: Derived, not listed. The list here was three names while the package
#: shipped four, so ``capabilities-v1`` would have gone into the wheel with
#: nothing checking it arrived -- and a schema absent from an install fails at
#: the reader, not here.
SCHEMAS = tuple(f"{name}.schema.json" for name in SCHEMA_NAMES)


@pytest.fixture(scope="session")
def built_distributions(tmp_path_factory):
    return build_distributions(tmp_path_factory)


@pytest.fixture
def fresh_install(tmp_path):
    return fresh_install_factory(tmp_path)


def _resource_probe(install: Install) -> dict[str, object]:
    program = (
        f"SCHEMA_FILES = {list(SCHEMAS)!r}\n"
        + """
import base64
import importlib.util
import json
from pathlib import Path
from _rheplicant_bootstrap.presets import read_installed_preset

snapshot = read_installed_preset("rhino_v1")
spec = importlib.util.find_spec("rheplicant")
root = Path(tuple(spec.submodule_search_locations)[0])
schemas = {}
for name in SCHEMA_FILES:
    schemas[name] = base64.b64encode(
        (root / "config" / "schemas" / name).read_bytes()
    ).decode("ascii")
print(json.dumps({
    "preset": base64.b64encode(snapshot.input_bytes).decode("ascii"),
    "sha256": snapshot.sha256,
    "schemas": schemas,
}))
"""
    )
    return json.loads(install.python_run(program).stdout)


@pytest.mark.parametrize("artifact", ["direct-wheel", "sdist-wheel"])
def test_installed_wheel_exposes_cli_presets_schemas_and_scripts(
    fresh_install, built_distributions, artifact, tmp_path
):
    install = fresh_install(built_distributions[artifact])
    config = tmp_path / f"{artifact}.yaml"
    target = tmp_path / f"{artifact}-generated-results"
    value = document(output=target)
    value["outputs"]["write"] = {"arrays": True, "assembly": True}
    config.write_text(yaml.safe_dump(value, sort_keys=False))

    clean = install.python_run(
        "import sys, _rheplicant_bootstrap; "
        "assert 'jax' not in sys.modules; "
        "assert 'jaxlib' not in sys.modules; "
        "assert 'rheplicant' not in sys.modules"
    )
    assert clean.stdout == ""
    validate = install.run(["validate", str(config)])
    assert validate.returncode == 0, validate.stderr
    assert validate.stdout == "configuration valid: base + 0 variants\n"

    stdin = install.run(
        ["validate", "-", "--base-dir", str(tmp_path)],
        input=config.read_text(),
    )
    assert stdin.returncode == 0, stdin.stderr
    script = tmp_path / f"{artifact}.py"
    generated = install.run(["script", str(config), "-o", str(script)])
    assert generated.returncode == 0, generated.stderr
    executed = subprocess.run(
        [os.fspath(install.python), os.fspath(script)],
        cwd=install.cwd,
        env=install.env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert executed.returncode == 0, executed.stderr
    assert (target / "config.input.yaml").read_bytes() == config.read_bytes()
    generated_manifest = (target / "products.json").read_bytes()
    generated_arrays = (target / "runs/n-666f7277617264/arrays.npz").read_bytes()
    generated_assembly = (target / "layers/base/assembly.json").read_bytes()

    direct_target = tmp_path / f"{artifact}-direct-results"
    direct_config = tmp_path / f"{artifact}-direct.yaml"
    direct_value = document(output=direct_target)
    direct_value["outputs"]["write"] = {"arrays": True, "assembly": True}
    direct_config.write_text(yaml.safe_dump(direct_value, sort_keys=False))
    direct = install.run(["run", str(direct_config)])
    assert direct.returncode == 0, direct.stderr
    assert (direct_target / "products.json").read_bytes() == generated_manifest
    assert (direct_target / "runs/n-666f7277617264/arrays.npz").read_bytes() == generated_arrays
    assert (direct_target / "layers/base/assembly.json").read_bytes() == generated_assembly

    resources = _resource_probe(install)
    expected = PRESET.read_bytes()
    assert base64.b64decode(resources["preset"]) == expected
    assert resources["sha256"] == hashlib.sha256(expected).hexdigest()
    for name in SCHEMAS:
        schema = json.loads(base64.b64decode(resources["schemas"][name]))
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def test_direct_and_sdist_wheels_have_the_same_closed_file_list(
    built_distributions,
):
    listings = []
    for key in ("direct-wheel", "sdist-wheel"):
        with zipfile.ZipFile(built_distributions[key]) as archive:
            names = tuple(row.filename for row in archive.infolist())
        assert len(names) == len(set(names))
        assert "rheplicant/config/presets/rhino_v1.yaml" in names
        # Derived, for the reason SCHEMAS is derived at the top of this file:
        # spelled out, this was three names against a package that shipped
        # four, so capabilities-v1 could ship or stop shipping with nothing
        # here noticing. The equality below closes the list, but only against
        # the OTHER wheel -- both can be wrong together.
        for schema in SCHEMAS:
            assert f"rheplicant/config/schemas/{schema}" in names, schema
        assert "rheplicant/gui/static/index.html" in names
        assert any(
            name.startswith("rheplicant/gui/static/assets/") and name.endswith(".js")
            for name in names
        )
        assert any(
            name.startswith("rheplicant/gui/static/assets/") and name.endswith(".css")
            for name in names
        )
        assert any(name.startswith("_rheplicant_bootstrap/") for name in names)
        assert any(name.endswith(".dist-info/entry_points.txt") for name in names)
        assert not any(
            "/tests/" in name
            or "__pycache__" in name
            or ".rheplicant-" in name
            or name.endswith(".pyc")
            for name in names
        )
        listings.append(names)
    assert listings[0] == listings[1]


def test_fresh_wheel_launches_gui_api_and_static_assets(
    fresh_install,
    built_distributions,
):
    install = fresh_install(built_distributions["direct-wheel"], extras=("gui",))
    with running_gui(install) as base_url:
        with urllib.request.urlopen(base_url + "/", timeout=5) as response:
            markup = response.read().decode("utf-8")
        assert "Rheplicant YAML config editor" in markup
        asset = re.search(r'src="(/[^"]+\.js)"', markup)
        assert asset is not None
        with urllib.request.urlopen(base_url + asset.group(1), timeout=5) as response:
            assert response.status == 200
            assert "javascript" in response.headers.get_content_type()
            assert response.read(1)

        request = urllib.request.Request(
            base_url + "/api/sessions",
            data=json.dumps({"yaml_text": "model: {}\nruns: []\n"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.load(response)
        assert response.status == 201
        assert payload["document"]["yaml_text"] == "model: {}\nruns: []\n"


def test_fresh_gui_wheel_contains_and_runs_the_scientific_worker(
    fresh_install, built_distributions
):
    install = fresh_install(built_distributions["direct-wheel"], extras=("gui",))
    worker_document = synthetic_document()
    worker_document["defaults"] = ["rhino_v1"]
    worker_document["observation"]["pointing"] = {"materialise": []}
    worker_document["outputs"] = {
        "dir": "priced-wheel",
        "clobber": False,
        "write": {"arrays": {"format": "npz"}},
    }
    completed = subprocess.run(
        [
            os.fspath(install.python),
            "-m",
            "_rheplicant_bootstrap.gui_worker",
            "validate",
        ],
        input=yaml.safe_dump(worker_document, sort_keys=False).encode("utf-8", "strict"),
        cwd=install.cwd,
        env=install.env,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    prefix = b"\x1eRHEPLICANT_GUI_JOB "
    encoded = completed.stdout.rsplit(prefix, 1)[1].split(b"\n", 1)[0]
    frame = json.loads(encoded.decode("utf-8", "strict"))
    # A53 rides along because the fixture places placeholder operators, and an
    # INSTALLED wheel reporting it is the point of checking here: the
    # capability levels are ClassVars on shipped classes, so a wheel that
    # dropped them would answer differently from the source tree.
    assert frame["status"] == "ok"
    assert frame["result"]["layers"] == 2
    assert [(one["check"], one["severity"]) for one in frame["result"]["findings"]] == [
        ("A53", "report")
    ]


def test_wheel_and_editable_preset_discovery_are_byte_identical(fresh_install, built_distributions):
    wheel = fresh_install(built_distributions["direct-wheel"])
    editable = fresh_install(PROJECT_ROOT, editable=True)
    wheel_row = _resource_probe(wheel)
    editable_row = _resource_probe(editable)
    assert (wheel_row["preset"], wheel_row["sha256"]) == (
        editable_row["preset"],
        editable_row["sha256"],
    )


# --- where bayesmith is looked for -------------------------------------------
#
# The fresh-venv tests above skip when the release is absent, and a skip is a
# thinner environment, never a pass. So the lookup that decides "absent" has to
# be right from every place the suite runs, and a worktree is one of them:
# until 2026-09-24 it looked beside the worktree, not beside the main checkout,
# and all five skipped while the release was on disk.

_USER = ("-c", "user.name=rheplicant", "-c", "user.email=tests@example.invalid")


def _git_environment(root: Path) -> dict[str, str]:
    """The inherited environment with no user, system or redirecting git state.

    No global config (HOME is the temporary root), so signing or hooks the
    developer has configured cannot fail a commit here; no GIT_* variables, so
    nothing points git at the repository the suite itself runs in; no bayesmith
    override, so the lookup takes the git path. The ceiling stops git at the
    temporary root, so a layout meant to have no repository has none.
    """
    environ = {
        key: value
        for key, value in os.environ.items()
        if key != BAYESMITH_VARIABLE and not key.startswith("GIT_")
    }
    return {
        **environ,
        "HOME": os.fspath(root),
        "XDG_CONFIG_HOME": os.fspath(root / ".config"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CEILING_DIRECTORIES": os.fspath(root),
    }


def _repository(path: Path, env: dict[str, str]) -> Path:
    path.mkdir(parents=True)
    _run(["git", "init", "-q"], cwd=path, env=env)
    _run(["git", *_USER, "commit", "-q", "--allow-empty", "-m", "root"], cwd=path, env=env)
    return path


def _main_with_worktree(root: Path, env: dict[str, str]) -> tuple[Path, Path]:
    """`<root>/projects/rheplicant` and a worktree of it where the app makes them."""
    main = _repository(root / "projects" / "rheplicant", env)
    worktree = main / ".claude" / "worktrees" / "w"
    _run(["git", "worktree", "add", "-q", "--detach", os.fspath(worktree)], cwd=main, env=env)
    return main, worktree


def test_bayesmith_is_one_directory_from_a_worktree_and_from_the_main_checkout(tmp_path):
    root = tmp_path.resolve()
    env = _git_environment(root)
    main, worktree = _main_with_worktree(root, env)
    # Pinned to the layout, not only to each other: two lookups that were
    # wrong the same way would agree.
    expected = root / "projects" / "bayesmith"
    assert locate_bayesmith(main, env).path == expected
    assert locate_bayesmith(worktree, env).path == expected


def test_this_checkout_looks_where_its_main_checkout_would():
    """The same property on the real layout, against an independent oracle.

    `git worktree list` names the main working tree first; the lookup reaches
    it through `--git-common-dir` instead. From the main checkout this is
    trivially true, and from a worktree it is the case that used to skip.
    """
    listed = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    lines = listed.stdout.splitlines()
    if listed.returncode != 0 or not lines or "bare" in lines[1:2]:
        pytest.skip(
            f"{PROJECT_ROOT} is not in a repository with a main working tree, "
            "so there is no main checkout to compare this one with"
        )
    main = Path(lines[0].removeprefix("worktree ")).resolve()
    environ = {key: value for key, value in os.environ.items() if key != BAYESMITH_VARIABLE}
    expected = main.parent / "bayesmith"
    assert locate_bayesmith(PROJECT_ROOT, environ).path == expected
    assert locate_bayesmith(main, environ).path == expected


def test_the_variable_names_the_checkout_over_git(tmp_path):
    root = tmp_path.resolve()
    env = _git_environment(root)
    _, worktree = _main_with_worktree(root, env)
    named = locate_bayesmith(worktree, {**env, BAYESMITH_VARIABLE: os.fspath(root / "elsewhere")})
    assert named.path == root / "elsewhere"
    assert BAYESMITH_VARIABLE in named.basis
    # Empty is unset, so an exported but blank variable does not name the cwd.
    blank = locate_bayesmith(worktree, {**env, BAYESMITH_VARIABLE: ""})
    assert blank.path == root / "projects" / "bayesmith"


def test_a_git_dir_left_in_the_environment_does_not_move_the_lookup(tmp_path):
    # A hook or a mutation script can leave GIT_DIR set. Obeyed, it makes git
    # answer for that repository with the project root as its work tree, and
    # bayesmith would be looked for beside the other repository.
    root = tmp_path.resolve()
    env = _git_environment(root)
    _, worktree = _main_with_worktree(root, env)
    other = _repository(root / "elsewhere" / "other", env)
    redirected = {**env, "GIT_DIR": os.fspath(other / ".git")}
    assert locate_bayesmith(worktree, redirected).path == root / "projects" / "bayesmith"


def _no_git_on_path(root: Path, env: dict[str, str]) -> tuple[Path, dict[str, str]]:
    _, worktree = _main_with_worktree(root, env)
    return worktree, {**env, "PATH": os.fspath(root / "no-bin")}


def _not_a_repository(root: Path, env: dict[str, str]) -> tuple[Path, dict[str, str]]:
    project = root / "projects" / "rheplicant"
    project.mkdir(parents=True)
    return project, env


def _inside_another_repository(root: Path, env: dict[str, str]) -> tuple[Path, dict[str, str]]:
    # An unpacked sdist somewhere under someone else's checkout: git answers
    # for THAT repository, whose root is not this project's.
    outer = _repository(root / "outer", env)
    project = outer / "projects" / "rheplicant"
    project.mkdir(parents=True)
    return project, env


def _worktree_of_a_bare_repository(root: Path, env: dict[str, str]) -> tuple[Path, dict[str, str]]:
    seed = _repository(root / "seed", env)
    bare = root / "projects" / "rheplicant.git"
    _run(["git", "clone", "-q", "--bare", os.fspath(seed), os.fspath(bare)], cwd=root, env=env)
    project = root / "projects" / "rheplicant"
    _run(["git", "worktree", "add", "-q", "--detach", os.fspath(project)], cwd=bare, env=env)
    return project, env


_NO_MAIN_CHECKOUT: dict[str, Callable[[Path, dict[str, str]], tuple[Path, dict[str, str]]]] = {
    "no-git-on-path": _no_git_on_path,
    "not-a-repository": _not_a_repository,
    "inside-another-repository": _inside_another_repository,
    "worktree-of-a-bare-repository": _worktree_of_a_bare_repository,
}


@pytest.mark.parametrize("layout", sorted(_NO_MAIN_CHECKOUT))
def test_without_a_main_checkout_bayesmith_is_the_project_roots_sibling(tmp_path, layout):
    """Where git cannot name a main working tree, the old answer stands.

    Each layout with git in it is built so that trusting git's output there
    would land somewhere other than `project.parent`: the worktree's main
    checkout, the outer repository's root, or the parent of `rheplicant.git`.
    """
    root = tmp_path.resolve()
    project, env = _NO_MAIN_CHECKOUT[layout](root, _git_environment(root))
    assert locate_bayesmith(project, env).path == project.parent / "bayesmith"


def test_the_skip_names_the_manifest_it_looked_for(tmp_path):
    checkout = BayesmithCheckout(tmp_path / "bayesmith", "named by this test")
    with pytest.raises(pytest.skip.Exception) as skipped:
        verified_release(checkout)
    message = str(skipped.value)
    assert os.fspath(checkout.manifest) in message
    assert "named by this test" in message
    assert BAYESMITH_VARIABLE in message
    assert "this machine does not have" not in message


def test_a_release_that_disagrees_with_its_manifest_fails_rather_than_skips(tmp_path):
    checkout = BayesmithCheckout(tmp_path / "bayesmith", "named by this test")
    checkout.release.mkdir(parents=True)
    wheel = checkout.release / "bayesmith-0.10.0-py3-none-any.whl"
    wheel.write_bytes(b"the recorded build")
    record = {"sha256": hashlib.sha256(b"the recorded build").hexdigest()}
    checkout.manifest.write_text(json.dumps({"artifacts": {wheel.name: record}}))
    assert verified_release(checkout) == checkout.release

    wheel.write_bytes(b"some other build")
    with pytest.raises(AssertionError, match="does not match the release manifest"):
        verified_release(checkout)
