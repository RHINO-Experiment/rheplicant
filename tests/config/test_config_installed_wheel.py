from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from tests.config.test_config_cli import document
from tests.config.test_config_document import synthetic_document

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UV = "uv"
PRESET = PROJECT_ROOT / "src/rheplicant/config/presets/rhino_v1.yaml"
SCHEMAS = (
    "provenance-v1.schema.json",
    "diagnostics-v1.schema.json",
    "products-v1.schema.json",
)


def _run(arguments, *, cwd=PROJECT_ROOT, env=None):
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"command failed: {arguments!r}\n{completed.stdout}\n{completed.stderr}"
    )
    return completed


def _single(directory: Path, suffix: str) -> Path:
    rows = tuple(path for path in directory.iterdir() if path.name.endswith(suffix))
    assert len(rows) == 1, rows
    return rows[0]


@pytest.fixture(scope="session")
def built_distributions(tmp_path_factory):
    root = tmp_path_factory.mktemp("config-distributions")
    direct = root / "direct"
    sdist = root / "sdist"
    derived = root / "from-sdist"
    _run([UV, "build", "--wheel", "--out-dir", str(direct), "--clear"])
    _run([UV, "build", "--sdist", "--out-dir", str(sdist), "--clear"])
    archive = _single(sdist, ".tar.gz")
    _run(
        [
            UV,
            "build",
            str(archive),
            "--wheel",
            "--out-dir",
            str(derived),
            "--clear",
        ]
    )
    return {
        "direct-wheel": _single(direct, ".whl"),
        "sdist-wheel": _single(derived, ".whl"),
        "root": root,
    }


@dataclass(frozen=True)
class Install:
    python: Path
    command: Path
    gui_command: Path
    cwd: Path
    env: dict[str, str]

    def run(self, arguments, *, input=None):
        return subprocess.run(
            [os.fspath(self.command), *arguments],
            cwd=self.cwd,
            env=self.env,
            input=input,
            check=False,
            capture_output=True,
            text=True,
        )

    def python_run(self, program: str):
        return _run(
            [os.fspath(self.python), "-c", program],
            cwd=self.cwd,
            env=self.env,
        )


@pytest.fixture
def fresh_install(tmp_path):
    counter = 0

    def install(
        source: Path,
        *,
        editable: bool = False,
        extras: tuple[str, ...] = (),
    ) -> Install:
        nonlocal counter
        counter += 1
        root = tmp_path / f"install-{counter}"
        venv = root / "venv"
        cwd = root / "cwd"
        cwd.mkdir(parents=True)
        _run([UV, "venv", "--clear", str(venv)])
        arguments = [
            UV,
            "pip",
            "install",
            "--python",
            os.fspath(venv / "bin/python"),
        ]
        if editable:
            arguments.append("--editable")
        requirement = os.fspath(source)
        if extras:
            requirement += f"[{','.join(extras)}]"
        arguments.append(requirement)
        _run(arguments)
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        return Install(
            venv / "bin/python",
            venv / "bin/rheplicant",
            venv / "bin/rheplicant-gui",
            cwd,
            env,
        )

    return install


def _resource_probe(install: Install) -> dict[str, object]:
    program = """
import base64
import importlib.util
import json
from pathlib import Path
from _rheplicant_bootstrap.presets import read_installed_preset

snapshot = read_installed_preset("rhino_v1")
spec = importlib.util.find_spec("rheplicant")
root = Path(tuple(spec.submodule_search_locations)[0])
schemas = {}
for name in (
    "provenance-v1.schema.json",
    "diagnostics-v1.schema.json",
    "products-v1.schema.json",
):
    schemas[name] = base64.b64encode(
        (root / "config" / "schemas" / name).read_bytes()
    ).decode("ascii")
print(json.dumps({
    "preset": base64.b64encode(snapshot.input_bytes).decode("ascii"),
    "sha256": snapshot.sha256,
    "schemas": schemas,
}))
"""
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
    assert (
        direct_target / "runs/n-666f7277617264/arrays.npz"
    ).read_bytes() == generated_arrays
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
        assert "rheplicant/config/schemas/provenance-v1.schema.json" in names
        assert "rheplicant/config/schemas/diagnostics-v1.schema.json" in names
        assert "rheplicant/config/schemas/products-v1.schema.json" in names
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
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    process = subprocess.Popen(
        [
            os.fspath(install.gui_command),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=install.cwd,
        env=install.env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                with urllib.request.urlopen(base_url + "/", timeout=1) as response:
                    markup = response.read().decode("utf-8")
                break
            except (OSError, urllib.error.URLError):
                if process.poll() is not None or time.monotonic() >= deadline:
                    stdout, stderr = process.communicate(timeout=5)
                    pytest.fail(f"GUI did not start.\nstdout:\n{stdout}\nstderr:\n{stderr}")
                time.sleep(0.1)

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
    finally:
        process.terminate()
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)


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
        input=yaml.safe_dump(worker_document, sort_keys=False).encode(
            "utf-8", "strict"
        ),
        cwd=install.cwd,
        env=install.env,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8", "replace"
    )
    prefix = b"\x1eRHEPLICANT_GUI_JOB "
    encoded = completed.stdout.rsplit(prefix, 1)[1].split(b"\n", 1)[0]
    frame = json.loads(encoded.decode("utf-8", "strict"))
    assert frame == {
        "status": "ok",
        "result": {"findings": [], "layers": 2},
    }


def test_wheel_and_editable_preset_discovery_are_byte_identical(
    fresh_install, built_distributions
):
    wheel = fresh_install(built_distributions["direct-wheel"])
    editable = fresh_install(PROJECT_ROOT, editable=True)
    wheel_row = _resource_probe(wheel)
    editable_row = _resource_probe(editable)
    assert (wheel_row["preset"], wheel_row["sha256"]) == (
        editable_row["preset"],
        editable_row["sha256"],
    )
