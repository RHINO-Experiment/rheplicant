"""Which git commit the software audit row may attribute to rheplicant.

The row used to ask git from the installed distribution's location. For a
wheel that location is ``site-packages``, so a wheel installed into a virtual
environment inside any git work tree recorded that repository's commit,
dirty flag and diff hash as rheplicant's own. A commit is now recorded only
for an editable install whose checkout contains the code actually running.

Each test builds a throwaway git repository and a ``*.dist-info`` directory
by hand and presents them through ``importlib.metadata``; nothing is
installed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

import _rheplicant_bootstrap.audit.software as software

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

GIT_FACTS = ("git_commit", "dirty", "tracked_diff_sha256")


def _git(cwd: Path, *arguments: str) -> str:
    completed = subprocess.run(
        (
            "git",
            "-c",
            "user.name=audit test",
            "-c",
            "user.email=audit@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *arguments,
        ),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    package = root / "src" / "rheplicant"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "initial")
    return root


def _dist_info(site_packages: Path, direct_url: object) -> metadata.Distribution:
    info = site_packages / "rheplicant-0.0.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: rheplicant\nVersion: 0.0.0\n",
        encoding="utf-8",
    )
    if direct_url is not None:
        (info / "direct_url.json").write_text(json.dumps(direct_url), encoding="utf-8")
    return metadata.PathDistribution(info)


def _present(monkeypatch, distribution: metadata.Distribution, package: Path) -> None:
    def distribution_named(name: str) -> metadata.Distribution:
        return distribution if name == "rheplicant" else metadata.distribution(name)

    monkeypatch.setattr(
        software,
        "metadata",
        SimpleNamespace(
            distribution=distribution_named,
            version=metadata.version,
            PackageNotFoundError=metadata.PackageNotFoundError,
        ),
    )
    monkeypatch.setattr(software, "_package_directory", lambda: package)


def _assert_no_git_facts(facts) -> None:
    for name in GIT_FACTS:
        assert (facts[name], facts[f"{name}_reason"]) == (None, "not_a_git_checkout"), name


@pytest.mark.parametrize(
    "direct_url",
    [
        None,
        {"url": "file:///wheels/rheplicant-0.0.0-py3-none-any.whl", "archive_info": {}},
        {"url": "REPOSITORY", "dir_info": {}},
        {"url": "REPOSITORY", "dir_info": {"editable": False}},
        {"url": "REPOSITORY", "dir_info": {"editable": "true"}},
    ],
    ids=["index", "wheel-file", "local-dir", "editable-false", "editable-string"],
)
def test_a_non_editable_install_inside_a_work_tree_records_no_commit(
    monkeypatch, repository, direct_url
) -> None:
    if direct_url is not None and direct_url["url"] == "REPOSITORY":
        direct_url = {**direct_url, "url": repository.as_uri()}
    site_packages = repository / ".venv" / "lib" / "python3.12" / "site-packages"
    distribution = _dist_info(site_packages, direct_url)
    package = site_packages / "rheplicant"
    package.mkdir()
    _present(monkeypatch, distribution, package)

    facts = software._project_facts()

    _assert_no_git_facts(facts)
    assert facts["source_root"] == str(site_packages.resolve())
    software.validate_software({**software.collect_software(), "rheplicant": facts})


@pytest.mark.parametrize("authority", ["", "localhost"], ids=["file-uri", "file-localhost"])
def test_an_editable_install_records_the_commit_of_its_checkout(
    monkeypatch, tmp_path, repository, authority
) -> None:
    distribution = _dist_info(
        tmp_path / "venv" / "site-packages",
        {"url": f"file://{authority}{repository.as_posix()}", "dir_info": {"editable": True}},
    )
    _present(monkeypatch, distribution, repository / "src" / "rheplicant")

    assert software._git_root() == (repository.resolve(), None)
    facts = software._project_facts()

    assert facts["git_commit"] == _git(repository, "rev-parse", "HEAD")
    assert facts["git_commit_reason"] is None
    assert (facts["dirty"], facts["dirty_reason"]) == (False, None)
    assert facts["tracked_diff_sha256"] == hashlib.sha256(b"").hexdigest()

    (repository / "src" / "rheplicant" / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    assert software._project_facts()["dirty"] is True


def test_an_editable_install_running_code_from_elsewhere_records_no_commit(
    monkeypatch, tmp_path, repository
) -> None:
    """The metadata names one checkout while the imported package lives in
    another directory, as when PYTHONPATH points at a second work tree. The
    first checkout's commit does not describe the code that is running."""
    distribution = _dist_info(
        tmp_path / "venv" / "site-packages",
        {"url": repository.as_uri(), "dir_info": {"editable": True}},
    )
    elsewhere = tmp_path / "elsewhere" / "src" / "rheplicant"
    elsewhere.mkdir(parents=True)
    _present(monkeypatch, distribution, elsewhere)

    assert software._git_root() == (None, "not_a_git_checkout")
    _assert_no_git_facts(software._project_facts())


@pytest.mark.parametrize(
    "direct_url",
    [
        {"url": "https://example.invalid/rheplicant", "dir_info": {"editable": True}},
        {"dir_info": {"editable": True}},
        ["not", "a", "mapping"],
    ],
    ids=["not-a-file-url", "no-url", "not-a-mapping"],
)
def test_editable_metadata_without_a_local_checkout_records_no_commit(
    monkeypatch, tmp_path, repository, direct_url
) -> None:
    distribution = _dist_info(repository / ".venv" / "site-packages", direct_url)
    _present(monkeypatch, distribution, repository / "src" / "rheplicant")

    _assert_no_git_facts(software._project_facts())


def test_an_editable_checkout_on_another_host_records_no_commit(
    monkeypatch, tmp_path, repository
) -> None:
    """``file://fileserver/path`` names a path on another machine. Read as a
    local path it would be this repository, which is not what it names."""
    distribution = _dist_info(
        tmp_path / "venv" / "site-packages",
        {
            "url": f"file://fileserver{repository.as_posix()}",
            "dir_info": {"editable": True},
        },
    )
    _present(monkeypatch, distribution, repository / "src" / "rheplicant")

    _assert_no_git_facts(software._project_facts())


def test_the_package_directory_is_where_rheplicant_is_imported_from() -> None:
    """The one test that runs ``_package_directory`` itself; the others
    replace it to point into their throwaway repository."""
    import rheplicant

    assert software._package_directory() == Path(rheplicant.__file__).parent
