# Releasing

`rheplicant` publishes to PyPI via **Trusted Publishing** (OIDC) from
`.github/workflows/publish.yml`. No API tokens are stored anywhere: GitHub
mints a short-lived identity that PyPI verifies against a trusted-publisher
config you set once.

## PyPI trusted-publisher setup

Publishing is authorized by a PyPI **trusted publisher** matched against the
workflow's OIDC identity. Manage it under the project's publishing settings
(<https://pypi.org/manage/project/rheplicant/settings/publishing/>):

| Field | Value |
|---|---|
| PyPI Project Name | `rheplicant` |
| Owner | `RHINO-Experiment` |
| Repository name | `rheplicant` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

**If the GitHub repo is transferred or renamed, add a new trusted publisher
for the new `owner/repo` before the next release.** The OIDC identity carries
the *current* owner, so a stale publisher entry makes publishing fail even
though old GitHub URLs redirect. (This repo moved `zzhang0123` →
`RHINO-Experiment`; the `zzhang0123` publisher can be deleted once the
`RHINO-Experiment` one is added.)

The `pypi` GitHub Environment is created automatically on the first run; add
protection rules under **Settings → Environments** for a manual approval gate.

## Cutting a release

1. Bump `version` in `pyproject.toml` — the **single source of truth**. The
   package's `__version__` is read back from the installed distribution
   metadata (`importlib.metadata.version`), so never hardcode a version string
   anywhere else in the source. Update `CHANGELOG.md`, then commit.
2. Tag it: `git tag -a vX.Y.Z -m "rheplicant X.Y.Z" && git push origin vX.Y.Z`.
3. Publish a **GitHub Release** for that tag (Releases → Draft a new release →
   choose the tag → Publish). This triggers `publish.yml`, which builds the
   sdist + wheel, runs `twine check`, and uploads to PyPI over OIDC.

You can also run the workflow manually from **Actions → Publish to PyPI → Run
workflow** (it builds and publishes whatever is on the default branch).

Each PyPI version is immutable — a version number can be uploaded only once.
