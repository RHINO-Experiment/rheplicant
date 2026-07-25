# Releasing

`rheplicant` publishes to PyPI via **Trusted Publishing** (OIDC) from
`.github/workflows/publish.yml`. No API tokens are stored anywhere: GitHub
mints a short-lived identity that PyPI verifies against a trusted-publisher
config you set once.

## One-time PyPI setup

Before the first release, register the trusted publisher at
<https://pypi.org/manage/account/publishing/> (the "pending publisher" form,
since the project does not exist on PyPI yet):

| Field | Value |
|---|---|
| PyPI Project Name | `rheplicant` |
| Owner | `zzhang0123` |
| Repository name | `rheplicant` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

The `pypi` GitHub Environment is created automatically on the first run; add
protection rules to it under **Settings → Environments** if you want a manual
approval gate before each publish.

## Cutting a release

1. Bump `version` in `pyproject.toml`, update `CHANGELOG.md`, commit.
2. Tag it: `git tag -a vX.Y.Z -m "rheplicant X.Y.Z" && git push origin vX.Y.Z`.
3. Publish a **GitHub Release** for that tag (Releases → Draft a new release →
   choose the tag → Publish). This triggers `publish.yml`, which builds the
   sdist + wheel, runs `twine check`, and uploads to PyPI over OIDC.

You can also run the workflow manually from **Actions → Publish to PyPI → Run
workflow** (it builds and publishes whatever is on the default branch).

Each PyPI version is immutable — a version number can be uploaded only once.
