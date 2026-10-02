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

1. Bump `version` in `pyproject.toml`, the single source of truth. The
   package's `__version__` is read back from the installed distribution
   metadata (`importlib.metadata.version`), so never hardcode a version string
   anywhere else in the source. Update `CHANGELOG.md` and the README test
   count, then commit.
2. Read `README.md` as the page PyPI will show for this version: it is the
   long description (`readme = "README.md"`), and it cannot be edited after
   the upload. The install section says which version PyPI serves; update
   that sentence in the commit that is tagged.
3. Tag the last commit, after every documentation change:
   `git tag -a vX.Y.Z -m "rheplicant X.Y.Z"`.
4. Push the branch, then the tag: `git push origin main`, then
   `git push origin vX.Y.Z`. Read the Docs builds `latest` from main.
5. Publish a **GitHub Release** for that tag (Releases → Draft a new release →
   choose the tag → Publish). This triggers `publish.yml`, which builds the
   sdist and wheel, installs the wheel into a fresh environment, checks that
   the bootstrap imports without JAX and that `rheplicant validate` accepts a
   document, runs `twine check`, and uploads to PyPI over OIDC.
6. Confirm `pip install rheplicant==X.Y.Z` in a fresh environment.

You can also run the workflow manually from **Actions → Publish to PyPI → Run
workflow** (it builds and publishes whatever is on the default branch).

A PyPI version number can be uploaded only once. A tag pushed without a
Release uploads nothing.

**Do not publish a Release for `v0.9.0`.** That tag fails on a vector latent
under `kind: nuts` (see the 0.9.1 changelog entry). It is pushed so that the
history is complete; 0.9.1 is the first 0.9 version to upload.
