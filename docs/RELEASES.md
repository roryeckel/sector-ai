# Releasing

Three stages, two human decisions: a bump PR, a draft release created on
merge, and a human publish click that ships to PyPI and GHCR.

A draft release creates **no tag**. Deleting a bad draft therefore leaves
nothing public, which is the reason the publish step is separate from the merge.

## Stages

| stage | workflow | trigger | outcome |
| --- | --- | --- | --- |
| 1. Prepare | `release-prepare.yml` | manual dispatch with a version | `release/vX.Y.Z` branch + PR labelled `release` |
| 2. Draft | `release-draft.yml` | release PR merged (or dispatch) | draft release at the merge commit, notes generated |
| 3. Publish | `release-publish.yml` | draft published (or dispatch with a tag) | PyPI + release assets, then a clean-venv install check |
| Image | `docker-image.yml` | release published, or push to `master` | `X.Y.Z`, `X.Y`, `latest`; `edge` on master |

## Cutting a release

1. **Check the target version.** It is a single literal in `pyproject.toml`;
   `python scripts/version.py get` prints it. Nothing else carries the version:
   the package reads it from installed metadata at runtime, and the README has
   no version numbers, so the bump touches one line.
2. **Dispatch `Release prepare`** with the version, no leading `v` (`1.4.0`,
   not `v1.4.0`). It refuses to continue if the tag or branch already exists, if
   the version does not sort above the current one, or if ruff or pytest fail.
   Use `dry_run` to rehearse without pushing.
3. **Review and merge the `Release vX.Y.Z` PR.** The merge commit is the release
   commit; the draft is pinned to it, so a later push to `master` cannot change
   what gets released.
4. **Review the draft release.** Its notes come from `.github/release.yml`
   categories and exclude the bump PR. Edit the notes if you want.
5. **Publish the draft.** That is the irreversible step: PyPI refuses a version
   it already has, and the Docker tags start pointing at this image.

## What gets verified, and where

- Prepare: semver, forward-only version, duplicate tag/branch, ruff, format,
  `pytest --cov --cov-fail-under=95`.
- Draft: re-runs the same suite at the merge commit, and asserts
  `pyproject.toml` matches the version in the branch name.
- Publish: asserts the tag matches `pyproject.toml`, runs `twine check`, then
  installs `sector-ai==X.Y.Z` from PyPI into a clean venv and requires
  `sector-ai --version` to print `sector-ai X.Y.Z`.
- Docker: pulls the pushed manifest (not the local build cache) and runs
  `--version` inside the image, so a broken `ENTRYPOINT` or a missing wheel
  fails the workflow.

Pull requests that touch `src/**`, `pyproject.toml` or the Dockerfile also build
the image and start it, including a check that a missing token produces the
readable message rather than a traceback.

## Recovery

| symptom | fix |
| --- | --- |
| Prepare failed before pushing | Re-dispatch. Nothing was created. |
| Prepare failed after pushing the branch | Delete `release/vX.Y.Z`, then re-dispatch. |
| Draft has the wrong notes | Edit the draft release. A draft is editable and creates no tag. |
| Draft deleted after the PR merged | Dispatch `Release draft` with the version and `target_ref` = the merge commit. |
| Publish failed at PyPI | Re-run the failed jobs. `skip-existing` makes the upload idempotent, so a partial run is safe. |
| Publish failed after PyPI succeeded | Re-run only the failed jobs; the upload is skipped and the assets and verification still run. |
| Docker build failed after PyPI | Re-run the failed Docker workflow, or dispatch `Docker image` with the tag. A dispatch restores `X.Y.Z` and `X.Y`; `latest` stays with the release it was published for. |
| Wrong version published | Do not delete it. Yank it on PyPI, mark the release as a pre-release or delete it, then cut a patch version. `npm` has `deprecate`; PyPI has `yank`. |

## One-time repository setup

These are settings, not files:

1. **PyPI trusted publisher.** On <https://pypi.org/manage/account/publishing/>
   add a pending publisher: project `sector-ai`, owner `roryeckel`, repository
   `sector-ai`, workflow `release-publish.yml`, environment `pypi`. The name was
   unclaimed when this was written, so the first publish creates the project.
2. **`pypi` environment.** The publish job targets it. No secrets are needed;
   OIDC supplies the credential. Add a required reviewer if you want a second
   pair of eyes on the ship click.
3. **GHCR visibility.** Packages created by `GITHUB_TOKEN` start private. After
   the first push, set `sector-ai` to public in the package settings, or nobody
   can pull it anonymously.
4. **Labels.** `breaking`, `dependencies`, `skip-changelog` for the changelog
   categories, and `release` for the bump PR (which also keeps it out of the
   notes). `bug`, `enhancement` and `documentation` already exist.
5. **Branch protection is deliberately not required.** The release PR is opened
   by `GITHUB_TOKEN`, and pushes made with it do not trigger `ci.yml`; branch
   protection that requires checks would block that PR. If you enable it later,
   either add a fine-grained personal access token that opens the PR, or
   dispatch `ci.yml` at the release branch. The draft stage re-runs the suite
   itself, so the missing CI run is covered either way.

### Why PyPI has an environment and the image workflow does not

`release-publish.yml` runs in the `pypi` environment because the environment
name is part of the OIDC claim PyPI verifies: it narrows *who may upload*, not
just whether a GitHub-side gate fired. `docker-image.yml` authenticates with
`GITHUB_TOKEN` and `packages: write`, and has no OIDC claim for an environment
to constrain. It also runs on every `master` push for `edge`, and
`environment:` cannot be made conditional per trigger, so a gate there would
pause ordinary pushes too. Revisit this if keyless image signing
(cosign/Sigstore) is added, where the environment name becomes part of the
signing identity.

## First release

If `v1.3.0` was never tagged, start at the version that contains these changes
(`1.4.0`) and publish it through all three stages. `v1.3.0` is not worth
reconstructing: nothing was ever published under it.
