---
name: release
description: Cut a sector-ai release through the GitHub Actions pipeline (prepare, draft, publish).
argument-hint: "[patch|minor|major|estimate|X.Y.Z]"
disable-model-invocation: true
---

Release `$ARGUMENTS`. If no version or bump type is given, infer it from
`estimate` and ask before dispatching.

Three workflows own the release. Do **not** hand-edit `pyproject.toml`, do
**not** run `git tag`, and do **not** publish without the user explicitly
saying so. The full runbook and recovery table are in `docs/RELEASES.md`.

| Stage | Workflow | Trigger | Result |
| --- | --- | --- | --- |
| Prepare | `release-prepare.yml` | dispatch with `version` | `release/vX.Y.Z` branch + PR |
| Draft | `release-draft.yml` | release PR merged | draft release, no tag |
| Publish | `release-publish.yml` | draft published | PyPI + assets + install check |

## Preconditions

- On `master`, clean tree: `git switch master && git pull --ff-only && git status --short`.
- `gh auth status` succeeds and the token may dispatch workflows.
- Stop if `vX.Y.Z` or `release/vX.Y.Z` already exists; prepare refuses to reuse
  either, so pick another version or delete the stale ref first.

## 1. Pick the version

```bash
python scripts/version.py get                  # current version
git describe --tags --abbrev=0                 # last released tag
git log --oneline <last-tag>..HEAD             # changes since then
gh pr list --state merged --limit 20           # labels help classify them
```

- **major** — breaking change to config, CLI or public behaviour.
- **minor** — new feature.
- **patch** — fixes, docs, dependencies only.
- `estimate` means infer the bump and present `old -> new` with the reason;
  wait for confirmation before step 2.

## 2. Prepare

```bash
gh workflow run release-prepare.yml -f version=X.Y.Z
gh run list --workflow release-prepare.yml --limit 1
gh run watch <run-id> --exit-status
```

Use `-f dry_run=true` to bump and test without pushing. The workflow refuses
a version that does not sort above the current one, and runs ruff, format and
`pytest --cov --cov-fail-under=95` on the bumped tree.

- Failed before pushing: re-dispatch, nothing was created.
- Failed after pushing: delete `release/vX.Y.Z` first, then re-dispatch.

## 3. Merge the release PR

```bash
gh pr list --label release --state open
gh pr diff <number>            # expect exactly one changed line: pyproject.toml
gh pr merge <number> --merge   # merge commit; never squash this one
```

Merging is the first human decision. The merge commit is pinned as the release
commit, so later pushes to `master` cannot change what ships. Branch protection
is deliberately off: the PR is opened with `GITHUB_TOKEN`, so pushes from it do
not trigger `ci.yml`; the draft stage re-runs the suite instead.

## 4. Review the draft

`release-draft.yml` fires on the merge, re-runs the suite at the merge commit,
deletes `release/vX.Y.Z`, and creates a **draft**. A draft creates no tag, so
deleting a bad one leaves nothing public.

```bash
gh run list --workflow release-draft.yml --limit 1
gh release view vX.Y.Z
```

Notes come from `.github/release.yml` categories and exclude the release PR.
Edit the title or notes if you want (`gh release edit vX.Y.Z --title ...`).

- Draft deleted after the PR merged: dispatch `release-draft.yml` with
  `version` and `target_ref` set to the merge commit.

## 5. Publish — ask first

This is the irreversible ship decision: PyPI refuses a version it already has,
and the Docker tags start moving. Do not run this until the user says publish.

```bash
gh release edit vX.Y.Z --draft=false
gh run list --workflow release-publish.yml --limit 1
gh run watch <run-id> --exit-status
```

Publishing triggers `release-publish.yml` (build, `twine check`, trusted
publishing to PyPI, clean-venv install check) and `docker-image.yml`
(`X.Y.Z`, `X.Y`, `latest`). Confirm the PyPI page and
`ghcr.io/roryeckel/sector-ai:X.Y.Z` before reporting success.

## Recovery

Use the table in `docs/RELEASES.md`. The important rules:

- A failed publish is re-runnable; the PyPI upload is idempotent.
- Never delete a wrong PyPI version — yank it and cut a patch release.
- A failed Docker build after PyPI succeeds is a separate re-run, not a new
  release.
