# Continuous integration

`.github/workflows/roadmap-leases.yml` is the single automatic **CI** entrypoint.
The historical filename is retained; it now routes the platform workflows as
well as enforcing roadmap ownership. Other workflows are reusable and may also
be dispatched manually for focused investigations.

## What runs

| Change/event | Checks |
|---|---|
| Task claim, renewal, release, completion metadata; roadmap/design documents and model tests | Roadmap + CI contract tests, then the stable result gate: two short Ubuntu jobs |
| Known root prose, ordinary `docs/*.md` outside the generated reference, roadmap tooling/history | Same quick checks |
| `docs/reference/**`, `tools/reference/**`, reference-artifact unit test | Quick checks + complete reference-artifact reproduction |
| Shared source, shaders, API/registry, build inputs, examples, platform code/tests, CI files, any unknown path | Quick checks + every existing platform suite |
| Mixed PR | Union of all changed paths; documentation never exempts accompanying code |
| Nightly (04:23 UTC) or manual **CI** dispatch | Full suite regardless of changed files |

Full verification retains shader compilation/reflection/host tests on Linux,
Windows and macOS; all three installed-SDK checks; mobile iPad/Android builds and
Android UI emulator; three-OS frame timing; Windows loader contracts; and complete
reference regeneration. This is build/test coverage, not a new line-coverage
measurement or a substitute for the physical-device F8 qualification matrix.

The initial policy intentionally optimizes coordination/prose traffic, not every
individual C++ dependency. Core/platform code still gets comprehensive checks.
Add narrower platform lanes only after auditing their shared inputs and adding
routing regression tests. In particular, don't exempt all of `docs/`: generated
API headers, specifications and reference artifacts require real checks.

## Trigger, diff and cancellation policy

- PR `opened`, `synchronize` and `reopened` events run against the complete
  base/head merge-base diff. Branch pushes do **not** start duplicate shader jobs.
- Main pushes use `before..after`; merge queues use their group base/head range.
  Main still verifies integration of code changes. Metadata-only main pushes
  use the quick path too. Closing a PR alone does not start another run.
- Diff collection uses checked-out full Git history and NUL-separated paths,
  not a truncated changed-files API or native workflow path filters. Renames
  include both old and new paths. Missing objects, malformed events, zero/unknown
  base revisions and other unavailable diffs select the full suite, never none.
- A new revision cancels an older run of the **same PR**. Non-PR runs have unique
  concurrency groups: a later metadata-only main push must not cancel an older
  code-change build and leave its platform checks unexecuted.
- Cancellation affects future runs using this configuration, not existing runs
  queued under old workflow definitions. Clear that backlog only after checking
  its revisions and whether the corresponding PR still needs the evidence.

No secrets are passed to the reusable workflows. Tokens retain read-only contents
permission. PR routing never uses `pull_request_target` or executes code with a
privileged write token. Workflow/policy changes themselves select all suites.

## Stable merge checks

Require **CI required** and **Roadmap lease contract**, with up-to-date branches
or the existing serialized merge procedure. This change does not alter GitHub
administrator settings. Do not require the individual conditional platform jobs:
they may correctly be absent/skipped for a documentation-only PR. Previous check
names (such as `Shader portability required`) are now nested in reusable suites;
an administrator must migrate any external rules referring to those names.

The entrypoint has no `paths`/`paths-ignore` filter. GitHub warns that skipping
an entire required workflow with path filters can leave its checks pending;
job-level selection keeps the stable aggregate check available on every PR.
See [GitHub's trigger documentation](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow).

The gate requires routing and contract tests to succeed, every selected suite to
succeed, and every unselected suite to be skipped. Failed/cancelled routing,
missing/malformed outputs, selected-but-skipped jobs, and failed/cancelled suites
cannot become a green gate. The run summary reports selected/not-applicable
suites; not-applicable is **not** successful platform execution evidence.

CI is infrastructure maintenance and may use a coordination-only planning PR.
That exception covers workflows, `tools/ci/`, `tests/ci/` and this policy document;
it does not authorize bundled runtime edits or relax claim-only PR boundaries.

## Local verification

```bash
python3 -m pip install -r tools/ci/requirements.txt
python3 -m unittest discover -s tests/ci -p 'test_*.py'
python3 -m unittest discover -s tests/roadmap -p 'test_*.py'
python3 tools/roadmap.py check --base origin/main
```

CI tests exercise path classification, real Git diffs (including 3,102 changed
paths and unusual filenames), PR/push/merge-group ranges, fail-closed selection,
result-gate failures, and the actual workflow graph/permissions/concurrency.
