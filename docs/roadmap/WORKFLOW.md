# Roadmap leases and pull requests

`docs/roadmap/tasks/<ID>.json` is the canonical task record. The generated index
in `docs/ROADMAP.md` is the human work queue. `main` is the ownership record;
local edits, pushed branches and open PRs are only proposals until merged.
This is cooperative Git coordination, not an identity/authentication service.

## Pick one bounded task

1. Fetch current `main`; inspect open lease PRs to avoid duplicate proposals.
2. Run `python3 tools/roadmap.py list --ready` and read the selected task card.
3. Check its machine requirements, acceptance checks, dependencies and locks.
   F8 is the next architecture/performance priority; honor existing leases and
   finish outstanding foundation reviews. Independent ready tasks may run concurrently.
4. Reserve one leaf ID per claim PR. Do not claim an entire F-series area. If a
   leaf is too large for one focused implementation/review, split it in a
   planning PR before claiming. Keep old IDs and history; never renumber/reuse.

A task with unfinished dependencies is waiting, even if its stored status is
`open`. Shared-area locks are exclusive across leased tasks, including expired
ones awaiting release/reclaim. Locks identify common implementation areas;
they are not a complete file-access permission system. The reviewer must also
check overlapping files and semantic changes. Routine edits to shared build
files still need merge review even when the tasks have different locks.

## Initial push: claim-only PR

Start from current `origin/main` in a clean branch/worktree. Example identity:
agent `codex-linux-01`, machine `linux-laptop`. Use a stable, human-identifiable
agent/session name and a real machine label; never use credentials as identifiers.

```bash
git fetch origin
git switch -c lease/F1-01/linux-01 origin/main
python3 tools/roadmap.py claim F1-01 \
  --owner codex-linux-01 --machine linux-laptop \
  --branch work/F1-01/linux-01 --hours 24
python3 tools/roadmap.py check --base origin/main
git add docs/roadmap/tasks/F1-01.json docs/ROADMAP.md
git commit -m "chore(roadmap): lease F1-01"
git push -u origin lease/F1-01/linux-01
```

Open a PR targeting `main`, titled `lease(F1-01): geometry allocation plan`.
Include owner, machine, intended work branch, exact scope, dependencies/locks
and acceptance checks. Use a Markdown body file with `gh pr create --body-file`
or the GitHub UI. The initial push changes only the task card and generated
roadmap. Do not include implementation or another task reservation.

**Wait until the claim PR is merged.** Fetch again and verify that the task on
`origin/main` contains your exact lease token, owner, machine and work branch.
If a competing claim merged first, abandon your claim and choose another ready
task. A green/open PR does not establish ownership.

```bash
git fetch origin
git show origin/main:docs/roadmap/tasks/F1-01.json
git switch -c work/F1-01/linux-01 origin/main
```

The UUID token distinguishes this reservation from every previous reservation,
even when the owner and machine names are reused. Preserve it for maintenance
and completion commands. Task scope, dependencies and locks are frozen while
leased. If implementation needs a wider contract/shared area, release the task,
merge a planning change, then claim the revised task with a new token.

## Work, renew, release and reclaim

Keep the implementation branch synchronized with current `main`. Recheck the
canonical token before beginning a session, after pulling, and before pushing a
completion. Stop task work if the lease expired or its token changed. Preserve
unfinished code on its branch; never overwrite the new owner's task metadata.

A lease lasts at most 24 hours from its UTC heartbeat. Renew before expiry if
work/review will run longer. The helper edits files only: it never fetches,
creates PRs, pushes, merges, starts a background heartbeat or silently acquires
ownership. Renewal therefore requires another small PR merged before expiry.

Use a separate lease-maintenance branch from current `main`, so unfinished
implementation does not enter the renewal/release PR:

```bash
python3 tools/roadmap.py renew F1-01 --token <current-token> --hours 24
python3 tools/roadmap.py release F1-01 --token <current-token> \
  --reason 'Paused; unfinished code retained on work/F1-01/linux-01'
```

Run **one** operation, check against current main, commit the card/index, push
and merge its PR. An active owner may release voluntarily. Other agents must
not release another owner's live lease; obtain an explicit owner handoff first.
For a crashed/unreachable machine, wait for expiry and use explicit reclaim:

```bash
python3 tools/roadmap.py reclaim F1-01 \
  --owner codex-mac-02 --machine mac-studio \
  --branch work/F1-01/mac-02 --hours 24 \
  --reason 'Prior lease expired; inspected retained branch and adopting its scope'
```

Reclaim appends the previous lease to history and generates a new token. It
cannot take a live lease. Expiry alone does not release shared locks or mark a
task open. A reclaim/release PR must merge before any new work begins. An expired
owner uses the same reclaim process, with a fresh token. A blocked owner should
record the blocker and retained branch in the PR, then release unless actively
resolving it; do not renew indefinitely without progress.

## Completion PR

Finish the card's acceptance checks, retain reviewable evidence and update
capability/qualification documentation as needed. Fetch/merge current main into
the work branch **before** recording completion. If a renewal changed the card,
retain the current main lease/history and rerun completion against that record.
Never resolve a metadata conflict by restoring an older token or heartbeat.

```bash
python3 tools/roadmap.py complete F1-01 --token <current-token> \
  --reason 'Shared geometry allocation plan and exact-capacity recovery implemented' \
  --evidence 'PASS: <exact command>; <driver/device/toolchain>; <tracked report or artifact URL>' \
  --evidence 'PASS: <negative/recovery check>; <reviewable reference>'
python3 tools/roadmap.py check --base origin/main --branch work/F1-01/linux-01
```

Replace placeholders with actual evidence. A result only on a machine's ignored
local disk is insufficient for another machine to review; attach the artifact
or commit a scoped report. Include limitations and unsupported cases. Execution,
reference regression, physical quality and performance remain separate claims.
A failed required acceptance check leaves the task open/leased, not done.

Commit implementation, tests, the completed card and generated roadmap to the
reserved work branch; push and open/update its completion PR. Include task ID,
lease token, claim PR, scope, test commands/results and artifact links in the PR
body. The metadata proposes `done`; canonical completion and lock release happen
only when the completion PR merges into `main`. CI verifies its token/history
against main. Partial implementation PRs may remain drafts with a failing lease
check until they carry a valid completion; do not merge them as finished work.

A task whose purpose is investigation or planning may complete with a reviewed
reproduction/specification plus new, bounded follow-up cards. It does not close
the larger physics gate. Foundation `*-99` reviews must add any discovered gaps
as dependencies and separate tasks before claiming/completing the review; all
advertised modes still require their own evidence.

## States and planning

| Stored state | Meaning |
|---|---|
| `open` | Unowned; ready only when dependencies are done and locks are free |
| `leased` | Merged reservation with token, owner, machine, branch and UTC deadline |
| `done` | Accepted completion/evidence on main; no current lease |
| `deferred` | Deliberately inactive; a planning PR must activate it before claiming |

History is append-only and records retired lease, outcome, time, reason and
completion evidence. Completed IDs stay completed; regressions get new task IDs.
Dependencies form an acyclic graph. A new task must land open/deferred before
being claimed. Planning PRs may change unleased scope, split work, add tasks or
activate deferred tasks. They may not silently rewrite a live owner's scope,
reuse a completed ID or mark unleased work done.

Bootstrap creates historical completed `*-00` cards with their existing commit
evidence. That exception applies only when introducing the first catalog.
Previously started work is not retroactively leased. Preserve its branch,
identify or add the matching bounded task through planning, merge its claim,
then update the implementation branch to that lease before completion. Do not
silently discard or take over another machine's unfinished changes.

Coordination-only maintenance uses a planning PR; it is not a bypass for physics,
platform, runtime or SDK implementation. Existing historical roadmap snapshots
are context only and must never become competing task queues.

Entirely completed categories are collapsed automatically in the generated index.
Completed F5 is archived out of the live index. Its cards and history stay
canonical, and a new open/deferred F5 follow-up makes the category visible again;
never delete completed cards to clean the work queue.
For F8, follow [the 60 Hz plan](F8-REALTIME-60HZ.md), including the frozen candidate
revision, complete kernel/device matrix and separate legacy/new quality profiles.

## Merge serialization and checks

`CI / Roadmap lease contract` validates task structure, dependency
cycles, exclusive locks, generated-index consistency and PR state transitions.
It also tests competing claims, expiry/reclaim and stale completions. It reads
current fetched main, rejects outdated branches, and requires implementation
PRs to complete one previously merged lease. Claim/renew/release/reclaim PRs may
change only their one card and the generated index. No network or physics build
is needed for the local lease checks:

```bash
python3 tools/roadmap.py check
python3 -m unittest discover -s tests/roadmap -p 'test_*.py'
```

CI routing is documented in [the CI policy](../continuous-integration.md).
Coordination/prose-only PRs run the quick contract checks and stable result gate;
shared code, CI configuration and unknown paths retain full platform verification.
Workflow/CI-policy maintenance is coordination-only; runtime changes still need
a merged task lease. Claim/renew/release PRs still change only their card/index.

For automated multi-machine merging, make `CI required` and the lease check
required status checks and require up-to-date branches. Each merge candidate
permits one lease operation; merge-queue groups containing multiple lease
operations are rejected.
Use one-at-a-time merges for lease-affecting PRs. This repository
change does not configure those administrator settings. Until they are enabled,
one merge coordinator must fetch current main, update the PR, rerun the check
and merge one lease-affecting PR at a time. A check against an old base is not
an atomic reservation. Do not force-push main or bypass this serialization.

The check validates cooperative records, not physical truth or ownership of a
GitHub identity. Review still verifies that the PR author represents the recorded
agent, respects scope, supplies adequate evidence and may release the lease.
