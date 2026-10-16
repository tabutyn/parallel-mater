# Parallel Mater agent coordination

For work selected from `docs/ROADMAP.md`, read `docs/roadmap/WORKFLOW.md` first.
Task cards in `docs/roadmap/tasks/` are canonical; the roadmap index is generated.

- Fetch current main and choose one ready leaf task matching this machine.
- Push a claim-only PR updating that card and the roadmap. Do not implement the
  task until the claim merges and current main contains your exact lease token.
- Use the reserved work branch. Respect dependencies, shared-area locks and the
  24-hour UTC lease. Renew through a separate lease-only PR before expiry.
- Stop if your token changes or the lease expires; preserve unfinished work.
  Expired reservations require an explicit reclaim/release PR, not silent reuse.
- Complete through an implementation PR with the matching token and reviewable
  evidence. Passing execution does not establish physical quality. Never relax
  tolerances or rewrite goldens merely to close a task.
- Update cards with `tools/roadmap.py`, regenerate the index and run its checks.
  Freeze scope/locks during a lease; release and replan if the task must expand.
- Coordination-only task definitions/workflow maintenance use planning PRs.
  This exception does not authorize implementation without a merged lease.
- Keep HTTPS workstation remotes. Never stage/discard another worktree's changes.
  Follow explicit user instructions about publishing; local lease edits do not
  by themselves authorize network actions or establish ownership.

The initial lease-workflow bootstrap has no pre-existing task to claim. Once
merged, all roadmap implementation follows the lease protocol above.
