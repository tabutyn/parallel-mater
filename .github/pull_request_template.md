## Work type

- [ ] Lease claim / renewal / release / reclaim (metadata only)
- [ ] Task completion (implementation and evidence)
- [ ] Planning / coordination maintenance (no implementation)

Task ID:
Lease token (if applicable):
Owner / machine:
Claim PR (for completion):
Reserved work branch:

## Scope and result

Describe the bounded task and resulting behavior. For a lease request, state
scope, dependencies, shared-area locks and UTC expiry. Ownership starts only
after the claim merges; see `docs/roadmap/WORKFLOW.md`.

## Evidence

List exact checks/results and reviewable artifact links. Separate execution,
reference comparison, physical quality and timing; list unsupported cases and
remaining failures. Planning PRs describe their catalog/protocol checks.

- [ ] Fetched current main; lease token/history and dependencies still match.
- [ ] Task card and generated roadmap agree (`python3 tools/roadmap.py check`).
- [ ] Changes respect the task scope and shared-area locks.
- [ ] Completion meets every acceptance check, or this remains an unfinished draft.
