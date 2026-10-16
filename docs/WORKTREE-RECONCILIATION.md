# Preserved workstation work — 2026-10-16

The active checkout is `parallel-mater-portability`. Two older dirty worktrees
remain intact. Verified snapshots in ignored
`out/worktree-backups/2026-10-16/` contain their branch/HEAD metadata, staged and
unstaged binary patches, complete changed/untracked files and SHA-256 hashes.
They were inspected, not staged, reset, merged wholesale or deleted.

| Older checkout | Work disposition |
|---|---|
| `parallel-mater`, HEAD `3143a53` | Linux deployment/launcher work overlaps the current deployment path. Old `World` buffer wrappers and AVBD generated-field suffix fixes target retired interfaces; do not revive them. Preserve remaining UI/branding changes until a focused comparison resolves their intent. |
| `parallel-mater-slang-kernels`, HEAD `deb158a` | Unique hinge/concave-contact work remains: closed-shell normals, swept contact feature recovery, breakable-hinge handling, substep momentum replay, debug capture and altered hinge assets/goldens. These routines/tests are not all present in current main. Preserve as a physics change with its reproduction; do not silently import a changed golden or assume the refactor superseded it. |

The second patch changes the hinge GLB from SHA-256
`db5447b00a33d29990e2f0d441d154c0509dee99689fc940261c1c92c2fe2a20`
to `80a5f22de8c7662397d7405eff6720232389e5163fd12dc586b39deb9f784e7b`.
That is a changed test input, not evidence that the old physical failure passed.
Reconcile fixture changes, solver changes and golden recapture separately.

Active `build-foundations` and `build-desktop-switch` configurations now use
`out/toolchains/slang-2026.18-linux-x86_64`, copied with all 4,888 file hashes
verified against the old pinned compiler package; `slangc -version` reports
2026.18. Its copy manifest is retained under `out/toolchains`. cgltf now comes
from the pinned in-tree source. Desktop GLFW resolves to the system package;
obsolete FetchContent cache overrides pointing at the old checkout were removed.
Both active CMake caches and generated build rules were checked for old-worktree
paths. Legacy checkout removal still awaits disposition of the unique physics
and asset work above.
