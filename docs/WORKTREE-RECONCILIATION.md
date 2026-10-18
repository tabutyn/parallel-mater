# Preserved and retired workstation work

Two dirty legacy checkouts were captured on 2026-10-16 before cleanup. The
ignored local archive at `out/worktree-backups/2026-10-16/` contains their
branch and HEAD metadata, staged and unstaged binary patches, complete changed
and untracked files, and SHA-256 hashes. F6-08 compared every live changed file,
patch, status record, branch and HEAD with that archive before retiring either
working state.

Reviewable dispositions remain in:

- `docs/roadmap/evidence/F6-06-legacy-launcher-reconciliation.json` for all
  10 launcher, branding and retired World/compiler-shim files;
- `docs/roadmap/evidence/F6-07/` for all 59 text hunks, two binary fixture
  changes and the raw debug capture;
- `docs/roadmap/evidence/F6-09-hinge-fixture-qualification.json` for the
  rejected five-body hinge fixture;
- `docs/roadmap/evidence/F6-10-physics-capture-replay.json` for the retained
  backend-neutral capture/replay behavior; and
- `docs/roadmap/evidence/F6-11-rigid-contact-adjudication.json` for all 26
  retained solver hunks.

The `parallel-mater-slang-kernels` worktree was removed after those records
passed their archive and disposition tests. Its branch and commit remain in the
shared Git database as an additional recovery path. The original
`parallel-mater` directory owns that shared Git database, so deleting the
directory would break every linked worktree. Its archived dirty state and
obsolete build directories were removed, and the checkout was repurposed as a
clean, fast-forwarded `main` checkout. It contains no legacy working changes.

Active `build-foundations` and `build-desktop-switch` configurations use the
independent copy at `out/toolchains/slang-2026.18-linux-x86_64`; all 4,888 files
were hash-verified when copied. cgltf comes from the pinned in-tree source and
desktop GLFW comes from the system package. Both active CMake trees regenerate
without the removed worktree, and neither their cache/build rules nor the
installed Gallery launcher references `parallel-mater-slang-kernels`.

The final machine-level checks and archive hashes are recorded in
`docs/roadmap/evidence/F6-08-retired-worktrees.json`.
