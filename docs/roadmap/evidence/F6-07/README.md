# F6-07 retained hinge/contact snapshot

This directory preserves the 2026-10-16 working files and patch from commit
`deb158aeb75cd9eae02d78acbc06d9417a958ade`. The archive is evidence and input
for F6-09 through F6-11. It is not an implementation patch to apply wholesale.

- `snapshot-working-files.tar.gz` contains the exact 21 changed/untracked files.
- `snapshot-unstaged.patch` provides reviewable text hunks and binary change markers.
- `snapshot-metadata.json` records every working-file SHA-256.
- `hunk-dispositions.json` assigns all 62 retained items to bounded follow-up work.
- `cuda-reproduction.json` records the archived and current-main physical CUDA runs.

The archived CUDA golden combines a different fixture with solver changes and is
quarantined. F6-09 may generate a new golden only from its reviewed revision.
