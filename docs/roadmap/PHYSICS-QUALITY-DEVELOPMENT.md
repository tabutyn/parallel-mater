# Improve physics one system at a time

Maintainer direction, 2026-11-02: the current simulations are not convincing or
compelling. Passing execution, regression, or performance tests does not settle
that concern. Subsequent maintainer direction on the same date sets the order:
land the retained Mac repair and finish the existing F8 integration,
qualification and release sequence before starting scene repairs. The later
scene-quality work improves one system and one concrete behavior at a time.
On 2026-11-03, the maintainer assigned that later work to its own F9 series.

This decision changes development merge and release gates. It supersedes older
instructions that make full F8 candidate qualification a prerequisite for
scoped improvements or development previews. It does **not** change the solver
architecture or declare any failed measurement a pass.

## Keep the solver limits

- One public API and one production execution path; no legacy/quality/F8 mode
  selection for applications or the Gallery.
- Exactly one 1/60-second simulation advance and one shared dynamic spatial
  update per tick. Existing iteration, dispatch, submission, readback, and
  synchronization budgets remain enforced. No extra substeps or hidden retries.
- Physics equations remain shared Slang. Native adapters provide device
  services. Keep existing resource ownership, failure recovery, and explicit
  capability behavior.
- Fix or replace a poor solver equation within those limits when the scoped
  physics task requires it. Preserving the old trajectory bit-for-bit is an
  optimization check, not a requirement to preserve incorrect physics in a
  quality-improvement task. Unaffected behavior still needs regression checks.

If a convincing solution cannot fit those limits, record that finding. Do not
silently increase the work budget or weaken the physical requirements.

## What a focused change needs to merge

1. Name the system, the specific visible problem, and the intended physical
   behavior. Start with a small reproducible scene and explicit material,
   timestep, scale, and initial-state inputs. A whole-family rewrite is not the
   default unit of work.
2. Preserve comparable before/after motion evidence from the actual Gallery:
   short clips or an ordered frame sequence with camera, inputs, source, device,
   and backend identified. Capture state or trajectories alongside the images
   so rendering changes cannot conceal a physics defect.
3. Add or use independent checks for the behavior being repaired: for example
   contact separation, strain, volume, buoyancy, or momentum/energy with external
   work and dissipation accounted for. A stable image or a faster run alone
   does not establish a better simulation. Explain changes to any expected
   result using those checks; do not simply replace a golden with new output.
4. Pass the affected numerical, lifecycle, resource, and interaction regression
   tests and required build/portable CI. Shared schedule or cross-family edits
   require checks for the consumers actually affected. Unrelated physical
   device rows and the complete mixed workload suite need not pass first.
5. Enforce the unchanged solver-work budgets. Record timing when available,
   including stalls and required waits; report missing or noisy measurements
   honestly. The whole-project 10 ms physics p95 and 16.667 ms frame targets
   are tracked release goals, not universal merge blockers for these tasks.
   Do not advertise an unmeasured or missed target as achieved.
6. Update the installed Gallery on the development target and verify its
   revision/backend. Document what improved, what remains poor, and which
   systems/devices were exercised. Completing one fix does not qualify an
   entire family or every interaction.

Physics tasks may improve equations within the retained limits. Adapter-only
optimizations still preserve the physics result. A genuine physical regression,
crash, data race, broken resource lifetime, or exceeded work budget remains a
merge blocker in the affected scope.

## Development releases and final qualification

Main and explicitly labeled development Gallery/SDK previews can receive these
scoped improvements. A preview can ship for one available platform; it does not
wait for the complete Windows/Linux/macOS/iPad/Android hardware matrix. Its
notes state the tested scope and outstanding quality/performance limitations.
This is release labeling, not a second simulation mode or backend physics fork.

Keep immutable candidate manifests, raw failures, and completed task history.
The historical frozen capture still evaluates with its original thresholds.
Never combine different revisions into a qualified candidate or describe a
failed performance baseline as a passing negative-control suite.

F8-74 (whole-candidate refreeze), F8-78 (frozen Windows performance target), and
F8-99 (aggregate exit review) are active again to finish the existing F8 work.
Their original qualification rules remain intact. They do not restore aggregate
timing as a merge blocker for the scoped F8-73 development repair. Broad final
qualification still requires its own passing device/workload evidence.

F9-01 through F9-06 are deferred until the F8-99 exit review completes. They
are post-exit quality follow-ups, not prerequisites for that exit. Completing
F8 does not itself establish that the scenes are convincing.

## F9 scene repair series

F9 owns the scene-quality improvements after F8. These six deferred cards had
no lease, evidence or execution history when moved on 2026-11-03. Only their
IDs changed; their scope, dependencies, locks and acceptance checks are intact.
The previous IDs remain reserved and their original definitions stay in Git
history. The roadmap checker permits only this explicit migration; it still
rejects deletion or renumbering of started/completed work and unrelated IDs.

| Previous ID | Current task | System |
|---|---|---|
| F8-79 | [F9-01](tasks/F9-01.json) | Rigid-body contacts and settling |
| F8-80 | [F9-02](tasks/F9-02.json) | Liquid motion and settling |
| F8-81 | [F9-03](tasks/F9-03.json) | Cloth drape and motion |
| F8-82 | [F9-04](tasks/F9-04.json) | Soft-body deformation and recovery |
| F8-83 | [F9-05](tasks/F9-05.json) | Rope hanging and swinging |
| F8-84 | [F9-06](tasks/F9-06.json) | Smoke transport and buoyancy |

Every F9 card retains its F8-99 prerequisite and remains deferred until that
exit review. F9 completion is not required to close F8. Solver limits and the
focused-change acceptance above continue to apply.

## Immediate queue

F8-73 is merged under its revised development acceptance. Complete remaining
F8-78 performance work,
F8-74 candidate refreeze, physical qualification rows, qualified delivery and
F8-99 exit review according to dependencies and available hardware.

After that exit, reactivate F9-01 through F9-06 through planning. They cover a
first visible improvement to rigid bodies, liquids, cloth, soft bodies, ropes,
and smoke respectively. No all-six completion barrier precedes their individual
merges. Follow existing leases and locks.

These are new follow-ups to previously completed implementation tasks, not a
rewrite of their history. F8-73 remains the bounded Mac adapter/Gallery repair;
its revised acceptance separates correctness from aggregate timing so that
repair can provide a working development base. No task becomes complete merely
because its merge gate changed.
