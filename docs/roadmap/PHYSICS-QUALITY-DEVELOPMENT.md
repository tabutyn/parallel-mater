# Improve physics one system at a time

Maintainer direction, 2026-11-02: the current simulations are not convincing or
compelling. Passing execution, regression, or performance tests does not settle
that concern. Subsequent maintainer direction on the same date sets the order:
land the retained Mac repair and finish the existing F8 integration,
qualification and release sequence before starting scene repairs. The later
scene-quality work improves one system and one concrete behavior at a time.
On 2026-11-03, the maintainer assigned that later work to its own F9 series.
Later that day, the maintainer also deferred performance-only blockers to F9
so the broken scenes can be repaired before final timing optimization.

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

F8-74 (integration-candidate refreeze), F8-78 (integrated Windows validation),
and F8-99 (integration exit review) remain active. Apply the timing deferral
below throughout the remaining F8 device, delivery and exit sequence. Broad
60 Hz performance qualification requires later passing device/workload evidence.

F9-01 through F9-06 are deferred until the F8-99 exit review completes. They
are post-exit quality follow-ups, not prerequisites for that exit. Completing
F8 does not itself establish that the scenes are convincing.

## Timing deferral approved on 2026-11-03

The remaining F8 cards accept integration and correctness with disclosed timing
debt. CPU/GPU/physics/frame latency and simulated/wall-rate targets become
informational for F8 completion. The original 10 ms physics p95, 16.667 ms frame
target, 0.99 simulated/wall ratio and other frozen per-tier timing targets carry
forward into F9. This decision supersedes earlier F8 instructions requiring
those wall-clock targets before refreeze, device acceptance, delivery or exit.

Every capture must still execute its required cases, scenarios and workloads at
the frozen counts and sample sizes, retain all samples and stalls, and identify
exact source/device/artifact provenance. Evaluate physical/numerical assertions,
resource ownership, lifecycle, rejected-step recovery and independent work
budgets separately. A failure in any of those checks blocks the affected F8
acceptance. Missing required device evidence remains an open qualification row.

Keep the original validator outputs and thresholds for historical captures.
Record every failure by assertion and classify timing-only debt explicitly.
For each canonical negative control, retain the injected-defect result and the
unmodified baseline result separately. When baseline timing already fails,
record the control as unqualified for final performance acceptance. Demonstrate
the affected correctness/work/recovery rejection independently. F9 owns the
remaining timing-dependent control qualification. Any executable reporting
change needs a bounded implementation card and merged lease.

F8-78 captures the integrated Windows repair and these separate verdicts.
F8-74 freezes an integration candidate and routes fresh device rows under this
policy. F8-31 publishes its tested scope and limitations. F8-99 inventories all
outstanding timing debt before exiting. F9-07 owns Windows GTX 1080 optimization
against the original targets after scene repairs. F9-08 defines the fresh
candidate and bounded device/performance qualification tasks needed for a broad
60 Hz claim. Both follow the F8 exit, alongside the existing F9 scene series.

## F9 scene repair series

### Recorded PEG PAINT scene-quality debt

The maintainer assigned broken Gallery scene repairs to F9 before the Windows
v10 qualification run. The 2026-11-04 audit gives that direction a concrete
reproduction. Installed PEG PAINT completes the normal 240 fixed ticks with
zero capacity overflows, then fails its existing deposition check with 7 painted
texels against 100 required. Liquid Y bounds end at -11.9014 to -11.4821. The
other 27 scene checks and an interactive startup pass. The frozen public and
primitive physical, numerical, capability, resource/recovery and independent
work-budget checks also pass. Full results are retained in
[the scene-debt record](evidence/F8-26-CANDIDATE-V10-GALLERY-DEBT.json).

F8-26 may accept this exact recorded scene-quality debt under F9-09 while
publishing separate integration, Gallery-quality and timing verdicts. Keep the
deposition assertion failed in the v10 evidence. F8-99 carries the same named
debt into F9 after its integration exit. Every frozen core check, complete frame
execution and the remaining required Gallery checks still gate F8-26. Additional
faults require their own reviewed classification and bounded task.

F9-09 repairs this concrete PEG PAINT behavior after F8-99. F9-07 timing work
waits for that repair along with the first six scene improvements. This records
the existing scene-deferral direction while preserving the original failure,
threshold, workload and source identity.

### Initial scene task migration

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

Every F9 card includes an F8-99 prerequisite and remains deferred until that
exit review. F9 completion is not required to close F8. Solver limits and the
focused-change acceptance above continue to apply.

## Immediate queue

F8-73, F8-78, F8-74 and the candidate-v10 physical observation/delivery cards
are merged. The preliminary F8-99 review did not close the series: five
required rows retain non-timing correctness, capability or exact-capture
failures. Follow the bounded chain in
[`F8-99-EXIT-AUDIT.json`](evidence/F8-99-EXIT-AUDIT.json): repair F8-88 through
F8-92, freeze F8-93, physically qualify F8-94 through F8-97, deliver F8-98 and
then repeat F8-99. Keep candidate-v10 failures immutable and carry the
specifically recorded F9-09 scene debt.

After that exit, reactivate F9-01 through F9-06 through planning. They cover a
first visible improvement to rigid bodies, liquids, cloth, soft bodies, ropes,
and smoke respectively. No all-six completion barrier precedes their individual
merges. Follow existing leases and locks.
F9-09 addresses the recorded PEG PAINT defect after the F8 exit.
F9-07 then addresses Windows timing after those scene improvements.
F9-08 plans the remaining exact-candidate performance qualification.

These are new follow-ups to previously completed implementation tasks, not a
rewrite of their history. F8-73 remains the bounded Mac adapter/Gallery repair;
its revised acceptance separates correctness from aggregate timing so that
repair can provide a working development base. No task becomes complete merely
because its merge gate changed.
