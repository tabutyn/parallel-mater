# F8-04: fixed-tick presentation clock

Status: implemented client contract for the explicit
`realtime-single-step-v1` Gallery frame profile. This clock profile does not by
itself advertise solver execution support or select the API simulation profile;
those remain negotiated by the F8 runtime work. `legacy-v1` remains the default.

## Tick clock

The real-time clock owns a monotonic accumulator with a fixed `1/60 s` tick and
one solver substep. A render callback can consume at most one tick. Any other
whole tick due after that callback is counted as dropped and discarded; it is
never caught up and `dt` is never enlarged. Consequently a sustained 30 Hz
display advances 30 ticks, or 0.5 simulated seconds, per wall-clock second.
Displays above 60 Hz issue zero-tick callbacks between tick boundaries.

A backwards clock, pause, disabled simulation, reset, scene replacement,
background/foreground transition, or workload/profile change clears presentation
debt. An elapsed interval above 250 ms is a stall: it submits no tick, reports
the discarded debt, and publishes alpha zero. The legacy profile retains its
existing target-rate timestep, approximately 240 substeps/second, four-step
catch-up limit, and alpha behavior under its existing name.

The frame plan is captured once before the tick decision. Because a real-time
callback admits at most one tick, controls, camera-derived gravity and kinematic
targets are applied exactly once at that tick boundary. A zero-tick render does
not apply them. Later F8 schedule work must reject a failed tick atomically; the
client clock must call `record_step` only after that successful tick.

## Presentation snapshots

Interpolation is previous-to-current with alpha clamped to `[0,1]`; the clock
publishes `[0,1)` and never extrapolates. Normal interpolation adds one completed
tick of presentation latency.

- Rigid positions use linear interpolation. Orientations use normalized,
  shortest-hemisphere quaternion interpolation.
- Fluid particles match previous state by stable ID. A birth has no match and
  snaps to current; a removed ID is absent from the current frame.
- Cloth, soft-body and rope outputs interpolate only while the affected domain
  cardinality is unchanged. A topology change snaps that domain to current.
- Legacy smoke output lacks stable IDs. Equal-cardinality snapshots interpolate
  by slot; any spawn/removal snaps the whole smoke domain to current.
- Paint and other discrete fields always use current state.
- Reset seeds previous and current with the same reset state, so alpha zero
  cannot expose the state from before reset.

Host rendering keeps these snapshots only on its already-required host-output
path. Native GPU rendering receives alpha with the public output-buffer request.
Each draw that has a previous GPU region interpolates in the graphics shader;
a draw without one uses current output (`alpha = 1`) in place. After reset or
before the first successful tick, GPU history is invalid and all draws use
current output even though the clock reports alpha zero. This makes the visible
previous/current reset pair identical without a CPU copy or physics readback.

## Reporting and performance meaning

`GalleryPresentationState` reports the frame-profile name, interpolation alpha,
presentation latency, and cumulative simulated-seconds/active-wall-seconds ratio
for the current clock epoch. Pausing, backgrounding, reset, scene replacement,
or profile change starts a new epoch. Active stalls remain wall time, so their
slowdown remains visible.

Host work frames/second and simulation/wall rate are different measurements.
A 30 Hz overloaded run correctly reports about `0.5`; it cannot satisfy a 60 Hz
physics performance gate merely because the display continues presenting.

## Executable coverage

`parallel-mater-gallery-frame-policy-tests` covers 30/60/90/120/144 Hz, jitter,
stall, pause/background/resume, one-tick admission, fixed `dt`, one substep,
alpha and simulation/wall reporting. The session-clock contract verifies the
same profile reaches the shared input/kinematic path only once per callback.
`parallel-mater-gallery-presentation-interpolation-tests` covers orientation,
stable-ID birth/removal, topology changes and reset snapshots. The GPU renderer
contract verifies unavailable previous output snaps to current without readback.
