# F8-23 replacement-candidate CPU failure

F8-23 executed replacement candidate `realtime-single-step-v1-fd78fa8`
through the installed public API on its registered
`linux-thomas-laptop-slang-cpu` target. The relocated install and one-body
public observations pass, but the exact `cpu-reference-v1`
`rigid-brick-oracle` workload does not.

The required first tick returned `PM_ERROR_CAPACITY_EXCEEDED`. Its completed
public statistics query records 20 dispatches, 1,421 workgroups, one numerical
iteration, one spatial build and one spatial publication. The fixed workload
was not reduced: eight live/capacity rigid bodies, eight live/capacity
constraints and contact capacity 2,048.

The tracked raw capture is
[`F8-23-CANDIDATE-V2-CPU-FAILURE.json`](evidence/F8-23-CANDIDATE-V2-CPU-FAILURE.json).
It binds candidate manifest SHA-256
`f4f1c7807dd50b07857c568b9df3e8e2ccd6fd3870907238dc9dcfab45b493af`,
the exact embedded source revision, production driver SHA-256
`4a457b2288c3fe4474a73acc0f602b0eda7bb7153dbcdee3c7f66e9646622a47`
and the installed qualification harness.

This is a required-row failure, not partial qualification. F8-23 must stay
open. F8-37 must repair the protected telemetry or schedule envelope, and
F8-38 must freeze a new immutable candidate and route every affected device
qualification task to it before F8-23 can be claimed again.
