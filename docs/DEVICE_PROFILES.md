# Device-adaptive gallery scenes

The Rigid Body gallery scene builds its brick walls at runtime. A profile selects
brick count, uniform brick scale, and separated wall count. Bricks remain
independent triangle-mesh rigid bodies; scale changes mass by volume and lets the
physics backend derive inertia from the scaled triangles.

Both galleries render at 1920×1080 internally. Press `P` in the Rigid Body scene
to edit the three wall values. `Tab` changes fields, `Enter` applies and restarts,
`C` runs calibration, `V` verifies the latest passing calibration, and `R`
restarts the current configuration. CUDA and Metal also accept
`--brick-count`, `--brick-scale`, `--brick-planes`, `--calibrate`, and
`--profiles-file`.

Headless runs ignore hardware profiles unless the brick values are supplied on
the command line, so regression tests retain the same scene on every machine.
Interactive releases read the catalog bundled beside the executable; merely
launching the app never opens a source checkout or external development drive.

Interactive CUDA and Metal startup select bricks automatically from the current
physics GPU and the bundled data, in this order:

1. Use an exact compatible `verified_profiles` entry when one exists.
2. Otherwise use `measurements` for the same GPU model/variant, backend, OS family,
   power mode, scene/solver version and render resolution. Allow up to 256 MiB
   difference in usable VRAM for driver reservations. Prefer records from the
   same machine model, CPU and driver; among those, choose the largest stable
   measured configuration with quiet **median <=16.67 ms** and collision
   **p95 <=33.33 ms** (typical 60 FPS / difficult moments 30 FPS).
3. Without suitable data, start with eight bricks, scale 2, one wall plane.

Startup prints the GPU, brick count, scale, planes and selection source.
The RTX 3050 Ti Laptop GPU records currently select **24 bricks, scale 2,
one wall plane**. These short measurements give a startup recommendation, not
sustained qualification or a guarantee of no dropped simulation time. They do
not populate `verified_profiles`. Published GPU specifications never determine
a brick count, and records from another backend or solver are not extrapolated.
The earlier one-brick certification remains withdrawn.

Explicit `--brick-count`, `--brick-scale` or `--brick-planes` values override the
automatic selection. Headless regression scenes retain their fixed defaults.
CUDA can report its normal startup decision without opening a window:

```sh
./build-gallery/parallel-mater-gallery --print-brick-profile
```

On Linux the running executable's actual path locates its bundled catalog even
when launched through `PATH` from another directory. DMI supplies the machine
model independently of whether the shell exports a hostname.

Calibration measures completed physics and rendering during quiet support and
scripted centered/off-center impacts. A candidate passes only when both phases
have a 95th percentile at or below 15 ms, every measured frame is at or below
30 ms, simulation time keeps pace, and body state remains stable. A short preset
search is followed by a three-minute sustained qualification. Measurements are
saved in the user's application-state directory; they do not enter the tracked
catalog until a person verifies the visual result.
If the one-brick preset cannot qualify at 1080p, calibration stops and records
that failed result rather than inventing a usable capacity. `--calibrate` exits
after writing the local result.

`config/device-profiles.json` contains three kinds of records:

- `hardware` records transcribe published manufacturer specifications and link
  to the official source.
- `verified_profiles` contain measured, human-approved scene configurations.
- `measurements` contain short-run timings and provenance used for startup
  recommendations. Saving a verified profile preserves these records.

Missing manufacturer values remain absent. Compute-unit counts, memory
bandwidth, and power figures provide search context; they never predict or
certify a brick count. The measured preset ladder and compatible prior profiles
guide calibration. Contact solving contains dependent work and Metal contact
storage grows approximately with the square of rigid-body count, so theoretical
shader throughput is not a reliable capacity conversion.

Verified profile matching includes machine/GPU variant, memory, backend, operating
system/driver, power mode, 1080p dimensions, scene version, and solver version.
An incompatible verified profile is skipped; measurements or the modest startup
fallback above supply the interactive configuration instead.

The AVBD migration uses `cuda-avbd-v1` and `metal-avbd-v1` solver keys for
both profile matching and new calibration results. Earlier `*-rigid-v1`
profiles remain on disk but are not reused; recalibrate before verifying a
profile for the new solver. This does not change D3D12's solver.

Metal's verify action asks for a destination, which can be the tracked catalog
or a standalone export when no checkout is present. CUDA writes a standalone
local export unless `--profiles-file` explicitly identifies the tracked catalog.
Catalog replacement is atomic, serializes concurrent writers, and rejects an
external edit detected between read and rename. AMD, Google, Intel, and Qualcomm
records are research entries only until a future runtime backend can measure
them.

The 60/30 FPS result applies only to the recorded qualification workload,
resolution, software versions, power mode, and test date. A profile does not
claim the same result for arbitrary future workloads or operating-system stalls.
