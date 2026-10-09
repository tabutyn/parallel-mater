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

`config/device-profiles.json` contains two kinds of records:

- `hardware` records transcribe published manufacturer specifications and link
  to the official source.
- `verified_profiles` contain measured, human-approved scene configurations.

Missing manufacturer values remain absent. Compute-unit counts, memory
bandwidth, and power figures provide search context; they never predict or
certify a brick count. The measured preset ladder and compatible prior profiles
guide calibration. Contact solving contains dependent work and Metal contact
storage grows approximately with the square of rigid-body count, so theoretical
shader throughput is not a reliable capacity conversion.

Profile matching includes machine/GPU variant, memory, backend, operating
system/driver, power mode, 1080p dimensions, scene version, and solver version.
An incompatible or unknown device uses the versioned built-in configuration
until calibration produces a replacement.

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
