# F8-02 verified Pixel 7 legacy baseline

Result: **passed provenance and capture checks**. This is a baseline of the old
30 Hz, eight-substep profile, not evidence that it meets the F8 real-time budget.
The source was clean commit `7a6d0b5fe8e2509257d2d5bfb7efe365a0251c0b`.

The runner installed `org.parallelmater.gallery` on physical Pixel 7
`panther`, then read back the installed `base.apk`. Local and installed APK
SHA-256 both equal
`59f276ac9b81eaf104051eef298b0157d2839f43a6538e02b4cd732d369e6fbb`.
Local and installed bundled Vulkan driver SHA-256 both equal
`2c1bd2bb76ba5f33c968302161d21ac13ca69110d6c5c6cf750a415f3b9b67cb`.
The APK runtime reported the same source revision, `dirty: false`, patch identity
`clean`, the pinned toolchain, direct Vulkan presentation and fresh per-scene
nonces derived from run nonce `18a5b71a242848fa8b1e5f521d331d79`.

Device build: Android 17/API 37, security patch 2026-10-05, fingerprint
`google/panther/panther:17/CP3A.261005.002.A1/16269273:user/release-keys`.
Thermal status remained 0. Virtual skin was 30.84 °C at start and 31.39 °C at
end; battery was 29.40 °C and 29.00 °C respectively. No selected fatal,
validation or signal diagnostics occurred.

| Fixed workload | Live / capacity | Dispatches/tick | Physics mean / frame p95 | GPU mean / p95 | CPU record mean / p95 | FPS | Simulation/wall |
|---|---|---:|---:|---:|---:|---:|---:|
| Rigid Body | rigid 6 / 6 | 138 | 103.919 / 116.783 ms | 96.667 / 108.395 ms | 2.347 / 3.042 ms | 8.54 | 0.285× |
| Fluid | fluid 128 / 128; rigid 1 / 1 | 1,163 | 116.874 / 148.417 ms | 85.686 / 97.496 ms | 20.452 / 29.620 ms | 7.59 | 0.253× |
| Smoke Soft Body | smoke 128 / 128; soft nodes 3,520 / 3,520; rigid 2 / 2 | 6,841 | 599.972 / 630.758 ms | 521.888 / 543.802 ms | 61.561 / 72.901 ms | 1.62 | 0.054× |

`profile.json` contains all 60 per-tick samples for each workload, p50/p95/p99
CPU record, submit-to-fence and GPU distributions, plus per-phase timings and
dispatch counts. `device.log` is the corresponding raw logcat capture. Workload
count readback happened only after the matched timing window was frozen.

The dominant measured phase costs reinforce the F8 architecture motivation:

- Fluid neighbor sorting used 784 of 1,163 dispatches/tick and averaged
  51.28 ms GPU time.
- Smoke Soft Body used 3,520 dispatches for contact cleanup, 1,760 for contact
  generation and 640 for soft constraints. Those three phases averaged about
  330 ms GPU time together.
- These are measurements of the legacy implementation only. They do not qualify
  O3, rank-one, cooperative or other unfinished P3-02 experiments.

Reproduction commands and the quarantine of earlier unverified comparisons are
in [`../F8-02-P3-02-QUARANTINE.md`](../F8-02-P3-02-QUARANTINE.md).
