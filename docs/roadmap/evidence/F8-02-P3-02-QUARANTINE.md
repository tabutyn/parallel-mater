# F8-02 quarantine of pre-provenance Android experiments

The unfinished `work/P3-02/thomas-laptop` branch and its local output directories
remain preserved. They are historical debugging material, not performance
qualification. The v1 runner recorded the hash of the APK named by `--apk`, but
did not install that APK or compare it with the installed package. It also had no
fresh run nonce or dirty-patch identity. Consequently, a report's local APK hash
does not prove that APK produced the report.

Do not use any relative timing below to choose or reject an optimization. In
particular, **O3, rank-one and cooperative comparisons are unqualified** until
remeasured with the F8-02 v2 runner and a verified deployment. The source labels
and observed numbers are retained here without inferring causality:

| Local label | Report SHA-256 | Reported source | Rigid physics / GPU / FPS | Status |
|---|---|---|---|---|
| `P3-02-rigid-o3` | `321e42ac81d8a8aa86f9ca51ad46017b275816ed584a228d188a0c4365c68be1` | `05ead375cb8ee70eebb4b414c362b65b824a2b15` | 96.370 ms / 88.065 ms / 9.32 | unqualified |
| `P3-02-rigid-o3-run` | `55ccd1003062e81cbff4830c25107b6bee33f507b5d176411ce1b1a6e987e010` | `b22ccb05a4046978a3def535ccf9a4b681a980df` | 102.777 ms / 94.900 ms / 8.56 | unqualified |
| `P3-02-rigid-rank-one` | `31bc47896b925f3ad0832888025f050484b2e6cd7fb14399e61e412c557b06da` | `b22ccb05a4046978a3def535ccf9a4b681a980df` | 101.091 ms / 93.199 ms / 8.81 | unqualified |
| `P3-02-rigid-cooperative-module` | `333d5fb76a8c1af26887adf85ee82304aa43fb9a98cd52eff1f837f279876fc0` | `b22ccb05a4046978a3def535ccf9a4b681a980df` | 102.034 ms / 93.467 ms / 8.70 | unqualified |

The v1 reports' `passed: true` meant only that their captured log had no selected
fatal pattern and the final runtime report named direct Vulkan presentation. It
did not qualify build deployment. The same rule applies to all other
`out/android/P3-02-*/profile.json` comparisons.

Original incomplete/failed attempts are also retained as failures, not silently
dropped: `P3-02-rigid-compact-ldlt`, `-compact-ldlt-deduplicated`,
`-cooperative-deduplicated`, `-cooperative-solve`, `-explicit-ldlt`, and
`-schur` produced no report. The `P3-02-baseline`, `-baseline-chunked`,
`-optimized`, and `-optimized-chunked` directories retained APKs but no report;
their APK SHA-256 values are respectively
`ff811c42234834d12feced277077c1e86cf0793b59d7b5dab2249d8c3c76f649`,
`142a1f5768dcff3f2270fca4d77ae700fcec4ef784e382fdb126ac64d7cb1b9a`,
`10294b1c7a7a26a426c9adf9dc324d708287b6670c593f209716ea02ccdf5220`,
and `91e49b22aa80cd7602275ab95db10cf5f7fd56ec495bbc5f6a597a6eb654f9bd`.
No runtime is attributed to those files.

## Reproducible F8-02 capture

From a clean checkout of the candidate revision, with the pinned Android SDK and
Slang 2026.18 compiler available:

```bash
ANDROID_HOME=/path/to/android-sdk \
SLANGC=/path/to/slang-2026.18/bin/slangc \
./scripts/Build-AndroidGallery.sh

adb devices -l
python3 scripts/Test-AndroidVulkanSchedule.py \
  --serial PIXEL_7_SERIAL \
  --apk examples/android/app/build/outputs/apk/debug/app-debug.apk \
  --output out/android/F8-02-legacy-baseline
```

The v2 runner requires the exact Pixel 7, installs the APK, byte-compares the
installed base APK and bundled Vulkan driver, checks the APK-embedded source and
toolchain identity, and requires a fresh nonce for every scene. It captures the
fixed Rigid Body, Fluid and Smoke Soft Body workloads. `profile.json` is the
machine-readable result and `device.log` retains the source log. Missing devices,
installations, mismatched bytes, stale reports, thermal throttling and fatal
runtime diagnostics fail the run.
