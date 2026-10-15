#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The same native Gallery smoke test on any available physics driver."""
import csv
import math
from pathlib import Path
import subprocess
import sys
import tempfile


def run(executable, *arguments, succeeds=True):
    result = subprocess.run([executable, *map(str, arguments)], capture_output=True, text=True, timeout=120)
    if (result.returncode == 0) != succeeds:
        raise AssertionError(f"{result.args}: exit {result.returncode}\n{result.stdout}\n{result.stderr}")
    return result


def main():
    executable = sys.argv[1]
    with tempfile.TemporaryDirectory(prefix="pm-frame-timing-") as temporary:
        directory = Path(temporary)
        normal = directory / "normal.ppm"
        timed = directory / "timed.ppm"
        output = directory / "timings.csv"
        common = ("--constraint-point", "--frames", "3")
        run(executable, *common, "--headless", normal)
        assert not list(directory.glob("*.csv")), "Unrequested CSV output"
        run(executable, *common, "--headless", timed, "--frame-timings", output)
        assert timed.read_bytes().startswith(b"P6\n"), "Invalid final image"
        assert timed.read_bytes() == normal.read_bytes(), "Timing changed final rigid scene image"
        with output.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == ["frame", "scene", "mode", "physics_steps", "total_ms",
                                        "physics_ms", "render_ms", "present_ms", "other_ms"]
            rows = list(reader)
        assert len(rows) == 3, "Expected one sample per simulated/rendered frame"
        for index, row in enumerate(rows, 1):
            assert int(row["frame"]) == index and int(row["physics_steps"]) == 1
            assert row["mode"] == "headless" and row["present_ms"] == "" and row["scene"]
            values = {key: float(row[key]) for key in ("total_ms", "physics_ms", "render_ms", "other_ms")}
            assert all(math.isfinite(value) and value >= 0 for value in values.values())
            assert values["physics_ms"] > 0 and values["render_ms"] > 0, "Each sample must simulate and render"
            assert abs(values["total_ms"] - sum(values[key] for key in ("physics_ms", "render_ms", "other_ms"))) < 0.00001
        result = run(executable, *common, "--headless", timed, "--frame-timings", directory, succeeds=False)
        assert "Frame timing CSV open failed" in result.stderr
        run(executable, "--frame-timings", succeeds=False)
        run(executable, "--calibrate", "--frame-timings", output, succeeds=False)
        run(executable, "--print-brick-profile", "--frame-timings", output, succeeds=False)
    print("Gallery headless frame timing passed")


if __name__ == "__main__":
    main()
