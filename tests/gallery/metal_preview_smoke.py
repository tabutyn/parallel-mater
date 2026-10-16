#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Opt-in Apple-GPU Gallery preview checks; not all-domain conformance."""
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def capture(executable, directory, name, options, frames, reference=None, domains=()):
    image = directory / (name + ".ppm")
    states = directory / (name + ".txt")
    environment = os.environ.copy()
    environment.pop("PM_DRIVER_FILES", None)
    environment.pop("PM_DRIVER_PATH", None)
    if reference:
        environment["PM_DRIVER_FILES"] = reference
    result = subprocess.run([executable, *options, "--frames", str(frames),
                             "--physics-capture", str(states), "--headless", str(image)],
                            env=environment, capture_output=True, text=True, timeout=300)
    if result.returncode:
        raise AssertionError(f"{name}: {result.stdout}\n{result.stderr}")
    expected = "Slang CPU" if reference else "Metal GPU preview"
    assert expected in result.stdout, (name, "wrong physics driver", result.stdout)
    data = image.read_bytes()
    assert data.startswith(b"P6\n") and len(data) > 10000, (name, "missing rendered image")
    rows = {}
    lines = states.read_text().splitlines()
    summary = lines[1].split()
    counts = dict(zip(summary[::2], map(int, summary[1::2])))
    for domain in domains:
        assert counts.get(domain, 0) > 0, (name, "empty domain output", domain, counts)
    for line in lines:
        if line.startswith(("fluid ", "debug ", "contact ")):
            for word in line.split()[1:]:
                try:
                    value = float(word)
                except ValueError:
                    continue
                assert math.isfinite(value), (name, "nonfinite output", line)
        if not line.startswith("rigid "):
            continue
        words = line.split()
        numbers = [float(value) for value in words[2:] if value not in
                   ("position", "orientation", "linear_velocity", "angular_velocity")]
        assert len(numbers) == 13 and all(math.isfinite(value) for value in numbers), (name, line)
        assert max(abs(value) for value in numbers[:3]) < 1000, (name, "unbounded position", line)
        assert abs(sum(value * value for value in numbers[3:7]) - 1) < 0.001, (name, "non-unit rotation", line)
        rows[int(words[1])] = numbers
    assert rows, (name, "empty rigid output")
    return rows


def main():
    executable = sys.argv[1]
    reference = sys.argv[2] if len(sys.argv) > 2 else None
    cases = [(name, ["--constraint-" + name]) for name in
             ("fixed", "point", "hinge", "piston", "generic", "motor-spring")]
    cases.append(("bricks", ["--brick-count", "16"]))
    with tempfile.TemporaryDirectory(prefix="pm-metal-gallery-") as temporary:
        directory = Path(temporary)
        for name, options in cases:
            short = capture(executable, directory, name + "-short", options, 3)
            if reference:
                baseline = capture(executable, directory, name + "-cpu", options, 3, reference)
                assert short.keys() == baseline.keys(), (name, "body count differs")
                error = max(abs(a - b) for body in short for a, b in zip(short[body], baseline[body]))
                print(f"{name}: three-frame CPU/GPU maximum state difference {error:.6g}", flush=True)
                assert error < 0.005, (name, "short-run reference mismatch", error)
            capture(executable, directory, name + "-long", options, 120)
            print(f"{name}: 120 GPU frames, finite states, normalized rotations, final image passed", flush=True)
    print("Metal rigid Gallery preview smoke passed")


if __name__ == "__main__":
    main()
