#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Current Gallery domains and pairs; execution smoke, NOT physics conformance."""
from pathlib import Path
import sys
import tempfile

from metal_preview_smoke import capture


def main():
    executable = sys.argv[1]
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    cases = [
        ("fluid", ("fluid",)), ("fluid-rigid", ("fluid", "rigid")),
        ("peg-paint", ("fluid",)),
        ("cloth", ("cloth",)), ("cloth-tear", ("cloth",)), ("cloth-paint", ("cloth",)),
        ("water-cloth", ("fluid", "cloth")), ("soft-body", ("soft_body",)),
        ("soft-body-rigid", ("soft_body", "rigid")),
        ("soft-body-cloth", ("soft_body", "cloth")),
        ("soft-body-fluid", ("soft_body", "fluid")), ("rope", ("rope",)),
        ("rope-fluid", ("rope", "fluid")), ("rope-soft-body", ("rope", "soft_body")),
        ("rope-cloth", ("rope", "cloth")), ("smoke", ("smoke",)),
        ("smoke-water", ("smoke", "fluid")), ("smoke-softbody", ("smoke", "soft_body")),
        ("smoke-cloth", ("smoke", "cloth")), ("smoke-rope", ("smoke", "rope")),
    ]
    with tempfile.TemporaryDirectory(prefix="pm-metal-domains-") as temporary:
        directory = Path(temporary)
        for name, domains in cases:
            options = ["--" + name]
            if "fluid" in domains:
                options += ["--fluid-particles", "512"]
            # Paint's existing coverage check requires time for brush contact.
            steps = max(frames, 120) if "paint" in name else frames
            capture(executable, directory, name, options, steps, domains=domains)
            print(f"{name}: {steps} GPU steps, populated domain output, final image passed", flush=True)
    print("Current-domain execution smoke passed; roadmap modes and full qualification remain pending")


if __name__ == "__main__":
    main()
