#!/usr/bin/env python3
"""Adapt the installed conformance command contract to a physical iPad app."""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import uuid


BUNDLE = "com.ballrollergames.parallelmater.gallery"


def run(*command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def copy_from(device, source, destination, timeout="30"):
    run("xcrun", "devicectl", "device", "copy", "from", "--device", device,
        "--domain-type", "appDataContainer", "--domain-identifier", BUNDLE,
        "--source", source, "--destination", destination, "--timeout", timeout,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    device = os.environ.get("PM_IPAD_DEVICE")
    if not device:
        raise SystemExit("PM_IPAD_DEVICE is required")
    arguments = sys.argv[1:]
    output = None
    if "--output" in arguments:
        index = arguments.index("--output")
        if index + 1 >= len(arguments):
            raise SystemExit("--output requires a path")
        output = pathlib.Path(arguments[index + 1]).resolve()

    token = str(uuid.uuid4())
    validation = os.environ.get("PM_TEST_VALIDATION") == "1"
    command = ["xcrun", "devicectl", "device", "process", "launch",
               "--device", device, "--terminate-existing", BUNDLE,
               "--pm-job-token", token]
    if validation:
        command.append("--pm-validation")
    command.extend(arguments)
    run(*command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    timeout = float(os.environ.get("PM_IPAD_CONFORMANCE_TIMEOUT", "7200"))
    deadline = time.monotonic() + timeout
    with tempfile.TemporaryDirectory(prefix="pm-ipad-command-") as temporary:
        temporary = pathlib.Path(temporary)
        status_path = temporary / "status.json"
        while time.monotonic() < deadline:
            try:
                copy_from(device,
                          f"Documents/conformance-jobs/{token}/status.json",
                          status_path, "15")
                status = json.loads(status_path.read_text(encoding="utf-8"))
                if status.get("completed"):
                    break
            except (subprocess.CalledProcessError, FileNotFoundError,
                    json.JSONDecodeError):
                pass
            time.sleep(1)
        else:
            raise SystemExit(f"physical iPad command timed out after {timeout}s")

        stdout_path = temporary / "stdout.txt"
        copy_from(device, f"Documents/conformance-jobs/{token}/stdout.txt",
                  stdout_path, "30")
        contents = stdout_path.read_text(encoding="utf-8")
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            copied = temporary / "output"
            copy_from(device, f"Documents/conformance-jobs/{token}/output",
                      copied, "300")
            # Scenario commands use distinct output directories, while the
            # workload producer deliberately appends each selected workload to
            # one directory.  Preserve prior files in the latter case.
            shutil.copytree(copied, output, dirs_exist_ok=True)
        sys.stdout.write(contents)
        return int(status.get("return_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
