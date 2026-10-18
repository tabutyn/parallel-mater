#!/usr/bin/env python3
"""Capture Android Gallery control and accessibility evidence on a phone AVD."""
import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import time
import xml.etree.ElementTree as ET


PACKAGE = "org.parallelmater.gallery"
ACTIVITY = PACKAGE + "/.GalleryActivity"
BOUNDS = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")
INTERNAL_TERMS = ("roadmap", "qualification", "f7-")


class Device:
    def __init__(self, adb, serial, output, timeout):
        self.adb = adb
        self.serial = serial
        self.output = output
        self.timeout = timeout

    def run(self, *command, binary=False):
        result = subprocess.run(
            [self.adb, "-s", self.serial, *command], check=True,
            capture_output=True, text=not binary)
        return result.stdout

    def hierarchy(self):
        dump = self.run("shell", "uiautomator", "dump",
                        "/sdcard/pm-gallery-window.xml")
        raw = self.run("exec-out", "cat", "/sdcard/pm-gallery-window.xml")
        start = raw.find("<?xml")
        if start < 0:
            start = raw.find("<hierarchy")
        end = raw.rfind("</hierarchy>")
        if start < 0 or end < start:
            raise ET.ParseError(
                "incomplete Android hierarchy; dump output=" + dump.strip()[:200])
        return ET.fromstring(raw[start:end + len("</hierarchy>")])

    @staticmethod
    def matches(node, value):
        return node.get("text") == value or node.get("content-desc") == value

    def wait_node(self, value, present=True):
        deadline = time.monotonic() + self.timeout
        last = None
        while time.monotonic() < deadline:
            try:
                last = self.hierarchy()
            except (subprocess.CalledProcessError, ET.ParseError):
                time.sleep(1)
                continue
            matches = [node for node in last.iter() if self.matches(node, value)]
            if bool(matches) == present:
                return matches[0] if matches else None, last
            time.sleep(1)
        slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:48]
        if last is not None:
            ET.ElementTree(last).write(
                self.output / f"failure-{slug}.xml", encoding="utf-8",
                xml_declaration=True)
        try:
            (self.output / f"failure-{slug}.png").write_bytes(
                self.run("exec-out", "screencap", "-p", binary=True))
            (self.output / f"failure-{slug}-logcat.txt").write_text(
                self.run("logcat", "-d"), encoding="utf-8")
        except subprocess.CalledProcessError:
            pass
        state = "appear" if present else "disappear"
        raise RuntimeError(f"Timed out waiting for {value!r} to {state}")

    def tap(self, value):
        node, _ = self.wait_node(value)
        self.tap_node(node, value)

    def tap_node(self, node, value):
        match = BOUNDS.fullmatch(node.get("bounds", ""))
        if not match:
            raise RuntimeError(f"No usable bounds for {value!r}")
        left, top, right, bottom = map(int, match.groups())
        self.run("shell", "input", "tap", str((left + right) // 2),
                 str((top + bottom) // 2))

    def maybe_tap(self, value):
        try:
            tree = self.hierarchy()
        except (subprocess.CalledProcessError, ET.ParseError):
            return False
        matches = [node for node in tree.iter() if self.matches(node, value)]
        if not matches:
            return False
        self.tap_node(matches[0], value)
        return True

    def launch_scene(self, index, name, expected_control):
        self.run("shell", "am", "start", "-f", "0x20000000", "-n", ACTIVITY,
                 "--ei", "pm_scene", str(index))
        self.wait_node(expected_control)
        _, tree = self.wait_node("Choose scene: " + name)
        self.audit(tree, name)
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        ET.ElementTree(tree).write(
            self.output / f"{index:02d}-{slug}.xml", encoding="utf-8",
            xml_declaration=True)
        return tree

    def screenshot(self, name):
        data = self.run("exec-out", "screencap", "-p", binary=True)
        path = self.output / f"{name}.png"
        path.write_bytes(data)
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError(f"Android did not produce a PNG for {name}")

    @staticmethod
    def audit(tree, context):
        missing = []
        visible_text = []
        for node in tree.iter():
            text = node.get("text", "").strip()
            description = node.get("content-desc", "").strip()
            if text:
                visible_text.append(text)
            if description:
                visible_text.append(description)
            if (node.get("clickable") == "true" and
                    node.get("enabled") == "true" and not (text or description)):
                missing.append(node.get("class", "unknown"))
        if missing:
            raise RuntimeError(
                f"{context}: enabled clickable controls lack accessible text: {missing}")
        joined = "\n".join(visible_text).lower()
        if "error:" in joined:
            raise RuntimeError(f"{context}: Gallery reported an error")
        found = [term for term in INTERNAL_TERMS if term in joined]
        if found:
            raise RuntimeError(f"{context}: internal terms visible to users: {found}")


def require_controls(tree, expected, context):
    descriptions = {node.get("content-desc", "") for node in tree.iter()}
    missing = [value for value in expected if value not in descriptions]
    if missing:
        raise RuntimeError(f"{context}: missing semantic controls {missing}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("out/android/gallery-ui"))
    parser.add_argument("--revision", default=os.environ.get("GITHUB_SHA", "unknown"))
    parser.add_argument("--apk", type=pathlib.Path, default=pathlib.Path(
        "examples/android/app/build/outputs/apk/debug/app-debug.apk"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = Device(args.adb, args.serial, args.output, args.timeout)

    device.run("shell", "am", "force-stop", PACKAGE)
    device.run("shell", "am", "start", "-n", ACTIVITY)
    time.sleep(1)
    device.maybe_tap("Got it")
    _, initial = device.wait_node("Choose scene: RIGID BODY")
    ET.ElementTree(initial).write(
        args.output / "initial-rigid-body.xml", encoding="utf-8",
        xml_declaration=True)

    records = []
    global_controls = {
        "Reset scene", "Scene information", "Focus scene action",
        "Zoom out", "Zoom in",
    }

    gravity = device.launch_scene(0, "RIGID BODY", "Tilt gravity left")
    require_controls(gravity, global_controls | {
        "Tilt gravity left", "Tilt gravity forward", "Tilt gravity back",
        "Tilt gravity right",
    }, "gravity")
    device.tap("Tilt gravity left")
    device.tap("Pause")
    device.wait_node("Resume")
    device.tap("Resume")
    device.wait_node("Pause")
    device.tap("Focus scene action")
    device.tap("Reset scene")
    device.screenshot("gravity-controls")
    records.append({"scene": "rigid-body", "actions": [
        "gravity", "focus", "pause", "resume", "reset"]})

    force = device.launch_scene(5, "CONSTRAINT: GENERIC", "Push forward")
    require_controls(force, {
        "Push left", "Push forward", "Push back", "Push right",
    }, "authored force")
    device.tap("Push forward")
    records.append({"scene": "constraint-generic", "actions": ["authored force"]})

    motor = device.launch_scene(
        6, "CONSTRAINT: MOTOR + SPRING", "Drive forward")
    require_controls(motor, {
        "Steer left", "Drive forward", "Drive reverse", "Steer right",
    }, "motor and steering")
    device.tap("Drive forward")
    device.tap("Steer left")
    device.screenshot("motor-steering-controls")
    records.append({"scene": "constraint-motor-spring",
                    "actions": ["motor", "steering"]})

    released = device.launch_scene(2, "CONSTRAINT: POINT", "Release")
    require_controls(released, {"Release"}, "constraint release")
    device.tap("Release")
    device.wait_node("Restore")
    device.tap("Restore")
    device.wait_node("Release")
    records.append({"scene": "constraint-point",
                    "actions": ["release", "restore"]})

    dump = device.launch_scene(7, "DUMP", "Raise bed")
    require_controls(dump, {"Drive forward", "Steer left", "Raise bed"}, "dump")
    device.tap("Raise bed")
    device.wait_node("Lower bed")
    device.screenshot("dump-controls")
    device.tap("Lower bed")
    device.wait_node("Raise bed")
    records.append({"scene": "dump-truck",
                    "actions": ["motor", "steering", "raise bed", "lower bed"]})

    fluid = device.launch_scene(8, "FLUID", "Focus scene action")
    directional = {"Push left", "Drive forward", "Tilt gravity left"}
    descriptions = {node.get("content-desc", "") for node in fluid.iter()}
    leaked = sorted(directional & descriptions)
    if leaked:
        raise RuntimeError(f"camera-only scene exposes directional actions: {leaked}")
    device.tap("Scene information")
    _, info = device.wait_node("Done")
    device.audit(info, "Info")
    ET.ElementTree(info).write(
        args.output / "common-info.xml", encoding="utf-8",
        xml_declaration=True)
    info_text = "\n".join(node.get("text", "") for node in info.iter())
    for phrase in ("Drag to orbit", "Requires:", "Renderer:"):
        if info_text.count(phrase) != 1:
            raise RuntimeError(
                f"Info must contain one {phrase!r} section, found {info_text.count(phrase)}")
    device.screenshot("common-info")
    device.tap("Done")
    records.append({"scene": "fluid", "actions": ["focus", "info"],
                    "directional_controls_hidden": True})

    model = device.run("shell", "getprop", "ro.product.model").strip()
    apk_sha256 = hashlib.sha256(args.apk.read_bytes()).hexdigest()
    runner_sha256 = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    report = {
        "schema": "parallel-mater-android-gallery-ui/v1",
        "revision": args.revision,
        "device": model,
        "serial": args.serial,
        "backend": "slang-cpu",
        "artifacts": {
            "apk_sha256": apk_sha256,
            "runner_sha256": runner_sha256,
        },
        "result": "pass",
        "accessibility": {
            "clickable_controls_named": True,
            "checked_internal_terms": list(INTERNAL_TERMS),
            "common_info_sections_not_duplicated": True,
        },
        "scenes": records,
        "screenshots": sorted(path.name for path in args.output.glob("*.png")),
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: Android Gallery semantic controls and help on {model}")


if __name__ == "__main__":
    main()
