#!/usr/bin/env python3
import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts/Test-AndroidGalleryPerformance.py"
SPEC = importlib.util.spec_from_file_location("android_gallery_performance", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AndroidGalleryPerformanceTests(unittest.TestCase):
    def test_frame_window_parser(self):
        values = MODULE.parse_frame_windows(
            "FrameWindow scene=Fluid.glb backend=vulkan frames=119 count=60 "
            "mean_ms=120.500 median_ms=119.000 p95_ms=140.250 "
            "physics_ms=117.000 render_ms=3.500 present_ms=0.000 "
            "fps=7.75 particles=128 bodies=4")
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0]["scene"], "Fluid.glb")
        self.assertEqual(values[0]["count"], 60)
        self.assertEqual(values[0]["particles"], 128)
        self.assertEqual(values[0]["p95_ms"], 140.25)
        self.assertEqual(values[0]["fps"], 7.75)

    def test_scene_ready_delimits_each_measurement(self):
        log = """FrameWindow scene=Old.glb backend=vulkan frames=59 count=60 mean_ms=1.0 median_ms=1.0 p95_ms=1.0 physics_ms=1.0 render_ms=0.0 present_ms=0.0 fps=60.0 particles=0 bodies=1
Scene ready: Current.glb; backend=vulkan; device=GPU; bodies=2; particles=3
FrameWindow scene=Current.glb backend=vulkan frames=59 count=60 mean_ms=2.0 median_ms=2.0 p95_ms=2.0 physics_ms=1.0 render_ms=1.0 present_ms=0.0 fps=30.0 particles=3 bodies=2
"""
        ready = list(MODULE.SCENE_READY.finditer(log))
        values = MODULE.parse_frame_windows(log[ready[-1].start():])
        self.assertEqual(ready[-1].group("scene"), "Current.glb")
        self.assertEqual([item["scene"] for item in values], ["Current.glb"])

    def test_thermal_parser_uses_current_hal_section(self):
        state = MODULE.parse_thermal("""Thermal Status: 0
Cached temperatures:
 Temperature{mValue=90.0, mType=1, mName=G3D, mStatus=3}
Current temperatures from HAL:
 Temperature{mValue=40.0, mType=1, mName=G3D, mStatus=0}
 Temperature{mValue=25.0, mType=2, mName=battery, mStatus=0}
Current cooling devices from HAL:
 CoolingDevice{mValue=0, mType=3, mName=thermal-gpufreq-0}
""")
        self.assertEqual(state["status"], 0)
        self.assertEqual(state["temperatures_c"]["G3D"], 40.0)
        self.assertEqual(state["sensor_statuses"]["G3D"], 0)


if __name__ == "__main__":
    unittest.main()
