# SPDX-License-Identifier: MIT
from pathlib import Path
import re
import unittest


COMPONENT = Path(__file__).resolve().parents[1]


def runner_sources(cmake_text: str, root: Path) -> list[Path]:
    match = re.search(
        r"add_library\(parallel-mater-conformance-runner\s+STATIC\s+(.*?)\)",
        cmake_text,
        re.DOTALL,
    )
    if match is None:
        raise ValueError("missing conformance runner target")
    paths = []
    for token in match.group(1).split():
        path = (root / token).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(f"runner source escapes conformance component: {token}") from error
        paths.append(path)
    return paths


class SourceClosureTests(unittest.TestCase):
    def test_runner_sources_are_present_inside_component(self):
        sources = runner_sources(
            (COMPONENT / "CMakeLists.txt").read_text(encoding="utf-8"),
            COMPONENT,
        )
        self.assertGreater(len(sources), 0)
        for source in sources:
            self.assertTrue(source.is_file(), source)

    def test_parent_relative_runner_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "escapes conformance component"):
            runner_sources(
                "add_library(parallel-mater-conformance-runner STATIC "
                "../../tests/api/private.cpp)",
                COMPONENT,
            )

    def test_runner_local_includes_are_present_inside_component(self):
        cmake = (COMPONENT / "CMakeLists.txt").read_text(encoding="utf-8")
        for source in runner_sources(cmake, COMPONENT):
            text = source.read_text(encoding="utf-8")
            for include in re.findall(r'^\s*#\s*include\s*"([^"]+)"', text, re.MULTILINE):
                header = (source.parent / include).resolve()
                self.assertTrue(header.is_relative_to(COMPONENT), include)
                self.assertTrue(header.is_file(), header)


if __name__ == "__main__":
    unittest.main()
