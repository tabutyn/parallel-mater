#!/usr/bin/env python3
"""Generate link-complete no-op wrappers for unadvertised CPU domains."""

from __future__ import annotations

import argparse
import pathlib
import re


DECLARATION = re.compile(
    r"(?P<declaration>(?:\[\[nodiscard\]\]\s+)?(?:void|PmResult)\s+"
    r"[A-Za-z_][A-Za-z0-9_]*\s*\(.*?\)\s*noexcept)\s*;",
    re.DOTALL,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("headers", nargs="+", type=pathlib.Path)
    arguments = parser.parse_args()

    definitions: list[str] = []
    includes: list[str] = []
    for header in arguments.headers:
        includes.append(f'#include "simulation/{header.name}"')
        source = header.read_text(encoding="utf-8")
        for match in DECLARATION.finditer(source):
            declaration = re.sub(
                r"\s*=\s*nullptr", "", match.group("declaration")
            )
            return_type = declaration.replace("[[nodiscard]]", "", 1).lstrip()
            body = " {}" if return_type.startswith("void ") else (
                " { return PM_ERROR_FEATURE_NOT_PRESENT; }")
            definitions.append(declaration + body)

    if not definitions:
        raise RuntimeError("no dispatch declarations found")

    output = "\n".join([
        "// SPDX-License-Identifier: MIT",
        "// Generated wrappers for capabilities the serial driver does not advertise.",
        *includes,
        "",
        "namespace parallel_mater::simulation {",
        "",
        "\n\n".join(definitions),
        "",
        "} // namespace parallel_mater::simulation",
        "",
    ])
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
