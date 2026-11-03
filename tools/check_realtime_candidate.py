#!/usr/bin/env python3
"""Validate the active realtime qualification candidate."""

from check_realtime_candidate_v10 import (
    ACTIVE_CI_RUN_ID,
    ACTIVE_ID,
    ACTIVE_SOURCE_REVISION,
    ACTIVE_SOURCE_TREE,
    CANDIDATE_PATH,
    REQUIRED_CLOSURES,
    V9_ID,
    V9_SHA256,
    main,
    validate_candidate,
)


__all__ = [
    "ACTIVE_CI_RUN_ID",
    "ACTIVE_ID",
    "ACTIVE_SOURCE_REVISION",
    "ACTIVE_SOURCE_TREE",
    "CANDIDATE_PATH",
    "REQUIRED_CLOSURES",
    "V9_ID",
    "V9_SHA256",
    "validate_candidate",
]


if __name__ == "__main__":
    raise SystemExit(main())
