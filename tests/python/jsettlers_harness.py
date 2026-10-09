"""Tests that need the JSettlers harness call `require()`: a missing jar or build fails the test with
the command that provides it (never a skip; the full suite must run everything)."""

import pytest

from oracle.jsettlers import missing


def require(main: str = "ArenaMain") -> None:
    why = missing(main)
    if why is not None:
        pytest.fail(f"the JSettlers harness is needed: {why}")
