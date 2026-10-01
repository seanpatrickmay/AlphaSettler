"""Catanatron adapter: the only code that imports Catanatron (GPL-3.0, imported at runtime,
never copied). Submodules import it; this module does not, so callers can read the install
requirement when Catanatron is missing."""

import os
from contextlib import contextmanager

CATANATRON_REQUIREMENT = (
    "catanatron @ git+https://github.com/bcollazo/catanatron.git@ecf931181b9a65bb4116a2153fb78c16f1438e00"
)


def catanatron_available() -> bool:
    """Whether Catanatron is installed. The import happens here, when called, never when this
    module loads, so a caller can tell a missing oracle from a bug in the oracle's own modules.
    Only Catanatron itself being absent counts as missing; any other import error (a broken
    install, a missing dependency of Catanatron) propagates."""
    import importlib

    try:
        importlib.import_module("catanatron")
    except ModuleNotFoundError as e:
        if e.name != "catanatron":
            raise
        return False
    return True


@contextmanager
def hash_seed_zero():
    """PYTHONHASHSEED=0 while spawn workers start inside the block (they read it at startup;
    Catanatron's move order depends on it), then the parent's own value is restored."""
    previous = os.environ.get("PYTHONHASHSEED")
    os.environ["PYTHONHASHSEED"] = "0"
    try:
        yield
    finally:
        if previous is None:
            del os.environ["PYTHONHASHSEED"]
        else:
            os.environ["PYTHONHASHSEED"] = previous
