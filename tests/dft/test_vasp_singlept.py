# ruff: noqa: E402
"""
Unit-tests for VaspSinglePointMaker without importing real atomate2.
"""

from __future__ import annotations

import sys
import types

# ---------------------------------------------------------------------------
# Build fake atomate2 -> vasp.flows.core.StaticMaker
# ---------------------------------------------------------------------------
core_mod = types.ModuleType("atomate2.vasp.flows.core")


class DummyStaticMaker:  # noqa: D101  (docstring not needed in test)
    def __init__(self, *args, **kwargs): ...
    def make(self, *args, **kwargs):
        return "dummy"


core_mod.StaticMaker = DummyStaticMaker  # type: ignore[attr-defined]
sys.modules["atomate2"] = types.ModuleType("atomate2")
sys.modules["atomate2.vasp"] = types.ModuleType("atomate2.vasp")
sys.modules["atomate2.vasp.flows"] = types.ModuleType("atomate2.vasp.flows")
sys.modules["atomate2.vasp.flows.core"] = core_mod

from pymatgen.core import Lattice, Element, Structure
from atomate2_ffforge.dft.vasp_singlept_maker import VaspSinglePointMaker  # noqa: E402


def _toy():
    return Structure(Lattice.cubic(4.0), [Element("Li")], [[0, 0, 0]])


def test_local_override():
    maker = VaspSinglePointMaker(q_override="local")
    assert maker.make([_toy()]) == "dummy"


def test_auto_route():
    maker = VaspSinglePointMaker()
    assert maker.make([_toy()]) == "dummy"
