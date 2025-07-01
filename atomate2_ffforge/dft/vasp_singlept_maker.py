"""
Maker that creates a VASP single-point static calculation flow, with smart
queue selection (local or SLURM).

Usage
-----
>>> maker = VaspSinglePointMaker(q_override="local")
>>> flow  = maker.make([structure1, structure2])
"""

from __future__ import annotations

from typing import Sequence, Literal

from pymatgen.core import Structure
from jobflow import Flow
from atomate2.vasp.flows.core import StaticMaker

from .queue_policies import queue_adapter


class VaspSinglePointMaker(StaticMaker):
    """
    Thin wrapper on atomate2 StaticMaker that injects a QueueAdapter.

    Parameters
    ----------
    q_override
        Force job to run 'local' or 'slurm'.  None → auto based on label.
    """

    label: str = "DFT"  # used by queue_policies
    name: str = "VaspSinglePointMaker"

    def __init__(
        self,
        *,
        q_override: Literal["local", "slurm", None] = None,
        **static_kwargs,
    ):
        super().__init__(**static_kwargs)
        self.q_override = q_override

    # ---------------------------------------------------------------------
    # jobflow entry point
    # ---------------------------------------------------------------------
    def make(
        self,
        structures: Sequence[Structure],
        **kw,
    ) -> Flow:
        """
        Build a jobflow Flow that runs VASP static calculations.

        Parameters
        ----------
        structures
            Sequence of pymatgen Structure objects.
        kw
            Additional keyword args forwarded to parent `.make`.
        """
        qadapter = queue_adapter(self.label, self.q_override)
        return super().make(
            structures=structures,
            queue_adapter=qadapter,
            **kw,
        )
