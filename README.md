# FFForge (atomate2_ffforge)

**FFForge** is a modular add-on for the [atomate2](https://github.com/materialsproject/atomate2)
workflow engine that turns raw structures into production-ready
machine-learning force fields *automatically*:

INPUT structures → DIRECT sampler → DFT single-points
▲ |
└──────── active-learning loop ◄──┘


Key features
------------

* **DIRECT sampling core**  
  * Graph/GNN, SOAP, or composition-fallback encoders  
  * PCA whitening + BIRCH clustering + deterministic stratified pick  
  * Domain presets for cathodes, polar perovskites, catalysts, polymers,
    electrolytes

* **End-to-end jobflow**  
  * VASP single-point Maker, PANNA trainer, MD/extrapolation monitor  
  * Local vs. SLURM queue policies and NERSC SF-API launcher ready out-of-the-box

* **Active-learning metrics API**  
  * Real-time learning-curve forecast exposed via FastAPI for web dashboards

* **Dev-friendly**  
  * PEP-8 compliant, Black/Ruff/Mypy gated by pre-commit  
  * `pip install -e .[dev]` brings a full test suite (`pytest -q`) online in minutes

Why FFForge?
------------

> *“Getting a robust MLIP shouldn’t require weeks of manual dataset curation or
> cluster-shell gymnastics.”*

FFForge front-loads diversity with DIRECT, then closes the gap with a
data-efficient active learner—so you spend your CPU/GPU budget only where it
matters.

---

### Installation (dev)

```bash
git clone git@github.com:rsilvabuarque/atomate2_ffforge.git
cd atomate2_ffforge
conda env create -f environment.yml -p ~/conda_envs/ffforge
conda activate ~/conda_envs/ffforge
pip install -e .[dev]
pytest -q          # sanity check

Quick Start

from atomate2_ffforge.flows import ff_initial_flow
from pymatgen.core import Structure

struct = Structure.from_file("LiFePO4.cif")
flow = ff_initial_flow(tag="battery", seed_structure=struct, dft_budget=2000)
flow.run()  # launches DIRECT → DFT → PANNA
