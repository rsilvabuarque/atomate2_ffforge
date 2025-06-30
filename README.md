# FFForge

**FFForge** is a plug‑in for the [`atomate2`](https://github.com/materialsproject/atomate2) / [`jobflow`](https://github.com/materialsproject/jobflow) ecosystem that *automatically* turns raw crystal, slab, polymer or electrolyte structures into production‑ready machine‑learning interatomic potentials (MLIPs).

<p align="center">
  <img src="docs/assets/ffforge_workflow.png" alt="FFForge active‑learning workflow" width="80%">
</p>

---

## ✨ Key Features

| Layer | Highlights |
|-------|------------|
| **DIRECT sampling** | • Graph/GNN latent, SOAP, or composition‑fallback encoders  <br>• PCA whitening & BIRCH clustering  <br>• Deterministic, stratified subset selection |
| **DFT wrapper** | • VASP single‑point Maker (other codes pluggable)  <br>• Queue policy: local vs. SLURM/SF‑API compute nodes |
| **MLIP training** | • PANNA as default engine (ACE, NequIP coming)  <br>• Learning‑curve forecast (E<sub>MAE</sub>, F<sub>RMSE</sub>) |
| **Active learning** | • Uncertainty‑driven MD loop with novelty check  <br>• Real‑time metrics exposed via FastAPI for dashboards |
| **Dev‑friendly** | • PEP‑8, *black*, *ruff*, *mypy* gated by pre‑commit  <br>• One‑command dev install: `pip install -e .[dev]` |

---

## 🚀 Quick Start (developer install)

```bash
git clone git@github.com:rsilvabuarque/atomate2_ffforge.git
cd atomate2_ffforge

conda env create -f environment.yml -p ~/conda_envs/ffforge
conda activate ~/conda_envs/ffforge

pip install -e .[dev]
pytest -q
```

### Minimal example

```python
from atomate2_ffforge.flows import ff_initial_flow
from pymatgen.core import Structure

struct = Structure.from_file("LiFePO4.cif")

flow = ff_initial_flow(tag="battery", seed_structure=struct, dft_budget=2000)
flow.run()   # DIRECT → VASP → PANNA
```

---

## 🏛️ Architecture

```
input_gen/     ← system‑specific structure builders
sampling/      ← DIRECT (encode → PCA → BIRCH → pick)
dft/           ← VASP single‑point Maker + parsers
mlip/          ← PANNA trainer + MD driver
al/            ← active‑learning loop & metrics
nersc/         ← SF‑API queue launcher
flows/         ← user‑facing jobflow entry points
```

Supported **domain tags**:

```
battery            Liₓ cathodes / anodes
polar_perovskite   ferroelectric slabs & interfaces
surface_cat        heterogeneous catalysts
polymer            DMA fragments, strained
electrolyte        SeaUrchin solvation clusters
```

---

## 🤝 Contributing

1. Fork & clone  
2. `pre-commit install`  
3. Create a feature branch (`git checkout -b feat/<topic>`)  
4. Format / lint / type‑check (`black`, `ruff`, `mypy`)  
5. Add tests under `tests/`  
6. Open a PR – CI must be green

---

## 📜 Citation

```text
@software{ffforge_2025,
  author       = {R. S. Buarque and contributors},
  title        = {{FFForge}: Machine‑Learning Force‑Field Workflows for atomate2},
  year         = 2025,
  url          = {https://github.com/rsilvabuarque/atomate2_ffforge},
}
```

---

## 📝 License

Licensed under the **MIT License**. See `LICENSE` for details.
