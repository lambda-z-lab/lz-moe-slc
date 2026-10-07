# Stochastic Lagrangian Control and Proportional Bias Routing for Mixture-of-Experts

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch CPU](https://img.shields.io/badge/PyTorch-CPU%20Compatible-orange.svg)](https://pytorch.org/)
[![DeepSeek](https://img.shields.io/badge/Baseline-DeepSeek-4D6BFE)](https://github.com/lambda-z-lab/lz-moe-slc)
[![Deep Learning](https://img.shields.io/badge/Paradigm-Deep%20Learning-success)](https://github.com/lambda-z-lab/lz-moe-slc)
[![Control Theory](https://img.shields.io/badge/Domain-Stochastic%20Control-blueviolet)](https://github.com/lambda-z-lab/lz-moe-slc)
[![Optimization](https://img.shields.io/badge/Field-Constrained%20Optimization-informational)](https://github.com/lambda-z-lab/lz-moe-slc)
[![Lambda-Z-Lab](https://img.shields.io/badge/Research-Lambda--Z--Lab-purple.svg)](https://github.com/lambda-z-lab)

Official research paper draft and companion reference implementation for:  
> **"Stochastic Lagrangian Control and Proportional Bias Routing for Mixture-of-Experts"**

---

## 📄 Read the Paper

You can read the full pre-print draft directly from this repository:
* **[Download/View Current Draft (PDF)](main.pdf)**

### Abstract
This paper presents a formalization and comparative analysis of auxiliary-loss-free routing mechanisms for sparsely-gated Mixture-of-Experts (MoE) layers, introducing Stochastic Augmented Lagrangian Control (SLC) as a rigorous optimization framework and benchmarking it against existing proportional bias heuristics. Standard MoE routing relies on empirical auxiliary losses that lack formal convergence guarantees and suffer from extreme sensitivity to fixed weighting coefficients. We formulate MoE load balancing as a constrained utility-maximization problem and analyze two distinct structural solutions: (i) our proposed Lagrangian router, which solves the associated saddle-point problem via a robust two-timescale scheme featuring momentum-filtered dual ascent and slack filtering; and (ii) a proportional bias router inspired by DeepSeek-V3 that adjusts per-expert selection biases via a mean-preserving feedback integrator driven by hard token assignments. By establishing a formal economic correspondence where the negative of the router bias acts as a heuristic dual price, we unify both architectures under a single primal-dual convergence framework. Finally, we demonstrate that the stability of SLC under local mini-batch updates uniquely facilitates rigorous Mixture-of-Experts research on memory-limited consumer hardware.

---

## 💻 Installation & Environment Setup

1. **Clone the repository and set up a virtual environment:**
   ```bash
   python -m venv --system-site-packages .venv
   source .venv/bin/activate
   pip install --upgrade pip
   ```

2. **Install the package and CPU-only PyTorch:**
   ```bash
   pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu --no-cache-dir
   ```

---

## 🚀 Quick Start & Benchmarking

You can verify the routing mechanisms and test stochastic load-skew recovery directly through interactive Python (`IPython`):

```bash
python -m IPython
```

```python
In [1]: import torch
In [2]: from benchmarks.benchmark_routers import run_benchmark
In [3]: torch.manual_seed(42)
   ...: run_benchmark("SLC", steps=150)
   ...: run_benchmark("DeepSeekBias", steps=150)
```

### Expected Benchmark Output
```text
--- Benchmarking Router Variant: SLC ---
Step [50/150]  | Primal: 0.3029 | Dual: 0.3036 | Gap: 0.0007
Step [100/150] | Primal: 0.3026 | Dual: 0.3025 | Gap: 0.0001
Step [150/150] | Primal: 0.3023 | Dual: 0.3024 | Gap: 0.0000
Completed 150 steps in 0.3387 seconds.

--- Benchmarking Router Variant: DeepSeekBias ---
Step [50/150]  | Primal: 0.3399 | Dual: 0.3274 | Gap: 0.0124
Step [100/150] | Primal: 0.3392 | Dual: 0.3374 | Gap: 0.0018
Step [150/150] | Primal: 0.3386 | Dual: 0.3395 | Gap: 0.0010
Completed 150 steps in 0.3406 seconds.
```

---

## 🧪 Running Unit Tests

To validate tensor shapes, unit-sphere centroid tracking, and calibration diagnostics:
```bash
pytest tests/test_routers.py
```

---

## 📖 Citation

If you build upon this research, utilize the theoretical framework, or reference this code base, please cite it using the following BibTeX entry:

```bibtex
@article{rehn2026stochastic,
  title={Stochastic Lagrangian Control and Proportional Bias Routing for Mixture-of-Experts},
  author={Rehn, Carl Johan},
  journal={arXiv preprint arXiv:26xx.xxxxx},
  year={2026}
}

@software{rehn2026lzmoeslc,
  author = {Rehn, Carl Johan},
  title = {Stochastic Lagrangian Control and Proportional Bias Routing for Mixture-of-Experts: Companion Code},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.23214045},
  url = {https://github.com/lambda-z-lab/lz-moe-slc}
}
```

---

## 🛡️ License

This project is open-source under the terms of the [MIT License](LICENSE).

