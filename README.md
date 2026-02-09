# Superformula Geometric Fusion (SGF) for Robust Multimodal Anomaly Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

**Implementation of the paper: "Geometric Regularization for Safe Multimodal Fusion in Industrial Robotics"**

This repository contains the PyTorch implementation of **Superformula Geometric Fusion (SGF)**, a novel autoencoder architecture that enforces intrinsic safety constraints on the latent space of multimodal systems. By parameterizing the fusion manifold using the differentiable **Gielis Superformula**, SGF ensures global boundedness, Lipschitz stability, and topological separability of anomalies.

---

## 🎥 Demo & Visualization

### Learned Manifold Evolution
Watch how the latent space evolves from an initial isotropic sphere to a complex, data-driven supershape that captures the joint topology of visual and kinematic data.

![Manifold Evolution](manifold_evolution.gif)

### Anomaly Detection via Topological Voids
When sensor data conflicts with visual input (e.g., dead sensor), the geometric constraints force the representation to collapse into disjoint "voids."

[Clean t-SNE](tsne.png)

---

## 🚀 Key Features

* **Geometric Regularization:** Replaces standard Euclidean latent spaces with a closed, bounded Riemannian manifold defined by learnable Gielis parameters $\theta_G = \{m, n_1, n_2, n_3\}$.
* **Intrinsic Safety:** Mathematically guaranteed **Global Boundedness** (Theorem 4.1) prevents latent explosion under catastrophic input failure.
* **Lipschitz Stability:** The curvature of the supershape actively dampens input noise, ensuring **Lipschitz Continuity** (Theorem 4.2).
* **Unsupervised Anomaly Detection:** Detects logical inconsistencies (e.g., frozen sensors) by measuring **Volumetric Collapse** (Theorem 4.3) without requiring labeled failure data.

---

## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/diyarino/superformula-geometric-fusion.git](https://github.com/diyarino/superformula-geometric-fusion.git)
    cd superformula-geometric-fusion
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *Key requirements:* `torch`, `numpy`, `matplotlib`, `scikit-learn`, `tqdm`.

---

## 📂 Dataset

This framework is benchmarked on high-fidelity industrial digital twins (MuJoCo, ABB Single-Arm, ABB Dual-Arm) as described in:

Altinses, D., & Schwung, A. (2025). Performance benchmarking of multimodal data-driven approaches in industrial settings. Machine Learning with Applications, 100691.


---


## 🤝 Citation

If you use this code or the SGF architecture in your research, please cite our paper:

```bibtex
@article{altinses2026SGF,
  title={Geometric Regularization for Safe Multimodal Fusion in Industrial Robotics},
  author={Diyar Altinses and Andreas Schwung},
  journal={arXiv preprint arXiv:26XX.XXXXX},
  year={2026}
}

```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

```



