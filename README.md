# Physics-Informed Neural Network for Learning Relativistic Compton Scattering Kinematics

A Physics-Informed Neural Network (PINN) implemented in PyTorch to learn the kinematics of relativistic Compton scattering from sparse noisy data while enforcing fundamental physical conservation laws.

The model combines supervised learning with physics-based regularization, enabling it to produce physically consistent predictions even when training data are limited.

---

## Project Overview

Compton scattering is one of the fundamental interactions between photons and electrons. Instead of relying solely on experimental data, this project incorporates known physical laws directly into the neural network training process.

The network predicts key kinematic quantities of Compton scattering while minimizing both data error and violations of physical constraints.

This project demonstrates the application of Physics-Informed Machine Learning to a relativistic scattering problem.

---

## Physics Background

The dataset is generated from the relativistic Compton scattering equations using normalized units where

- Electron rest mass energy: \(m_ec^2 = 1\)
- Photon energies are normalized by \(m_ec^2\)

The generated quantities include

- Scattered photon energy
- Electron kinetic energy
- Electron momentum
- Electron scattering angle

During training, the network is constrained using fundamental conservation laws instead of relying only on labeled data.

---

## Physics Constraints

The physics-informed loss includes

- Energy conservation
- Momentum conservation (x-direction)
- Momentum conservation (y-direction)
- Relativistic energy-momentum relation

The total loss is

\[
L = L_{\text{data}} + \lambda L_{\text{physics}}
\]

where

- \(L_{\text{data}}\) is the supervised regression loss
- \(L_{\text{physics}}\) penalizes violations of physical laws
- \(\lambda\) controls the strength of the physics regularization

---

## Neural Network

The project compares several fully connected neural network architectures with different depths and widths.

Features:

- PyTorch implementation
- Multiple hidden-layer configurations
- ReLU activation
- Adam optimizer
- Early stopping
- Multiple random seeds for reproducibility

---

## Dataset

Synthetic data are generated directly from the analytical Compton scattering equations.

The dataset includes

- Training samples
- Sparse physics anchor points
- Unlabeled collocation points
- Extrapolation test data

Gaussian noise is added to simulate measurement uncertainty.

---

## Evaluation Metrics

Model performance is evaluated using

- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
- Energy conservation residual
- Momentum conservation residual
- Relativistic consistency residual

These metrics evaluate both predictive accuracy and physical consistency.

---

## Repository Structure

```
.
├── Compton_PINN.ipynb
├── README.md
├── requirements.txt
├── figures/
└── results/
```

---

## Requirements

```
torch
numpy
pandas
matplotlib
scikit-learn
```

Install dependencies with

```bash
pip install -r requirements.txt
```

---

## Running the Project

Open the notebook

```bash
jupyter notebook Compton_PINN.ipynb
```

Run all cells to

- Generate the dataset
- Train the neural network
- Compare PINN and baseline models
- Visualize predictions
- Evaluate physical consistency

---

## Results

The Physics-Informed Neural Network produces predictions that satisfy conservation laws more consistently than a purely data-driven neural network while maintaining competitive prediction accuracy.

The project illustrates how incorporating physical knowledge into machine learning can improve model reliability and extrapolation.

---

## Future Improvements

Possible extensions include

- Predicting only independent physical variables
- Alternative physics-loss weighting strategies
- Bayesian uncertainty estimation
- Comparison with additional neural network architectures
- Extension to other particle interaction problems

---

## Technologies

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Author

**Pritam Dhakal**

M.Sc. Physics

Research interests:
- Computational Physics
- Machine Learning
- Particle Physics
- Physics-Informed Neural Networks
