# Physics-Informed Neural Network for Relativistic Compton Scattering

A PyTorch implementation of a Physics-Informed Neural Network (PINN) for learning the kinematics of relativistic Compton scattering from sparse noisy data. The model combines supervised learning with physics-based constraints to produce predictions that satisfy fundamental conservation laws.

The synthetic dataset is generated analytically from the relativistic Compton scattering equations, while the neural network is trained using both data loss and physics-informed loss.

---

## Physics Model

The project considers the scattering of a photon by a stationary electron. All quantities are expressed in normalized units where the electron rest-mass energy is

**mc² = 1**

### Scattered Photon Energy

The scattered photon energy is calculated using the Compton scattering equation

E′ = E / (1 + E(1 − cosφ))

where

- E is the incident photon energy
- E′ is the scattered photon energy
- φ is the photon scattering angle

### Electron Kinetic Energy

The recoil electron kinetic energy is obtained from energy conservation

K = E − E′

### Electron Momentum

The electron momentum is calculated from the relativistic energy-momentum relation

(pc)² = (K + 1)² − 1

### Electron Scattering Angle

The recoil angle is computed using momentum conservation

tan(θe) = E′ sinφ / (E − E′ cosφ)

These analytical equations are used to generate the synthetic dataset.

---

## Physics-Informed Loss

The neural network predicts

- Scattered photon energy (E′)
- Electron kinetic energy (K)
- Electron momentum (p)
- Electron scattering angle (θe)

Instead of minimizing only the prediction error, the PINN also enforces the following physical constraints during training:

### Energy Conservation

E = E′ + K

### Momentum Conservation (x-direction)

E = E′ cosφ + p cosθe

### Momentum Conservation (y-direction)

0 = E′ sinφ − p sinθe

### Relativistic Energy-Momentum Relation

(K + 1)² = (pc)² + 1

The total training loss is

Total Loss = Data Loss + λ × Physics Loss

where λ controls the contribution of the physics-informed regularization.

---

## Features

- Physics-Informed Neural Network implemented in PyTorch
- Analytical Compton scattering dataset generation
- Sparse noisy supervised training data
- Physics anchor points
- Unlabeled collocation points
- Multiple neural network architectures
- Early stopping
- Multiple random seeds for reproducibility
- RMSE and R² evaluation
- Physics residual analysis
- Extrapolation testing

---

## Evaluation

The trained models are evaluated using

- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
- Energy conservation residual
- Momentum conservation residual
- Relativistic energy-momentum residual

These metrics measure both predictive accuracy and physical consistency.

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

**Research Interests**

- Computational Physics
- Machine Learning
- High Energy Physics
- Quantum Computing
- Physics-Informed Neural Networks
