# Neural Cosmology Reconstruction

A non-parametric cosmological reconstruction project using Artificial Neural Networks (ANNs) to learn the Hubble parameter H(z) directly from observational data.

This project explores modern machine learning techniques applied to computational cosmology, focusing on model-independent reconstruction of the cosmic expansion history without explicitly imposing a fiducial cosmological model during training.

---

# Overview

Traditional cosmological analyses usually assume a specific cosmological model such as:

- LambdaCDM
- wCDM
- CPL parametrizations
- Phantom Dark Energy scenarios

and then estimate cosmological parameters through Bayesian inference techniques such as:

- Markov Chain Monte Carlo (MCMC)
- Nested Sampling
- Fisher Matrix Forecasting

In contrast, this project investigates a purely data-driven reconstruction approach.

Instead of fitting cosmological parameters directly, the neural network learns the mapping:

```text
z -> H(z)
```

where:

- z is the cosmological redshift
- H(z) is the Hubble expansion parameter

The reconstruction is performed directly from observational measurements using TensorFlow/Keras.

---

# Scientific Motivation

Modern cosmology faces several open problems:

- Hubble tension
- Dark energy nature
- Possible deviations from LambdaCDM
- Phantom Dark Energy scenarios
- Model-dependent reconstruction biases

Non-parametric reconstruction methods provide an alternative way to study the expansion history of the Universe while minimizing assumptions about the underlying cosmological model.

Artificial Neural Networks offer a powerful framework for learning smooth functional relationships from sparse and noisy astrophysical datasets.

---

# Methodology

The ANN receives cosmological redshift values as input and reconstructs the corresponding Hubble parameter.

## Input

```text
z
```

## Output

```text
H(z)
```

The network is trained directly on observational Hubble data.

No fiducial cosmological model is explicitly imposed during training.

---

# Neural Network Architecture

The reconstruction uses a fully connected feedforward neural network:

```python
model = Sequential([

    Input(shape=(1,)),

    Dense(256, activation='elu'),
    Dense(256, activation='elu'),
    Dense(256, activation='elu'),

    Dense(1)

])
```

---

# Activation Function

The ELU (Exponential Linear Unit) activation function is adopted.

ELU activation provides:

- smooth gradients
- stable convergence
- improved regression performance
- better behavior for continuous cosmological functions

---

# Loss Function

The model is trained using the Mean Absolute Error (MAE).

MAE is more robust against outliers compared to MSE and performs well for sparse cosmological datasets.

---

# Training Strategy

The project uses:

- Adam optimizer
- Early stopping regularization
- Training/validation split
- Progress monitoring with tqdm

Early stopping prevents overfitting and restores the best network weights.

---

# Features

- Non-parametric cosmological reconstruction
- TensorFlow/Keras implementation
- Direct training on observational data
- Model-independent ANN reconstruction
- Smooth continuous interpolation
- Training and validation loss monitoring
- Cosmological visualization plots
- tqdm training progress bar
- Modular neural network architecture

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your_username/neural-cosmology-reconstruction.git
```

Enter the project directory:

```bash
cd neural-cosmology-reconstruction
```

Install dependencies:

```bash
pip install numpy pandas matplotlib tensorflow tqdm scipy
```

---

# Dependencies

Main libraries used in this project:

- TensorFlow
- NumPy
- Pandas
- Matplotlib
- tqdm
- SciPy

---

# Usage

Run the notebook:

```bash
jupyter notebook
```

or execute the Python script:

```bash
python neural.py
```

---

# Dataset

The project uses observational measurements of the Hubble parameter H(z), typically obtained from:

- Cosmic Chronometers
- BAO measurements
- Galaxy surveys

Dataset structure:

| z | H(z) | sigma_H |
|---|---|---|
| redshift | Hubble parameter | observational uncertainty |

---

# Reconstruction Pipeline

The reconstruction workflow is:

## 1. Load observational data

```python
df = pd.read_table("Hz.txt")
```

## 2. Split train/test samples

```python
train/test split
```

## 3. Train ANN

```python
model.fit(...)
```

## 4. Reconstruct continuous expansion history

```python
H_rec = model.predict(z_grid)
```

## 5. Visualize reconstruction

- observational data
- ANN reconstruction
- training loss
- validation loss

---

# Example Outputs

The code generates:

- reconstructed H(z) curves
- observational comparison plots
- training loss evolution
- validation loss evolution

---

# Scientific Applications

This project can be extended toward:

- Dark Energy reconstruction
- Phantom cosmology analysis
- w(z) reconstruction
- Gaussian Process comparisons
- Bayesian Neural Networks
- Physics-Informed Neural Networks (PINNs)
- Simulation-Based Inference (SBI)
- Cosmological emulators
- Multi-probe cosmological reconstruction

---

# Future Improvements

Planned future developments include:

- covariance matrix incorporation
- Bayesian uncertainty estimation
- ensemble neural networks
- Gaussian Process comparison
- neural posterior estimation
- normalizing flows
- PINNs for Friedmann equations
- reconstruction of dynamical dark energy models

---

# Results Interpretation

The ANN reconstructs smooth approximations for the cosmic expansion history directly from observational measurements.

This approach avoids explicitly assuming a cosmological model during training and enables exploratory studies of:

- expansion history
- dark energy behavior
- model-independent cosmology

---

# Research Context

This project is related to current research topics in:

- computational cosmology
- machine learning in astrophysics
- cosmological reconstruction
- observational cosmology
- dark energy phenomenology
- non-parametric inference

---

# Repository Structure

```text
.
├── neural.ipynb
├── neural.py
├── Hz.txt
├── README.md
└── figures/
```

---

# Author

Adriel de Oliveira Aquino

MSc student working in theoretical and computational cosmology, focusing on:

- dark energy phenomenology
- Bayesian inference
- cosmological reconstruction
- machine learning in cosmology
- computational astrophysics

---

# License

This project is intended for academic and research purposes.
