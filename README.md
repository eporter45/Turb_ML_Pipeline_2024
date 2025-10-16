# 🌪️ Turbulence Modeling Pipeline (2024)

A first-generation pipeline for **data-driven turbulence modeling** using supervised and neural approaches on the  
[Kaggle ML Turbulence Dataset](https://www.kaggle.com/datasets/ryleymcconkey/ml-turbulence-dataset).

This project represents the **2024 iteration** of my turbulence modeling research — focused on building, training, and evaluating multiple machine learning architectures for predicting flow-field properties in turbulent regimes.

---

## 🎯 Project Overview

The goal of this project was to design an **end-to-end turbulence modeling workflow** that leverages open data and modern ML techniques to approximate turbulence statistics, velocity fields, and energy dissipation rates.

Using the Kaggle ML Turbulence Dataset, I constructed a modular pipeline that:
- Ingests and preprocesses large-scale simulation data  
- Engineers spatial and physical features relevant to turbulent flow  
- Trains multiple ML architectures for regression and pattern discovery  
- Benchmarks predictive accuracy and generalization behavior  

This marks the **first iteration** of my turbulence modeling work — serving as a prototype for more physics-informed, hybrid ML–CFD systems planned for future versions.

---

## 🧠 Modeling Approaches

| Category | Example Models | Description |
|-----------|----------------|--------------|
| **Classical ML** | Linear Regression, Random Forest, XGBoost | Baseline models for interpretability and feature relevance |
| **Neural Networks** | Fully Connected NN, CNNs | Learned mappings from flow field features to turbulent quantities |
| **Hybrid / Experimental** | Dimensionality-scaled networks | Early exploration of physics-informed regularization techniques |

Each model was evaluated on metrics such as RMSE, MAE, and R² across test splits, with visualization of flow predictions and error maps.

---

## 🧰 Tools & Environment

**Language:** Python  
**Frameworks:** PyTorch, Scikit-learn, NumPy, pandas, Matplotlib  
**Hardware:** Local GPU (CUDA) + Jupyter Lab workflow  

**Recommended Environment Setup**
```bash
conda create -n turb_pipeline python=3.11 numpy pandas matplotlib scikit-learn torch jupyterlab
conda activate turb_pipeline
