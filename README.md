# ScratchNet Studio (v3.1)
**Author:** MD Saifullah Baig.A  
**Version:** 2.0  
**Status:** Active

**A Professional, Modular Deep Learning Framework built from scratch in Python.**

## 📌 Overview
ScratchNet Studio is a lightweight, education-focused Neural Network framework that implements Deep Learning fundamentals—including **Backpropagation**, **Optimizers (Adam/SGD)**, and **Activation Functions**—using only NumPy.

This version (v3.1) introduces a modular architechture for gui implementation

## 🚀 Key Features (New in v3)

### 1. Vectorized Neural Engine (v3.0)
* **High Performance:** Replaced loop-based learning with **Matrix Operations** (Vectorization), allowing for instant predictions and faster training.
* **Mini-Batch Gradient Descent:** Updates weights using stable batches (default: 10) instead of stochastic single-sample updates.
* **Smart Shuffling:** Automatically shuffles training data every epoch to prevent overfitting.

### 2. Modular Architecture
* **Backend Controller (v2.0):** Logic is strictly separated from the UI using a dedicated `Neural_Network_Backend` class.
* **Smart Connect:** The architecture builder automatically detects and matches input/output shapes to prevent errors.

---

## 📂 Project Structure

```text
Project_Root/
│
├── Neural_Network_GUI.py       # [ENTRY POINT] Run this file to start the App
├── README.md                   # Documentation
├── CHANGELOG.md                # Version History
│
└── py_code/                    # Core Logic Module
    ├── __init__.py             # Package initializer
    ├── Neural_Network_Engine.py # The Math (Layers, Activations, Optimizers)
    └── Neural_Network_Main.py   # The Backend Controller (Data Loading, State)
```

> **Note:** Detailed installation and usage documentation is currently being written and will be updated shortly.