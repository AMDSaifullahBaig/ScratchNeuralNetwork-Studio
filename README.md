# Modular Neural Network Engine
**Author:** MD Saifullah Baig.A  
**Version:** 2.0  
**Status:** Active

## 📌 Overview
This repository contains a modular, scratch-built Deep Learning framework in Python. It is designed to demystify the internal mechanics of deep learning by implementing **Backpropagation**, **Optimizers (SGD, Adam)**, and **Dynamic Layer Stacking** entirely from scratch using NumPy, without relying on auto-differentiation libraries like PyTorch or TensorFlow.

## 🚀 Key Features
* **Modular Architecture:** Build networks by stacking layers dynamically (similar to Keras Sequential API).
* **Advanced Optimizers:** Custom implementation of **Adam** (Adaptive Moment Estimation) and **SGD** (Stochastic Gradient Descent).
* **Vectorized Operations:** High-performance matrix computations using NumPy.
* **Activation Functions:** Includes Sigmoid, Tanh, and ReLU with their respective derivatives.
* **Visualization:** Real-time tracking of training loss.

## 📂 Project Structure
```text
Neural_Network_Scratch_App/
│
├── Neural_Network_Engine.py            # The Core Computation Library
├── Neural_Network_Main.py              # Main Execution Script
├── Neural_Network_Engine.ipynb         #Easy Reference material
├── Neural_Network_Main.ipynb           #Easy Reference material
├── requirements.txt                    # List of dependencies
├── README.md                           # Project Documentation
└── .gitignore                          # Ignored files (.venv, __pycache__)
```
## 💻 Usage
To run the diabetes regression example:
```bash
python Neural_Network_Main.py 
```

> **Note:** Detailed installation and usage documentation is currently being written and will be updated shortly.