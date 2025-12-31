# Changelog - Modular Neural Network Framework

All notable changes to this project will be documented in this file.

### 🚀 Major Releases

###  Organised files into folders 
#### **Neural Network Engine (v3.0)**
*Major performance overhaul converting the engine from simple loops to matrix operations.*
* **Vectorization Support:** Updated `Predict` method to handle full input matrices ($N \times M$) in a single operation using NumPy broadcasting. Removed slow `for` loops over samples.
* **Mini-Batch Gradient Descent:** Refactored `Training_model` to update weights after batches of 32 samples instead of every single sample (SGD).
* **GUI Hooks:** Added a `callback` parameter to the training loop. This allows external interfaces to listen for epoch progress and interrupt training safely.
* **Data Shuffling:** Added automatic index shuffling at the start of every epoch to prevent order-based overfitting.

#### **Backend Controller (v2.0)**
*New architecture to separate logic from the User Interface.*
* **New Class `Neural_Network_Backend`:** Acts as the bridge between the raw Engine and the GUI.
* **State Management:** Handles the `layer_stack` configuration, allowing users to "Undo" or "Reset" layers before building the model.
* **Threaded Training:** Implements `train_loop` with a bridge function to run the blocking training process on a background thread while keeping the GUI responsive.
* **Automated Preprocessing:** `load_data` now automatically applies Standard Scaling ($\mu=0, \sigma=1$) to both "Diabetes" and "XOR" datasets.


### 🐛 Bug Fixes
* Fixed a version mismatch where the GUI was sending `batch_size` arguments to an old v1.0 Engine.
* Fixed "Vanishing Gradient" issues on the Diabetes dataset by enforcing Standard Scaling on inputs.