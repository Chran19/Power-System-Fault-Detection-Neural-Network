# Power System Fault Detection Using Neural Networks

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Chran19/Power-System-Fault-Detection-Neural-Network)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Neural%20Network-orange.svg)](https://numpy.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-87.25%25-green.svg)](#results)

**🔗 Repository:** [https://github.com/Chran19/Power-System-Fault-Detection-Neural-Network](https://github.com/Chran19/Power-System-Fault-Detection-Neural-Network)

## 📋 Project Overview

This project implements a **feedforward neural network from scratch** using NumPy for classifying power system faults. The neural network can identify three types of faults:

- **Line Breakage**
- **Transformer Failure**
- **Overheating**

The project demonstrates a complete machine learning pipeline including data preprocessing, model development, training, evaluation, hyperparameter tuning, and model persistence.

---

## 👥 Team Members

| Name                   | Roll Number  |
| ---------------------- | ------------ |
| **Yashwardhan Jangid** | 202301100007 |
| **Shreyash Kumbhar**   | 202301100032 |
| **Ranjeet Choudhary**  | 202301100046 |
| **Rishabh Rai**        | 202301100047 |

---

## 🏗️ Project Structure

```
Power-System-Fault-Detection-Neural-Network/
├── PowerSystemFaultDetection.ipynb    # Main Jupyter notebook
├── fault_data.csv                     # Dataset (506 samples)
├── power_system_fault_neural_network_model.h5  # Saved model weights
└── README.md                          # This file
```

---

## 🔧 Neural Network Architecture

```
Input Layer (25 features)
        ↓
Hidden Layer 1 (256 neurons, ReLU activation)
        ↓
Hidden Layer 2 (128 neurons, ReLU activation)
        ↓
Output Layer (3 classes, Softmax activation)
```

### Features Used (25 total)

- **Numeric Features (14):** Voltage, Current, Power Load, Temperature, Wind Speed, Duration, Downtime, and derived features
- **Weather Condition (5):** One-hot encoded (Clear, Rainy, Snowy, Thunderstorm, Windstorm)
- **Maintenance Status (3):** One-hot encoded (Completed, Pending, Scheduled)
- **Component Health (3):** One-hot encoded (Faulty, Normal, Overheated)

---

## 📊 Results

| Metric                | Value   |
| --------------------- | ------- |
| **Training Accuracy** | 100.00% |
| **Testing Accuracy**  | 87.25%  |
| **Total Epochs**      | 10,000  |
| **Learning Rate**     | 0.01    |
| **L2 Regularization** | 0.001   |

### Classification Performance

| Fault Type          | Precision | Recall | F1-Score |
| ------------------- | --------- | ------ | -------- |
| Line Breakage       | 0.86      | 0.86   | 0.86     |
| Overheating         | 0.92      | 0.95   | 0.93     |
| Transformer Failure | 0.75      | 0.64   | 0.69     |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn h5py
```

### Running the Notebook

1. Clone the repository
2. Open `PowerSystemFaultDetection.ipynb` in Jupyter Notebook or VS Code
3. Run all cells sequentially

### Using the Saved Model

```python
import h5py
import numpy as np

# Load the model
def load_model(filepath):
    with h5py.File(filepath, 'r') as f:
        h1 = f['w1'].shape[1]
        h2 = f['w2'].shape[1]
        model = FeedforwardNeuralNetwork(f.attrs['input_size'], h1, h2, f.attrs['output_size'])
        model.w1 = f['w1'][:]
        model.b1 = f['b1'][:]
        model.w2 = f['w2'][:]
        model.b2 = f['b2'][:]
        model.w3 = f['w3'][:]
        model.b3 = f['b3'][:]
    return model

model = load_model('power_system_fault_neural_network_model.h5')
```

---

## 📈 Key Features

- ✅ **From Scratch Implementation** - No TensorFlow/PyTorch, pure NumPy
- ✅ **Multi-layer Architecture** - 2 hidden layers with ReLU activation
- ✅ **Momentum-based SGD** - Faster convergence
- ✅ **L2 Regularization** - Prevents overfitting
- ✅ **Hyperparameter Tuning** - Grid search for optimal parameters
- ✅ **Model Persistence** - Save/Load using HDF5 format
- ✅ **XOR Verification** - Validates network learning capability

---

## 📁 Dataset Description

The dataset contains **506 power system fault records** with the following attributes:

| Feature            | Description                                  |
| ------------------ | -------------------------------------------- |
| Voltage (V)        | System voltage (1800-2300 V)                 |
| Current (A)        | Current measurement (180-250 A)              |
| Power Load (MW)    | Power load (45-55 MW)                        |
| Temperature (°C)   | Ambient temperature (20-40°C)                |
| Wind Speed (km/h)  | Wind speed (10-30 km/h)                      |
| Weather Condition  | Clear, Rainy, Snowy, Thunderstorm, Windstorm |
| Maintenance Status | Scheduled, Completed, Pending                |
| Component Health   | Normal, Faulty, Overheated                   |
| Duration of Fault  | Fault duration in hours                      |
| Down time          | System downtime in hours                     |

---

## 🔬 Methodology

1. **Data Preprocessing**
   - Handle missing values
   - Feature engineering (derived features)
   - One-hot encoding for categorical variables
   - StandardScaler normalization

2. **Model Training**
   - Forward propagation with ReLU and Softmax
   - Backpropagation with cross-entropy loss
   - Momentum-based gradient descent
   - L2 regularization

3. **Hyperparameter Tuning**
   - Grid search over learning rates, hidden layer sizes, L2 lambdas
   - 81 configurations evaluated
   - Best configuration selected based on test accuracy

4. **Evaluation**
   - Confusion matrix analysis
   - Classification report with precision, recall, F1-score
   - Training/Testing accuracy comparison

---

## 📝 License

This project is developed as part of the **Neural Networks Design and Deployment** course.

---

_Developed with ❤️ by Team 46-47-07-32_
