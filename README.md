# Custom Neural Network for Insurance Prediction

A custom implementation of a neural network from scratch using Python and NumPy to predict insurance eligibility based on age and affordability factors.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [File Structure](#file-structure)
- [Contributing](#contributing)

## 🔍 Overview

This project implements a custom neural network from scratch without using high-level machine learning frameworks like TensorFlow or PyTorch for the core neural network logic. The model is designed to predict insurance eligibility based on two input features: age and affordability.

## ✨ Features

- **Custom Neural Network Implementation**: Built from scratch using only NumPy
- **Sigmoid Activation Function**: Custom implementation of the sigmoid activation
- **Gradient Descent Optimization**: Manual implementation of gradient descent algorithm
- **Logarithmic Loss Function**: Custom log-loss implementation for binary classification
- **Data Preprocessing**: Feature scaling for optimal training performance
- **Model Evaluation**: Accuracy scoring and prediction analysis

## 📊 Dataset

The dataset consists of 20 samples with the following features:

- **Age**: Customer age (ranging from 22 to 41 years)
- **Affordability**: Binary indicator (1 = can afford, 0 = cannot afford)
- **Predicted Insurance**: Target variable (1 = eligible, 0 = not eligible)

### Sample Data Structure
```python
{
    "age": [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, ...],
    "affordibility": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, ...],
    "predictedInsurance": [1, 1, 1, 1, 0, 1, 1, 0, 0, 0, ...]
}
```

## 🏗️ Model Architecture

The neural network consists of:

- **Input Layer**: 2 features (age, affordability)
- **Output Layer**: 1 neuron with sigmoid activation
- **Parameters**: 2 weights (w1, w2) and 1 bias term
- **Activation Function**: Sigmoid function for binary classification

### Mathematical Foundation

**Forward Pass:**
```
weighted_sum = w1 * age + w2 * affordability + bias
output = sigmoid(weighted_sum) = 1 / (1 + e^(-weighted_sum))
```

**Loss Function (Log Loss):**
```
loss = -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
```

## 🚀 Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab

### Required Packages
```bash
pip install pandas numpy scikit-learn tensorflow
```

### Clone the Repository
```bash
git clone <repository-url>
cd custom_NN
```

## 💻 Usage

1. **Open the Jupyter Notebook:**
   ```bash
   jupyter notebook CustomNN.ipynb
   ```

2. **Run the cells in sequence:**
   - Import libraries and create dataset
   - Define helper functions (sigmoid, log_loss)
   - Define the NN class
   - Split and scale the data
   - Train the model
   - Make predictions and evaluate

3. **Example Usage:**
   ```python
   # Create and train the model
   customModel = NN()
   customModel.fit(X_train_scaled, y_train, epochs=1000, loss_threshold=0.4631)
   
   # Make predictions
   predictions = customModel.predict(X_test_scaled)
   binary_predictions = (predictions > 0.5).astype(int)
   
   # Evaluate accuracy
   accuracy = accuracy_score(y_test, binary_predictions)
   print(f"Model Accuracy: {accuracy}")
   ```

## 🔧 Implementation Details

### Custom Neural Network Class (`NN`)

**Initialization:**
- Weights (w1, w2) initialized to 1
- Bias initialized to 0

**Key Methods:**
- `fit(X, y, epochs, loss_threshold)`: Train the model using gradient descent
- `predict(X_test)`: Make predictions on test data
- `gradient_descent()`: Core optimization algorithm

**Training Process:**
1. Forward propagation to compute predictions
2. Calculate log loss
3. Compute gradients for weights and bias
4. Update parameters using gradient descent
5. Repeat until convergence or max epochs

### Data Preprocessing

- **Feature Scaling**: Age values are scaled by dividing by 100
- **Train-Test Split**: 80-20 split with random_state=42 for reproducibility

## 📈 Results

The model achieves binary classification for insurance prediction with the following characteristics:

- **Training**: Uses gradient descent with learning rate 0.1
- **Convergence**: Stops when loss threshold (0.4631) is reached
- **Output**: Probability scores converted to binary predictions using 0.5 threshold
- **Evaluation**: Accuracy calculated using scikit-learn's accuracy_score

## 📁 File Structure

```
custom_NN/
├── CustomNN.ipynb          # Main Jupyter notebook with implementation
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies (if created)
```

## 🛠️ Key Components

### 1. Sigmoid Function
```python
def sigmoid_numpy(n):
    return 1/(1+np.exp(-n))
```

### 2. Log Loss Function
```python
def log_loss(y_true, y_predicted):
    epsilon = 1e-15
    y_predicted_new = [max(i,epsilon) for i in y_predicted]
    y_predicted_new = [min(i,1-epsilon) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    return -np.mean(y_true*np.log(y_predicted_new)+(1-y_true)*np.log(1-y_predicted_new))
```

### 3. Neural Network Class
The `NN` class implements the complete neural network with methods for training and prediction.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Notes

- This is an educational implementation to understand neural network fundamentals
- The model uses a simple architecture suitable for binary classification
- Feature scaling is applied only to the age variable for better convergence
- The implementation includes epsilon smoothing in log loss to prevent numerical instability

## 🏷️ Tags

`machine-learning` `neural-network` `python` `numpy` `binary-classification` `gradient-descent` `from-scratch` `insurance-prediction`

---

**Author**: [Your Name]  
**Date**: August 2025  
**Version**: 1.0
