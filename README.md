# Can a Machine Discover a Law of Nature?

## Neural Network Learning the Celsius-to-Fahrenheit Conversion Formula

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-Educational-green.svg)]()

> **Deep Learning Course - Lesson 28**
> An educational project demonstrating how neural networks can discover mathematical relationships from data alone.

---

## The Challenge

**Can a neural network discover the Celsius-to-Fahrenheit conversion formula without being explicitly programmed?**

The well-known formula is:
```
F = C × 1.8 + 32
```

Instead of programming this formula, we give the model only **7 example pairs** of temperature conversions:

| Celsius (Input) | Fahrenheit (Output) |
|:---------------:|:-------------------:|
| -40°C           | -40°F               |
| -10°C           | 14°F                |
| 0°C             | 32°F                |
| 8°C             | 46.4°F              |
| 15°C            | 59°F                |
| 22°C            | 71.6°F              |
| 38°C            | 100°F               |

**The Question:** Can the model infer the underlying mathematical relationship from these examples?

---

## The Solution: Neural Network Approaches

### Approach 1: Single Neuron (Perceptron)

The simplest possible neural network - a single neuron with one input and one output.

```python
# The model architecture
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])
```

**Mathematical Representation:**
```
output = weight × input + bias
   F   =   m   ×   C   +   b
```

After training for 500 epochs, the neuron learns:
- **Weight (m):** ~1.82 (close to 1.8!)
- **Bias (b):** ~28.9 (close to 32!)

**Result:** The model essentially **discovered** the formula `F = 1.8 × C + 32`!

### Approach 2: Multi-Layer Network (Deep Learning)

A more complex architecture with hidden layers:

```python
l0 = tf.keras.layers.Dense(units=4, input_shape=[1])  # 4 neurons
l1 = tf.keras.layers.Dense(units=4)                    # 4 neurons
l2 = tf.keras.layers.Dense(units=1)                    # 1 output

model = tf.keras.Sequential([l0, l1, l2])
```

**Result:** Also achieves accurate predictions (~211.9°F for 100°C), but the internal weights appear as seemingly random numbers distributed across multiple neurons.

### Key Insight: Interpretability vs. Power

| Aspect | Single Neuron | Multi-Layer Network |
|--------|---------------|---------------------|
| Prediction Accuracy | High | High |
| Interpretability | **Excellent** - we can read the formula | Low - weights look random |
| Complexity | Minimal | Higher |
| Use Case | Simple linear relationships | Complex patterns |

---

## The Learning Process

### Training Loop (500 Epochs)

```
1. INPUT      → Feed Celsius value
2. FORWARD    → Model makes a guess (prediction)
3. LOSS       → Calculate error (Mean Squared Error)
4. BACKWARD   → Compute gradients (how to adjust)
5. UPDATE     → Adjust weights to reduce error
6. REPEAT     → Go back to step 1
```

### Loss Function: Mean Squared Error (MSE)

Measures how far predictions are from actual values:
```
MSE = (1/n) × Σ(predicted - actual)²
```

### Optimizer: Adam

Adaptive learning rate optimizer that:
- Starts with larger steps
- Takes smaller steps as it approaches the minimum
- Prevents "jumping over" the optimal solution

### Learning Rate (η = 0.1)

Controls the step size during optimization:
```
W_new = W_old - η × ∂E/∂W
```

---

## Key Concepts for AI Students

### 1. The Perceptron

The **perceptron** is the fundamental building block of neural networks:

```
z = Σ(wᵢ × xᵢ) + b
output = activation(z)
```

Components:
- **Inputs (xᵢ):** Features/data points
- **Weights (wᵢ):** Learned importance of each input
- **Bias (b):** Shifts the decision boundary
- **Activation:** Introduces non-linearity

### 2. Why This Matters: The XOR Problem

A single perceptron cannot solve the XOR problem (non-linearly separable data). This limitation, proven by Minsky & Papert in 1969, led to the "AI Winter."

**Solution:** Multi-layer networks can solve XOR by adding hidden layers.

### 3. Gradient Descent

The optimization algorithm that enables learning:

```
W_new = W_old - α × ∇E(W)
```

- **α (alpha):** Learning rate
- **∇E(W):** Gradient of error with respect to weights

Think of it as "walking downhill" on the error surface to find the minimum.

### 4. Backpropagation

The algorithm for computing gradients in multi-layer networks:

1. **Forward Pass:** Compute outputs layer by layer
2. **Compute Loss:** Compare prediction to ground truth
3. **Backward Pass:** Propagate error gradients back through layers
4. **Update Weights:** Adjust each weight based on its contribution to error

---

## Project Structure

```
L28-C-F-function-by-Neural-Network/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── notebooks/
│   └── celsius_fahrenheit_neural_network.ipynb   # Interactive Jupyter notebook
│
├── src/
│   ├── __init__.py
│   └── celsius_to_fahrenheit.py       # Python module with all code
│
├── docs/
│   ├── class_materials/               # Course lecture notes (PDFs)
│   │   ├── L27- DL-01-Perceptron.pdf
│   │   ├── L28 - XORClassificationwithPerceptron.pdf
│   │   └── main.pdf
│   │
│   └── presentations/
│       └── task_presentation.pdf      # Original assignment presentation
│
└── assets/
    └── images/                        # Diagrams and visualizations
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/hadarwayn/L28-C-F-function-by-Neural-Network.git
cd L28-C-F-function-by-Neural-Network

# Install dependencies
pip install -r requirements.txt
```

### Run the Code

**Option 1: Python Script**
```bash
python src/celsius_to_fahrenheit.py
```

**Option 2: Jupyter Notebook**
```bash
jupyter notebook notebooks/celsius_fahrenheit_neural_network.ipynb
```

**Option 3: Google Colab**
Upload the notebook to [Google Colab](https://colab.research.google.com/) for GPU acceleration.

---

## Results Summary

### Single Neuron Model Output

```
Prediction for 100.0°C: 211.17°F
Expected (from formula): 212.0°F

Learned weights (m): 1.8226
Learned bias (b):    28.907

Discovered formula: F ≈ 1.82 × C + 28.9
Actual formula:     F = 1.80 × C + 32.0
```

### Learning Curve

The loss decreases dramatically in early epochs, then gradually converges:

```
Epoch 1:   Loss ≈ 1000+
Epoch 50:  Loss ≈ 100
Epoch 200: Loss ≈ 20
Epoch 500: Loss ≈ 1
```

---

## Historical Context

| Year | Event |
|------|-------|
| **1958** | Frank Rosenblatt invents the Perceptron |
| **1969** | Minsky & Papert prove perceptron limitations (XOR problem) |
| **1969-1980s** | "AI Winter" - funding and research decline |
| **1989** | Yann LeCun introduces CNNs for digit recognition |
| **2006** | Geoffrey Hinton introduces Deep Belief Networks |
| **2012+** | Deep Learning revolution with GPU computing |

---

## Key Takeaways

1. **Machine Learning = Numerical Optimization**
   The model doesn't "understand" temperature - it finds the mathematical function that minimizes prediction error.

2. **The Loss Function is the Compass**
   Everything in deep learning revolves around minimizing the loss function.

3. **Simplicity Can Be Powerful**
   A single neuron solved this problem beautifully and interpretably.

4. **Complexity Has Costs**
   Deeper networks are more powerful but less interpretable.

5. **Data Quality Matters**
   Only 7 well-chosen examples were enough to discover a fundamental physical relationship.

---

## Further Reading

- **Course Materials:** See `docs/class_materials/` for detailed lecture notes
- **TensorFlow Documentation:** [tensorflow.org](https://www.tensorflow.org/)
- **Deep Learning Book:** Goodfellow, Bengio, Courville - [deeplearningbook.org](https://www.deeplearningbook.org/)

---

## License

This project is for educational purposes as part of Dr. Yoram Segal's Deep Learning course.

---

## Author

**Course:** Deep Learning L28 - Dr. Yoram Segal
**Student Implementation:** Neural Network Temperature Conversion Discovery

---

*"The best way to understand machine learning is to see it discover something you already know."*
