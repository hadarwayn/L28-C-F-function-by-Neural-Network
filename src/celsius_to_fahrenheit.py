"""
Celsius to Fahrenheit Conversion using Neural Networks
========================================================

This module demonstrates how a neural network can discover the mathematical
relationship between Celsius and Fahrenheit temperatures (F = C * 1.8 + 32)
by learning from examples only, without being explicitly programmed.

Author: Based on Deep Learning Course L28
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def create_training_data():
    """
    Prepare the training data: 7 pairs of Celsius-Fahrenheit values.
    The model will learn the conversion formula from these examples only.
    """
    celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
    fahrenheit = np.array([-40, 14, 32, 46.4, 59, 71.6, 100], dtype=float)
    return celsius, fahrenheit


def build_single_neuron_model():
    """
    Build the simplest possible model: a single neuron (perceptron).

    This neuron will learn to compute: F = m * C + b
    Where m (weight) should converge to ~1.8 and b (bias) to ~32
    """
    layer = tf.keras.layers.Dense(units=1, input_shape=[1])
    model = tf.keras.Sequential([layer])
    return model, layer


def build_multi_layer_model():
    """
    Build a more complex model with multiple layers.

    Architecture:
    - Input layer: 4 neurons
    - Hidden layer: 4 neurons
    - Output layer: 1 neuron

    This demonstrates that deeper networks can solve the same problem,
    but the learned weights become harder to interpret.
    """
    l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
    l1 = tf.keras.layers.Dense(units=4)
    l2 = tf.keras.layers.Dense(units=1)

    model = tf.keras.Sequential([l0, l1, l2])
    return model, [l0, l1, l2]


def compile_model(model, learning_rate=0.1):
    """
    Compile the model with loss function and optimizer.

    - Loss: Mean Squared Error (MSE) - measures prediction error
    - Optimizer: Adam - adaptive learning rate optimization
    """
    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(learning_rate)
    )
    return model


def train_model(model, celsius, fahrenheit, epochs=500, verbose=False):
    """
    Train the model on the Celsius-Fahrenheit pairs.

    The training loop:
    1. Forward pass: compute prediction
    2. Compute loss: how far from correct answer
    3. Backward pass: compute gradients
    4. Update weights: adjust to reduce error
    """
    history = model.fit(celsius, fahrenheit, epochs=epochs, verbose=verbose)
    return history


def plot_training_history(history, title="Training Progress"):
    """
    Visualize how the loss decreased during training.

    A decreasing loss curve shows the model is learning.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.title(title)
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss Magnitude (MSE)')
    plt.grid(True)
    plt.show()


def inspect_learned_weights(layer, layer_name="Layer"):
    """
    Inspect the weights learned by a layer.

    For a single neuron learning C->F conversion:
    - Weight (m) should be close to 1.8
    - Bias (b) should be close to 32
    """
    weights, biases = layer.get_weights()
    print(f"\n{layer_name} - Learned Parameters:")
    print(f"  Weights (m): {weights.flatten()}")
    print(f"  Biases (b):  {biases.flatten()}")
    return weights, biases


def main():
    """
    Main demonstration: Can a machine discover a law of nature?
    """
    print("=" * 60)
    print("Neural Network Temperature Conversion Discovery")
    print("=" * 60)
    print("\nChallenge: Learn F = C * 1.8 + 32 from 7 examples only")
    print("The model does NOT know this formula - it must discover it!\n")

    # Prepare data
    celsius, fahrenheit = create_training_data()
    print("Training Data:")
    for c, f in zip(celsius, fahrenheit):
        print(f"  {c:6.1f}°C = {f:6.1f}°F")

    # === Single Neuron Model ===
    print("\n" + "=" * 60)
    print("Model 1: Single Neuron (Interpretable)")
    print("=" * 60)

    model_single, layer = build_single_neuron_model()
    model_single = compile_model(model_single)
    history_single = train_model(model_single, celsius, fahrenheit)

    # Test prediction
    test_celsius = 100.0
    prediction = model_single.predict(np.array([test_celsius]), verbose=0)
    print(f"\nPrediction for {test_celsius}°C: {prediction[0][0]:.2f}°F")
    print(f"Actual formula result: {test_celsius * 1.8 + 32}°F")

    # Inspect what the neuron learned
    weights, biases = inspect_learned_weights(layer, "Single Neuron")
    print(f"\nFormula discovered: F = {weights[0][0]:.4f} * C + {biases[0]:.4f}")
    print(f"Actual formula:     F = 1.8000 * C + 32.0000")

    # === Multi-Layer Model ===
    print("\n" + "=" * 60)
    print("Model 2: Multi-Layer Network (Black Box)")
    print("=" * 60)

    model_multi, layers = build_multi_layer_model()
    model_multi = compile_model(model_multi)
    history_multi = train_model(model_multi, celsius, fahrenheit)

    prediction_multi = model_multi.predict(np.array([test_celsius]), verbose=0)
    print(f"\nPrediction for {test_celsius}°C: {prediction_multi[0][0]:.2f}°F")

    # Inspect weights - they appear random/distributed
    print("\nMulti-layer weights (distributed, hard to interpret):")
    for i, layer in enumerate(layers):
        inspect_learned_weights(layer, f"Layer {i}")

    print("\n" + "=" * 60)
    print("Conclusion")
    print("=" * 60)
    print("""
    The single neuron DISCOVERED the formula F = 1.8*C + 32!

    Key Insights:
    1. Machine learning is numerical optimization
    2. The neuron found the same relationship humans derived
    3. Multi-layer networks work but are less interpretable
    4. This is the foundation of Deep Learning
    """)

    return history_single, history_multi


if __name__ == "__main__":
    history_single, history_multi = main()
