# Lab-2.4-Normalization-BatchNorm-vs-LayerNorm
This README explores **Normalization Layers**, specifically comparing **Batch Normalization** and **Layer Normalization**. These techniques are essential for stabilizing the training of deep networks by controlling the distribution of activations across layers.

---

### BatchNorm vs. LayerNorm

Training deep neural networks often suffers from "Internal Covariate Shift," where the distribution of each layer's inputs changes as the parameters of the prior layers change. Normalization layers mitigate this, allowing for higher learning rates and faster convergence.



## 🧪 Normalization Techniques

### 1. Batch Normalization (BatchNorm)
* **Mechanism:** Normalizes the activations using the mean and variance calculated across the **batch** dimension for each channel.
* **Best For:** Convolutional Neural Networks (CNNs).
* **Key Feature:** During training, it maintains a running average of mean and variance to use during inference (`model.eval()`).
* **In this Lab:** Implemented via `nn.BatchNorm2d` after convolutional layers and `nn.BatchNorm1d` after fully connected layers.

### 2. Layer Normalization (LayerNorm)
* **Mechanism:** Normalizes the activations across the **feature** dimension for each individual sample.
* **Best For:** Recurrent Neural Networks (RNNs) and Transformers. It is independent of batch size.
* **In this Lab:** Implemented via `nn.LayerNorm` in a standard Multi-Layer Perceptron (MLP).

---

## 🏗️ Experimental Setup

The lab compares four distinct model configurations to isolate the impact of normalization:

### Convolutional Models (CNN)
* **`CNN_NoNorm`**: A standard 2-layer CNN.
* **`CNN_BN`**: The same CNN architecture but with **BatchNorm** layers inserted after each convolution and before the final activation.

### Linear Models (MLP)
* **`MLP_NoNorm`**: A 3-layer fully connected network.
* **`MLP_LN`**: The same MLP architecture but with **LayerNorm** applied to the outputs of the hidden layers.

---

## 📈 Key Observations

| Model | Typical Behavior |
| :--- | :--- |
| **No Norm** | Slower convergence. Loss curves may be more "jittery" as the model struggles with varying activation scales. |
| **With Norm** | **Faster Convergence:** Reaches high accuracy in fewer epochs. **Smoother Loss:** The validation loss curves are typically more stable. |

### Why use Normalization?
1.  **Higher Learning Rates:** You can be more aggressive with your optimizer without the model diverging.
2.  **Regularization Effect:** BatchNorm adds a slight amount of noise to the training process (due to batch statistics), which can help reduce overfitting.
3.  **Reduces Initialization Sensitivity:** Models become less dependent on perfect weight initialization (like He or Xavier).

---

## 📊 Summary of Results

The script generates validation loss plots to visualize the "speed up" provided by normalization.

* **CNN Comparison:** You should notice the `CNN_BN` curve dropping much faster than the `CNN_NoNorm` curve.
* **MLP Comparison:** `MLP_LN` typically provides a more stable training path compared to the baseline MLP.

---

## 🚀 How to Run
1.  Execute the script to train all four models simultaneously.
2.  Review the console output for epoch-by-epoch accuracy.
3.  Observe the Matplotlib plots to see the visual gap in convergence speed between normalized and non-normalized models.

> **Technical Note:** When using BatchNorm, always remember to call `model.eval()` during testing. This ensures the model uses the **running statistics** learned during training rather than the statistics of the (potentially small) test batch.
