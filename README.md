# Self-Pruning Neural Networks: Learning Sparse Representations via Gated Weight Optimization

## 📌 Problem Statement

Large neural networks are computationally expensive and memory-intensive.
This project implements a **self-pruning neural network** that dynamically removes less important weights during training.

The key idea is to associate each weight with a learnable **gate parameter**, allowing the network to automatically decide which connections to keep or remove.

---

## 🧠 Approach

### 🔹 Prunable Linear Layer

* Each weight has a corresponding **gate score**
* Gates are obtained using **Sigmoid function (0 to 1)**
* Effective weight:

  weight × gate

If gate → 0 → connection is pruned

---

### 🔹 Sparsity Regularization

Total Loss:

Loss = CrossEntropy + λ × L1(gates)

* L1 penalty forces many gates → 0
* This creates a **sparse network**

---

### 🔹 Model Architecture

* Convolution layers for feature extraction
* Fully connected layers replaced with **PrunableLinear**

---

## 📊 Results

| Lambda | Accuracy | Sparsity |
| ------ | -------- | -------- |
| 1e-5   | 74.64%   | 6.18%    |
| 1e-4   | 72.94%   | 10.64%   |
| 1e-3   | 71.20%   | 13.45%   |

---

## 📉 Gate Distribution

![Gate Distribution](histogram.png)

---

## ⚖️ Key Observation

* Increasing λ increases sparsity
* But reduces accuracy
* Shows trade-off between efficiency and performance

---

## 🚀 How to Run

```bash
pip install torch torchvision matplotlib
python Self_Pruning_Neural_Network_for_CIFAR_10_Classification.py
```

OR run in Google Colab

---

## 🛠 Tech Stack

* Python
* PyTorch
* Matplotlib

---

## 📌 Conclusion

This project demonstrates how neural networks can learn to optimize their own structure, making them more efficient without manual pruning.
