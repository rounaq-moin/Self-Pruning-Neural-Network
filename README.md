# Self-Pruning Neural Network (CIFAR-10)

## 🚀 Overview

This project implements a self-pruning neural network that learns to remove unnecessary weights during training using learnable gates.

Unlike traditional pruning, this method dynamically learns sparsity during training itself.

---

## 🧠 Core Idea

Each weight is paired with a learnable gate:

g = sigmoid(gate_scores)

Effective weight:

W' = W × g

If g → 0, the weight is effectively pruned.

---

## 📉 Loss Function

Loss = CrossEntropy + λ × SparsityLoss

Where:

SparsityLoss = sum of all gate values

This encourages the network to:
- Keep important connections
- Remove redundant ones

---

## 📊 Results

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|--------------|-------------|
| 1.0    | 57.65        | 31.68       |
| 5.0    | 58.35        | 71.61       |
| 20.0   | 58.26        | 91.25       |

---

## 🔥 Key Insight

The model retains strong performance even after pruning more than 70% of weights.

---

## ⚙️ How to Run

```bash
pip install -r requirements.txt
python self-prune-NN.py
