# Self-Pruning Neural Network for CIFAR-10

## 1. Introduction

Large neural networks are often limited by memory footprint and inference cost. This project explores a self-pruning approach in which pruning is learned during training, instead of being applied as a separate post-training step.

The core idea is to attach a learnable gate to each weight, then regularize gates toward zero so unimportant connections are suppressed automatically.

---

## 2. Methodology

### 2.1 Prunable Layer

For each weight $W$, a learnable gate score $s$ is introduced. The gate value is:

$$
g = \sigma(s), \quad g \in (0,1)
$$

The effective weight used in the forward pass is:

$$
W' = W \odot g
$$

where $\odot$ is element-wise multiplication. If a gate moves close to 0, the corresponding weight is effectively removed.

### 2.2 Objective Function

Training minimizes classification loss plus a sparsity term on gates:

$$
\mathcal{L} = \mathcal{L}_{CE} + \lambda \cdot \mathbb{E}[g]
$$

In implementation, $\mathbb{E}[g]$ is the mean gate value across all prunable layers. Higher $\lambda$ increases pressure toward sparse connectivity.

Note: L1-style penalties generally push many values close to zero; exact zeros are typically obtained after explicit thresholding.

### 2.3 Model Architecture

Fully connected network on flattened CIFAR-10 images (3072 input features):

- `PrunableLinear(3072, 1024)` + BatchNorm + GELU + Dropout(0.2)
- `PrunableLinear(1024, 768)` + BatchNorm + GELU + Dropout(0.3)
- `PrunableLinear(768, 256)` + BatchNorm + GELU + Dropout(0.4)
- `PrunableLinear(256, 10)`

Total prunable weights: 4,131,328.

### 2.4 Training Configuration (Reproducibility)

- Dataset: CIFAR-10
- Data augmentation (train): RandomCrop(32, padding=4), RandomHorizontalFlip, ColorJitter(0.2, 0.2)
- Normalization: mean (0.4914, 0.4822, 0.4465), std (0.2470, 0.2435, 0.2616)
- Batch size: 128
- Epochs: 20
- Optimizer: Adam with parameter groups
- Learning rates:
- Weights/biases/BatchNorm: $1\times10^{-3}$
- Gate scores: $1\times10^{-2}$
- LR scheduler: StepLR(step size = 10, gamma = 0.5)
- Regularization coefficients tested: $\lambda \in \{1.0, 5.0, 20.0\}$

### 2.5 Sparsity Metric

Reported sparsity is computed from gate values using threshold $g < 0.05$:

$$
	ext{Sparsity}(\%) = \frac{\#\{g < 0.05\}}{\#\{g\}} \times 100
$$

This measures percentage of effectively pruned connections among prunable weights.

---

## 3. Results

| Lambda ($\lambda$) | Test Accuracy (%) | Sparsity (%) |
|---|---:|---:|
| 1.0 | 57.65 | 31.68 |
| 5.0 | 58.35 | 71.61 |
| 20.0 | 58.26 | 91.25 |

Approximate active weights after thresholding:

- $\lambda=1.0$: ~2.82M active weights
- $\lambda=5.0$: ~1.17M active weights
- $\lambda=20.0$: ~0.36M active weights

---

## 4. Discussion

### 4.1 Effect of Regularization Strength

- Low $\lambda$ (1.0): lower sparsity, larger active model capacity.
- Medium $\lambda$ (5.0): strong sparsification with the best accuracy in this run.
- High $\lambda$ (20.0): very aggressive pruning with only a small accuracy drop.

The results indicate that this architecture tolerates substantial connection removal while maintaining similar accuracy.

### 4.2 Accuracy-Sparsity Trade-off

The key practical result is that around 70%+ sparsity is achievable without degrading performance in this experiment. This suggests meaningful compression potential for deployment-constrained settings.

### 4.3 Gate Dynamics

Observed gate behavior during training:

- Mean gate value decreases over epochs.
- Many gates move toward the pruning threshold.
- A subset stays high, preserving important connections.

This is consistent with a bimodal tendency in gate distributions: one group near zero (pruned) and one retained group away from zero.

---

## 5. Limitations

- Results are from single runs per $\lambda$ (no mean/std across seeds).
- No explicit dense baseline is reported in this document.
- The model is an MLP; CIFAR-10 performance can likely improve with convolutional architectures.

---

## 6. Conclusion

This project demonstrates that a neural network can learn to prune itself during training by combining gated weights with sparsity regularization. On CIFAR-10, the method reaches high sparsity (up to 91.25%) while keeping accuracy in a narrow range (~58%), showing strong compression-performance behavior for this setup.

---

## 7. Future Work

- Add a dense baseline and report compression ratio vs baseline.
- Repeat experiments across multiple seeds and report mean +/- std.
- Extend to CNNs and structured pruning (channel/filter-level).
- Combine pruning with quantization for further model compression.
- Add gate histograms and accuracy-vs-sparsity plots.

---