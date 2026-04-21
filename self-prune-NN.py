# ==========================================
# Self-Pruning Neural Network (CIFAR-10)
# ==========================================
# This model learns to prune its own weights
# during training using learnable gates.
#
# Core idea:
# Each weight has a gate (0 → pruned, 1 → active)
# Gates are learned using L1 regularization.
# ==========================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import json


# ==========================================
# 1. PRUNABLE LINEAR LAYER
# ==========================================
class PrunableLinear(nn.Module):
    """
    A custom linear layer where each weight has a learnable gate.

    Standard Linear:
        output = x @ W + b

    Prunable Linear:
        output = x @ (W * sigmoid(gate_scores)) + b

    If gate → 0 → weight effectively removed.
    """

    def __init__(self, in_features, out_features):
        super().__init__()

        # Standard weight and bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # Learnable gate parameters (same shape as weights)
        # Initialized >0 so most weights start active
        self.gate_scores = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01 + 2.0
        )

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x):
        """
        Forward pass:
        1. Convert gate_scores → gates (0 to 1 using sigmoid)
        2. Multiply weights by gates
        3. Apply linear transformation
        """
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self):
        """
        Returns gate values (detached from graph)
        Used for analysis (no gradient needed)
        """
        return torch.sigmoid(self.gate_scores).detach()


# ==========================================
# 2. NETWORK ARCHITECTURE
# ==========================================
class SelfPruningNet(nn.Module):
    """
    Fully connected network using prunable layers.

    Architecture:
    3072 → 1024 → 768 → 256 → 10
    """

    def __init__(self):
        super().__init__()

        # Flatten CIFAR-10 image (3×32×32 → 3072)
        self.flatten = nn.Flatten()

        # Prunable layers
        self.fc1 = PrunableLinear(3072, 1024)
        self.fc2 = PrunableLinear(1024, 768)
        self.fc3 = PrunableLinear(768, 256)
        self.fc4 = PrunableLinear(256, 10)

        # BatchNorm for stability
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(768)
        self.bn3 = nn.BatchNorm1d(256)

        # Dropout to reduce overfitting
        self.d1 = nn.Dropout(0.2)
        self.d2 = nn.Dropout(0.3)
        self.d3 = nn.Dropout(0.4)

    def forward(self, x):
        """
        Forward pass through network.
        Each layer:
        Linear → BatchNorm → GELU → Dropout
        """
        x = self.flatten(x)
        x = self.d1(F.gelu(self.bn1(self.fc1(x))))
        x = self.d2(F.gelu(self.bn2(self.fc2(x))))
        x = self.d3(F.gelu(self.bn3(self.fc3(x))))
        return self.fc4(x)

    def prunable_layers(self):
        """Returns all prunable layers in the model"""
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def sparsity_loss(self):
        """
        L1 penalty on all gate values.

        Why L1?
        → Encourages values to become exactly zero
        → Leads to sparse network

        NOTE:
        Using SUM (not mean) as per assignment requirement
        """
        gates = []
        for layer in self.prunable_layers():
            g = torch.sigmoid(layer.gate_scores)
            gates.append(g.view(-1))

        return torch.cat(gates).sum()

    def overall_sparsity(self, threshold=0.05):
        """
        Calculates percentage of pruned weights.

        A weight is considered pruned if:
            gate < threshold
        """
        total, pruned = 0, 0

        for layer in self.prunable_layers():
            g = layer.get_gates()
            pruned += (g < threshold).sum().item()
            total  += g.numel()

        return pruned / total

    def all_gates(self):
        """
        Returns all gate values (for analysis/plotting)
        """
        vals = []
        for layer in self.prunable_layers():
            vals.append(layer.get_gates().cpu().numpy().flatten())

        return np.concatenate(vals)


# ==========================================
# 3. DATA LOADING
# ==========================================
def get_loaders(batch_size=128):
    """
    Loads CIFAR-10 dataset with augmentation.
    """

    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    # Training augmentations
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Test transformations (no augmentation)
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train = datasets.CIFAR10('./data', train=True, download=True, transform=train_tf)
    test  = datasets.CIFAR10('./data', train=False, download=True, transform=test_tf)

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2)
    )


# ==========================================
# 4. TRAINING FUNCTIONS
# ==========================================
def train_epoch(model, loader, optimizer, device, lam):
    """
    Trains model for one epoch.

    Loss = CrossEntropy + λ * SparsityLoss
    """
    model.train()

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)

        ce = F.cross_entropy(out, y)
        sp = model.sparsity_loss()

        loss = ce + lam * sp
        loss.backward()
        optimizer.step()


def evaluate(model, loader, device):
    """Evaluates model accuracy"""
    model.eval()

    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total


# ==========================================
# 5. EXPERIMENT RUNNER
# ==========================================
def run_experiment(lam, train_loader, test_loader, device):
    """
    Runs training for a specific lambda.

    lambda controls:
    - Higher → more pruning
    - Lower → better accuracy
    """

    model = SelfPruningNet().to(device)

    # Separate gate and weight parameters
    gate_params = [p for n,p in model.named_parameters() if 'gate_scores' in n]
    weight_params = [p for n,p in model.named_parameters() if 'gate_scores' not in n]

    optimizer = optim.Adam([
        {'params': weight_params, 'lr':1e-3},
        {'params': gate_params,   'lr':1e-2}  # faster gate learning
    ])

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(20):
        train_epoch(model, train_loader, opt, device, lam)

        acc = evaluate(model, test_loader, device)
        sp  = model.overall_sparsity()

        mean_gate = float(model.all_gates().mean())  # NEW METRIC

        print(f"Epoch {epoch:02d} | Acc={acc*100:.2f}% | Sparsity={sp*100:.2f}% | MeanGate={mean_gate:.3f}")

        scheduler.step()

    return {
    "lambda": float(lam),
    "accuracy": float(acc * 100),
    "sparsity": float(sp * 100),
    "mean_gate": float(mean_gate)
}


# ===============================
# Main
# ===============================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_loaders()

    lambdas = [1.0, 5.0, 20.0]

    results = []
    for lam in lambdas:
        print(f"\n==== Lambda {lam} ====")
        res = run(lam, train_loader, test_loader, device)
        results.append(res)

    # Save results
    out = []
    for r in results:
        out.append({
            "lambda": r["lambda"],
            "accuracy": round(r["acc"]*100,2),
            "sparsity": round(r["sparsity"]*100,2),
            "mean_gate": round(r["mean_gate"],4)
        })

    with open("results.json","w") as f:
        json.dump(out,f,indent=2)

    print("\nSaved results.json")


if __name__ == "__main__":
    main()
