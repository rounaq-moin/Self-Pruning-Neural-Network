import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# ===============================
# Prunable Linear Layer
# ===============================
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # Modified init
        self.gate_scores = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01 + 2.0
        )

        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        return F.linear(x, self.weight * gates, self.bias)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores).detach()


# ===============================
# Network
# ===============================
class SelfPruningNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.fc1 = PrunableLinear(3072, 1024)
        self.fc2 = PrunableLinear(1024, 768)
        self.fc3 = PrunableLinear(768, 256)
        self.fc4 = PrunableLinear(256, 10)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(768)
        self.bn3 = nn.BatchNorm1d(256)

        self.d1 = nn.Dropout(0.2)
        self.d2 = nn.Dropout(0.3)
        self.d3 = nn.Dropout(0.4)

    def forward(self, x):
        x = self.flatten(x)
        x = self.d1(F.gelu(self.bn1(self.fc1(x))))
        x = self.d2(F.gelu(self.bn2(self.fc2(x))))
        x = self.d3(F.gelu(self.bn3(self.fc3(x))))
        return self.fc4(x)

    def prunable_layers(self):
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def sparsity_loss(self):
        gates = []
        for layer in self.prunable_layers():
            gates.append(torch.sigmoid(layer.gate_scores).view(-1))
        return torch.cat(gates).mean()

    def overall_sparsity(self, threshold=0.05):
        total, pruned = 0, 0
        for layer in self.prunable_layers():
            g = layer.get_gates()
            pruned += (g < threshold).sum().item()
            total  += g.numel()
        return pruned / total

    def all_gates(self):
        vals = []
        for layer in self.prunable_layers():
            vals.append(layer.get_gates().cpu().numpy().flatten())
        return np.concatenate(vals)


# ===============================
# Data
# ===============================
def get_loaders(batch=128):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train = datasets.CIFAR10('./data', train=True, download=True, transform=train_tf)
    test  = datasets.CIFAR10('./data', train=False, download=True, transform=test_tf)

    return (
        DataLoader(train, batch_size=batch, shuffle=True, num_workers=2),
        DataLoader(test, batch_size=batch, shuffle=False, num_workers=2)
    )


# ===============================
# Training
# ===============================
def train_epoch(model, loader, opt, device, lam):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        out = model(x)

        ce = F.cross_entropy(out, y)
        sp = model.sparsity_loss()

        loss = ce + lam * sp
        loss.backward()
        opt.step()


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


# ===============================
# Experiment
# ===============================
def run(lam, train_loader, test_loader, device):
    model = SelfPruningNet().to(device)

    gate_params = [p for n,p in model.named_parameters() if 'gate_scores' in n]
    weight_params = [p for n,p in model.named_parameters() if 'gate_scores' not in n]

    opt = optim.Adam([
        {'params': weight_params, 'lr':1e-3},
        {'params': gate_params,   'lr':1e-2}
    ])

    scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

    for epoch in range(20):
        train_epoch(model, train_loader, opt, device, lam)

        acc = evaluate(model, test_loader, device)
        sp  = model.overall_sparsity()

        mean_gate = model.all_gates().mean()

        print(f"Epoch {epoch:02d} | Acc={acc*100:.2f}% | Sparsity={sp*100:.2f}% | MeanGate={mean_gate:.3f}")

        scheduler.step()

    return {
        "lambda": lam,
        "acc": acc,
        "sparsity": sp,
        "mean_gate": mean_gate,
        "gates": model.all_gates()
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

    # FIXED JSON SAVE (important)
    out = []
    for r in results:
        out.append({
            "lambda": float(r["lambda"]),
            "accuracy": float(r["acc"] * 100),
            "sparsity": float(r["sparsity"] * 100),
            "mean_gate": float(r["mean_gate"])
        })

    with open("results.json","w") as f:
        json.dump(out, f, indent=2)

    print("\nSaved results.json")


if __name__ == "__main__":
    main()