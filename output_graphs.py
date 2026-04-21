import json
import matplotlib.pyplot as plt
import numpy as np
import os

# ===============================
# Setup
# ===============================
plt.style.use('seaborn-v0_8')
os.makedirs("outputs", exist_ok=True)

# Load results
with open("results.json", "r") as f:
    results = json.load(f)

lambdas = np.array([r["lambda"] for r in results])
acc = np.array([r["accuracy"] for r in results])
sparsity = np.array([r["sparsity"] for r in results])

# ===============================
# 1. TRADE-OFF (MAIN GRAPH)
# ===============================
plt.figure(figsize=(7,5))
plt.plot(sparsity, acc, marker='o')

# Highlight best point
best_idx = np.argmax(acc)
plt.scatter(sparsity[best_idx], acc[best_idx], s=120, edgecolors='black')

# Annotate lambda values
for i in range(len(lambdas)):
    plt.annotate(
        f"λ={lambdas[i]}",
        (sparsity[i], acc[i]),
        xytext=(5,5),
        textcoords="offset points"
    )

plt.xlabel("Sparsity (%)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy–Sparsity Trade-off (Best Balance at λ = 5)")
plt.grid(alpha=0.3)

plt.savefig("outputs/tradeoff.png")
plt.show()

# ===============================
# 2. BUBBLE PLOT (VISUAL IMPACT)
# ===============================
plt.figure(figsize=(7,5))

sizes = lambdas * 200  # bubble size = lambda
plt.scatter(sparsity, acc, s=sizes, alpha=0.6)

for i in range(len(lambdas)):
    plt.text(sparsity[i]+1, acc[i], f"λ={lambdas[i]}")

plt.xlabel("Sparsity (%)")
plt.ylabel("Accuracy (%)")
plt.title("Bubble Plot (Size Represents λ)")
plt.grid(alpha=0.3)

plt.savefig("outputs/bubble.png")
plt.show()

# ===============================
# 3. COLOR GRADIENT (MODERN LOOK)
# ===============================
plt.figure(figsize=(7,5))

sc = plt.scatter(sparsity, acc, c=lambdas, cmap='viridis', s=120)

plt.colorbar(sc, label="Lambda")
plt.xlabel("Sparsity (%)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Sparsity (Color = λ)")
plt.grid(alpha=0.3)

plt.savefig("outputs/gradient.png")
plt.show()

print("\n✅ Selected graphs saved in 'outputs/' folder")
