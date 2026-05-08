import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ======================== Global Settings ========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
RESULT_DIR = "./mnist_compare_results"
os.makedirs(RESULT_DIR, exist_ok=True)

# ======================== Data Preparation ========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ======================== Model Definitions ========================
def get_model(name):
    if name == "mlp_small":
        return nn.Sequential(
            nn.Linear(784, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
    elif name == "mlp_base":
        return nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 10)
        )
    elif name == "mlp_deep":
        return nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 10)
        )
    elif name == "mlp_drop04":
        return nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 10)
        )

model_names = ["mlp_small", "mlp_base", "mlp_deep", "mlp_drop04"]
model_colors = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3"]
model_labels = [
    "Small MLP (128)",
    "Base MLP (512,256)",
    "Deep MLP (512,256,128,64)",
    "Base MLP + Dropout 0.4"
]

# ======================== Train & Eval ========================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for img, label in loader:
        img, label = img.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * img.size(0)
        correct += (out.argmax(1) == label).sum().item()
        total += label.size(0)
    return loss_sum / total, correct / total

def test_model(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    with torch.no_grad():
        for img, label in loader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            out = model(img)
            loss_sum += criterion(out, label).item() * img.size(0)
            correct += (out.argmax(1) == label).sum().item()
            total += label.size(0)
    return loss_sum / total, correct / total

# ======================== Run All Models ========================
results = {}
for name in model_names:
    print(f"\n===== Training: {name} =====")
    model = get_model(name).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for ep in range(1, EPOCHS+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = test_model(model, test_loader, criterion)
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {ep:2d} | TrainLoss {train_loss:.4f} | TrainAcc {train_acc:.4f} | ValAcc {val_acc:.4f}")
    results[name] = history

# ======================== Plot Accuracy Comparison ========================
plt.figure(figsize=(12, 5))
for i, name in enumerate(model_names):
    plt.plot(range(1, EPOCHS+1), [a*100 for a in results[name]["val_acc"]],
             color=model_colors[i], marker='o', label=model_labels[i])
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy Comparison of Different MLP Structures")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "accuracy_compare.png"), dpi=200)
plt.close()

# ======================== Plot Loss Comparison ========================
plt.figure(figsize=(12, 5))
for i, name in enumerate(model_names):
    plt.plot(range(1, EPOCHS+1), results[name]["val_loss"],
             color=model_colors[i], marker='s', label=model_labels[i])
plt.xlabel("Epoch")
plt.ylabel("Test Loss")
plt.title("Test Loss Comparison of Different MLP Structures")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "loss_compare.png"), dpi=200)
plt.close()

# ======================== Print Final Result Table ========================
print("\n" + "="*70)
print("MNIST MLP Comparison Final Results (Test Set)")
print("="*70)
print(f"{'Model Structure':<35}{'Final Acc':<12}{'Best Acc':<12}")
print("-"*70)

final_table = []
for i, name in enumerate(model_names):
    best_acc = max(results[name]["val_acc"])
    last_acc = results[name]["val_acc"][-1]
    print(f"{model_labels[i]:<35}{last_acc:<12.4f}{best_acc:<12.4f}")
    final_table.append([model_labels[i], last_acc, best_acc])

# ======================== Save Results ========================
with open(os.path.join(RESULT_DIR, "compare_result.txt"), "w", encoding="utf-8") as f:
    f.write("MNIST MLP Comparison Experiment Results\n")
    f.write("="*50 + "\n")
    f.write(f"{'Model Structure':<35}{'Final Acc':<12}{'Best Acc':<12}\n")
    f.write("-"*50 + "\n")
    for item in final_table:
        f.write(f"{item[0]:<35}{item[1]:<12.4f}{item[2]:<12.4f}\n")

print(f"\nAll results saved to: {os.path.abspath(RESULT_DIR)}")