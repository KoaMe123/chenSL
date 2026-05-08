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

# ======================== 全局设置 ========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001
RESULT_DIR = "./mnist_results"
os.makedirs(RESULT_DIR, exist_ok=True)

# ======================== 1. 数据准备与探索 ========================
transform = transforms.Compose([
    transforms.ToTensor(),                         # 转为Tensor并归一化到[0,1]
    transforms.Lambda(lambda x: x.view(-1))        # 展平为784维向量
])

# 下载并加载数据集
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"训练集样本数: {len(train_dataset)}")
print(f"测试集样本数: {len(test_dataset)}")

# ---- 数据分析与可视化 ----
# 1.1 类别分布图
train_labels = [label for _, label in train_dataset]
plt.figure(figsize=(8, 5))
sns.countplot(x=train_labels, palette="viridis")
plt.title("Distribution of Classes in Training Set")
plt.xlabel("Digit")
plt.ylabel("Count")
plt.savefig(os.path.join(RESULT_DIR, "class_distribution.png"), dpi=150)
plt.close()

# 1.2 样本展示：每个类别展示一例
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
axes = axes.flatten()
for digit in range(10):
    # 找到第一个属于该类的样本
    idx = train_labels.index(digit)
    img, label = train_dataset[idx]
    axes[digit].imshow(img.view(28, 28).numpy(), cmap="gray")
    axes[digit].set_title(f"Label: {label}")
    axes[digit].axis("off")
plt.suptitle("Sample Images from Each Class")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "class_samples.png"), dpi=150)
plt.close()

# 1.3 随机16张样本图
plt.figure(figsize=(8, 8))
indices = np.random.choice(len(train_dataset), 16, replace=False)
for i, idx in enumerate(indices):
    img, label = train_dataset[idx]
    plt.subplot(4, 4, i+1)
    plt.imshow(img.view(28, 28).numpy(), cmap="gray")
    plt.title(f"Label: {label}")
    plt.axis("off")
plt.suptitle("Random 16 Training Samples")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "random_samples.png"), dpi=150)
plt.close()

# ======================== 2. 多层感知机模型定义 ========================
class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[512, 256], num_classes=10, dropout=0.2):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = MLP().to(DEVICE)
print(model)
print(f"模型总参数量: {sum(p.numel() for p in model.parameters()):,}")

# ======================== 3. 训练配置 ========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# 记录训练历史
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

# ======================== 4. 训练循环 ========================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    loop = tqdm(loader, desc="Training")
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        loop.set_postfix(loss=loss.item(), acc=100.0*correct/total)
    return running_loss / total, correct / total

def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total

for epoch in range(1, EPOCHS+1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, test_loader, criterion)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

# ======================== 5. 训练曲线 ========================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS+1), history["train_loss"], marker="o", label="Train Loss")
plt.plot(range(1, EPOCHS+1), history["val_loss"], marker="s", label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS+1), [acc*100 for acc in history["train_acc"]], marker="o", label="Train Acc")
plt.plot(range(1, EPOCHS+1), [acc*100 for acc in history["val_acc"]], marker="s", label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "training_curves.png"), dpi=150)
plt.close()

# ======================== 6. 最终评估：混淆矩阵 ========================
all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix on Test Set")
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# 输出分类报告
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, digits=4))

# ======================== 7. 随机9张测试图预测展示 ========================
def imshow_predictions(images, labels, preds, save_path, title="Predictions"):
    plt.figure(figsize=(8, 8))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        img = images[i].view(28, 28).cpu().numpy()
        plt.imshow(img, cmap="gray")
        true_label = labels[i].item()
        pred_label = preds[i].item()
        color = "green" if true_label == pred_label else "red"
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# 随机选9张
test_images, test_labels = next(iter(test_loader))
# 为了保证9张，直接从整个测试集随机采样
indices = np.random.choice(len(test_dataset), 9, replace=False)
random_images = torch.stack([test_dataset[i][0] for i in indices]).to(DEVICE)
random_labels = torch.tensor([test_dataset[i][1] for i in indices])
with torch.no_grad():
    outputs = model(random_images)
    _, random_preds = outputs.max(1)
imshow_predictions(random_images, random_labels, random_preds,
                   os.path.join(RESULT_DIR, "random_9_predictions.png"),
                   title="Random 9 Test Predictions (Green: Correct, Red: Wrong)")

# ======================== 8. 错误预测样本分析（前9个错误） ========================
error_indices = []
for idx, (img, label) in enumerate(test_dataset):
    if len(error_indices) >= 9:
        break
    img_tensor = img.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(1).item()
    if pred != label:
        error_indices.append(idx)

if len(error_indices) >= 9:
    error_images = torch.stack([test_dataset[i][0] for i in error_indices[:9]]).to(DEVICE)
    error_labels = torch.tensor([test_dataset[i][1] for i in error_indices[:9]])
    with torch.no_grad():
        outputs = model(error_images)
        _, error_preds = outputs.max(1)
    imshow_predictions(error_images, error_labels, error_preds,
                       os.path.join(RESULT_DIR, "top_9_errors.png"),
                       title="Misclassified Examples (9 Samples)")
else:
    print("错误样本不足9张，跳过错误分析图。")

print(f"\n所有图表已保存至: {os.path.abspath(RESULT_DIR)}")