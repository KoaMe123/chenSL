# model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

# dataset
import os
import math
import glob
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, Subset, DataLoader
import matplotlib.pyplot as plt

# save result
import pickle

torch.manual_seed(2022)

try:
    device = torch.device("mps")
except:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f'Current Device : {device}')

# ==============================
# 全局路径（和你的项目结构匹配）
# ==============================
TRAIN_IMG_DIR = "./dataset/train"
TEST_IMG_DIR = "./dataset/test"
LABEL_CSV = "./dataset/labels.csv"
SUBMIT_CSV = "./dataset/sample_submission.csv"

img_names = glob.glob(os.path.join(TRAIN_IMG_DIR, "*.jpg"))
print(f'Total images in Dataset : {len(img_names)}')

# ==============================
# 数据集类（正确独立定义）
# ==============================
class DogDataset(Dataset):
    def __init__(self, img_path, csv_path):
        self.csv_path = csv_path
        self.transform = None
        self.img_names = glob.glob(os.path.join(img_path, "*.jpg"))

        if csv_path is not None:
            label_df = pd.read_csv(csv_path)
            self.label_idx2name = label_df['breed'].unique()
            self.label_name2idx = {name: i for i, name in enumerate(self.label_idx2name)}
            self.img2label = {}
            for _, row in label_df.iterrows():
                img_path_full = os.path.join(img_path, f"{row['id']}.jpg")
                self.img2label[img_path_full] = self.label_name2idx[row['breed']]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_path = self.img_names[index]
        label = -1
        if self.csv_path is not None:
            label = self.img2label[img_path]
            label = torch.tensor(label, dtype=torch.long)

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

# ==============================
# 全局预处理参数
# ==============================
channel_mean = [0.485, 0.456, 0.406]
channel_std = [0.229, 0.224, 0.225]

vit_train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.6),
    transforms.RandomRotation(degrees=30),
    transforms.ToTensor(),
    transforms.Normalize(mean=channel_mean, std=channel_std),
])

vit_valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=channel_mean, std=channel_std),
])

# ==============================
# 构建数据集
# ==============================
dataset = DogDataset(img_path=TRAIN_IMG_DIR, csv_path=LABEL_CSV)
indexes = list(range(len(dataset)))
train_idx, valid_idx = train_test_split(indexes, test_size=0.1, random_state=2022)

# 给子集设置 transform
class TransformSubset(Subset):
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform
    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transform:
            x = self.transform(x)
        return x, y
    # 🔥 修复 PyTorch 2.0+ 强制要求
    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]

train_dataset = TransformSubset(dataset, train_idx, vit_train_transform)
valid_dataset = TransformSubset(dataset, valid_idx, vit_valid_transform)

print(f"Train samples: {len(train_dataset)}")
print(f"Valid samples: {len(valid_dataset)}")

# ==============================
# DataLoader
# ==============================
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=0)

# ==============================
# 绘图展示样本
# ==============================
def show_samples(batch_img, batch_label, idx2name, num_samples=8):
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        img = batch_img[i].permute(1, 2, 0).cpu().numpy()
        img = img * channel_std + channel_mean
        img = img.clip(0, 1)
        plt.subplot(2, 4, i+1)
        plt.imshow(img)
        plt.title(idx2name[batch_label[i].item()])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 展示一批数据
batch_img, batch_label = next(iter(train_loader))
show_samples(batch_img, batch_label, dataset.label_idx2name, num_samples=8)

# ==============================
# ViT 模型
# ==============================
class PretrainViT(nn.Module):
    def __init__(self, num_classes=120):
        super().__init__()
        self.model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
        hidden_dim = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(hidden_dim, num_classes)

        # 冻结主干
        for name, param in self.model.named_parameters():
            if "heads" not in name:
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)

net = PretrainViT(num_classes=120)
net.to(device)

# ==============================
# 优化器与损失
# ==============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.009, momentum=0.9)

def get_accuracy(output, label):
    pred = torch.argmax(output, dim=1)
    return (pred == label).sum().item() / label.size(0)

# ==============================
# 训练 & 验证
# ==============================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    for img, label in loader:
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += get_accuracy(out, label)

    return total_loss / len(loader), total_acc / len(loader)

@torch.no_grad()
def valid_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    for img, label in loader:
        img, label = img.to(device), label.to(device)
        out = model(img)
        loss = criterion(out, label)
        total_loss += loss.item()
        total_acc += get_accuracy(out, label)
    return total_loss / len(loader), total_acc / len(loader)

# ==============================
# 开始训练
# ==============================
EPOCHS = 7
best_loss = float('inf')

train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(net, train_loader, optimizer, criterion, device)
    val_loss, val_acc = valid_one_epoch(net, valid_loader, criterion, device)

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(net.state_dict(), "net.pt")
        print("✅ Save best model")

    print(f"Epoch {epoch:2d} | train_loss:{train_loss:.3f} train_acc:{train_acc:.3f} | val_loss:{val_loss:.3f} val_acc:{val_acc:.3f}")

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

# ==============================
# 加载最优模型
# ==============================
net = PretrainViT(num_classes=120)
net.load_state_dict(torch.load("net.pt", map_location=device))
net.to(device)
net.eval()

# ==============================
# 测试集预测
# ==============================
class TestDataset(Dataset):
    def __init__(self, test_ids, img_dir, transform):
        self.test_ids = test_ids
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.test_ids)

    def __getitem__(self, idx):
        tid = self.test_ids[idx]
        path = os.path.join(self.img_dir, f"{tid}.jpg")
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, tid

submit_df = pd.read_csv(SUBMIT_CSV)
test_ids = submit_df["id"].values
test_dataset = TestDataset(test_ids, TEST_IMG_DIR, vit_valid_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class_names = list(dataset.label_idx2name)

# ==============================
# 生成提交文件
# ==============================
dfs = []
with torch.no_grad():
    for img, tid in tqdm(test_loader):
        img = img.to(device)
        out = net(img)
        prob = F.softmax(out, dim=1).cpu().numpy()
        df = pd.DataFrame(prob, columns=class_names)
        df.insert(0, "id", tid)
        dfs.append(df)

final_sub = pd.concat(dfs, ignore_index=True)
final_sub.to_csv("submit.csv", index=False)
print("✅ 提交文件已保存：submit.csv")