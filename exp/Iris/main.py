import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)
from matplotlib.colors import ListedColormap

# ===================== 全局设置 =====================
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.size"] = 11
np.random.seed(42)
RESULT_DIR = "./svm_iris_results"
import os
os.makedirs(RESULT_DIR, exist_ok=True)

# ===================== 1. 加载数据 =====================
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]  # 花瓣长、宽
y = iris.target
class_names = iris.target_names
feature_names = ["Petal Length", "Petal Width"]

# ===================== 图1：数据集类别数量统计 =====================
plt.figure()
sns.countplot(x=y)
plt.title("Iris Dataset Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks([0,1,2], class_names)
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/01_class_count.png", dpi=150)
plt.close()

# ===================== 图2：原始特征散点图 =====================
plt.figure()
for i, color in zip(range(3), ["red", "blue", "green"]):
    plt.scatter(X[y==i,0], X[y==i,1], c=color, label=class_names[i], edgecolors="k", s=60)
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Iris Data Scatter Plot")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/02_data_scatter.png", dpi=150)
plt.close()

# ===================== 图3：特征箱线图（查看分布） =====================
df_temp = pd.DataFrame(X, columns=feature_names)
df_temp["class"] = y
plt.figure()
df_temp.boxplot(column=feature_names, by="class", grid=False)
plt.suptitle("Feature Distribution by Class")
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/03_boxplot.png", dpi=150)
plt.close()

# ===================== 数据划分与标准化 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===================== 2. SVM 训练 =====================
svm = SVC(kernel="rbf", C=1, gamma="scale", random_state=42)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# ===================== 图4：SVM 决策边界（测试集） =====================
def plot_decision_boundary(X, y, model, title, save_path):
    h = 0.02
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure()
    cmap_light = ListedColormap(["#FFAAAA", "#AAAAFF", "#AAFFAA"])
    cmap_bold = ListedColormap(["#FF0000", "#0000FF", "#00FF00"])
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.5)
    plt.scatter(X[:,0], X[:,1], c=y, cmap=cmap_bold, edgecolor="k", s=60)
    plt.title(title)
    plt.xlabel("Petal Length (scaled)")
    plt.ylabel("Petal Width (scaled)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

plot_decision_boundary(
    X_test, y_test, svm,
    f"SVM Decision Boundary (Accuracy={acc:.2%})",
    f"{RESULT_DIR}/04_decision_boundary.png"
)

# ===================== 图5：混淆矩阵热力图 =====================
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/05_confusion_matrix.png", dpi=150)
plt.close()

# ===================== 图6：Precision / Recall / F1 柱状图 =====================
p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
metrics = pd.DataFrame({
    "Precision": p, "Recall": r, "F1": f1
}, index=class_names)

metrics.plot(kind="bar", figsize=(9,5))
plt.title("Precision, Recall, F1 Score for Each Class")
plt.ylim(0.8, 1.05)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/06_metrics_bar.png", dpi=150)
plt.close()

# ===================== 图7：10折交叉验证得分曲线 =====================
cv_scores = cross_val_score(svm, X_train, y_train, cv=10)
plt.figure()
plt.plot(range(1,11), cv_scores, marker="o", color="steelblue")
plt.ylim(0.9, 1.02)
plt.title("10-Fold Cross-Validation Scores")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/07_cross_validation.png", dpi=150)
plt.close()

# ===================== 图8：随机9个样本预测结果图 =====================
def plot_samples(X_test, y_test, y_pred, save_path):
    idx = np.random.choice(len(X_test), 9, replace=False)
    plt.figure(figsize=(9,9))
    for i in range(9):
        plt.subplot(3,3,i+1)
        x, yt, yp = X_test[idx[i]], y_test[idx[i]], y_pred[idx[i]]
        plt.scatter(x[0], x[1],
                    c="green" if yt==yp else "red", s=150)
        plt.title(f"True:{class_names[yt]}\nPred:{class_names[yp]}",
                  color="green" if yt==yp else "red")
        plt.xlim(X_test[:,0].min()-0.5, X_test[:,0].max()+0.5)
        plt.ylim(X_test[:,1].min()-0.5, X_test[:,1].max()+0.5)
        plt.grid(alpha=0.3)
    plt.suptitle("Random 9 Test Samples Prediction")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

plot_samples(X_test, y_test, y_pred, f"{RESULT_DIR}/08_sample_predictions.png")

# ===================== 保存结果文本 =====================
with open(f"{RESULT_DIR}/svm_report.txt", "w", encoding="utf-8") as f:
    f.write("===== SVM Iris Classification Report =====\n")
    f.write(f"Test Accuracy: {acc:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred, target_names=class_names))

print("✅ 全部完成！共生成 8 张实验图 + 1份报告文本")
print(f"路径：{os.path.abspath(RESULT_DIR)}")