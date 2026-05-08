import numpy as np
import matplotlib.pyplot as plt
import os

# ===================== 全局设置 =====================
np.set_printoptions(precision=4)
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['font.size'] = 11
RESULT_DIR = "./adaboost_results"
os.makedirs(RESULT_DIR, exist_ok=True)

# ===================== 1. 构造二分类数据集（对齐PPT思想） =====================
# 简单 2D 二分类数据：y = ±1
X = np.array([
    [1, 5], [2, 4], [3, 6], [4, 3], [5, 7],
    [6, 2], [7, 1], [8, 3], [9, 4], [10, 2]
])
y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])

# 3个弱分类器（基于x1阈值）
def h1(x): return 1 if x[0] < 2.5 else -1
def h2(x): return 1 if x[0] < 6.5 else -1
def h3(x): return 1 if x[0] < 8.5 else -1

weak_clfs = [h1, h2, h3]
names = ["h1(x1<2.5)", "h2(x1<6.5)", "h3(x1<8.5)"]

# 预测结果
H = np.array([[h(x) for h in weak_clfs] for x in X])
err = (H != y.reshape(-1,1)).astype(int)

print("===== Weak Classifiers Predictions =====")
print(H)
print("\n===== Error Matrix =====")
print(err)

# ===================== 图1：原始数据集散点图 =====================
plt.figure()
plt.scatter(X[y==1,0], X[y==1,1], c='red', label='class +1', s=100, edgecolor='k')
plt.scatter(X[y==-1,0], X[y==-1,1], c='blue', label='class -1', s=100, edgecolor='k')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Original 2-Class Dataset")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/01_dataset.png", dpi=150)
plt.close()

# ===================== 2. AdaBoost 训练（3轮，对齐PPT） =====================
n_samples = len(X)
D = np.ones(n_samples) / n_samples  # 初始化权重
n_rounds = 3
alphas = []
selected = []
Ds = [D.copy()]

for t in range(n_rounds):
    print(f"\n===== Round {t+1} =====")
    # 计算每个弱分类器误差
    errs = []
    for j in range(3):
        e = np.sum(D * (H[:,j] != y)) / np.sum(D)
        errs.append(e)
    idx = np.argmin(errs)
    e = errs[idx]
    a = 0.5 * np.log((1 - e) / e)

    # 更新权重
    z = 2 * np.sqrt(e * (1 - e))
    D = D * np.exp(-a * y * H[:,idx]) / z

    alphas.append(a)
    selected.append(idx)
    Ds.append(D.copy())

    print(f"Select: {names[idx]}")
    print(f"Error: {e:.4f}, Alpha: {a:.4f}")
    print(f"Weights: {D}")

# ===================== 3. 强分类器 =====================
F = np.sum([alphas[i]*H[:,selected[i]] for i in range(n_rounds)], axis=0)
pred = np.sign(F)
acc = np.mean(pred == y)

print("\n===== Strong Classifier =====")
print("F(x) =", F)
print("Predict:", pred)
print("True:", y)
print(f"Accuracy: {acc:.2%}")

# ===================== 图2：每轮权重变化 =====================
plt.figure(figsize=(10,5))
for i, d in enumerate(Ds):
    plt.plot(range(1,11), d, marker='o', label=f'Round {i}')
plt.xlabel("Sample Index")
plt.ylabel("Weight")
plt.title("Sample Weight Update During AdaBoost")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/02_weights.png", dpi=150)
plt.close()

# ===================== 图3：弱分类器误差 =====================
plt.figure()
plt.bar(names, [np.mean(err[:,0]), np.mean(err[:,1]), np.mean(err[:,2])], color=['r','g','b'])
plt.ylim(0, 0.6)
plt.title("Error Rate of Weak Classifiers")
plt.ylabel("Error")
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/03_weak_error.png", dpi=150)
plt.close()

# ===================== 图4：弱分类器权重系数 Alpha =====================
plt.figure()
selected_names = [names[i] for i in selected]
plt.bar(selected_names, alphas, color='orange')
plt.title("Alpha Coefficients of Selected Weak Classifiers")
plt.ylabel("Alpha")
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/04_alphas.png", dpi=150)
plt.close()

# ===================== 图5：强分类器结果 =====================
plt.figure()
correct = (pred == y)
plt.scatter(X[correct,0], X[correct,1], c='green', label='Correct', s=100, edgecolor='k')
plt.scatter(X[~correct,0], X[~correct,1], c='red', label='Wrong', s=100, edgecolor='k')
for i, (a, b) in enumerate(zip(pred, y)):
    plt.annotate(f"P:{a}\nT:{b}", (X[i,0]+0.2, X[i,1]), fontsize=9)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title(f"AdaBoost Strong Classifier Result\nAccuracy = {acc:.2%}")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/05_final_result.png", dpi=150)
plt.close()

# ===================== 图6：每轮分类结果 =====================
for t in range(n_rounds):
    idx = selected[t]
    p = H[:,idx]
    c = (p == y)
    plt.figure()
    plt.scatter(X[c,0], X[c,1], c='green', s=90, label='Correct')
    plt.scatter(X[~c,0], X[~c,1], c='red', s=90, label='Wrong')
    plt.title(f"Round {t+1}: {names[idx]}, Alpha={alphas[t]:.3f}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/06_round{t+1}.png", dpi=150)
    plt.close()

# ===================== 保存报告 =====================
with open(f"{RESULT_DIR}/adaboost_report.txt", 'w', encoding='utf-8') as f:
    f.write("===== AdaBoost Binary Classification Report =====\n")
    f.write(f"Final Accuracy: {acc:.4f}\n")
    f.write(f"Selected weak classifiers: {[names[i] for i in selected]}\n")
    f.write(f"Alphas: {alphas}\n")
    f.write(f"Predict: {pred}\n")
    f.write(f"True: {y}\n")

print("\n✅ All done! 9 images + report saved to:", RESULT_DIR)