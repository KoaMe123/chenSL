# 波士顿房价预测 - 完整版（控制台输出 + 图表保存 + 修复数据集）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ===================== 创建 image 文件夹（自动） =====================
if not os.path.exists('./image'):
    os.makedirs('./image')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示

# ===================== 加载数据集（修复版） =====================
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
df = pd.DataFrame(data, columns=feature_names)
df['PRICE'] = target

# ===================== 【恢复】控制台原始输出 =====================
print("========== 数据集前5行 ==========")
print(df.head())

print("\n========== 数据集描述统计 ==========")
print(df.describe())

print("\n========== 缺失值检查 ==========")
print(df.isnull().sum())

# ===================== 1. 房价分布直方图 =====================
plt.figure(figsize=(10, 5))
sns.histplot(df['PRICE'], bins=30, kde=True, color='blue')
plt.title('房价分布直方图 + 核密度估计')
plt.xlabel('房价')
plt.ylabel('数量')
plt.savefig('./image/1_price_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# ===================== 2. 各特征分布直方图 =====================
df.hist(figsize=(16, 12), bins=20, color='green')
plt.tight_layout()
plt.savefig('./image/2_all_features_hist.png', dpi=300)
plt.close()

# ===================== 3. 相关性热力图 =====================
plt.figure(figsize=(14, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('特征相关性热力图')
plt.savefig('./image/3_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# ===================== 4. 箱线图（检测异常值） =====================
plt.figure(figsize=(16, 8))
df.boxplot()
plt.title('各特征箱线图（异常值检测）')
plt.xticks(rotation=45)
plt.savefig('./image/4_boxplot_outliers.png', dpi=300, bbox_inches='tight')
plt.close()

# ===================== 5. 房间数 RM 与房价散点图 =====================
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df['RM'], y=df['PRICE'], alpha=0.6)
plt.title('平均房间数（RM） vs 房价')
plt.savefig('./image/5_rm_vs_price.png', dpi=300, bbox_inches='tight')
plt.close()

# ===================== 6. 低收入人群比例 LSTAT 与房价 =====================
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df['LSTAT'], y=df['PRICE'], alpha=0.6, color='red')
plt.title('低收入人群比例（LSTAT） vs 房价')
plt.savefig('./image/6_lstat_vs_price.png', dpi=300, bbox_inches='tight')
plt.close()

# ===================== 7. 犯罪率 CRIM 与房价 =====================
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df['CRIM'], y=df['PRICE'], alpha=0.6, color='orange')
plt.title('城镇犯罪率（CRIM） vs 房价')
plt.savefig('./image/7_crim_vs_price.png', dpi=300, bbox_inches='tight')
plt.close()

# ===================== 8. 成对关系图（最重要特征） =====================
top_features = ['RM', 'LSTAT', 'PTRATIO', 'PRICE']
sns.pairplot(df[top_features])
plt.savefig('./image/8_pairplot.png', dpi=300)
plt.close()

# ===================== 9. 房屋年龄 AGE 与房价 =====================
plt.figure(figsize=(10,5))
sns.scatterplot(x=df['AGE'], y=df['PRICE'], alpha=0.5, color='purple')
plt.title('房龄（AGE） vs 房价')
plt.savefig('./image/9_age_vs_price.png', dpi=300, bbox_inches='tight')
plt.close()

# ===================== 模型训练 =====================
X = df.drop('PRICE', axis=1)
y = df['PRICE']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 线性回归
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 随机森林
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 评估函数
def evaluate(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

evaluate(y_test, y_pred_lr, "Linear Regression")
evaluate(y_test, y_pred_rf, "Random Forest")

# ===================== 【恢复】模型对比表 =====================
comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'RMSE': [
        np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        np.sqrt(mean_squared_error(y_test, y_pred_rf))
    ],
    'R2': [
        r2_score(y_test, y_pred_lr),
        r2_score(y_test, y_pred_rf)
    ]
})
print("\n========== 模型对比表 ==========")
print(comparison)

# ===================== 10. 预测值 vs 真实值对比图 =====================
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_rf, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('真实房价')
plt.ylabel('预测房价')
plt.title('随机森林：真实值 vs 预测值')
plt.savefig('./image/10_pred_vs_true.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ 所有图表已保存到 ./image 文件夹")