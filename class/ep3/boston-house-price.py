# 导入所需库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing  # 替代波士顿房价数据集（原数据集已移除）
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（解决matplotlib中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
# plt.rcParams['font.sans-serif'] = ['PingFang SC']  # Mac
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 1. 数据加载与基本探索 ----------------------
# 加载加州房价数据集（替代波士顿房价数据集，数据结构类似）
housing = fetch_california_housing()
# 转换为DataFrame方便操作
df = pd.DataFrame(data=housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target  # 房价中位数（目标变量）

# 基本信息展示
print("数据集基本信息：")
print(f"数据集形状: {df.shape}")
print("\n前5行数据：")
print(df.head())
print("\n数据描述性统计：")
print(df.describe())

# ---------------------- 2. 数据可视化探索 ----------------------
# 创建画布，设置子图布局
fig = plt.figure(figsize=(16, 12))

# 子图1：房价分布直方图
ax1 = plt.subplot(2, 3, 1)
sns.histplot(df['MedHouseVal'], kde=True, color='skyblue', ax=ax1)
ax1.set_title('加州房价中位数分布', fontsize=12)
ax1.set_xlabel('房价中位数（万美元）')
ax1.set_ylabel('频数')

# 子图2：各特征与房价的相关性热力图
ax2 = plt.subplot(2, 3, 2)
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax2)
ax2.set_title('特征相关性热力图', fontsize=12)

# 子图3：平均收入与房价的散点图（最相关特征）
ax3 = plt.subplot(2, 3, 3)
sns.scatterplot(x='MedInc', y='MedHouseVal', data=df, alpha=0.6, ax=ax3)
ax3.set_title('平均收入 vs 房价中位数', fontsize=12)
ax3.set_xlabel('平均收入（万美元）')
ax3.set_ylabel('房价中位数（万美元）')

# 子图4：房屋年龄分布
ax4 = plt.subplot(2, 3, 4)
sns.histplot(df['HouseAge'], bins=20, color='orange', ax=ax4)
ax4.set_title('房屋年龄分布', fontsize=12)
ax4.set_xlabel('房屋年龄（年）')
ax4.set_ylabel('频数')

# 子图5：每户平均房间数 vs 房价
ax5 = plt.subplot(2, 3, 5)
sns.scatterplot(x='AveRooms', y='MedHouseVal', data=df, alpha=0.6, color='green', ax=ax5)
ax5.set_title('每户平均房间数 vs 房价中位数', fontsize=12)
ax5.set_xlabel('平均房间数')
ax5.set_ylabel('房价中位数（万美元）')

# 调整子图间距
plt.tight_layout()
plt.savefig('房价数据探索可视化.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------------- 3. 模型训练 ----------------------
# 划分特征和目标变量
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 模型评估
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n模型评估指标：")
print(f"均方误差(MSE): {mse:.4f}")
print(f"均方根误差(RMSE): {rmse:.4f}")
print(f"决定系数(R²): {r2:.4f}")

# ---------------------- 4. 预测结果可视化 ----------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 子图1：真实值 vs 预测值散点图
ax1.scatter(y_test, y_pred, alpha=0.6, color='blue')
# 添加参考线（完美预测线）
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_title('真实房价 vs 预测房价', fontsize=12)
ax1.set_xlabel('真实房价中位数（万美元）')
ax1.set_ylabel('预测房价中位数（万美元）')
ax1.text(0.1, 4.5, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

# 子图2：预测误差分布
error = y_test - y_pred
ax2.hist(error, bins=50, color='purple', alpha=0.7)
ax2.axvline(x=0, color='red', linestyle='--', lw=2)
ax2.set_title('预测误差分布', fontsize=12)
ax2.set_xlabel('误差（万美元）')
ax2.set_ylabel('频数')
ax2.text(-2, 150, f'平均误差 = {error.mean():.4f}', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('房价预测结果可视化.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------------- 5. 特征重要性可视化 ----------------------
# 计算特征重要性（线性回归的系数绝对值）
feature_importance = pd.DataFrame({
    '特征': X.columns,
    '系数': model.coef_
})
feature_importance['系数绝对值'] = abs(feature_importance['系数'])
feature_importance = feature_importance.sort_values('系数绝对值', ascending=False)

# 绘制特征重要性条形图
plt.figure(figsize=(10, 6))
sns.barplot(x='系数绝对值', y='特征', data=feature_importance, palette='viridis')
plt.title('特征重要性排序（线性回归系数）', fontsize=12)
plt.xlabel('系数绝对值')
plt.ylabel('特征名称')
plt.tight_layout()
plt.savefig('特征重要性可视化.png', dpi=300, bbox_inches='tight')
plt.show()