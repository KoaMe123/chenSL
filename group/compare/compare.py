import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ===================== 基础配置（修复中文+路径）=====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===================== 1. 读取4个模型数据 =====================
# 按你的文件命名匹配，确保路径正确
model_data = {
    "YOLOv8_原始": pd.read_csv("v8-results.csv"),  # YOLOv8 无Diff数据
    "YOLOv8_Diff": pd.read_csv("v8_with_diff-results.csv"),  # YOLOv8 加Diff数据
    "YOLOv26_原始": pd.read_csv("26-results.csv"),  # YOLOv26 无Diff数据
    "YOLOv26_Diff": pd.read_csv("26_with_diff-results.csv")  # YOLOv26 加Diff数据
}

# 定义颜色（区分模型+Diff，海报配色更协调）
colors = {
    "YOLOv8_原始": "#1f77b4",    # 蓝色
    "YOLOv8_Diff": "#ff7f0e",    # 橙色（Diff用暖色区分）
    "YOLOv26_原始": "#2ca02c",   # 绿色
    "YOLOv26_Diff": "#d62728"    # 红色（Diff用暖色区分）
}

# ===================== 2. 提取4模型最终指标 =====================
final_metrics = []
for name, df in model_data.items():
    final = df.iloc[-1]  # 取最后一轮数据
    final_metrics.append({
        "模型名称": name,
        "训练轮数": int(final["epoch"]),
        "mAP50": round(final["metrics/mAP50(B)"], 4),
        "mAP50-95": round(final["metrics/mAP50-95(B)"], 4),
        "精确率(P)": round(final["metrics/precision(B)"], 4),
        "召回率(R)": round(final["metrics/recall(B)"], 4),
        "验证框损失": round(final["val/box_loss"], 4),
        "验证分类损失": round(final["val/cls_loss"], 4)
    })

# 转为表格，方便查看和排序
metrics_df = pd.DataFrame(final_metrics)
# 按 mAP50 降序排序（突出最优模型）
metrics_df_sorted = metrics_df.sort_values("mAP50", ascending=False).reset_index(drop=True)

# ===================== 3. 打印4模型完整对比表 =====================
print("=" * 100)
print("📊 4模型完整性能对比表（按mAP50降序）")
print("=" * 100)
print(metrics_df_sorted.to_string(index=False))
print("=" * 100)

# 计算综合评分（mAP50 + mAP50-95，突出Diff效果）
metrics_df_sorted["综合评分"] = metrics_df_sorted["mAP50"] + metrics_df_sorted["mAP50-95"]
print("\n🏆 综合评分（mAP50 + mAP50-95）")
for _, row in metrics_df_sorted.iterrows():
    print(f"{row['模型名称']}：{row['综合评分']:.4f}")

# ===================== 4. 自动分析Diff效果 + 最优模型 =====================
print("\n✅ 关键结论分析：")
# 1. Diff数据增强效果
v8_diff_gain = metrics_df_sorted[metrics_df_sorted["模型名称"].str.contains("YOLOv8")]
v26_diff_gain = metrics_df_sorted[metrics_df_sorted["模型名称"].str.contains("YOLOv26")]

v8_gain = v8_diff_gain[v8_diff_gain["模型名称"].str.contains("Diff")]["mAP50"].iloc[0] - \
          v8_diff_gain[~v8_diff_gain["模型名称"].str.contains("Diff")]["mAP50"].iloc[0]
v26_gain = v26_diff_gain[v26_diff_gain["模型名称"].str.contains("Diff")]["mAP50"].iloc[0] - \
           v26_diff_gain[~v26_diff_gain["模型名称"].str.contains("Diff")]["mAP50"].iloc[0]

print(f"1. Diff数据增强效果：")
print(f"   - YOLOv8 加Diff后 mAP50 提升：{v8_gain:.4f}")
print(f"   - YOLOv26 加Diff后 mAP50 提升：{v26_gain:.4f}")

# 2. 最优模型
best_model = metrics_df_sorted.iloc[0]
print(f"\n2. 最优模型：{best_model['模型名称']}")
print(f"   - 最高 mAP50：{best_model['mAP50']}")
print(f"   - 综合评分第一：{best_model['综合评分']:.4f}")
print("=" * 100)

# ===================== 5. 绘制5张独立的专业对比图（每张单独保存） =====================
# 定义单张图的尺寸（适合海报，保持高清）
fig_size = (12, 8)

# 图1：4模型 mAP50 收敛曲线（核心对比）
plt.figure(figsize=fig_size)
for name, df in model_data.items():
    plt.plot(df["epoch"], df["metrics/mAP50(B)"],
             label=name, color=colors[name], linewidth=2.5, alpha=0.8)
plt.title("4模型 mAP50 收敛曲线对比", fontsize=18, fontweight='bold', pad=20)
plt.xlabel("训练轮数（Epoch）", fontsize=14)
plt.ylabel("mAP50", fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("1_mAP50收敛曲线.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ 图1已保存：1_mAP50收敛曲线.png")

# 图2：4模型 mAP50-95 收敛曲线
plt.figure(figsize=fig_size)
for name, df in model_data.items():
    plt.plot(df["epoch"], df["metrics/mAP50-95(B)"],
             label=name, color=colors[name], linewidth=2.5, alpha=0.8)
plt.title("4模型 mAP50-95 收敛曲线对比", fontsize=18, fontweight='bold', pad=20)
plt.xlabel("训练轮数（Epoch）", fontsize=14)
plt.ylabel("mAP50-95", fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("2_mAP50-95收敛曲线.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ 图2已保存：2_mAP50-95收敛曲线.png")

# 图3：4模型 验证框损失对比
plt.figure(figsize=fig_size)
for name, df in model_data.items():
    plt.plot(df["epoch"], df["val/box_loss"],
             label=name, color=colors[name], linewidth=2.5, alpha=0.8)
plt.title("4模型 验证集 Box Loss 对比", fontsize=18, fontweight='bold', pad=20)
plt.xlabel("训练轮数（Epoch）", fontsize=14)
plt.ylabel("Box Loss（越低越好）", fontsize=14)
plt.legend(loc='upper right', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("3_BoxLoss对比.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ 图3已保存：3_BoxLoss对比.png")

# 图4：4模型 最终 mAP50 柱状图（海报重点！）
plt.figure(figsize=fig_size)
x = np.arange(len(metrics_df_sorted))
bars = plt.bar(x, metrics_df_sorted["mAP50"],
               color=[colors[name] for name in metrics_df_sorted["模型名称"]],
               alpha=0.8, edgecolor='black', linewidth=1)
# 在柱子上标数值
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{height:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.title("4模型 最终 mAP50 对比", fontsize=18, fontweight='bold', pad=20)
plt.xlabel("模型", fontsize=14)
plt.ylabel("mAP50", fontsize=14)
plt.xticks(x, metrics_df_sorted["模型名称"], rotation=45, ha='right', fontsize=11)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("4_最终mAP50对比.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ 图4已保存：4_最终mAP50对比.png")

# 图5：4模型 精确率+召回率 双柱状图
plt.figure(figsize=fig_size)
x = np.arange(len(metrics_df_sorted))
width = 0.35
# 精确率柱子
p_bars = plt.bar(x - width/2, metrics_df_sorted["精确率(P)"],
                 width, label='精确率(P)', alpha=0.8, edgecolor='black', linewidth=1)
# 召回率柱子
r_bars = plt.bar(x + width/2, metrics_df_sorted["召回率(R)"],
                 width, label='召回率(R)', alpha=0.8, edgecolor='black', linewidth=1)
# 标数值
for bars in [p_bars, r_bars]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=11)
plt.title("4模型 精确率(P) vs 召回率(R)", fontsize=18, fontweight='bold', pad=20)
plt.xlabel("模型", fontsize=14)
plt.ylabel("数值（越高越好）", fontsize=14)
plt.xticks(x, metrics_df_sorted["模型名称"], rotation=45, ha='right', fontsize=11)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("5_精确率召回率对比.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ 图5已保存：5_精确率召回率对比.png")

print("\n🎉 所有5张图已单独保存完成！")
print("📁 生成的文件列表：")
print("   1. 1_mAP50收敛曲线.png")
print("   2. 2_mAP50-95收敛曲线.png")
print("   3. 3_BoxLoss对比.png")
print("   4. 4_最终mAP50对比.png")
print("   5. 5_精确率召回率对比.png")
print("\n💡 所有图片均为300dpi高清格式，可直接用于海报/论文/汇报展示！")