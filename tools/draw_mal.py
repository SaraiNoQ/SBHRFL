import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 数据准备 (模拟数据，请替换为你真实的实验数据)
# ==========================================
# 攻击强度标签
ratios = [0, 10, 20, 30]  # 对应 0%, 10%, 20%, 30%

# 方法列表
methods = ['FedAvg', 'FedProto', 'SBFRL (Ours)']

# 数据结构：每个方法的准确率列表
# FedAvg: 只有 0%, 10%, 20%
acc_fedavg = [63.70, 47.23, 45.1] 
# FedProto: 只有 0%, 10%, 20%
acc_fedproto = [80.03, 20.87, 2.34]
# Ours: 包含 0%, 10%, 20%, 30%
acc_ours = [98.05, 95.97, 92.18, 83.72]

# 将数据整合，方便循环处理
data = [
    {'name': 'FedAvg',   'acc': acc_fedavg,   'ratios': [0, 10, 20]},
    {'name': 'FedProto', 'acc': acc_fedproto, 'ratios': [0, 10, 20]},
    {'name': 'SBFRL',  'acc': acc_ours,     'ratios': [0, 10, 20, 30]}
]

# ==========================================
# 2. 绘图参数设置
# ==========================================
# 定义颜色映射：攻击强度 -> 颜色
# 0%: 蓝色, 10%: 橙色, 20%: 绿色, 30%: 红色 (使用柔和的色板)
colors = {
    0:  '#4c72b0', # Blue
    10: '#dd8452', # Orange
    20: '#55a868', # Green
    30: '#c44e52'  # Red
}

bar_width = 0.2       # 每个柱子的宽度
group_spacing = 1.0   # 不同方法组之间的中心距离
# 初始组中心位置
group_centers = [1.0, 2.0, 3.0] 

# 创建画布
fig, ax = plt.subplots(figsize=(10, 6))

# ==========================================
# 3. 核心绘图逻辑 (手动计算坐标)
# ==========================================

for i, item in enumerate(data):
    method_name = item['name']
    accuracies = item['acc']
    current_ratios = item['ratios']
    center_x = group_centers[i]
    n_bars = len(accuracies)
    
    # 计算该组内每个柱子的偏移量，使其居中对齐
    # 公式：offset = (index - (n-1)/2) * width
    offsets = [(j - (n_bars - 1) / 2) * bar_width for j in range(n_bars)]
    
    for j, (acc, ratio) in enumerate(zip(accuracies, current_ratios)):
        x_pos = center_x + offsets[j]
        color = colors[ratio]
        
        # 绘制柱子
        # zorder=3 保证柱子在网格线之上
        bars = ax.bar(x_pos, acc, width=bar_width, color=color, 
                      edgecolor='black', linewidth=0.8, zorder=3)
        
        # (可选) 在柱子上方显示数值
        ax.text(x_pos, acc + 1, f'{acc:.1f}', ha='center', va='bottom', fontsize=10, fontfamily="Times New Roman")

# ==========================================
# 4. 图表美化 (符合期刊规范)
# ==========================================

# 设置 X 轴标签 (方法名称)
ax.set_xticks(group_centers)
ax.set_xticklabels(methods, fontsize=14, fontweight='bold', fontfamily="Times New Roman")

ax.set_xlim(0.2, 4.2)

# 设置 Y 轴
ax.set_ylim(0, 105)
ax.set_ylabel('Global Accuracy (%)', fontsize=16, fontweight='bold', fontfamily="Times New Roman")
ax.set_yticks(np.arange(0, 101, 10))

# 设置标题 (根据你的要求，表达特定含义)
# title_text = "Robustness Evaluation under Targeted Poisoning and Noise Injection"
# ax.set_title(title_text, fontsize=14, pad=20)

# 添加网格线 (仅 Y 轴，置于底层)
ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)

# 创建自定义图例 (Legend)
# 因为前面的柱子是循环画的，直接调用 legend 会混乱，这里手动创建句柄
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors[0], edgecolor='black', label=r'$\alpha_{mal}=0\%$'),
    Patch(facecolor=colors[10], edgecolor='black', label=r'$\alpha_{mal}=10\%$'),
    Patch(facecolor=colors[20], edgecolor='black', label=r'$\alpha_{mal}=20\%$'),
    Patch(facecolor=colors[30], edgecolor='black', label=r'$\alpha_{mal}=30\%$')
]

ax.legend(handles=legend_elements, title="Malicious Ratio", 
          fontsize=10, title_fontsize=11, loc='upper right', frameon=True)

# 调整边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# ==========================================
# 5. 保存与显示
# ==========================================
plt.tight_layout()

# 保存为 SVG 矢量图，适合插入论文
plt.savefig('robustness_comparison.svg', format='svg', dpi=300)

print("绘图完成！图片已保存为 robustness_comparison.svg 和 robustness_comparison.png")
plt.show()