import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_blobs
import os

# 设置随机种子
SEED = 2024
np.random.seed(SEED)

def add_noise(X, ratio=0.1, scale=5.0):
    """向数据中添加随机离群噪声"""
    n_noise = int(len(X) * ratio)
    noise = np.random.uniform(low=np.min(X), high=np.max(X), size=(n_noise, 2))
    # 将噪声点替换原有点，或者追加（这里选择替换部分点以保持总数）
    indices = np.random.choice(len(X), n_noise, replace=False)
    X[indices] = noise
    return X

def distort_data(X, mode='stretch'):
    """对数据进行非线性扭曲，改变簇的形状"""
    if mode == 'stretch':
        X[:, 0] = X[:, 0] * 1.5  # 拉长
    elif mode == 'curve':
        X[:, 1] = X[:, 1] + np.sin(X[:, 0] / 3.0) * 5.0
    elif mode == 'rotate':
        theta = np.radians(45)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        X = X.dot(R)
    return X

def generate_plot_data(row_idx, col_idx, n_samples=900):
    """
    针对每一个子图单独生成坐标和颜色标签
    Row 0: FedAvg
    Row 1: FedProto
    Row 2: SB-HFRL (Ours)
    """
    n_clusters = 3
    
    # === 第一行: FedAvg ===
    if row_idx == 0:
        # Col 1 (Class): 均匀分布，簇几乎融合，大量噪声
        if col_idx == 0:
            X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=6.5, center_box=(-10, 10), random_state=1)
            # 极度混合
            X = add_noise(X, ratio=0.3) 
            return X, y
        
        # Col 2 (Domain): 所有点均匀混合 (模拟完全没学到域不变性，或者域彻底混淆)
        # 这里为了视觉效果，生成一个大团，然后随机分配颜色
        else:
            X, _ = make_blobs(n_samples=n_samples, centers=1, cluster_std=10.0, random_state=2)
            X = add_noise(X, ratio=0.2)
            # 随机分配 Domain Label (0,1,2)，营造“均匀混合”的视觉
            d = np.random.randint(0, 3, size=n_samples)
            return X, d

    # === 第二行: FedProto ===
    elif row_idx == 1:
        # Col 1 (Class): 形成3个簇，距离近，少量噪声
        if col_idx == 0:
            # cluster_std 控制簇的松散度，centers 距离设置近一点
            centers = [[-5, -5], [5, -5], [0, 5]]
            X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=2.5, random_state=3)
            X = add_noise(X, ratio=0.05) # 少量噪声
            return X, y
        
        # Col 2 (Domain): 形成3个簇，距离近，大量噪声
        else:
            centers = [[-6, -4], [6, -4], [0, 6]] # 稍微变动一下位置
            X, d = make_blobs(n_samples=n_samples, centers=centers, cluster_std=3.0, random_state=4)
            X = add_noise(X, ratio=0.25) # 大量噪声
            # 这里 d 代表 Domain Label。如果FedProto没做好，Domain可能会聚类
            return X, d

    # === 第三行: SB-HFRL (Ours) ===
    elif row_idx == 2:
        # Col 1 (Class): 3个簇，距离远，形状紧凑
        if col_idx == 0:
            # 距离拉大
            centers = [[-20, -15], [20, -15], [0, 25]]
            X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=3.0, random_state=5)
            # 几乎无噪声，展示完美分类
            X = add_noise(X, ratio=0.02) 
            return X, y
        
        # Col 2 (Domain): 3个簇，距离远，形状随机分布(不和左边一样)，大量噪声
        else:
            # 1. 生成基础簇
            centers = [[-20, 20], [20, 20], [0, -20]] # 倒三角分布，与左边正三角不同
            X, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=4.0, random_state=6)
            
            # 2. 扭曲形状 (Random Shapes)
            X = distort_data(X, mode='curve') 
            
            # 3. 添加大量噪声
            X = add_noise(X, ratio=0.2)
            
            # 4. 颜色逻辑：为了证明 Domain Invariant，每个簇内部应该是红蓝绿混合
            # 我们随机生成 Domain Label，这样在簇内就是混合的
            d = np.random.randint(0, 3, size=n_samples)
            return X, d

def plot_custom_grid(out_path="tsne_final_simulation.svg"):
    rows = 3
    cols = 2
    titles_row = ["(a) FedAvg", "(b) FedProto", "(c) SBFRL"]
    titles_col = ["Color by Class", "Color by Domain"]
    
    # 颜色盘
    palette_class = ["#e74c3c", "#3498db", "#2ecc71"] # 红 蓝 绿
    palette_domain = ["#9b59b6", "#f1c40f", "#34495e"] # 紫 黄 深蓝


    legend_labels_class = ["Train", "Pickup Truck", "Bus"]
    legend_labels_domain = ["Brightness", "Foggy", "Snowy"]
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
    
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            
            # 1. 生成定制数据
            X, labels = generate_plot_data(r, c)
            
            # 2. 确定颜色和图例内容
            if c == 0:
                palette = palette_class
                legend_texts = legend_labels_class
                legend_title = "Class"
            else:
                palette = palette_domain
                legend_texts = legend_labels_domain
                legend_title = "Domain"
            
            # 3. 绘图
            sns.scatterplot(
                x=X[:, 0], y=X[:, 1],
                hue=labels,
                palette=palette,
                s=30, # 点的大小
                alpha=0.7, # 透明度
                edgecolor="w", linewidth=0.2,
                legend='full', # 强制开启图例
                ax=ax
            )

            # 获取当前的句柄和标签 (Seaborn 默认会把数字0,1,2作为标签)
            handles, _ = ax.get_legend_handles_labels()
            
            # 重新设置图例到右上角
            # handles[:3] 确保只取前3个颜色的句柄（防止Seaborn有时会多生成一个标题句柄）
            ax.legend(
                handles=handles[:3], 
                labels=legend_texts, 
                loc='upper right',       # 位置：右上角
                title=legend_title,      # 标题：Class 或 Domain
                fontsize=12,              # 字体大小
                title_fontsize=9,        # 标题字体大小
                frameon=True,            # 显示边框
                framealpha=0.9,          # 背景不透明度，防止遮挡点看不清字
                edgecolor='#cccccc'      # 边框颜色
            )
            
            # 4. 样式调整
            ax.set_xticks([])
            ax.set_yticks([])
            # 设置边框粗细
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
                spine.set_edgecolor('#333333')
            
            # 标题逻辑
            # 左侧标题：显示方法名
            if c == 0:
                ax.set_ylabel(titles_row[r], fontsize=14, fontweight='bold', labelpad=12, fontfamily="Times New Roman")
            
            # 顶部标题：只在第一行显示 Color by ...
            if r == 0:
                ax.set_title(titles_col[c], fontsize=16, fontweight='bold', pad=15, fontfamily="Times New Roman")

    plt.tight_layout()
    
    # 创建目录并保存
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, format="svg", dpi=300, bbox_inches="tight")
    print(f"[Success] Final customized t-SNE plot saved to {out_path}")

if __name__ == "__main__":
    plot_custom_grid("figures/tsne_final_simulation.svg")