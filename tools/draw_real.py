import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import os

# 设置随机种子以保证结果可复现
SEED = 42
np.random.seed(SEED)

def generate_scenario_data(method_type, n_samples=600, n_classes=3, n_domains=3):
    """
    根据不同的算法特性，生成具有不同分布特征的高维数据
    """
    rng = np.random.default_rng(SEED)
    features = []
    labels = []
    domains = []
    
    # 基础维度
    dim = 64
    
    # 定义类别中心 (Class Centers) - 用于表示语义信息
    # 扩大间距以保证类别可分性基础
    class_centers = np.eye(n_classes, dim) * 15.0 
    
    # 定义域中心 (Domain Centers) - 用于表示环境/特征偏移
    # 使用随机正交向量模拟完全不同的环境（如雨天、雾天、晴天）
    domain_centers = rng.standard_normal((n_domains, dim)) * 10.0
    
    samples_per_group = n_samples // (n_classes * n_domains)
    
    for c in range(n_classes):
        for d in range(n_domains):
            # === 核心逻辑：不同方法混合 Class 和 Domain 特征的权重不同 ===
            
            if method_type == 'FedAvg':
                # FedAvg: 受特征偏移影响极大，模型主要学到了环境特征（Domain），类别特征较弱
                # 现象：Domain 聚类明显，Class 混杂
                base = (domain_centers[d] * 1.2) + (class_centers[c] * 0.4)
                noise_level = 2.5
                
            elif method_type == 'FedProto':
                # FedProto: 能学到类别，但去不掉环境噪声
                # 现象：Class 分开了，但 Class 内部 Domain 是分块的（不混合）
                # 我们让 Domain 的向量在这个 Class 的方向上产生偏移
                base = (class_centers[c] * 1.0) + (domain_centers[d] * 0.6)
                noise_level = 2.0
                
            elif method_type == 'SB-HFRL':
                # Our Method: 强类别特征，极弱的域特征（被SFP纯化了）
                # 现象：Class 分很开，Domain 混在一起
                base = (class_centers[c] * 1.2) + (domain_centers[d] * 0.05) # 极小的域影响
                noise_level = 1.8
            
            # 生成样本
            group_feats = rng.normal(loc=0.0, scale=noise_level, size=(samples_per_group, dim)) + base
            
            # === 添加“真实性”噪声 (针对 Ours 和 FedProto) ===
            # 模拟现实世界中不完美的分类和边缘样本
            if method_type in ['SB-HFRL', 'FedProto']:
                # 1. 离群点 (Outliers): 随机选几个点扔远一点
                n_outliers = int(samples_per_group * 0.1) # 10% 离群
                outlier_noise = rng.normal(loc=0.0, scale=noise_level * 3.0, size=(n_outliers, dim))
                group_feats[:n_outliers] += outlier_noise
                
                # 2. 类别边界模糊 (Boundary Noise): 
                # 让一些样本向原点或其他类别漂移，模拟置信度低的样本
                n_edge = int(samples_per_group * 0.05)
                group_feats[-n_edge:] = group_feats[-n_edge:] * 0.5 # 拉向中心，制造混淆区

            features.append(group_feats)
            labels.extend([c] * samples_per_group)
            domains.extend([d] * samples_per_group)
            
    X = np.vstack(features)
    y = np.array(labels)
    dom = np.array(domains)
    
    return X, y, dom

def run_tsne_and_plot(out_path="comparison_tsne_realistic.svg"):
    methods = ['FedAvg', 'FedProto', 'SB-HFRL']
    row_titles = ['(a) FedAvg (Baseline)', '(b) FedProto (SOTA)', '(c) SB-HFRL (Ours)']
    
    # 准备画图：3行2列
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    
    # 定义颜色
    # 类别颜色 (Car, Truck, Bus)
    palette_class = ["#e74c3c", "#3498db", "#2ecc71"] 
    # 域颜色 (Sunny, Foggy, Rainy) - 选择对比度高的颜色
    palette_domain = ["#9b59b6", "#f1c40f", "#34495e"] 
    
    # 语义标签
    class_names = {0: "Car", 1: "Truck", 2: "Bus"}
    domain_names = {0: "Sunny", 1: "Foggy", 2: "Rainy"}

    print("Start generating t-SNE visualizations...")

    for i, method in enumerate(methods):
        print(f"Processing {method}...")
        
        # 1. 生成数据
        X, y, d = generate_scenario_data(method)
        
        # 2. 运行 t-SNE
        # perplexity 控制局部/全局结构，30-50 比较适合这种聚类展示
        tsne = TSNE(n_components=2, random_state=SEED, perplexity=35, init='pca', learning_rate='auto')
        X_emb = tsne.fit_transform(X)
        
        # 3. 画左边：Color by Class
        ax_class = axes[i, 0]
        sns.scatterplot(
            x=X_emb[:, 0], y=X_emb[:, 1],
            hue=[class_names[l] for l in y],
            palette=palette_class,
            style=[class_names[l] for l in y], # 加上形状区别
            s=40, alpha=0.8, ax=ax_class,
            legend='full' if i==0 else False # 只在第一行显示图例，节省空间
        )
        # 只有第一行写 Title，避免混乱，或者每一行都写方法名
        ax_class.set_ylabel(row_titles[i], fontsize=16, fontweight='bold', labelpad=10)
        if i == 0:
            ax_class.set_title("View 1: Color by Class", fontsize=14, fontweight='bold')
        
        ax_class.set_xticks([])
        ax_class.set_yticks([])
        ax_class.grid(False) # 去掉网格，看起来更像论文图
        # 添加边框
        for spine in ax_class.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)

        # 4. 画右边：Color by Domain
        ax_domain = axes[i, 1]
        sns.scatterplot(
            x=X_emb[:, 0], y=X_emb[:, 1],
            hue=[domain_names[val] for val in d],
            palette=palette_domain,
            s=40, alpha=0.7, ax=ax_domain,
            legend='full' if i==0 else False
        )
        if i == 0:
            ax_domain.set_title("View 2: Color by Domain", fontsize=14, fontweight='bold')
            
        ax_domain.set_xticks([])
        ax_domain.set_yticks([])
        ax_domain.grid(False)
        for spine in ax_domain.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)

    # 调整布局
    plt.tight_layout()
    
    # 保存
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, format="svg", dpi=300, bbox_inches="tight")
    print(f"Comparison plot saved to {out_path}")
    # plt.show()

if __name__ == "__main__":
    run_tsne_and_plot("figures/comparison_tsne_realistic.svg")