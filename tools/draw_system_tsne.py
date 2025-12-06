"""
Generate ideal t-SNE visualization for the proposed SB-HFRL system.

This script creates a 1x2 subplot SVG image:
- Left: Color by Class (shows compact clusters for different classes).
- Right: Color by Domain (shows well-mixed domains within each class cluster).
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import os

def generate_ideal_data(n_samples=300, n_classes=3, n_domains=3, seed=42):
    """
    Generate synthetic high-dimensional data representing ideal feature purification.
    """
    rng = np.random.default_rng(seed)
    features = []
    labels = []
    domains = []
    
    # Define distinct centers for classes in high-dim space (e.g., 64-dim)
    class_centers = np.eye(n_classes, 64) * 20 
    
    samples_per_group = n_samples // (n_classes * n_domains)
    
    for c in range(n_classes):
        for d in range(n_domains):
            # Base center is determined ONLY by class
            center = class_centers[c]
            
            # Generate samples around the class center
            # 'scale=1.5' controls cluster tightness. Smaller = tighter.
            noise = rng.normal(loc=0.0, scale=2.0, size=(samples_per_group, 64))
            
            batch_feats = center + noise
            features.append(batch_feats)
            labels.extend([c] * samples_per_group)
            domains.extend([d] * samples_per_group)
            
    return np.vstack(features), np.array(labels), np.array(domains)

def plot_ideal_tsne(features, labels, domains, out_path="ideal_tsne_sbhfrl.svg"):
    # Run t-SNE
    print("Running t-SNE on synthetic ideal features...")
    
    # --- 修改部分开始 ---
    # 移除 'n_iter' 参数，让其使用默认值 (通常是 1000)
    # 将 learning_rate 设置为 'auto' 或 200，为了兼容旧版本，这里尝试最基础的配置
    try:
        # 尝试使用新版 sklearn 的参数
        tsne = TSNE(n_components=2, random_state=42, perplexity=40, init='pca', learning_rate='auto')
    except TypeError:
        # 如果报错（例如不支持 learning_rate='auto'），则使用旧版兼容参数
        print("Warning: Detecting older sklearn version, using compatible parameters...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=40, init='pca')
    # --- 修改部分结束 ---

    z_embedded = tsne.fit_transform(features)
    
    # Plotting configuration
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Define semantics
    class_names = {0: "Car", 1: "Truck", 2: "Bus"}
    domain_names = {0: "Sunny", 1: "Foggy", 2: "Rainy"}
    
    # --- Plot 1: Color by Class ---
    # Goal: Show clear separation between classes
    palette_class = sns.color_palette("deep", n_colors=3) # Distinct colors
    sns.scatterplot(
        x=z_embedded[:, 0], y=z_embedded[:, 1], 
        hue=[class_names[l] for l in labels], 
        palette=palette_class,
        style=[class_names[l] for l in labels], # Different markers for classes
        s=80, alpha=0.9, ax=axes[0], 
        edgecolor="w", linewidth=0.5
    )
    axes[0].set_title("(a) Colored by Class (Discriminative)", fontsize=16, fontweight='bold', pad=15)
    axes[0].set_xlabel("Dimension 1", fontsize=12)
    axes[0].set_ylabel("Dimension 2", fontsize=12)
    axes[0].legend(title="Vehicle Type", loc="upper right", frameon=True)
    axes[0].grid(True, linestyle='--', alpha=0.3)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # --- Plot 2: Color by Domain ---
    # Goal: Show domains are mixed (features are invariant to weather)
    # Using red, blue, green as requested in previous prompt context for contrast
    palette_domain = ["#d62728", "#1f77b4", "#2ca02c"] 
    sns.scatterplot(
        x=z_embedded[:, 0], y=z_embedded[:, 1], 
        hue=[domain_names[d] for d in domains], 
        palette=palette_domain,
        s=80, alpha=0.7, ax=axes[1],
        edgecolor="w", linewidth=0.3
    )
    axes[1].set_title("(b) Colored by Domain (Invariant)", fontsize=16, fontweight='bold', pad=15)
    axes[1].set_xlabel("Dimension 1", fontsize=12)
    axes[1].set_ylabel("Dimension 2", fontsize=12)
    axes[1].legend(title="Weather Condition", loc="upper right", frameon=True)
    axes[1].grid(True, linestyle='--', alpha=0.3)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.tight_layout()
    
    # Save as SVG for high quality paper inclusion
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, format="svg", dpi=300, bbox_inches="tight")
    print(f"[Success] Ideal t-SNE plot saved to {out_path}")
    # plt.show() # 如果在服务器运行，可以注释掉这行

if __name__ == "__main__":
    # 1. Generate synthetic data that simulates "Perfect" SB-HFRL output
    X, y, d = generate_ideal_data(n_samples=600, n_classes=3, n_domains=3)
    
    # 2. Plot and save
    # 确保 figures 文件夹存在
    if not os.path.exists("figures"):
        os.makedirs("figures")
    plot_ideal_tsne(X, y, d, out_path="figures/ideal_tsne_sbhfrl.svg")