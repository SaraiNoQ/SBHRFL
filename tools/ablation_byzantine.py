import matplotlib.pyplot as plt

def plot_ablation_feature_shift(
    base,
    base_sfp,
    base_sfp_attn,
    full_model,
    out_prefix="ablation_byzantine",
    title=None
):
    """
    Inputs: five lists (or 1D arrays) of length 100, values in [0, 100].
    Output: out_prefix + ".pdf" (vector) and out_prefix + ".png" (300dpi).
    """

    # --------- sanity checks ----------
    series = {
        "Base": base,
        "Base + Auditing": base_sfp,
        "Base + Multi-Factor Clustering": base_sfp_attn,
        "Base + Auditing + Multi-Factor Clustering": full_model,
    }
    for name, s in series.items():
        if len(s) != 100:
            raise ValueError(f"{name} length must be 100, got {len(s)}")
        if any((v < 0 or v > 100) for v in s):
            raise ValueError(f"{name} has value outside [0, 100]")

    # --------- journal-like typography ----------
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "pdf.fonttype": 42,   # embed TrueType (good for IEEE-style PDFs)
        "ps.fonttype": 42,
    })

    x = list(range(1, 101))

    fig, ax = plt.subplots(figsize=(6.2, 3.8))  # compact single-column friendly

    # Curves (no manual colors; use matplotlib default color cycle)
    ax.plot(x, base, linewidth=1.5, label="Base")
    ax.plot(x, base_sfp, linewidth=1.5, label="Base + Auditing")
    ax.plot(x, base_sfp_attn, linewidth=1.5, label="Base + Multi-Factor Clustering")
    ax.plot(x, full_model, linewidth=1.5, label="Base + Auditing + Multi-Factor Clustering")

    # Axes labels (English + Times New Roman)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy (%)")

    # Axis limits and ticks (adjust if you want tighter y-range)
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 80)
    ax.set_xticks([1, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 20, 40, 60, 80])

    # Grid and spines
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    # Legend
    ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="black")

    if title:
        ax.set_title(title)

    fig.tight_layout()

    # Save: vector PDF + high-res PNG
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_prefix}.pdf and {out_prefix}.png")


# ----------------- Example usage -----------------
# Replace the following dummy data with your 5 lists (each length 100).
if __name__ == "__main__":
    base = [6.29, 7.58, 9.30, 8.67, 6.29, 8.69, 8.54, 8.69, 8.18, 6.29, 
            8.69, 8.65, 6.29, 8.50, 7.20, 9.85, 7.60, 6.29, 8.34, 9.10, 6.29, 10.22, 8.73,
            6.29, 8.95, 7.64, 9.41, 8.06, 11.35, 9.88, 7.50, 6.29, 8.82,
            10.50, 9.15, 6.29, 9.67, 8.20, 12.40, 10.82, 8.93, 6.29, 9.30,
            7.85, 11.90, 10.18, 6.29, 8.75, 9.46, 7.90, 13.20, 11.40, 9.60,
            6.29, 10.05, 8.40, 12.85, 11.10, 9.82, 6.29, 9.15, 7.75, 10.95,
            8.90, 6.29, 9.80, 8.60, 12.10, 10.50, 9.30, 6.29, 8.45, 10.20,
            7.60, 11.70, 10.85, 9.10, 6.29, 8.70, 9.95, 8.25, 13.50, 12.30,
            10.40, 6.29, 9.55, 7.85, 11.25, 10.00, 8.60, 6.29, 9.40, 8.80,
            12.60, 11.75, 9.90, 6.29, 10.30, 8.95, 14.20]
    
    base_sfp = [10.96, 20.28, 16.70, 17.39, 17.10, 17.52, 17.39, 17.17, 17.59, 17.01, 
        12.94, 18.83, 17.50, 17.81, 17.34, 16.88, 17.52, 17.59, 19.01, 19.95, 
        22.97, 20.26, 24.73, 18.17, 18.23, 24.42, 24.82, 28.20, 28.84, 27.17, 
        22.41, 21.99, 22.68, 27.84, 32.31, 29.71, 25.33, 35.29, 32.80, 26.20, 
        34.11, 33.58, 33.98, 33.91, 26.55, 34.16, 38.78, 35.36, 33.11, 38.60, 
        25.77, 39.16, 35.71, 39.09, 36.31, 35.27, 41.03, 37.29, 31.09, 38.07, 
        40.29, 27.51, 42.23, 41.54, 29.15, 43.12, 31.40, 33.84, 31.29, 39.11, 
        31.47, 30.93, 32.44, 36.87, 35.87, 34.96, 45.21, 40.38, 43.27, 40.65, 
        43.47, 41.07, 42.38, 42.78, 41.38, 36.60, 35.49, 43.05, 35.20, 43.34, 
        43.54, 40.29, 43.54, 43.56, 45.92, 38.05, 45.63, 37.29, 38.47, 46.16]
    
    base_sfp_attn = [10.78, 16.41, 16.92, 17.46, 17.08, 20.01, 17.34, 16.92, 17.48, 17.41, 
        17.57, 18.19, 17.63, 17.90, 17.41, 16.94, 17.61, 18.17, 18.19, 19.41, 
        25.95, 20.35, 23.24, 22.66, 22.10, 25.24, 25.15, 29.35, 26.82, 27.91, 
        25.08, 25.22, 26.40, 27.28, 32.60, 25.62, 26.20, 35.27, 30.29, 26.80, 
        32.31, 33.49, 31.71, 31.80, 28.55, 32.51, 36.98, 36.38, 33.78, 39.76,
        30.57, 33.99, 38.51, 39.76, 38.95, 40.12, 41.33, 39.85, 41.56, 42.78, 
        41.20, 43.05, 44.18, 42.90, 43.75, 41.10, 44.30, 43.88, 43.95, 44.40, 
        45.60, 47.12, 46.80, 44.15, 43.25, 42.90, 45.10, 44.45, 44.70, 48.00,
        49.20, 48.85, 46.95, 45.10, 47.40, 48.60, 46.00, 47.75, 48.10, 48.90,
        49.30, 49.95, 50.40, 50.98, 50.50, 49.99, 50.60, 49.88, 48.70, 49.95]
    
    full_model =[10.85, 15.12, 14.79, 16.83, 20.35, 18.37, 18.35, 21.44, 22.41, 17.50, 
        20.79, 23.08, 22.21, 18.86, 26.31, 26.35, 23.70, 29.31, 35.16, 32.76, 
        37.65, 40.78, 41.32, 29.49, 32.35, 39.67, 40.23, 43.05, 44.39, 42.01, 
        39.92, 42.18, 43.87, 47.52, 43.72, 49.48, 37.36, 44.47, 47.57, 39.69, 
        47.14, 49.61, 49.28, 47.94, 44.59, 51.32, 50.83, 47.41, 49.14, 48.81,
        44.62, 45.18, 47.72, 45.31, 47.09, 49.52, 47.85, 50.11, 48.37, 49.88,
        48.42, 50.73, 51.09, 50.24, 52.37, 50.66, 51.18, 51.45, 53.91, 52.08,
        52.59, 51.82, 53.17, 52.31, 54.76, 53.02, 51.49, 52.87, 54.32, 54.79,
        55.14, 53.60, 52.88, 52.22, 53.67, 55.10, 55.46, 53.90, 53.31, 54.78,
        55.12, 54.61, 55.84, 55.30, 55.72, 55.19, 55.48, 55.02, 55.39, 54.95]

    plot_ablation_feature_shift(
        base, base_sfp, base_sfp_attn, full_model,
        out_prefix="ablation_byzantine",
        title=None  # e.g., "Ablation on Feature Shift"
    )
