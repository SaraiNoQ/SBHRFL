import matplotlib.pyplot as plt

def plot_ablation_feature_shift(
    base,
    base_sfp,
    base_sfp_attn,
    base_sfp_attn_align,
    full_model,
    out_prefix="ablation_feature_shift",
    title=None
):
    """
    Inputs: five lists (or 1D arrays) of length 100, values in [0, 100].
    Output: out_prefix + ".pdf" (vector) and out_prefix + ".png" (300dpi).
    """

    # --------- sanity checks ----------
    series = {
        "Base": base,
        "Base + SFP": base_sfp,
        "Base + SFP + Attention fuse": base_sfp_attn,
        "Base + SFP + Attention fuse + Align": base_sfp_attn_align,
        "Base + SFP + Attention fuse + Align + Retro": full_model,
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
    ax.plot(x, base_sfp, linewidth=1.5, label="Base + SFP")
    ax.plot(x, base_sfp_attn, linewidth=1.5, label="Base + SFP + Attention fuse")
    ax.plot(x, base_sfp_attn_align, linewidth=1.5, label="Base + SFP + Attention fuse + Align")
    ax.plot(x, full_model, linewidth=1.5, label="Base + SFP + Attention fuse + Align + Retro")

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
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="black")

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
    base = [10.50, 16.41, 18.55, 17.39, 20.61, 22.32, 20.99, 22.88, 24.35, 22.03, 
        23.46, 22.04, 25.71, 24.95, 25.58, 23.13, 22.22, 25.53, 22.44, 22.41, 
        24.38, 25.11, 23.13, 25.82, 26.09, 24.98, 23.75, 23.98, 23.91, 24.40, 
        24.13, 28.31, 23.89, 26.05, 24.91, 26.11, 29.98, 25.56, 29.16, 34.85, 
        35.02, 32.89, 32.61, 30.03, 29.16, 32.32, 28.07, 28.45, 33.72, 36.79, 
        32.25, 32.72, 31.43, 36.41, 38.47, 38.32, 31.61, 31.21, 39.54, 35.70, 
        38.12, 41.83, 41.65, 42.92, 41.55, 37.61, 42.28, 45.19, 37.14, 38.99, 
        38.97, 42.41, 43.63, 40.74, 44.23, 45.14, 42.85, 43.85, 46.52, 42.32, 
        41.59, 46.79, 43.05, 44.79, 41.35, 47.97, 44.19, 46.66, 44.53, 45.70, 
        47.45, 48.85, 48.39, 44.90, 49.32, 43.61, 46.14, 49.23, 48.23, 49.64]
    
    base_sfp = [9.30, 16.59, 17.08, 18.03, 26.17, 26.44, 29.20, 23.10, 29.44, 25.19, 
        31.44, 31.89, 32.91, 33.58, 34.58, 35.38, 32.22, 35.69, 38.58, 35.51, 
        37.45, 44.59, 38.60, 43.96, 44.59, 41.89, 42.76, 40.14, 41.21, 42.32, 
        42.63, 40.69, 42.70, 42.50, 44.94, 44.61, 48.17, 46.01, 49.23, 50.43, 
        48.86, 50.32, 51.61, 51.81, 53.10, 51.41, 50.74, 51.75, 51.50, 53.53, 
        52.44, 52.95, 55.08, 54.97, 55.33, 56.22, 54.33, 54.70, 53.55, 56.62, 
        55.53, 56.51, 55.68, 55.93, 56.60, 54.70, 55.68, 56.55, 57.80, 58.42, 
        58.15, 58.53, 56.95, 58.66, 59.73, 54.35, 56.53, 59.77, 60.28, 57.46, 
        58.93, 59.42, 59.84, 59.68, 59.93, 60.64, 58.11, 59.55, 60.40, 59.73, 
        60.11, 57.99, 60.69, 59.55, 60.18, 60.06, 60.35, 59.53, 60.73, 60.44]
    
    base_sfp_attn = [9.30, 19.46, 17.63, 19.39, 23.93, 25.26, 28.91, 27.53, 32.09, 28.09, 
        35.22, 37.14, 36.05, 38.89, 37.60, 42.96, 38.60, 38.74, 44.36, 43.98, 
        43.01, 47.63, 41.92, 47.88, 46.48, 46.05, 47.65, 44.32, 46.03, 46.52, 
        47.61, 47.61, 47.41, 50.39, 53.35, 52.41, 52.84, 52.79, 54.24, 58.24, 
        55.97, 57.33, 57.86, 55.73, 57.13, 56.26, 57.97, 58.79, 55.46, 58.13, 
        56.79, 56.75, 58.79, 60.44, 61.15, 60.35, 59.60, 59.48, 58.04, 61.64, 
        60.13, 61.40, 61.73, 62.11, 57.99, 60.69, 60.55, 61.18, 60.06, 61.49, 
        62.35, 61.53, 60.73, 60.44, 61.47, 59.13, 61.95, 60.80, 62.66, 61.51, 
        61.58, 60.64, 58.53, 60.64, 60.67, 61.98, 61.06, 62.02, 61.60, 62.64, 
        62.44, 61.89, 62.58, 61.29, 62.98, 62.69, 60.47, 63.42, 59.44, 62.98]
    
    
    base_sfp_attn_align = [9.34, 11.39, 15.50, 17.19, 21.26, 22.64, 25.91, 25.68, 31.87, 28.15, 
        35.91, 39.14, 38.05, 40.03, 38.38, 42.05, 35.45, 42.41, 45.65, 41.83, 
        44.50, 47.25, 43.65, 49.05, 49.99, 45.14, 48.12, 43.94, 45.92, 46.54, 
        47.97, 45.01, 49.14, 51.26, 54.84, 51.92, 52.06, 54.08, 55.73, 57.73, 
        54.10, 57.15, 57.79, 54.81, 56.26, 57.33, 57.77, 57.35, 55.84, 58.91, 
        57.37, 56.82, 60.42, 59.55, 61.00, 61.40, 62.73, 62.11, 57.99, 62.69,
        62.55, 63.18, 60.06, 63.49, 64.35, 61.53, 62.73, 61.44, 64.47, 59.13, 
        61.95, 64.80, 62.66, 62.51, 64.58, 64.64, 64.53, 65.64, 64.67, 63.98, 
        64.06, 64.02, 65.60, 63.64, 64.44, 64.89, 63.58, 64.29, 63.98, 63.69, 
        64.47, 64.42, 64.62, 64.60, 64.33, 63.04, 63.58, 65.05, 65.54, 65.08]
    
    full_model = [9.14, 16.88, 18.06, 16.17, 20.58, 24.11, 24.49, 25.80, 25.50, 27.87, 
        34.94, 33.15, 34.50, 38.99, 40.06, 40.26, 44.37, 44.68, 45.33, 46.35, 
        48.24, 50.33, 52.06, 52.70, 55.97, 55.93, 57.55, 55.44, 58.77, 56.26, 
        57.28, 56.64, 57.42, 61.40, 62.73, 62.11, 57.99, 62.69, 62.55, 63.18, 
        60.06, 63.49, 64.35, 61.53, 62.73, 61.44, 64.47, 59.13, 61.95, 64.80, 
        62.66, 62.51, 64.58, 64.64, 64.53, 65.64, 64.67, 63.98, 64.06, 64.02, 
        65.60, 63.64, 64.44, 64.89, 63.58, 64.29, 63.98, 63.69, 64.47, 64.42, 
        64.62, 64.60, 64.33, 63.04, 63.58, 61.44, 62.98, 64.47, 65.80, 63.49, 
        66.16, 66.07, 63.02, 65.96, 65.93, 66.31, 62.75, 65.71, 66.47, 64.15, 
        65.58, 65.40, 66.22, 65.93, 65.87, 65.98, 65.20, 66.60, 66.88, 66.15]

    plot_ablation_feature_shift(
        base, base_sfp, base_sfp_attn, base_sfp_attn_align, full_model,
        out_prefix="ablation_feature_shift",
        title=None  # e.g., "Ablation on Feature Shift"
    )
