import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "figure.titleweight": "bold",
    "lines.linewidth": 1.6,
    "axes.linewidth": 0.8,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "figure.autolayout": False
})

def main():
    root = Path(".")
    manifest_path = root / "data" / "quality_metrics" / "quality_manifest.csv"
    if not manifest_path.exists():
        print(f"Error: {manifest_path} not found.")
        return

    df = pd.read_csv(manifest_path)
    thick = df[df["smear_type"] == "thick"]
    thin = df[df["smear_type"] == "thin"]

    print(f"Loaded {len(df)} images: {len(thick)} thick, {len(thin)} thin.")

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # Colors
    c_thick = "#0072B2"  # Okabe-Ito Blue
    c_thin = "#D55E00"   # Okabe-Ito Vermilion

    # 1. Log-Laplacian Blur Variance
    ax = axes[0, 0]
    ax.set_facecolor("#FFFFFF")
    # Avoid log(0)
    v_thick = np.log10(np.clip(thick["blur_laplacian"], 1e-2, None))
    v_thin = np.log10(np.clip(thin["blur_laplacian"], 1e-2, None))
    
    bins = np.linspace(-1, 3.5, 45)
    ax.hist(v_thick, bins=bins, density=True, alpha=0.45, color=c_thick, label=f"Thick ($n={len(thick)}$)")
    ax.hist(v_thin, bins=bins, density=True, alpha=0.45, color=c_thin, label=f"Thin ($n={len(thin)}$)")
    
    # KDE lines
    kde_thick = stats.gaussian_kde(v_thick)
    kde_thin = stats.gaussian_kde(v_thin)
    x_grid = np.linspace(-1, 3.5, 200)
    ax.plot(x_grid, kde_thick(x_grid), color=c_thick, lw=2.0)
    ax.plot(x_grid, kde_thin(x_grid), color=c_thin, lw=2.0)
    
    # Medians
    m_thick = np.median(v_thick)
    m_thin = np.median(v_thin)
    ax.axvline(m_thick, color=c_thick, ls="--", lw=1.2, alpha=0.8)
    ax.axvline(m_thin, color=c_thin, ls="--", lw=1.2, alpha=0.8)
    
    ks_res = stats.ks_2samp(v_thick, v_thin)
    ax.set_title("(A) Focus Acuity (Log-Laplacian Variance)")
    ax.set_xlabel(r"$\log_{10}(\sigma^2_{\nabla^2})$ Focus Score")
    ax.set_ylabel("Probability Density")
    ax.legend(frameon=True, facecolor="#FAFAFA", edgecolor="#DDDDDD", loc="upper left")
    ax.grid(True)
    ax.text(0.97, 0.88, f"KS $p < 10^{{-15}}$\nThick med: {10**m_thick:.1f}\nThin med: {10**m_thin:.1f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="#F8F9FA", ec="#CCCCCC", alpha=0.9))

    # 2. Michelson Contrast
    ax = axes[0, 1]
    ax.set_facecolor("#FFFFFF")
    mc_thick = thick["michelson_contrast"]
    mc_thin = thin["michelson_contrast"]
    bins_mc = np.linspace(0.2, 1.0, 45)
    ax.hist(mc_thick, bins=bins_mc, density=True, alpha=0.45, color=c_thick, label="Thick")
    ax.hist(mc_thin, bins=bins_mc, density=True, alpha=0.45, color=c_thin, label="Thin")
    
    kde_mc_thick = stats.gaussian_kde(mc_thick)
    kde_mc_thin = stats.gaussian_kde(mc_thin)
    x_mc = np.linspace(0.2, 1.0, 200)
    ax.plot(x_mc, kde_mc_thick(x_mc), color=c_thick, lw=2.0)
    ax.plot(x_mc, kde_mc_thin(x_mc), color=c_thin, lw=2.0)
    
    m_mc_thick = np.median(mc_thick)
    m_mc_thin = np.median(mc_thin)
    ax.axvline(m_mc_thick, color=c_thick, ls="--", lw=1.2, alpha=0.8)
    ax.axvline(m_mc_thin, color=c_thin, ls="--", lw=1.2, alpha=0.8)
    
    ax.set_title("(B) Optical Michelson Contrast")
    ax.set_xlabel(r"Michelson Contrast $\frac{I_{\max} - I_{\min}}{I_{\max} + I_{\min}}$")
    ax.set_ylabel("Probability Density")
    ax.legend(frameon=True, facecolor="#FAFAFA", edgecolor="#DDDDDD", loc="upper left")
    ax.grid(True)
    ax.text(0.97, 0.88, f"Thick med: {m_mc_thick:.3f}\nThin med: {m_mc_thin:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="#F8F9FA", ec="#CCCCCC", alpha=0.9))

    # 3. Signal-to-Noise Ratio (dB)
    ax = axes[1, 0]
    ax.set_facecolor("#FFFFFF")
    snr_thick = thick["snr"]
    snr_thin = thin["snr"]
    bins_snr = np.linspace(0, 30, 45)
    ax.hist(snr_thick, bins=bins_snr, density=True, alpha=0.45, color=c_thick, label="Thick")
    ax.hist(snr_thin, bins=bins_snr, density=True, alpha=0.45, color=c_thin, label="Thin")
    
    kde_snr_thick = stats.gaussian_kde(snr_thick)
    kde_snr_thin = stats.gaussian_kde(snr_thin)
    x_snr = np.linspace(0, 30, 200)
    ax.plot(x_snr, kde_snr_thick(x_snr), color=c_thick, lw=2.0)
    ax.plot(x_snr, kde_snr_thin(x_snr), color=c_thin, lw=2.0)
    
    m_snr_thick = np.median(snr_thick)
    m_snr_thin = np.median(snr_thin)
    ax.axvline(m_snr_thick, color=c_thick, ls="--", lw=1.2, alpha=0.8)
    ax.axvline(m_snr_thin, color=c_thin, ls="--", lw=1.2, alpha=0.8)
    
    ax.set_title("(C) Signal-to-Noise Ratio (SNR)")
    ax.set_xlabel(r"Signal-to-Noise Ratio ($\mathrm{dB}$)")
    ax.set_ylabel("Probability Density")
    ax.legend(frameon=True, facecolor="#FAFAFA", edgecolor="#DDDDDD", loc="upper left")
    ax.grid(True)
    ax.text(0.97, 0.88, f"Thick med: {m_snr_thick:.1f} dB\nThin med: {m_snr_thin:.1f} dB",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="#F8F9FA", ec="#CCCCCC", alpha=0.9))

    # 4. Circular Ocular FOV Coverage Ratio
    ax = axes[1, 1]
    ax.set_facecolor("#FFFFFF")
    fov_thick = thick["fov_coverage_ratio"]
    fov_thin = thin["fov_coverage_ratio"]
    bins_fov = np.linspace(0.4, 1.0, 45)
    ax.hist(fov_thick, bins=bins_fov, density=True, alpha=0.45, color=c_thick, label="Thick")
    ax.hist(fov_thin, bins=bins_fov, density=True, alpha=0.45, color=c_thin, label="Thin")
    
    kde_fov_thick = stats.gaussian_kde(fov_thick)
    kde_fov_thin = stats.gaussian_kde(fov_thin)
    x_fov = np.linspace(0.4, 1.0, 200)
    ax.plot(x_fov, kde_fov_thick(x_fov), color=c_thick, lw=2.0)
    ax.plot(x_fov, kde_fov_thin(x_fov), color=c_thin, lw=2.0)
    
    m_fov_thick = np.median(fov_thick)
    m_fov_thin = np.median(fov_thin)
    ax.axvline(m_fov_thick, color=c_thick, ls="--", lw=1.2, alpha=0.8)
    ax.axvline(m_fov_thin, color=c_thin, ls="--", lw=1.2, alpha=0.8)
    
    ax.set_title("(D) Microscopic Field-of-View (FOV) Coverage")
    ax.set_xlabel("Effective Illuminated Aperture Ratio")
    ax.set_ylabel("Probability Density")
    ax.legend(frameon=True, facecolor="#FAFAFA", edgecolor="#DDDDDD", loc="upper left")
    ax.grid(True)
    ax.text(0.97, 0.88, f"Thick med: {m_fov_thick:.2f}\nThin med: {m_fov_thin:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="#F8F9FA", ec="#CCCCCC", alpha=0.9))

    plt.tight_layout()
    
    out_dir = root / "manuscript" / "figures" / "appendix"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "fig_s2_optical_physics.pdf"
    png_path = out_dir / "fig_s2_optical_physics.png"
    
    plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {pdf_path} and {png_path}")

if __name__ == "__main__":
    main()
