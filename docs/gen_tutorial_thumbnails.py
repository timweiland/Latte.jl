#!/usr/bin/env python3
"""Generate the tutorial-gallery thumbnails in docs/src/assets/thumbs/.

For each tutorial we curate ONE representative figure (the most striking +
self-explanatory at ~360px) from the built docs figures and letterbox it onto a
clean 16:9 canvas, so the gallery cards are visually consistent. sbc_calibration
produces no plot (it is an output/text tutorial), so we draw a schematic of its
central diagnostic — a calibrated, uniform rank histogram with the expected band.

Prerequisite: the tutorials must have been built once (`make docs`) so the source
figures exist under docs/build/.documenter/tutorials/. Re-run after a tutorial's
hero figure changes:  python3 docs/gen_tutorial_thumbnails.py
"""
import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "build", ".documenter", "tutorials")
OUT = os.path.join(HERE, "src", "assets", "thumbs")
CW, CH, PAD, BG = 960, 540, 0.94, (255, 255, 255)   # 16:9 canvas, white margin

# tutorial -> curated hero figure (chosen for thumbnail legibility/representativeness)
PICKS = {
    "bayesian_model_averaging":    "bayesian_model_averaging-16.png",  # posterior-averaged prediction
    "disease_mapping_spatial":     "disease_mapping_spatial-39.png",   # PA exceedance-probability choropleth
    "fisheries_state_space":       "fisheries_state_space-8.png",      # biomass + catches + CPUE indices
    "getting_started":             "getting_started-28.png",           # per-hospital posterior intervals
    "hmc_laplace_when":            "hmc_laplace_when-26.png",
    "nonlinear_regression_gam":    "nonlinear_regression_gam-12.png",  # fitted curve + credible band
    "posterior_predictive_checks": "posterior_predictive_checks-10.png",
    "spatial_spde":                "spatial_spde-27.png",              # Japan seismic-intensity field
    "barrier_coastline":           "barrier_coastline-27.png",         # barrier vs stationary intensity around Florida
    "spatio_temporal_separable":   "spatio_temporal_separable-24.png", # full space-time fitted field
    "temporal_trend_earthquakes":  "temporal_trend_earthquakes-10.png",# fitted trend + band
    "turing_handoff":              "turing_handoff-8.png",
    "tweedie_insurance":           "tweedie_insurance-24.png",         # posterior densities vs truth
}


def letterbox(fig_path, out_path):
    im = Image.open(fig_path).convert("RGBA")
    scale = min(CW * PAD / im.width, CH * PAD / im.height)
    nw, nh = round(im.width * scale), round(im.height * scale)
    im = im.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (CW, CH), BG)
    canvas.paste(im, ((CW - nw) // 2, (CH - nh) // 2), im)
    canvas.save(out_path, optimize=True)


def sbc_schematic(out_path):
    """Calibrated-inference reference: ranks are uniform; bars sit in the band."""
    rng = np.random.default_rng(0x05bc)
    B, N = 20, 600
    counts = np.bincount(rng.integers(0, B, size=N), minlength=B)
    exp, sd = N / B, np.sqrt(N * (1 / B) * (1 - 1 / B))
    BERRY, MOCHA, CARAMEL, BEAN = "#C04A2A", "#8B6F47", "#C9986A", "#3D2817"
    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=100)
    ax.bar(np.arange(B), counts, width=0.92, color=BERRY, alpha=0.62,
           edgecolor="white", linewidth=0.6, zorder=3)
    ax.axhspan(exp - 2 * sd, exp + 2 * sd, color=CARAMEL, alpha=0.18, zorder=1)
    ax.axhline(exp, color=MOCHA, ls="--", lw=1.6, zorder=2)
    ax.set_xlim(-0.7, B - 0.3); ax.set_ylim(0, exp + 3.2 * sd)
    ax.set_title("Calibrated inference → uniform rank histogram",
                 fontsize=18, color=BEAN, pad=14)
    ax.set_xlabel(r"rank of $\theta_{true}$ among posterior draws", fontsize=13, color=BEAN)
    ax.set_ylabel("frequency", fontsize=13, color=BEAN)
    ax.set_xticks([]); ax.tick_params(colors=MOCHA, labelsize=11)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(MOCHA)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    for tut, fig in PICKS.items():
        letterbox(os.path.join(SRC, fig), os.path.join(OUT, f"{tut}.png"))
        print(f"  {tut:30s} <- {fig}")
    sbc_schematic(os.path.join(OUT, "sbc_calibration.png"))
    print("  sbc_calibration                <- schematic (no plot in tutorial)")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
