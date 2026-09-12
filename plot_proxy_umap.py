#!/usr/bin/env python3
"""HemaScribe-style UMAP of VNN metabolic tasks + EM rate proxies."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache_bm")
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.lines import Line2D
from scipy import sparse

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_local" / "chip_metabolic_graph"
NET = Path("/cis/net/r41/data/iessien1/bone_marrow_results/chip_metabolic_graph")

SPECS = (
    {
        "stem": "mice",
        "qc": ROOT / "data/GSE209994/processed/gse209994_qc_preprocessed.h5ad",
        "cells": "mice_cells.csv",
        "lineage_key": "lineage",
        "lineage_val": "HSPC",
        "geno_map": {"WT": "WT", "Tet2_KO": "Tet2"},
        "treat_map": {"vehicle": "vehicle", "IL1b": "IL1"},
        "arm_order": ("WT_vehicle", "WT_IL1", "Tet2_vehicle", "Tet2_IL1"),
        "arm_labels": ("WT · vehicle", "WT · IL-1", "Tet2 · vehicle", "Tet2 · IL-1"),
        "features": (
            (
                "HMP shunt",
                "Synthesis of fructose-6-phosphate from erythrose-4-phosphate (HMP shunt)",
            ),
            (
                "Glycolysis",
                "ATP generation from glucose (hypoxic conditions) - glycolysis",
            ),
            ("Myc EM", "rate:Myc_EM"),
            ("OXPHOS EM", "rate:OXPHOS_EM"),
            ("Gln EM", "rate:Gln_EM"),
        ),
        "title": "Mice · McClatchy HSPC",
    },
    {
        "stem": "human",
        "qc": ROOT / "data/GSE285379/processed/gse285379_qc_preprocessed.h5ad",
        "cells": "human_cells.csv",
        "lineage_key": "compartment",
        "lineage_val": "HSPC",
        "geno_map": {"WT": "WT", "TET2_KO": "Tet2", "Tet2_KO": "Tet2", "TET2": "Tet2"},
        "treat_map": {"CTRL": "CTRL", "LPS": "LPS", "ctrl": "CTRL"},
        "arm_order": ("WT_CTRL", "WT_LPS", "Tet2_CTRL", "Tet2_LPS"),
        "arm_labels": ("WT · CTRL", "WT · LPS", "Tet2 · CTRL", "Tet2 · LPS"),
        "features": (
            (
                "HMP shunt",
                "Synthesis of fructose-6-phosphate from erythrose-4-phosphate (HMP shunt)",
            ),
            (
                "Glycolysis",
                "ATP generation from glucose (hypoxic conditions) - glycolysis",
            ),
            ("Myc EM", "rate:Myc_EM"),
            ("OXPHOS EM", "rate:OXPHOS_EM"),
            ("Gln EM", "rate:Gln_EM"),
        ),
        "title": "Human · GSE285379 HSPC",
    },
)

ARM_COLORS = {
    0: "#4C72B0",
    1: "#55A868",
    2: "#C44E52",
    3: "#8172B3",
}


def _find_cells(name: str) -> Path:
    for base in (OUT, NET, ROOT / "results" / "chip_metabolic_graph"):
        p = base / name
        if p.exists():
            return p
    raise FileNotFoundError(name)


def _to_dense(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def _load_query(spec: dict) -> ad.AnnData:
    raw = ad.read_h5ad(spec["qc"])
    if spec["stem"] == "human":
        key = spec["lineage_key"]
        if key not in raw.obs:
            key = "lineage" if "lineage" in raw.obs else None
        if key is None:
            adata = raw.copy()
        else:
            adata = raw[raw.obs[key].astype(str).eq(spec["lineage_val"])].copy()
        gcol = "genotype" if "genotype" in adata.obs else "Genotype"
        tcol = "treatment" if "treatment" in adata.obs else "Treatment"
    else:
        m = (
            raw.obs[spec["lineage_key"]].astype(str).eq(spec["lineage_val"])
            & raw.obs["genotype"].isin(list(spec["geno_map"]))
            & raw.obs["treatment"].astype(str).isin(list(spec["treat_map"]))
        )
        adata = raw[m].copy()
        gcol, tcol = "genotype", "treatment"

    adata.obs["genotype"] = (
        adata.obs[gcol].astype(str).map(lambda x: spec["geno_map"].get(x, x))
    )
    adata.obs["treatment"] = (
        adata.obs[tcol].astype(str).map(lambda x: spec["treat_map"].get(x, x))
    )
    adata.obs["arm"] = (
        adata.obs["genotype"].astype(str) + "_" + adata.obs["treatment"].astype(str)
    )
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=False)
    sc.pp.pca(adata, n_comps=30, use_highly_variable=True)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata)
    return adata


def _attach_scores(adata: ad.AnnData, cells_path: Path) -> list[str]:
    cells = pd.read_csv(cells_path)
    score_cols = [
        c for c in cells.columns if c not in {"sample_name", "genotype", "treatment"}
    ]
    if len(cells) != adata.n_obs:
        # align within sample by order of appearance
        raise ValueError(
            f"cell count mismatch: adata={adata.n_obs} cells.csv={len(cells)}"
        )
    for c in score_cols:
        adata.obs[c] = cells[c].to_numpy(dtype=float)
    return score_cols


def _save(fig: plt.Figure, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25, dpi=200)
    plt.close(fig)
    # best-effort mirror
    try:
        NET.mkdir(parents=True, exist_ok=True)
        fig2_path = NET / name
        if fig2_path != path:
            import shutil

            shutil.copy2(path, fig2_path)
    except OSError:
        pass
    print("wrote", path, flush=True)


def _plot_arm_umap(adata: ad.AnnData, spec: dict):
    um = adata.obsm["X_umap"]
    arms = list(spec["arm_order"])
    labels = list(spec["arm_labels"])
    codes = pd.Categorical(adata.obs["arm"].astype(str), categories=arms).codes
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    order = np.argsort(codes)
    ax.scatter(
        um[order, 0],
        um[order, 1],
        c=[ARM_COLORS.get(int(c), "0.7") for c in codes[order]],
        s=6,
        linewidths=0,
        rasterized=True,
    )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(spec["title"] + " · arm", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=ARM_COLORS[i],
            markersize=7,
            label=lab,
        )
        for i, lab in enumerate(labels)
        if i in ARM_COLORS
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
    )
    _save(fig, f"{spec['stem']}_umap_arm.png")


def _plot_feature_umaps(adata: ad.AnnData, spec: dict):
    um = adata.obsm["X_umap"]
    feats = [(lab, col) for lab, col in spec["features"] if col in adata.obs]
    if not feats:
        raise ValueError(f"{spec['stem']}: no feature columns in obs")
    n = len(feats)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows), squeeze=False
    )
    fig.subplots_adjust(wspace=0.25, hspace=0.35, right=0.92)
    for i, (lab, col) in enumerate(feats):
        ax = axs[i // ncols][i % ncols]
        v = adata.obs[col].to_numpy(dtype=float)
        v = np.nan_to_num(v, nan=0.0)
        # robust clip for color
        lo, hi = np.percentile(v, [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        sca = ax.scatter(
            um[:, 0],
            um[:, 1],
            c=np.clip(v, lo, hi),
            s=5,
            cmap="viridis",
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(lab, fontweight="bold", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        cb = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)
    for j in range(n, nrows * ncols):
        axs[j // ncols][j % ncols].axis("off")
    fig.suptitle(
        spec["title"] + " · metabolic tasks & EM rate proxies",
        fontweight="bold",
        fontsize=13,
        y=1.02,
    )
    _save(fig, f"{spec['stem']}_umap_proxies.png")


def _plot_combo(adata: ad.AnnData, spec: dict):
    """One page: arm UMAP + key proxies (HemaScribe-like multipanel)."""
    um = adata.obsm["X_umap"]
    feats = [(lab, col) for lab, col in spec["features"] if col in adata.obs][:4]
    fig = plt.figure(figsize=(11.5, 7.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1], wspace=0.28, hspace=0.32)

    ax0 = fig.add_subplot(gs[:, 0])
    arms = list(spec["arm_order"])
    labels = list(spec["arm_labels"])
    codes = pd.Categorical(adata.obs["arm"].astype(str), categories=arms).codes
    order = np.argsort(codes)
    ax0.scatter(
        um[order, 0],
        um[order, 1],
        c=[ARM_COLORS.get(int(c), "0.7") for c in codes[order]],
        s=5,
        linewidths=0,
        rasterized=True,
    )
    ax0.set_title("Arm", fontweight="bold")
    ax0.set_xticks([])
    ax0.set_yticks([])
    for spine in ax0.spines.values():
        spine.set_visible(False)
    ax0.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=ARM_COLORS[i],
                markersize=7,
                label=lab,
            )
            for i, lab in enumerate(labels)
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        frameon=False,
        fontsize=8,
        ncol=1,
    )

    slots = [(0, 1), (0, 2), (1, 1), (1, 2)]
    for (lab, col), (r, c) in zip(feats, slots):
        ax = fig.add_subplot(gs[r, c])
        v = np.nan_to_num(adata.obs[col].to_numpy(dtype=float), nan=0.0)
        lo, hi = np.percentile(v, [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        sca = ax.scatter(
            um[:, 0],
            um[:, 1],
            c=np.clip(v, lo, hi),
            s=4,
            cmap="magma",
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(lab, fontweight="bold", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(spec["title"], fontweight="bold", fontsize=14, y=0.98)
    _save(fig, f"{spec['stem']}_umap_landscape.png")


def run_spec(spec: dict):
    print(f"{spec['stem']}: loading + UMAP", flush=True)
    adata = _load_query(spec)
    cells = _find_cells(spec["cells"])
    _attach_scores(adata, cells)
    # keep only arms we know
    keep = adata.obs["arm"].isin(spec["arm_order"])
    adata = adata[keep].copy()
    _plot_arm_umap(adata, spec)
    _plot_feature_umaps(adata, spec)
    _plot_combo(adata, spec)
    # write embedding for reuse
    emb = pd.DataFrame(
        {
            "UMAP1": adata.obsm["X_umap"][:, 0],
            "UMAP2": adata.obsm["X_umap"][:, 1],
            "arm": adata.obs["arm"].astype(str).to_numpy(),
            "genotype": adata.obs["genotype"].astype(str).to_numpy(),
            "treatment": adata.obs["treatment"].astype(str).to_numpy(),
        },
        index=adata.obs_names.astype(str),
    )
    for lab, col in spec["features"]:
        if col in adata.obs:
            emb[col] = adata.obs[col].to_numpy(dtype=float)
    OUT.mkdir(parents=True, exist_ok=True)
    emb.to_csv(OUT / f"{spec['stem']}_umap_scores.csv")
    print(f"{spec['stem']}: done n={adata.n_obs}", flush=True)


def main():
    for spec in SPECS:
        cells = _find_cells(spec["cells"])
        if not spec["qc"].exists():
            print(f"skip {spec['stem']}: missing qc {spec['qc']}", flush=True)
            continue
        print(f"using cells {cells}", flush=True)
        run_spec(spec)


if __name__ == "__main__":
    main()
