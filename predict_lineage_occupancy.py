#!/usr/bin/env python3
"""Downstream lineage occupancy on frozen VNN embeddings (full Python).

HemaScribe/HemaScape-style: predict HSPC subtype + branch occupancy, then
map onto a UMAP of the *frozen* multi-head VNN state (not raw RNA UMAP).
PAGA + DPT follow MultiLin/CellRank-style myeloid trajectory scripts.
"""
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "chip_metabolic_graph"
LOCAL_OUT = ROOT / "results_local" / "chip_metabolic_graph"
NET = Path("/cis/net/r41/data/iessien1/bone_marrow_results/chip_metabolic_graph")

# HemaScribe HSPC.annot palette (README / man/figures)
HSPC_COLORS = {
    "HSC": "navy",
    "STHSC": "mediumturquoise",
    "MPP2": "blue",
    "FcG_neg_MPP3": "deeppink",
    "FcG_pos_MPP3": "darkmagenta",
    "MPP4": "gold",
    "GMP": "seagreen",
    "MkP": "lightgreen",
    "EryP": "tomato",
    "CLP": "slateblue",
    "NotHSPC": "grey87",
}

# Marker modules → HemaScribe fine labels (score_genes; not Seurat TransferData)
MOUSE_MARKERS = {
    "HSC": ["Procr", "Hoxb5", "Mecom", "Ly6a", "Slamf1", "Cdkn1c", "Mllt3"],
    "STHSC": ["Cd34", "Cd48", "Itga2b", "Gata2", "Myc"],
    "MPP2": ["Cd34", "Flt3", "Cd48", "Slamf1", "Neo1", "Cd9"],
    "FcG_neg_MPP3": ["Flt3", "Cd48", "Cd34", "Csfr1", "Ikzf1"],
    "FcG_pos_MPP3": ["Fcgr3", "Fcgr2b", "Mpo", "Elane", "Cebpe", "Csf1r"],
    "MPP4": ["Flt3", "Il7r", "Dntt", "Rag1", "Satb1"],
    "GMP": ["Mpo", "Elane", "Cebpe", "Csf3r", "Ctsg", "Prtn3"],
    "MkP": ["Pf4", "Itga2b", "Gp1bb", "Gp9", "Vwf", "Tubb1"],
    "EryP": ["Gata1", "Klf1", "Car1", "Car2", "Epor", "Hemgn"],
    "CLP": ["Il7r", "Rag1", "Dntt", "Flt3", "Cd93", "Notch1"],
}

HUMAN_MARKERS = {
    "HSC": ["PROCR", "HOXB5", "MECOM", "HLA-DRA", "AVP", "CRHBP", "MLLT3"],
    "STHSC": ["CD34", "CD38", "MYC", "CDK6"],
    "MPP2": ["CD34", "FLT3", "CD48", "NEO1"],
    "FcG_neg_MPP3": ["FLT3", "CD48", "CSF1R", "IKZF1"],
    "FcG_pos_MPP3": ["FCGR3A", "MPO", "ELANE", "CEBPE", "CSF1R"],
    "MPP4": ["FLT3", "IL7R", "DNTT", "RAG1", "SATB1"],
    "GMP": ["MPO", "ELANE", "CEBPE", "CSF3R", "CTSG", "AZU1"],
    "MkP": ["PF4", "ITGA2B", "GP1BB", "GP9", "VWF", "TUBB1"],
    "EryP": ["GATA1", "KLF1", "CA1", "CA2", "EPOR", "HEMGN"],
    "CLP": ["IL7R", "RAG1", "DNTT", "FLT3", "CD7", "NOTCH1"],
}

BRANCH_MAP = {
    "HSC": "stem",
    "STHSC": "stem",
    "MPP2": "MegE",
    "FcG_neg_MPP3": "myeloid",
    "FcG_pos_MPP3": "myeloid",
    "MPP4": "lymphoid",
    "GMP": "myeloid",
    "MkP": "MegE",
    "EryP": "MegE",
    "CLP": "lymphoid",
    "NotHSPC": "other",
}

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
        "markers": MOUSE_MARKERS,
        "title": "Mice · McClatchy HSPC",
        "proxy_cols": (
            ("HMP shunt", "Synthesis of fructose-6-phosphate from erythrose-4-phosphate (HMP shunt)"),
            ("Glycolysis", "ATP generation from glucose (hypoxic conditions) - glycolysis"),
            ("Myc EM", "rate:Myc_EM"),
            ("OXPHOS EM", "rate:OXPHOS_EM"),
        ),
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
        "markers": HUMAN_MARKERS,
        "title": "Human · GSE285379 HSPC",
        "proxy_cols": (
            ("HMP shunt", "Synthesis of fructose-6-phosphate from erythrose-4-phosphate (HMP shunt)"),
            ("Glycolysis", "ATP generation from glucose (hypoxic conditions) - glycolysis"),
            ("Myc EM", "rate:Myc_EM"),
            ("OXPHOS EM", "rate:OXPHOS_EM"),
        ),
    },
)


def _dests() -> list[Path]:
    out = []
    for d in (LOCAL_OUT, OUT, NET):
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write_probe"
            probe.write_text("ok")
            probe.unlink()
            out.append(d)
        except OSError:
            continue
    if not out:
        LOCAL_OUT.mkdir(parents=True, exist_ok=True)
        out = [LOCAL_OUT]
    return out


DESTS = _dests()


def _save(name: str, obj):
    last = None
    for dest in DESTS:
        try:
            path = dest / name
            if hasattr(obj, "savefig"):
                obj.savefig(path, bbox_inches="tight", pad_inches=0.25, dpi=200)
            elif isinstance(obj, ad.AnnData):
                obj.write_h5ad(path)
            else:
                obj.to_csv(path, index=isinstance(obj, pd.DataFrame) and obj.index.name is not None)
            print("wrote", path, flush=True)
            return path
        except OSError as e:
            last = e
    raise last  # type: ignore[misc]


def _find_cells(name: str) -> Path:
    for base in (LOCAL_OUT, NET, OUT):
        p = base / name
        if p.exists():
            return p
    raise FileNotFoundError(name)


def _load_hspc(spec: dict) -> ad.AnnData:
    raw = ad.read_h5ad(spec["qc"])
    key = spec["lineage_key"]
    if key not in raw.obs and "lineage" in raw.obs:
        key = "lineage"
    if key in raw.obs:
        adata = raw[raw.obs[key].astype(str).eq(spec["lineage_val"])].copy()
    else:
        adata = raw.copy()

    if spec["stem"] == "mice":
        m = adata.obs["genotype"].isin(list(spec["geno_map"])) & adata.obs[
            "treatment"
        ].astype(str).isin(list(spec["treat_map"]))
        adata = adata[m].copy()

    gcol = "genotype" if "genotype" in adata.obs else "Genotype"
    tcol = "treatment" if "treatment" in adata.obs else "Treatment"
    adata.obs["genotype"] = adata.obs[gcol].astype(str).map(
        lambda x: spec["geno_map"].get(x, x)
    )
    adata.obs["treatment"] = adata.obs[tcol].astype(str).map(
        lambda x: spec["treat_map"].get(x, x)
    )
    adata.obs["arm"] = (
        adata.obs["genotype"].astype(str) + "_" + adata.obs["treatment"].astype(str)
    )
    keep = adata.obs["arm"].isin(spec["arm_order"])
    adata = adata[keep].copy()
    return adata


def _attach_vnn(adata: ad.AnnData, cells_path: Path) -> np.ndarray:
    cells = pd.read_csv(cells_path)
    if len(cells) != adata.n_obs:
        raise ValueError(f"n mismatch adata={adata.n_obs} cells={len(cells)}")
    score_cols = [
        c
        for c in cells.columns
        if c not in {"sample_name", "genotype", "treatment"}
    ]
    for c in score_cols:
        adata.obs[c] = cells[c].to_numpy(dtype=float)
    X = cells[score_cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0)
    Z = StandardScaler().fit_transform(X)
    adata.obsm["X_vnn"] = Z.astype(np.float32)
    adata.uns["vnn_features"] = score_cols
    return Z


def _present_genes(adata: ad.AnnData, genes: list[str]) -> list[str]:
    var = set(map(str, adata.var_names))
    return [g for g in genes if g in var]


def _predict_hspc_annot(adata: ad.AnnData, markers: dict[str, list[str]]) -> None:
    """Marker-module scores → soft occupancy → hard HSPC.annot (HemaScribe names)."""
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    # work on log-norm copy for scoring
    scor = adata.copy()
    sc.pp.normalize_total(scor, target_sum=1e4)
    sc.pp.log1p(scor)

    labels = list(markers)
    scores = np.zeros((scor.n_obs, len(labels)), dtype=np.float64)
    used = {}
    for j, lab in enumerate(labels):
        genes = _present_genes(scor, markers[lab])
        used[lab] = genes
        if len(genes) < 2:
            continue
        sc.tl.score_genes(scor, gene_list=genes, score_name=f"_ms_{lab}", use_raw=False)
        scores[:, j] = scor.obs[f"_ms_{lab}"].to_numpy(dtype=float)

    # softmax occupancy
    scores = np.nan_to_num(scores, nan=0.0)
    scores = scores - scores.max(axis=1, keepdims=True)
    occ = np.exp(scores)
    occ = occ / np.clip(occ.sum(axis=1, keepdims=True), 1e-12, None)

    # Symphony-like label transfer: kNN smooth occupancy in frozen VNN space
    Z = adata.obsm["X_vnn"]
    nn = NearestNeighbors(n_neighbors=min(30, max(5, adata.n_obs // 50)), metric="euclidean")
    nn.fit(Z)
    idx = nn.kneighbors(Z, return_distance=False)
    occ_s = occ[idx].mean(axis=1)

    hard = np.asarray(labels)[occ_s.argmax(axis=1)]
    # only observed labels as categories (empty cats break sc.pl.paga)
    order = [c for c in HSPC_COLORS if c in set(hard)]
    adata.obs["HSPC.annot"] = pd.Categorical(hard, categories=order)
    br = [BRANCH_MAP.get(h, "other") for h in hard]
    br_order = [c for c in ("stem", "myeloid", "MegE", "lymphoid", "other") if c in set(br)]
    adata.obs["branch_pred"] = pd.Categorical(br, categories=br_order)
    for j, lab in enumerate(labels):
        adata.obs[f"occupancy:{lab}"] = occ_s[:, j]
    adata.obs["annot_confidence"] = occ_s.max(axis=1)
    adata.uns["marker_genes_used"] = {k: v for k, v in used.items()}
    adata.obsm["X_occupancy"] = occ_s.astype(np.float32)


def _embed_frozen(adata: ad.AnnData) -> None:
    sc.pp.neighbors(adata, use_rep="X_vnn", n_neighbors=15, metric="euclidean")
    sc.tl.umap(adata, min_dist=0.3)
    adata.obsm["X_umap_vnn"] = adata.obsm["X_umap"].copy()


def _paga_dpt(adata: ad.AnnData) -> None:
    sc.tl.paga(adata, groups="HSPC.annot")
    sc.pl.paga(adata, plot=False)
    sc.tl.umap(adata, init_pos="paga")
    adata.obsm["X_umap_paga"] = adata.obsm["X_umap"].copy()
    # restore VNN umap as primary for occupancy panels
    adata.obsm["X_umap"] = adata.obsm["X_umap_vnn"]

    # DPT root = cell with highest HSC occupancy among high-conf HSC
    is_hsc = adata.obs["HSPC.annot"].astype(str).eq("HSC").to_numpy()
    if is_hsc.any():
        conf = adata.obs["occupancy:HSC"].to_numpy()
        conf = np.where(is_hsc, conf, -np.inf)
        root = int(np.argmax(conf))
    else:
        root = 0
    adata.uns["iroot"] = root
    sc.tl.diffmap(adata, n_comps=15)
    sc.tl.dpt(adata)
    adata.obs["pseudotime_pred"] = adata.obs["dpt_pseudotime"].astype(float)


def _scatter_cat(ax, um, labels, colors, title, order=None):
    labs = pd.Categorical(labels, categories=order) if order is not None else pd.Categorical(labels)
    codes = labs.codes
    order_ix = np.argsort(codes)
    c = [colors.get(str(labs.categories[c]), "0.7") if c >= 0 else "0.7" for c in codes]
    ax.scatter(
        um[order_ix, 0],
        um[order_ix, 1],
        c=np.asarray(c, dtype=object)[order_ix],
        s=5,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title, fontweight="bold")
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
            markerfacecolor=colors[str(cat)],
            markersize=6,
            label=str(cat),
        )
        for cat in labs.categories
        if str(cat) in colors and (labs == cat).any()
    ]
    return handles


def _plot_lineage_umap(adata: ad.AnnData, spec: dict):
    um = adata.obsm["X_umap_vnn"]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.0))
    h1 = _scatter_cat(
        axs[0],
        um,
        adata.obs["HSPC.annot"].astype(str),
        HSPC_COLORS,
        "Predicted HSPC.annot",
        order=list(HSPC_COLORS),
    )
    axs[0].legend(
        handles=h1,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
    )
    branch_colors = {
        "stem": "#1f77b4",
        "myeloid": "#2ca02c",
        "MegE": "#d62728",
        "lymphoid": "#9467bd",
        "other": "0.7",
    }
    h2 = _scatter_cat(
        axs[1],
        um,
        adata.obs["branch_pred"].astype(str),
        branch_colors,
        "Predicted branch",
        order=["stem", "myeloid", "MegE", "lymphoid", "other"],
    )
    axs[1].legend(
        handles=h2,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        spec["title"] + " · frozen VNN embedding",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    _save(f"{spec['stem']}_umap_lineage_pred.png", fig)
    plt.close(fig)


def _plot_occupancy(adata: ad.AnnData, spec: dict):
    """Arm × lineage occupancy (prediction summary) + UMAP soft occupancy for top lineages."""
    tab = (
        pd.crosstab(
            adata.obs["arm"].astype(str),
            adata.obs["HSPC.annot"].astype(str),
            normalize="index",
        )
        .reindex(index=list(spec["arm_order"]))
        .fillna(0.0)
    )
    cols = [c for c in HSPC_COLORS if c in tab.columns]
    tab = tab[cols]
    _save(f"{spec['stem']}_lineage_occupancy.csv", tab.reset_index())

    fig = plt.figure(figsize=(12.0, 7.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1, 1], height_ratios=[1.1, 1], hspace=0.35, wspace=0.3)

    axb = fig.add_subplot(gs[0, :])
    bottom = np.zeros(len(tab))
    x = np.arange(len(tab))
    for col in cols:
        axb.bar(
            x,
            tab[col].to_numpy(),
            bottom=bottom,
            color=HSPC_COLORS[col],
            width=0.7,
            label=col,
        )
        bottom = bottom + tab[col].to_numpy()
    axb.set_xticks(x)
    axb.set_xticklabels(list(spec["arm_labels"]), rotation=0)
    axb.set_ylabel("Lineage occupancy")
    axb.set_ylim(0, 1)
    axb.set_title("Predicted lineage occupancy by arm", fontweight="bold")
    axb.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=8,
        ncol=1,
    )
    for spine in ("top", "right"):
        axb.spines[spine].set_visible(False)

    um = adata.obsm["X_umap_vnn"]
    # show three high-interest occupancy maps
    focus = ["HSC", "GMP", "EryP"]
    for i, lab in enumerate(focus):
        ax = fig.add_subplot(gs[1, i])
        col = f"occupancy:{lab}"
        if col not in adata.obs:
            ax.axis("off")
            continue
        v = adata.obs[col].to_numpy(dtype=float)
        lo, hi = np.percentile(v, [5, 95])
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
        ax.set_title(f"P({lab} | VNN)", fontweight="bold", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(spec["title"] + " · lineage occupancy", fontweight="bold", y=0.98)
    _save(f"{spec['stem']}_umap_lineage_occupancy.png", fig)
    plt.close(fig)


def _plot_paga_pseudotime(adata: ad.AnnData, spec: dict):
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.8))
    # median VNN-UMAP position per predicted label → PAGA node layout
    labs = adata.obs["HSPC.annot"]
    um = adata.obsm["X_umap_vnn"]
    pos = np.zeros((len(labs.cat.categories), 2), dtype=float)
    for i, cat in enumerate(labs.cat.categories):
        m = labs.to_numpy() == cat
        pos[i] = um[m].mean(axis=0) if m.any() else 0.0
    adata.uns["paga"]["pos"] = pos
    sc.pl.paga(
        adata,
        ax=axs[0],
        show=False,
        pos=pos,
        node_size_scale=1.5,
        edge_width_scale=0.4,
        fontsize=7,
        frameon=False,
        color="dpt_pseudotime",
        cmap="gnuplot2",
    )
    axs[0].set_title("PAGA (HSPC.annot)", fontweight="bold")

    pt = adata.obs["pseudotime_pred"].to_numpy(dtype=float)
    med = float(np.nanmedian(pt)) if np.isfinite(pt).any() else 0.0
    pt = np.nan_to_num(pt, nan=med)
    sca = axs[1].scatter(
        um[:, 0],
        um[:, 1],
        c=pt,
        s=5,
        cmap="gnuplot2",
        linewidths=0,
        rasterized=True,
    )
    axs[1].set_title("Predicted pseudotime (DPT)", fontweight="bold")
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    for spine in axs[1].spines.values():
        spine.set_visible(False)
    fig.colorbar(sca, ax=axs[1], fraction=0.046, pad=0.02)
    fig.suptitle(spec["title"] + " · trajectory on frozen VNN", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(f"{spec['stem']}_umap_paga_pseudotime.png", fig)
    plt.close(fig)


def _plot_proxies_on_pred(adata: ad.AnnData, spec: dict):
    um = adata.obsm["X_umap_vnn"]
    feats = [(lab, col) for lab, col in spec["proxy_cols"] if col in adata.obs]
    fig, axs = plt.subplots(2, 2, figsize=(9.5, 8.0))
    for ax, (lab, col) in zip(axs.ravel(), feats):
        v = np.nan_to_num(adata.obs[col].to_numpy(dtype=float), nan=0.0)
        lo, hi = np.percentile(v, [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        sca = ax.scatter(
            um[:, 0],
            um[:, 1],
            c=np.clip(v, lo, hi),
            s=4,
            cmap="viridis",
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(lab, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(
        spec["title"] + " · VNN tasks/rates on predicted landscape",
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    _save(f"{spec['stem']}_umap_proxies_on_pred.png", fig)
    plt.close(fig)


def _export(adata: ad.AnnData, spec: dict):
    cols = [
        "arm",
        "genotype",
        "treatment",
        "HSPC.annot",
        "branch_pred",
        "annot_confidence",
        "pseudotime_pred",
    ]
    occ_cols = [c for c in adata.obs.columns if c.startswith("occupancy:")]
    df = adata.obs[cols + occ_cols].copy()
    df["UMAP1"] = adata.obsm["X_umap_vnn"][:, 0]
    df["UMAP2"] = adata.obsm["X_umap_vnn"][:, 1]
    for lab, col in spec["proxy_cols"]:
        if col in adata.obs:
            df[col] = adata.obs[col].to_numpy(dtype=float)
    df.index.name = "cell"
    _save(f"{spec['stem']}_lineage_predictions.csv", df.reset_index())

    # slim h5ad for reuse
    keep = ad.AnnData(
        X=sparse.csr_matrix(adata.obsm["X_vnn"]),
        obs=df,
        obsm={
            "X_vnn": adata.obsm["X_vnn"],
            "X_umap_vnn": adata.obsm["X_umap_vnn"],
            "X_occupancy": adata.obsm["X_occupancy"],
        },
    )
    _save(f"{spec['stem']}_lineage_pred.h5ad", keep)


def run_spec(spec: dict):
    print(f"== {spec['stem']}: load + attach frozen VNN", flush=True)
    adata = _load_hspc(spec)
    cells = _find_cells(spec["cells"])
    _attach_vnn(adata, cells)
    print(f"== {spec['stem']}: predict HSPC.annot / occupancy", flush=True)
    _predict_hspc_annot(adata, spec["markers"])
    print(f"== {spec['stem']}: UMAP + PAGA + DPT on X_vnn", flush=True)
    _embed_frozen(adata)
    _paga_dpt(adata)
    _plot_lineage_umap(adata, spec)
    _plot_occupancy(adata, spec)
    _plot_paga_pseudotime(adata, spec)
    _plot_proxies_on_pred(adata, spec)
    _export(adata, spec)
    print(
        f"{spec['stem']} done n={adata.n_obs} "
        f"annot=\n{adata.obs['HSPC.annot'].value_counts().to_string()}",
        flush=True,
    )


def main():
    print("writable dests:", [str(d) for d in DESTS], flush=True)
    for spec in SPECS:
        if not spec["qc"].exists():
            print("skip missing", spec["qc"], flush=True)
            continue
        run_spec(spec)


if __name__ == "__main__":
    main()
