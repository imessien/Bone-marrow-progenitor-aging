"""Age-core BM atlas: scGen joint → GMP vs agedHSC fate package.

Absorption fate on the neighbor graph (late GMP vs agedHSC), then fate UMAP,
driver-gene correlation, and ending GSEA.

Usage:
  source .venv/bin/activate
  python explore.py              # fate UMAP + drivers + GSEA
  python explore.py --train      # rebuild scGen + UMAP + DPT, then fate
  python -c "import explore; explore._self_check()"
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache_bone_marrow")

BONE = Path("/cis/net/r41/data/iessien1/bone")
RESULTS = Path("/cis/net/r41/data/iessien1/bone_marrow_results")
JOINT_OUT = RESULTS / "joint_hsc_aging"
MANIFEST = JOINT_OUT / "preprocess_manifest.json"
JOINT_H5AD = JOINT_OUT / "age_core_scgen.h5ad"
LABELED_H5AD = JOINT_OUT / "age_core_scgen_labeled.h5ad"
JOINT_FULL_H5AD = JOINT_OUT / "age_core_fullgene.h5ad"

AXIS_LINEAGES = ("HSPC", "Myeloid_prog")
TYPE_ORDER = ("HSC", "agedHSC", "MPP", "GMP")
TYPE_COLOR = {
    "HSC": "#2a6f6f",
    "agedHSC": "#b85c38",
    "MPP": "#6b7c3d",
    "GMP": "#4a5568",
}
FINE_MARKERS = {
    "HSC": ("Procr", "Hlf", "Mecom", "Hoxa9"),
    "agedHSC": ("Vwf", "Wwtr1", "Clca3a1"),
    "MPP": ("Flt3", "Cd34", "Kit", "Ly6a"),
    "GMP": ("Elane", "Mpo", "Ctsg", "Ms4a3", "Cebpe"),
}
FATE_GMP = "fate_GMP"
FATE_AGED = "fate_agedHSC"
GMP_DPT_Q = 0.80


def _as_csr(X):
    return X.tocsr() if sparse.issparse(X) else sparse.csr_matrix(X)


def _init_gpu() -> None:
    import cupy as cp
    import rmm
    import torch
    from rmm.allocators.cupy import rmm_cupy_allocator

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for rapids-singlecell.")
    rmm.reinitialize(managed_memory=False, pool_allocator=False, devices=0)
    cp.cuda.set_allocator(rmm_cupy_allocator)


def load_manifest(path: Path = MANIFEST) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; run: python preprocess.py --dataset age_core --annotate"
        )
    return json.loads(path.read_text())


def _age_bin_from_months(m) -> str:
    try:
        x = float(m)
    except (TypeError, ValueError):
        return "unassigned"
    if m is None or x != x:
        return "unassigned"
    if x <= 2.5:
        return "early"
    if x <= 8.0:
        return "mid"
    if x >= 18.0:
        return "late"
    return "unassigned"


def _harmonize_obs(a, dataset_key: str):
    if "gsm" not in a.obs.columns:
        a.obs["gsm"] = (
            a.obs["sample_name"].astype(str) if "sample_name" in a.obs else dataset_key
        )
    if "rep" not in a.obs.columns:
        a.obs["rep"] = "NA"
    if "age_group" not in a.obs.columns and "age_label" in a.obs.columns:
        lab = a.obs["age_label"].astype(str)
        a.obs["age_group"] = lab.map(
            {
                "young": "young",
                "juvenile": "young",
                "1mo": "young",
                "adult": "adult",
                "6mo": "adult",
                "old": "old",
                "20mo": "old",
            }
        ).fillna(lab)
    if "age_months" in a.obs:
        a.obs["age_months"] = pd.to_numeric(a.obs["age_months"], errors="coerce")
        a.obs["age_bin"] = a.obs["age_months"].map(_age_bin_from_months).astype(str)
    else:
        a.obs["age_bin"] = "unassigned"
    a.obs["cell_type"] = a.obs["lineage"].astype(str)
    a.obs["dataset_key"] = dataset_key
    for col in a.obs.columns:
        if a.obs[col].dtype == object or str(a.obs[col].dtype) == "category":
            a.obs[col] = a.obs[col].astype(str)
    return a


def load_age_core_axis(*, gene_join: str = "inner", exclude_datasets=("su2024",)):
    import anndata as ad
    import scanpy as sc

    exclude = set(exclude_datasets)
    entries = [e for e in load_manifest() if e["dataset"] not in exclude]
    parts = []
    for e in entries:
        a = sc.read_h5ad(e["path"])
        a.var_names_make_unique()
        a.obs_names_make_unique()
        a = _harmonize_obs(a, e["dataset"])
        keep = a.obs["lineage"].astype(str).isin(AXIS_LINEAGES)
        if e["dataset"] == "gse246464":
            ng = a.obs["n_genes"] if "n_genes" in a.obs else a.obs["n_genes_by_counts"]
            keep = keep & (ng.astype(float) >= 2000)
            keep = keep & (a.obs["pct_counts_mt"].astype(float) < 10.0)
        a = a[keep].copy()
        if a.n_obs == 0:
            continue
        if "counts" not in a.layers:
            raise RuntimeError(f"{e['path']} missing layers['counts']")
        a.layers["counts"] = _as_csr(a.layers["counts"])
        if sparse.issparse(a.X):
            xmax = float(a.X.data.max()) if a.X.nnz else 0.0
        else:
            xmax = float(np.asarray(a.X).max()) if a.n_obs else 0.0
        if xmax > 50:
            a.X = a.layers["counts"].copy()
            sc.pp.normalize_total(a, target_sum=1e4)
            sc.pp.log1p(a)
        a.obs_names = [f"{e['dataset']}_{x}" for x in a.obs_names]
        print(f"  {e['dataset']}: {a.n_obs:,} × {a.n_vars:,}")
        parts.append(a)
    if not parts:
        raise RuntimeError("No age-core cells after filtering")
    joint = ad.concat(parts, join=gene_join, merge="same", index_unique=None)
    joint.obs_names_make_unique()
    joint.X = _as_csr(joint.X)
    if "counts" in joint.layers:
        joint.layers["counts"] = _as_csr(joint.layers["counts"])
    nnz = np.asarray(joint.X.sum(axis=0)).ravel()
    joint = joint[:, nnz > 0].copy()
    print(f"joint: {joint.n_obs:,} × {joint.n_vars:,}")
    return joint


def _latent_key(adata) -> str:
    for k in ("corrected_latent", "X_scgen", "latent"):
        if k in adata.obsm:
            return k
    raise KeyError(f"no scGen latent; have {list(adata.obsm)}")


def run_scgen(joint, *, max_epochs: int = 100, n_top_genes: int = 7000):
    import rapids_singlecell as rsc
    import scgen
    import torch

    _init_gpu()
    adata = joint.copy()
    rsc.get.anndata_to_GPU(adata, convert_all=True)
    rsc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor="seurat_v3",
        layer="counts",
        batch_key="technical_batch",
    )
    rsc.get.anndata_to_CPU(adata, convert_all=True)
    adata.raw = adata.copy()
    JOINT_OUT.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(JOINT_FULL_H5AD)

    hvg = adata[:, adata.var["highly_variable"]].copy()
    hvg.obs["cell_type"] = hvg.obs["lineage"].astype(str)
    scgen.SCGEN.setup_anndata(hvg, batch_key="technical_batch", labels_key="cell_type")
    model = scgen.SCGEN(hvg, n_latent=30)
    model.train(
        max_epochs=max_epochs,
        batch_size=128,
        early_stopping=True,
        early_stopping_patience=25,
        accelerator="gpu" if torch.cuda.device_count() else "cpu",
        devices=1,
    )
    model.save(JOINT_OUT / "scgen_batch_removal", overwrite=True)
    corrected = model.batch_removal()
    corrected.obs = hvg.obs.loc[corrected.obs_names].copy()
    if corrected.raw is None and adata.raw is not None:
        corrected.raw = adata.raw
    corrected.obsm["X_scgen"] = corrected.obsm["latent"].copy()
    return corrected


def run_umap_dpt(corrected, *, n_dcs: int = 15):
    import rapids_singlecell as rsc
    import scanpy as sc

    use_rep = _latent_key(corrected)
    _init_gpu()
    rsc.get.anndata_to_GPU(corrected)
    rsc.pp.neighbors(corrected, n_neighbors=15, use_rep=use_rep)
    rsc.tl.umap(corrected)
    rsc.tl.diffmap(corrected, n_comps=n_dcs)
    rsc.get.anndata_to_CPU(corrected)
    corrected.uns["umap_use_rep"] = use_rep

    hspc = corrected.obs["lineage"].astype(str) == "HSPC"
    if not bool(hspc.any()):
        raise RuntimeError("No HSPC cells to root DPT")
    if "Procr" in corrected.var_names:
        x = corrected[:, "Procr"].X
        if hasattr(x, "toarray"):
            x = x.toarray()
        scores = np.where(hspc.to_numpy(), np.asarray(x).ravel(), -np.inf)
        root_ix = int(np.argmax(scores))
    else:
        root_ix = int(np.flatnonzero(hspc.to_numpy())[0])
    corrected.uns["iroot"] = root_ix
    sc.tl.dpt(corrected, n_dcs=min(10, n_dcs - 1))
    return corrected


def ensure_display_type(adata: AnnData) -> None:
    """Leiden on scGen latent → HSC/agedHSC/MPP/GMP; else lineage map."""
    import scanpy as sc

    if "display_type" in adata.obs:
        return
    if "leiden_named" in adata.obs:
        adata.obs["display_type"] = (
            adata.obs["leiden_named"].astype(str).str.split("_").str[-1]
        )
        return

    key = _latent_key(adata)
    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, n_neighbors=15, use_rep=key)
    sc.tl.leiden(
        adata,
        resolution=1.0,
        key_added="leiden_fine",
        random_state=0,
        flavor="igraph",
        n_iterations=2,
    )
    score_cols = []
    for name, genes in FINE_MARKERS.items():
        present = [g for g in genes if g in adata.var_names]
        if not present:
            continue
        col = f"fine_{name}"
        sc.tl.score_genes(adata, present, score_name=col, use_raw=False)
        score_cols.append(col)
    if not score_cols:
        adata.obs["display_type"] = (
            adata.obs["lineage"]
            .astype(str)
            .map({"HSPC": "HSC", "Myeloid_prog": "GMP"})
            .fillna("MPP")
        )
        return
    names = {}
    for cl in adata.obs["leiden_fine"].astype(str).unique():
        m = adata.obs["leiden_fine"].astype(str) == cl
        means = {
            c.replace("fine_", ""): float(adata.obs.loc[m, c].mean()) for c in score_cols
        }
        names[cl] = max(means, key=means.get)
    adata.obs["display_type"] = adata.obs["leiden_fine"].astype(str).map(names)


def _umap_xy(adata: AnnData) -> np.ndarray:
    if "X_umap" not in adata.obsm:
        raise KeyError("Missing X_umap — run --train first")
    return np.asarray(adata.obsm["X_umap"])[:, :2]


def _row_stochastic(conn) -> sparse.csr_matrix:
    conn = conn.tocsr()
    rs = np.asarray(conn.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    return sparse.diags(1.0 / rs) @ conn


def select_fate_terminals(
    adata: AnnData, *, gmp_dpt_q: float = GMP_DPT_Q
) -> tuple[np.ndarray, np.ndarray]:
    """Absorbing masks: late-DPT GMP vs all agedHSC."""
    ensure_display_type(adata)
    if "dpt_pseudotime" not in adata.obs:
        raise KeyError("Missing dpt_pseudotime — run --train first")
    dt = adata.obs["display_type"].astype(str)
    dpt = pd.to_numeric(adata.obs["dpt_pseudotime"], errors="coerce").to_numpy()
    gmp = (dt == "GMP").to_numpy()
    if not gmp.any():
        raise RuntimeError("No GMP cells for fate terminal A")
    thr = float(np.nanquantile(dpt[gmp], gmp_dpt_q))
    term_gmp = gmp & np.isfinite(dpt) & (dpt >= thr)
    term_aged = (dt == "agedHSC").to_numpy()
    if not term_gmp.any():
        raise RuntimeError("Empty late-GMP terminal — lower gmp_dpt_q")
    if not term_aged.any():
        raise RuntimeError("No agedHSC cells for fate terminal B")
    both = term_gmp & term_aged
    if both.any():
        term_gmp = term_gmp & ~both
        term_aged = term_aged & ~both
    return term_gmp, term_aged


def compute_fate_probs(
    adata: AnnData,
    *,
    gmp_dpt_q: float = GMP_DPT_Q,
    max_iter: int = 400,
    tol: float = 1e-5,
) -> AnnData:
    """Neighbor absorption → obs[fate_GMP], obs[fate_agedHSC]."""
    if "connectivities" not in adata.obsp:
        raise KeyError("Missing neighbors connectivities — run --train first")
    term_gmp, term_aged = select_fate_terminals(adata, gmp_dpt_q=gmp_dpt_q)
    T = _row_stochastic(adata.obsp["connectivities"])

    def _absorb(hit: np.ndarray, miss: np.ndarray) -> np.ndarray:
        p = np.zeros(adata.n_obs, dtype=np.float64)
        p[hit] = 1.0
        p[miss] = 0.0
        for _ in range(max_iter):
            p_new = T @ p
            p_new[hit] = 1.0
            p_new[miss] = 0.0
            if float(np.max(np.abs(p_new - p))) < tol:
                return p_new
            p = p_new
        return p

    p_gmp = _absorb(term_gmp, term_aged)
    p_aged = _absorb(term_aged, term_gmp)
    s = p_gmp + p_aged
    s[s == 0] = 1.0
    adata.obs[FATE_GMP] = p_gmp / s
    adata.obs[FATE_AGED] = p_aged / s
    adata.obs["terminal_GMP"] = term_gmp
    adata.obs["terminal_agedHSC"] = term_aged
    adata.uns["fate_terminals"] = {
        "gmp_dpt_q": gmp_dpt_q,
        "n_terminal_GMP": int(term_gmp.sum()),
        "n_terminal_agedHSC": int(term_aged.sum()),
    }
    print(
        f"  terminals: GMP={int(term_gmp.sum())} agedHSC={int(term_aged.sum())} "
        f"(dpt_q={gmp_dpt_q})"
    )
    return adata


def _grid_flow(xy: np.ndarray, T, *, n_grid: int = 18):
    disp = np.asarray(T @ xy) - xy
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    dx = (xmax - xmin) / max(n_grid, 1)
    dy = (ymax - ymin) / max(n_grid, 1)
    gx, gy, gu, gv = [], [], [], []
    for x0 in np.linspace(xmin, xmax, n_grid):
        for y0 in np.linspace(ymin, ymax, n_grid):
            m = (
                (xy[:, 0] >= x0 - dx / 2)
                & (xy[:, 0] < x0 + dx / 2)
                & (xy[:, 1] >= y0 - dy / 2)
                & (xy[:, 1] < y0 + dy / 2)
            )
            if int(m.sum()) < 5:
                continue
            d = disp[m].mean(axis=0)
            if float(np.linalg.norm(d)) < 1e-6:
                continue
            gx.append(x0)
            gy.append(y0)
            gu.append(d[0])
            gv.append(d[1])
    return np.asarray(gx), np.asarray(gy), np.asarray(gu), np.asarray(gv)


def plot_fate_umap(adata: AnnData, out_dir: Path = JOINT_OUT) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_display_type(adata)
    if FATE_GMP not in adata.obs:
        compute_fate_probs(adata)
    xy = _umap_xy(adata)
    fate = pd.to_numeric(adata.obs[FATE_GMP], errors="coerce").to_numpy()
    lab = adata.obs["display_type"].astype(str).to_numpy()
    T = _row_stochastic(adata.obsp["connectivities"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sc0 = axes[0].scatter(
        xy[:, 0], xy[:, 1], c=fate, s=2, alpha=0.7, cmap="RdBu_r", vmin=0, vmax=1, rasterized=True
    )
    plt.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04, label="P(GMP sink)")
    axes[0].set_title("Fate: GMP vs agedHSC")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    for t in TYPE_ORDER:
        m = lab == t
        if not m.any():
            continue
        axes[1].scatter(
            xy[m, 0], xy[m, 1], s=2, alpha=0.65, c=TYPE_COLOR.get(t, "#555"), label=t, rasterized=True
        )
        axes[1].annotate(
            t,
            (float(xy[m, 0].mean()), float(xy[m, 1].mean())),
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            color=TYPE_COLOR.get(t, "#222"),
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.75,
                "edgecolor": TYPE_COLOR.get(t, "#222"),
            },
        )
    gx, gy, gu, gv = _grid_flow(xy, T)
    if len(gx):
        axes[1].quiver(
            gx, gy, gu, gv, color="k", alpha=0.55, width=0.003, angles="xy", scale_units="xy"
        )
    axes[1].legend(markerscale=3, frameon=False, fontsize=8, loc="best")
    axes[1].set_title("Types + flow")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.suptitle("Competing fates: GMP-committed vs agedHSC-persistent", y=1.02)
    fig.tight_layout()
    path = out_dir / "panels_age_bin_fate.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")
    return path


def gene_fate_correlations(adata: AnnData) -> pd.DataFrame:
    """Pearson corr with P(GMP); corr_agedHSC = −corr_GMP for binary fates."""
    if FATE_GMP not in adata.obs:
        compute_fate_probs(adata)
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    mean_expr = X.mean(axis=0)
    Xc = X - mean_expr
    Xc /= Xc.std(axis=0) + 1e-8
    f = pd.to_numeric(adata.obs[FATE_GMP], errors="coerce").to_numpy(dtype=np.float64)
    f = (f - f.mean()) / (f.std() + 1e-8)
    corr_gmp = (Xc.T @ f) / n
    return pd.DataFrame(
        {
            "gene": adata.var_names.astype(str),
            "corr_GMP": corr_gmp,
            "corr_agedHSC": -corr_gmp,
            "mean_expr": mean_expr,
        }
    )


def plot_driver_corr(corr: pd.DataFrame, out_dir: Path = JOINT_OUT) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc0 = ax.scatter(
        corr["corr_GMP"],
        corr["mean_expr"],
        c=corr["corr_GMP"],
        s=10,
        alpha=0.65,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        rasterized=True,
    )
    plt.colorbar(sc0, ax=ax, fraction=0.046, pad=0.04, label="corr P(GMP)")
    ends = pd.concat([corr.nlargest(15, "corr_GMP"), corr.nsmallest(15, "corr_GMP")])
    for _, r in ends.drop_duplicates("gene").iterrows():
        ax.annotate(r["gene"], (r["corr_GMP"], r["mean_expr"]), fontsize=6, alpha=0.9)
    ax.axvline(0, color="#888", lw=0.6)
    ax.set_xlabel("corr with P(GMP-committed)  [left ← agedHSC | GMP → right]")
    ax.set_ylabel("mean expression")
    ax.set_title("Driver genes: GMP vs agedHSC fate")
    fig.tight_layout()
    path = out_dir / "drivers_fate_corr.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    csv = out_dir / "drivers_fate_corr.csv"
    corr.sort_values("corr_GMP", ascending=False).to_csv(csv, index=False)
    print(f"  → {path}")
    print(f"  → {csv}")
    return path


def run_ending_gsea(corr: pd.DataFrame, out_dir: Path = JOINT_OUT) -> Path | None:
    import gseapy as gp

    out_dir.mkdir(parents=True, exist_ok=True)
    rnk = (
        corr[["gene", "corr_GMP"]]
        .rename(columns={"corr_GMP": "score"})
        .dropna()
        .sort_values("score", ascending=False)
    )
    rnk.to_csv(out_dir / "fate_prerank.rnk", sep="\t", index=False, header=False)
    try:
        res = gp.prerank(
            rnk=rnk,
            gene_sets="GO_Biological_Process_2023",
            organism="Mouse",
            outdir=str(out_dir / "gsea_fate"),
            min_size=10,
            max_size=500,
            permutation_num=200,
            seed=0,
            verbose=False,
        )
    except Exception as e:
        print(f"  GSEA skipped: {e}")
        return None
    table = res.res2d if hasattr(res, "res2d") else None
    if table is None or len(table) == 0:
        print("  GSEA: no enriched terms")
        return None
    out = out_dir / "gsea_fate_ending.csv"
    table.to_csv(out, index=False)
    print(f"  → {out} ({len(table)} terms)")
    return out


def run_fate_package(*, h5ad: Path | None = None) -> list[Path]:
    import scanpy as sc

    path = h5ad or (LABELED_H5AD if LABELED_H5AD.exists() else JOINT_H5AD)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run: python explore.py --train")
    print(f"Loading {path}…")
    adata = sc.read_h5ad(path)
    print("Fate absorption (GMP vs agedHSC)…")
    compute_fate_probs(adata)
    written: list[Path] = [plot_fate_umap(adata, JOINT_OUT)]
    print("Driver correlations…")
    corr = gene_fate_correlations(adata)
    written.append(plot_driver_corr(corr, JOINT_OUT))
    print("Ending GSEA…")
    g = run_ending_gsea(corr, JOINT_OUT)
    if g is not None:
        written.append(g)
    try:
        adata.write_h5ad(LABELED_H5AD)
        print(f"  → updated {LABELED_H5AD} with fate columns")
    except OSError as e:
        print(f"  (skip h5ad write: {e})")
    return written


def train(*, max_epochs: int = 100) -> Path:
    warnings.filterwarnings("ignore", category=FutureWarning)
    print("Loading age-core…")
    joint = load_age_core_axis()
    print("scGen…")
    corrected = run_scgen(joint, max_epochs=max_epochs)
    print("UMAP + DPT…")
    corrected = run_umap_dpt(corrected)
    ensure_display_type(corrected)
    corrected.write_h5ad(JOINT_H5AD)
    corrected.write_h5ad(LABELED_H5AD)
    print(f"wrote {JOINT_H5AD}")
    return JOINT_H5AD


def _self_check() -> None:
    rng = np.random.default_rng(0)
    n = 300
    adata = AnnData(
        X=rng.normal(size=(n, 8)).astype(np.float32),
        obs=pd.DataFrame(
            {
                "lineage": ["HSPC"] * 150 + ["Myeloid_prog"] * 150,
                "display_type": ["HSC"] * 80 + ["agedHSC"] * 40 + ["MPP"] * 80 + ["GMP"] * 100,
                "dpt_pseudotime": np.concatenate(
                    [rng.uniform(0, 0.2, 120), rng.uniform(0.2, 0.5, 80), rng.uniform(0.5, 1.0, 100)]
                ),
            }
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(8)]),
    )
    adata.obsm["X_umap"] = rng.normal(size=(n, 2)).astype(np.float32)
    knn = sparse.random(n, n, density=0.05, random_state=0, format="csr")
    knn = knn + knn.T
    knn.setdiag(0)
    knn.eliminate_zeros()
    adata.obsp["connectivities"] = knn
    compute_fate_probs(adata, gmp_dpt_q=0.5, max_iter=50)
    assert FATE_GMP in adata.obs
    out = Path("/tmp/explore_fate_check")
    assert plot_fate_umap(adata, out).exists()
    assert plot_driver_corr(gene_fate_correlations(adata), out).exists()
    print("explore._self_check: OK", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", action="store_true", help="Rebuild scGen + UMAP/DPT first")
    p.add_argument("--max-epochs", type=int, default=100)
    args = p.parse_args()
    if args.train:
        train(max_epochs=args.max_epochs)
    run_fate_package()
