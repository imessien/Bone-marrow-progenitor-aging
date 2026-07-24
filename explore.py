"""Age-core mouse BM HSC→GMP aging atlas (CLI).

Pipeline:
  1. Load QC'd age-core h5ads from ``preprocess_manifest.json``
  2. Keep HSC→GMP axis (``HSPC`` + ``Myeloid_prog``); Elias uses marker lineage
  3. Shared-gene concat (``gene_join=inner``); ``batch_key=technical_batch``
  4. scGen ``batch_removal`` → UMAP on ``corrected_latent``
  5. Optional: young↔old condition transfer on HSPC
  6. DPT pseudotime + age_bin / marker stream plots via ``plotting``

Outputs: ``results/joint_hsc_aging/`` on the bone store
(``/cis/net/r41/data/iessien1/bone/results/joint_hsc_aging``).

Usage:
  source .venv/bin/activate
  python preprocess.py --dataset age_core --annotate
  python explore.py --skip-transfer
  python explore.py --pseudotime-only
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

# numba/scanpy cache fails in some sandboxes without an explicit cache dir
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache_bone_marrow")

BONE = Path("/cis/net/r41/data/iessien1/bone")
RESULTS = BONE / "results" / "joint_hsc_aging"
MANIFEST = RESULTS / "preprocess_manifest.json"
JOINT_OUT = RESULTS
JOINT_H5AD = JOINT_OUT / "age_core_scgen.h5ad"
JOINT_FULL_H5AD = JOINT_OUT / "age_core_fullgene.h5ad"
TRANSFER_H5AD = JOINT_OUT / "age_core_young_old_transfer.h5ad"
UMAP_PNG = JOINT_OUT / "umap_age_core_scgen.png"
ELIAS_UMAP_PNG = JOINT_OUT / "umap_elias_gsm_check.png"
AGE_BIN_UMAP_PNG = JOINT_OUT / "umap_age_bin_scgen.png"
MARKER_STREAM_PNG = JOINT_OUT / "stream_markers_scgen.png"

# HSC→GMP axis markers (same panels as preprocess); used for stream QC plots
AXIS_MARKERS: tuple[str, ...] = (
    "Procr",
    "Kit",
    "Cd34",
    "Flt3",
    "Mpo",
    "Elane",
    "Ms4a3",
    "Cebpe",
)
SU_HOLDOUT_JSON = JOINT_OUT / "su_holdout_age_summary.json"

AXIS_LINEAGES = ("HSPC", "Myeloid_prog")
ELIAS_GSMS = (
    "GSM7869307",  # HSC young rep1
    "GSM7869308",  # HSC old rep1
    "GSM7869309",  # HSC young rep2
    "GSM7869310",  # HSC old rep2
)
ELIAS_DATASET = "Elias2025"
ELIAS_BATCH = "Elias_GSE246464_10x"


def _as_csr(X):
    from scipy import sparse

    if sparse.issparse(X):
        return X.tocsr()
    return sparse.csr_matrix(X)


def _init_gpu() -> None:
    import cupy as cp
    import rmm
    import torch
    from rmm.allocators.cupy import rmm_cupy_allocator

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for rapids-singlecell HVG / neighbors.")
    rmm.reinitialize(managed_memory=False, pool_allocator=False, devices=0)
    cp.cuda.set_allocator(rmm_cupy_allocator)


def load_manifest(path: Path = MANIFEST) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; run: python preprocess.py --dataset age_core --annotate"
        )
    return json.loads(path.read_text())


def _age_bin_from_months(m) -> str:
    """early ≤2.5; mid (2.5, 8]; late ≥18; else unassigned."""
    if m is None:
        return "unassigned"
    try:
        x = float(m)
    except (TypeError, ValueError):
        return "unassigned"
    if x != x:  # NaN
        return "unassigned"
    if x <= 2.5:
        return "early"
    if x <= 8.0:
        return "mid"
    if x >= 18.0:
        return "late"
    return "unassigned"


def _harmonize_obs(a, dataset_key: str):
    """Fill shared obs columns used by scGen + UMAP checks."""
    import pandas as pd

    if "gsm" not in a.obs.columns:
        if "sample_name" in a.obs:
            a.obs["gsm"] = a.obs["sample_name"].astype(str)
        else:
            a.obs["gsm"] = dataset_key
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
    # Binary condition for young↔old transfer (drop adult from transfer train)
    ag = a.obs["age_group"].astype(str)
    a.obs["age_condition"] = ag.where(ag.isin(["young", "old"]), other="exclude")
    if "age_months" in a.obs:
        a.obs["age_months"] = pd.to_numeric(a.obs["age_months"], errors="coerce")
        a.obs["age_bin"] = a.obs["age_months"].map(_age_bin_from_months).astype(str)
    else:
        a.obs["age_bin"] = "unassigned"
    a.obs["cell_type"] = a.obs["lineage"].astype(str)
    a.obs["dataset_key"] = dataset_key
    for col in (
        "dataset",
        "technical_batch",
        "lineage",
        "age_label",
        "age_group",
        "age_bin",
        "age_condition",
        "genotype",
        "gsm",
        "sample_name",
        "rep",
        "cell_type",
        "dataset_key",
    ):
        if col in a.obs:
            a.obs[col] = a.obs[col].astype(str)
    return a


def load_age_core_axis(
    *,
    lineages: tuple[str, ...] = AXIS_LINEAGES,
    wt_only: bool = False,
    gene_join: str = "inner",
    exclude_datasets: tuple[str, ...] = ("su2024",),
) -> "ad.AnnData":
    """Shared-gene concat of QC'd age-core sets on the HSC→GMP axis.

    ``gene_join='inner'`` (default) avoids outer-join zero-fill artifacts that
    isolate Smart-seq2 (Su) and multiome RNA (Elias) in UMAP space.

    Su2024 is excluded by default: Smart-seq2 depth/feature space does not mix
    with 10x under scGen batch_removal even after symbol/QC fixes (pinpoint
    streaks). Pass ``exclude_datasets=()`` to force-include.
    """
    import anndata as ad
    import numpy as np
    import scanpy as sc
    from scipy import sparse

    if gene_join not in {"inner", "outer"}:
        raise ValueError(f"gene_join must be inner|outer, got {gene_join}")

    exclude = set(exclude_datasets)
    entries = [e for e in load_manifest() if e["dataset"] not in exclude]
    skipped = sorted({e["dataset"] for e in load_manifest()} & exclude)
    if skipped:
        print(f"  excluding from joint: {skipped}")
    parts = []
    for e in entries:
        a = sc.read_h5ad(e["path"])
        a.var_names_make_unique()
        a.obs_names_make_unique()
        a = _harmonize_obs(a, e["dataset"])
        keep = a.obs["lineage"].astype(str).isin(lineages)
        if wt_only and "genotype" in a.obs:
            keep = keep & (a.obs["genotype"].astype(str) == "WT")
        # Extra artifact gate for Elias multiome / Su Smart-seq2
        if e["dataset"] == "gse246464":
            ng = a.obs["n_genes"] if "n_genes" in a.obs else a.obs["n_genes_by_counts"]
            keep = keep & (ng.astype(float) >= 2000)
            keep = keep & (a.obs["pct_counts_mt"].astype(float) < 10.0)
            if "ribo_frac" in a.obs:
                keep = keep & (a.obs["ribo_frac"].astype(float) < 0.25)
            if "score_HSPC" in a.obs:
                keep = keep & (
                    (a.obs["lineage"].astype(str) != "HSPC")
                    | (a.obs["score_HSPC"].astype(float) >= 0.2)
                )
        if e["dataset"] == "su2024":
            ng = a.obs["n_genes_by_counts"] if "n_genes_by_counts" in a.obs else a.obs.get("n_genes")
            if ng is not None:
                keep = keep & (ng.astype(float) >= 2000)
            keep = keep & (a.obs["pct_counts_mt"].astype(float) < 5.0)
            if "ribo_frac" in a.obs:
                keep = keep & (a.obs["ribo_frac"].astype(float) < 0.15)
        a = a[keep].copy()
        if a.n_obs == 0:
            print(f"  skip {e['dataset']}: 0 cells after lineage/QC filter")
            continue
        if "counts" not in a.layers:
            raise RuntimeError(f"{e['path']} missing layers['counts']")
        a.layers["counts"] = _as_csr(a.layers["counts"])
        # Prefer log-normalized X from preprocess; fall back to counts→normalize
        if sparse.issparse(a.X):
            xmax = float(a.X.data.max()) if a.X.nnz else 0.0
        else:
            xmax = float(np.asarray(a.X).max()) if a.n_obs else 0.0
        if xmax > 50:  # looks like raw counts left in X
            a.X = a.layers["counts"].copy()
            sc.pp.normalize_total(a, target_sum=1e4)
            sc.pp.log1p(a)
        tag = e["dataset"]
        a.obs_names = [f"{tag}_{x}" for x in a.obs_names]
        print(
            f"  {tag}: {a.n_obs:,} × {a.n_vars:,} | "
            f"batch={a.obs['technical_batch'].iloc[0]} | "
            f"ages={a.obs['age_group'].value_counts().to_dict()}"
        )
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
    print(
        f"gene-{gene_join} joint: {joint.n_obs:,} cells × {joint.n_vars:,} genes | "
        f"batches={sorted(joint.obs['technical_batch'].unique())}"
    )
    return joint


def assert_elias_gsms(joint) -> dict[str, int]:
    """Confirm Elias GSE246464 young/old HSC replicates are present."""
    elias = joint.obs["technical_batch"].astype(str) == ELIAS_BATCH
    if not elias.any():
        # fallback on dataset name
        elias = joint.obs["dataset"].astype(str) == ELIAS_DATASET
    sub = joint.obs.loc[elias]
    counts = sub["gsm"].astype(str).value_counts().to_dict()
    missing = [g for g in ELIAS_GSMS if counts.get(g, 0) == 0]
    if missing:
        raise RuntimeError(
            f"Elias GSMs missing from joint object: {missing}; have={counts}"
        )
    print("Elias GSE246464 GSM counts:", {g: counts[g] for g in ELIAS_GSMS})
    return counts


def run_scgen_batch_removal(joint, *, max_epochs: int = 100, n_top_genes: int = 7000):
    """HVG → scGen batch_removal with batch_key=technical_batch."""
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
    print(f"full-gene → {JOINT_FULL_H5AD}")

    hvg = adata[:, adata.var["highly_variable"]].copy()
    hvg.obs["cell_type"] = hvg.obs["lineage"].astype(str)
    scgen.SCGEN.setup_anndata(
        hvg,
        batch_key="technical_batch",
        labels_key="cell_type",
    )
    model = scgen.SCGEN(hvg, n_latent=30)
    n_gpu = torch.cuda.device_count()
    model.train(
        max_epochs=max_epochs,
        batch_size=128,
        early_stopping=True,
        early_stopping_patience=25,
        accelerator="gpu" if n_gpu else "cpu",
        devices=1,
    )
    model.save(JOINT_OUT / "scgen_batch_removal", overwrite=True)
    corrected = model.batch_removal()
    corrected.obs = hvg.obs.loc[corrected.obs_names].copy()
    if corrected.raw is None and adata.raw is not None:
        corrected.raw = adata.raw
    corrected.obsm["X_scgen"] = corrected.obsm["latent"].copy()
    corrected.uns["integration"] = {
        "method": "scGen",
        "batch_key": "technical_batch",
        "labels_key": "cell_type",
        "lineages": list(AXIS_LINEAGES),
        "n_gpu": int(n_gpu),
        "elias_gsms": list(ELIAS_GSMS),
    }
    # hvg retains pre-correction X for a separate young↔old condition model
    return corrected, model, hvg


def resolve_scgen_rep(adata, use_rep: str = "corrected_latent") -> str:
    """Prefer corrected_latent → X_scgen → latent (matches scGen tutorial + this repo)."""
    if use_rep in adata.obsm:
        return use_rep
    if use_rep == "corrected_latent" and "X_scgen" in adata.obsm:
        return "X_scgen"
    if "latent" in adata.obsm:
        return "latent"
    raise KeyError(f"No embedding {use_rep}; have {list(adata.obsm)}")


def run_umap(corrected, *, use_rep: str = "corrected_latent"):
    """Neighbors/UMAP on scGen low-dim space (GPU via rapids-singlecell)."""
    import rapids_singlecell as rsc

    use_rep = resolve_scgen_rep(corrected, use_rep)
    _init_gpu()
    rsc.get.anndata_to_GPU(corrected)
    rsc.pp.neighbors(corrected, n_neighbors=15, use_rep=use_rep)
    rsc.tl.umap(corrected)
    rsc.get.anndata_to_CPU(corrected)
    corrected.uns["umap_use_rep"] = use_rep
    return corrected


def run_pseudotime(corrected, *, use_rep: str = "corrected_latent", n_dcs: int = 15):
    """GPU neighbors/diffmap on scGen latent, then scanpy DPT rooted in HSPC.

    Writes ``obs['dpt_pseudotime']``. Reuses existing neighbors if ``use_rep`` matches.
    """
    import numpy as np
    import rapids_singlecell as rsc
    import scanpy as sc

    use_rep = resolve_scgen_rep(corrected, use_rep)
    _init_gpu()
    rsc.get.anndata_to_GPU(corrected)
    need_neighbors = (
        "neighbors" not in corrected.uns
        or corrected.uns.get("umap_use_rep") != use_rep
    )
    if need_neighbors:
        rsc.pp.neighbors(corrected, n_neighbors=15, use_rep=use_rep)
        rsc.tl.umap(corrected)
        corrected.uns["umap_use_rep"] = use_rep
    rsc.tl.diffmap(corrected, n_comps=n_dcs)
    rsc.get.anndata_to_CPU(corrected)

    # Root = HSPC cell with highest Procr (fallback: first HSPC)
    hspc = corrected.obs["lineage"].astype(str) == "HSPC"
    if not bool(hspc.any()):
        raise RuntimeError("No HSPC cells to root DPT")
    if "Procr" in corrected.var_names:
        x = corrected[:, "Procr"].X
        if hasattr(x, "toarray"):
            x = x.toarray()
        scores = np.asarray(x).ravel()
        scores = np.where(hspc.to_numpy(), scores, -np.inf)
        root_ix = int(np.argmax(scores))
    else:
        root_ix = int(np.flatnonzero(hspc.to_numpy())[0])
    corrected.uns["iroot"] = root_ix
    sc.tl.dpt(corrected, n_dcs=min(10, n_dcs - 1))
    corrected.uns["dpt"] = {
        "root_index": root_ix,
        "root_lineage": "HSPC",
        "use_rep": use_rep,
        "n_dcs": n_dcs,
    }
    print(
        f"DPT rooted at cell {root_ix} "
        f"(lineage={corrected.obs['lineage'].iloc[root_ix]}, rep={use_rep})"
    )
    return corrected


def plot_integration_umaps(corrected, out_png: Path = UMAP_PNG, elias_png: Path = ELIAS_UMAP_PNG):
    import matplotlib.pyplot as plt
    import scanpy as sc

    # Cohort overview (include calendar age_bin)
    sc.pl.umap(
        corrected,
        color=["technical_batch", "dataset", "lineage", "age_group", "age_bin"],
        ncols=3,
        wspace=0.45,
        show=False,
    )
    fig = plt.gcf()
    fig.set_size_inches(12, 10)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Elias GSM check: highlight the four expected libraries
    obs = corrected.obs.copy()
    is_elias = obs["technical_batch"].astype(str) == ELIAS_BATCH
    gsm = obs["gsm"].astype(str)
    obs["elias_gsm"] = "other"
    for g in ELIAS_GSMS:
        obs.loc[is_elias & (gsm == g), "elias_gsm"] = g
    corrected.obs["elias_gsm"] = obs["elias_gsm"].astype(str)
    sc.pl.umap(
        corrected,
        color=["elias_gsm", "age_group", "rep"],
        ncols=3,
        wspace=0.4,
        show=False,
        title=["Elias GSM7869307–910", "age_group", "rep"],
    )
    fig2 = plt.gcf()
    fig2.set_size_inches(14, 4.5)
    fig2.savefig(elias_png, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"UMAP → {out_png} (use_rep={corrected.uns.get('umap_use_rep')})")
    print(f"Elias GSM check → {elias_png}")
    return out_png, elias_png


def plot_age_bin_umap(corrected, out_png: Path = AGE_BIN_UMAP_PNG) -> Path:
    """UMAP focused on calendar age_bin (+ lineage / genotype context)."""
    import matplotlib.pyplot as plt
    import scanpy as sc

    colors = ["age_bin", "lineage", "age_group"]
    if "genotype" in corrected.obs:
        colors.append("genotype")
    sc.pl.umap(corrected, color=colors, ncols=2, wspace=0.4, show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"age_bin UMAP → {out_png}")
    return out_png


def plot_marker_streams(corrected, out_png: Path = MARKER_STREAM_PNG) -> Path:
    """Branch streams colored by age_bin + axis marker genes (custom plotting)."""
    import matplotlib.pyplot as plt
    import plotting as pl

    if "dpt_pseudotime" not in corrected.obs:
        raise KeyError("dpt_pseudotime missing; call run_pseudotime first")
    genes = [g for g in AXIS_MARKERS if g in corrected.var_names]
    fig = pl.plot_marker_branch_streams(
        corrected,
        genes=genes,
        pt_key="dpt_pseudotime",
        branch_key="lineage",
        context_color="age_bin",
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"marker streams → {out_png} (genes={genes})")
    return out_png


def run_young_old_transfer(joint_hvg, *, celltype: str = "HSPC", max_epochs: int = 100):
    """Second scGen: batch_key=age_condition for young→old prediction on HSPC."""
    import scgen
    import torch

    train = joint_hvg[
        (joint_hvg.obs["age_condition"].isin(["young", "old"]))
        & (joint_hvg.obs["genotype"].astype(str) == "WT")
    ].copy()
    scgen.SCGEN.setup_anndata(
        train,
        batch_key="age_condition",
        labels_key="cell_type",
    )
    model = scgen.SCGEN(train, n_latent=30)
    n_gpu = torch.cuda.device_count()
    model.train(
        max_epochs=max_epochs,
        batch_size=128,
        early_stopping=True,
        early_stopping_patience=25,
        accelerator="gpu" if n_gpu else "cpu",
        devices=1,
    )
    model.save(JOINT_OUT / "scgen_young_old", overwrite=True)
    pred, delta = model.predict(
        ctrl_key="young",
        stim_key="old",
        celltype_to_predict=celltype,
    )
    pred.obs["age_condition"] = "pred_old"
    pred.obs["age_group"] = "pred_old"
    pred.obs["lineage"] = celltype
    pred.obs["cell_type"] = celltype
    pred.obs["technical_batch"] = "scGen_young_to_old"
    pred.obs["dataset"] = "scGen_transfer"
    print(
        f"young→old transfer ({celltype}): pred={pred.n_obs:,} cells, "
        f"delta_l2={float((delta ** 2).sum() ** 0.5):.3f}"
    )
    return pred, delta, model


def su_holdout_age_summary(*, joint_genes: set[str] | None = None) -> dict:
    """Su2024 adult/juvenile holdout — shared genes only, no atlas retrain.

    Su stays out of the primary scGen UMAP. Use it only for ages the 10x core
    lacks (adult, juvenile). Do **not** stack scANVI/scArches on scGen latents.
    """
    import scanpy as sc

    entries = {e["dataset"]: e for e in load_manifest()}
    if "su2024" not in entries:
        raise FileNotFoundError("su2024 missing from preprocess_manifest.json")
    su = sc.read_h5ad(entries["su2024"]["path"])
    su = _harmonize_obs(su, "su2024")
    keep = su.obs["lineage"].astype(str).isin(AXIS_LINEAGES)
    su = su[keep].copy()

    if joint_genes is None and JOINT_H5AD.exists():
        ref = sc.read_h5ad(JOINT_H5AD, backed="r")
        joint_genes = set(map(str, ref.var_names))
    elif joint_genes is None:
        # fall back: intersection of non-Su QC sets
        genes = None
        for e in load_manifest():
            if e["dataset"] == "su2024":
                continue
            g = set(map(str, sc.read_h5ad(e["path"], backed="r").var_names))
            genes = g if genes is None else genes & g
        joint_genes = genes or set()

    su_genes = set(map(str, su.var_names))
    shared = sorted(su_genes & joint_genes)
    su_only = su_genes - joint_genes
    missing = joint_genes - su_genes
    axis = ["Procr", "Kit", "Flt3", "Cd34", "Ly6a", "Hlf", "Mecom", "Hoxa9", "Mpo", "Elane"]

    # Restrict to shared genes for any downstream score / transfer check
    su_shared = su[:, shared].copy() if shared else su[:, []].copy()

    age_tab = (
        su.obs.groupby(["age_group", "age_label", "lineage"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    # Unique-age subset the 10x core lacks (juvenile=1mo→young, adult=3mo)
    n_holdout = {
        "juvenile_young": int((su.obs["age_label"].astype(str) == "juvenile").sum()),
        "adult": int((su.obs["age_group"].astype(str) == "adult").sum()),
        "old_reference_only": int((su.obs["age_group"].astype(str) == "old").sum()),
    }

    summary = {
        "role": (
            "holdout ages only (adult + juvenile); excluded from primary scGen UMAP"
        ),
        "n_cells_axis": int(su.n_obs),
        "n_holdout_ages": n_holdout,
        "genes": {
            "su_total": len(su_genes),
            "joint_ref": len(joint_genes),
            "shared_with_joint": len(shared),
            "su_not_in_joint": len(su_only),
            "joint_missing_in_su": len(missing),
            "axis_markers_in_su": {m: m in su_genes for m in axis},
            "axis_markers_in_shared": {m: m in set(shared) for m in axis},
            "note": (
                "Su-not-in-joint is dominated by Gm/contig/miRNA/ERCC-like names — "
                "Smart-seq2 annotation, not unique HSC biology. "
                "Procr/Kit/Flt3 are missing from Su vs joint."
            ),
        },
        "age_table": age_tab.to_dict(orient="records"),
        "no_scanvi_on_scgen": True,
    }

    JOINT_OUT.mkdir(parents=True, exist_ok=True)
    SU_HOLDOUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Su holdout: {su.n_obs:,} axis cells | "
        f"shared genes with joint={len(shared):,} | "
        f"juvenile={n_holdout['juvenile_young']}, adult={n_holdout['adult']}, "
        f"old(ref)={n_holdout['old_reference_only']}"
    )
    print(
        f"  markers in shared: "
        f"{ {m: summary['genes']['axis_markers_in_shared'][m] for m in axis} }"
    )
    print(f"  summary → {SU_HOLDOUT_JSON}")
    # keep shared-gene object available for density checks without atlas train
    hold = su_shared[su_shared.obs["age_group"].astype(str).isin(["adult", "young"])].copy()
    hold_path = JOINT_OUT / "su_holdout_adult_juvenile_sharedgenes.h5ad"
    hold.write_h5ad(hold_path)
    print(f"  holdout h5ad (adult+juvenile, shared genes) → {hold_path} ({hold.n_obs:,}×{hold.n_vars:,})")
    return summary


def run_headless(
    *,
    max_epochs: int = 100,
    skip_transfer: bool = False,
    gene_join: str = "inner",
    include_su: bool = False,
) -> Path:
    import scanpy as sc

    warnings.filterwarnings("ignore", category=FutureWarning)
    sc.settings.verbosity = 1
    exclude = () if include_su else ("su2024",)
    print(
        f"Loading age-core axis (HSPC + Myeloid_prog), "
        f"gene_join={gene_join}, exclude={exclude}…"
    )
    joint = load_age_core_axis(gene_join=gene_join, exclude_datasets=exclude)
    elias_counts = assert_elias_gsms(joint)
    print("scGen batch_removal (technical_batch)…")
    corrected, _, hvg = run_scgen_batch_removal(joint, max_epochs=max_epochs)
    corrected = run_umap(corrected, use_rep="corrected_latent")
    plot_integration_umaps(corrected)
    print("DPT pseudotime (GPU neighbors/diffmap → scanpy dpt)…")
    corrected = run_pseudotime(corrected, use_rep="corrected_latent")
    plot_age_bin_umap(corrected)
    plot_marker_streams(corrected)
    corrected.uns["elias_gsm_counts"] = elias_counts
    corrected.uns["gene_join"] = gene_join
    corrected.uns["exclude_datasets"] = list(exclude)
    corrected.write_h5ad(JOINT_H5AD)
    print(f"wrote {JOINT_H5AD} ({corrected.n_obs:,} × {corrected.n_vars:,})")

    if not skip_transfer:
        print("scGen young↔old transfer (HSPC)…")
        pred, _delta, _m = run_young_old_transfer(
            hvg, celltype="HSPC", max_epochs=max_epochs
        )
        pred.write_h5ad(TRANSFER_H5AD)
        print(f"wrote {TRANSFER_H5AD}")
    return JOINT_H5AD


def run_pseudotime_on_saved(*, h5ad: Path = JOINT_H5AD) -> Path:
    """Add DPT + age_bin/marker plots to an existing scGen joint without retraining."""
    import scanpy as sc

    if not h5ad.exists():
        raise FileNotFoundError(f"Missing {h5ad}; run explore.py first")
    print(f"Loading {h5ad}…")
    corrected = sc.read_h5ad(h5ad)
    if "age_bin" not in corrected.obs and "age_months" in corrected.obs:
        corrected.obs["age_bin"] = corrected.obs["age_months"].map(
            _age_bin_from_months
        ).astype(str)
    corrected = run_pseudotime(corrected, use_rep="corrected_latent")
    plot_age_bin_umap(corrected)
    plot_marker_streams(corrected)
    corrected.write_h5ad(h5ad)
    print(f"updated {h5ad} with dpt_pseudotime")
    return h5ad


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument(
        "--skip-transfer",
        action="store_true",
        help="Only batch_removal + UMAP (skip young↔old scGen)",
    )
    parser.add_argument(
        "--gene-join",
        choices=("inner", "outer"),
        default="inner",
        help="Gene concat mode (inner=shared genes; outer fills missing with 0)",
    )
    parser.add_argument(
        "--include-su",
        action="store_true",
        help="Include Su2024 Smart-seq2 in joint (default: exclude; does not mix with 10x)",
    )
    parser.add_argument(
        "--su-holdout",
        action="store_true",
        help=(
            "Summarize Su2024 adult/juvenile holdout on genes shared with the "
            "10x joint (no UMAP retrain; no scANVI-on-scGen)"
        ),
    )
    parser.add_argument(
        "--pseudotime-only",
        action="store_true",
        help=(
            "GPU neighbors/diffmap + DPT on existing age_core_scgen.h5ad; "
            "write age_bin UMAP + marker branch streams (no scGen retrain)"
        ),
    )
    args = parser.parse_args()
    if args.su_holdout:
        su_holdout_age_summary()
    elif args.pseudotime_only:
        run_pseudotime_on_saved()
    else:
        run_headless(
            max_epochs=args.max_epochs,
            skip_transfer=args.skip_transfer,
            gene_join=args.gene_join,
            include_su=args.include_su,
        )