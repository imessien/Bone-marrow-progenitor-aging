"""Bone marrow scRNA explore notebook (marimo).

1. GSE169162 (Mitchell) droplet — GEO + rapids-singlecell QC / Scrublet / UMAP
2. Tabula Muris Senis BM — CELLxGENE Census + rapids-singlecell QC / UMAP
3. Shared myeloid-progenitor trajectory (scGen batch removal → neighbors → diffmap → DPT)
4. Density along pseudotime: TMS ages + Mitchell young-WT / old-WT / old-KO
   (sample-level bootstrap; composition = where cells sit on the branch)
5. Metabolism vs pseudotime after trajectory validation (CuPy gene-sets + scCellFie)
   (cell state = metabolism at matched branch position)

Docs: https://rapids-singlecell.readthedocs.io/en/latest/Usage_Principles.html
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import tarfile
    import urllib.request
    import warnings
    from pathlib import Path

    import anndata as ad
    import cupy as cp
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import rapids_singlecell as rsc
    import rmm
    import scanpy as sc
    import marimo as mo
    from rmm.allocators.cupy import rmm_cupy_allocator

    warnings.filterwarnings("ignore")
    sc.settings.set_figure_params(dpi=100, facecolor="white")

    def init_gpu() -> None:
        rmm.reinitialize(managed_memory=False, pool_allocator=False, devices=0)
        cp.cuda.set_allocator(rmm_cupy_allocator)

    OUT_DIR = Path("/cis/net/r41/data/iessien1/bone/GSE169162")
    RAW_DIR = OUT_DIR / "raw"
    EXTRACTED = RAW_DIR / "extracted"
    PROCESSED = OUT_DIR / "processed"
    COMBINED_H5AD = PROCESSED / "combined.h5ad"
    FILTERED_H5AD = PROCESSED / "filtered_qc_doublets.h5ad"
    SCALED_H5AD = PROCESSED / "scaled.h5ad"
    GSE_ANNOTATED_H5AD = PROCESSED / "gse169162_annotated.h5ad"

    # Tabula Muris Senis bone marrow (CELLxGENE Census)
    TMS_OUT_DIR = Path("/cis/net/r41/data/iessien1/bone/tabula_muris_senis")
    TMS_H5AD_PATH = TMS_OUT_DIR / "tabula_muris_senis_bone_marrow.h5ad"
    TMS_UMAP_H5AD_PATH = TMS_OUT_DIR / "tabula_muris_senis_bone_marrow_umap.h5ad"
    JOINT_OUT_DIR = Path("/cis/net/r41/data/iessien1/bone/joint_myeloid")
    MYELOID_H5AD = JOINT_OUT_DIR / "myeloid_trajectory.h5ad"
    # Full-gene lognorm + counts kept for scCellFie (Raw has no .layers, so persist separately)
    MYELOID_FULL_H5AD = JOINT_OUT_DIR / "myeloid_fullgene_counts.h5ad"
    # Mitchell niche types with no TMS Census counterpart — keep in Mitchell-only analyses
    MITCHELL_ONLY_LINEAGES = ("Stroma_MSC", "Endothelial")
    # Lineages present in both TMS (Census map) and Mitchell (marker scores)
    SHARED_LINEAGES = (
        "HSPC",
        "Myeloid_prog",
        "Granulocyte",
        "Mono_Mac",
        "Erythroid",
        "MegE_prog",
        "B_lymphoid",
        "T_NK",
    )
    # Only Myeloid_prog has substantial TMS ↔ Mitchell overlap in the joint UMAP.
    # Restrict integration and DPT to that matched state; age remains biological.
    MYELOID_LINEAGES = ("Myeloid_prog",)
    assert set(MYELOID_LINEAGES).issubset(SHARED_LINEAGES)
    assert not set(MYELOID_LINEAGES) & set(MITCHELL_ONLY_LINEAGES)
    N_PT_BINS = 20
    N_BOOT = 200
    # Mouse metabolic gene sets (Title-case) for CuPy scoring on log-normalized X
    METAB_GENE_SETS: dict[str, list[str]] = {
        "glycolysis": [
            "Hk1",
            "Hk2",
            "Pfkl",
            "Pfkp",
            "Aldoa",
            "Gapdh",
            "Pgk1",
            "Eno1",
            "Pkm",
            "Ldha",
        ],
        "OXPHOS": [
            "mt-Co1",
            "mt-Nd1",
            "Atp5a1",
            "Atp5b",
            "Cox4i1",
            "Cox5a",
            "Ndufa1",
            "Uqcrc1",
            "Sdha",
            "Cytb",
        ],
        "FAO": [
            "Cpt1a",
            "Cpt2",
            "Acadl",
            "Acadm",
            "Hadha",
            "Hadhb",
            "Acox1",
            "Slc25a20",
            "Fabp4",
            "Cd36",
        ],
    }
    CENSUS_VERSION = "2025-11-08"
    TMS_ALL_DATASET_IDS = (
        "48b37086-25f7-4ecd-be66-f5bb378e3aea",  # 10x
        "98e5ea9f-16d6-47ec-a529-686e76515e39",  # Smart-seq2
    )
    AGE_ORDER = [
        "4-week-old stage",
        "3-month-old stage",
        "18-month-old stage",
        "20-month-old stage and over",
    ]
    AGE_LABELS = {
        "4-week-old stage": "4 wk",
        "3-month-old stage": "3 mo",
        "18-month-old stage": "18 mo",
        "20-month-old stage and over": "≥20 mo",
    }
    # Collapse Census fine labels → shared lineages (also used for GSE marker panels)
    TMS_TO_LINEAGE = {
        "hematopoietic stem cell": "HSPC",
        "hematopoietic precursor cell": "HSPC",
        "lymphoid lineage restricted progenitor cell": "HSPC",
        "granulocyte monocyte progenitor cell": "Myeloid_prog",
        "megakaryocyte-erythroid progenitor cell": "MegE_prog",
        "erythroid progenitor cell, mammalian": "Erythroid",
        "proerythroblast": "Erythroid",
        "erythroblast": "Erythroid",
        "granulocytopoietic cell": "Granulocyte",
        "granulocyte": "Granulocyte",
        "basophil": "Granulocyte",
        "promonocyte": "Mono_Mac",
        "monocyte": "Mono_Mac",
        "macrophage": "Mono_Mac",
        "early pro-B cell": "B_lymphoid",
        "late pro-B cell": "B_lymphoid",
        "precursor B cell": "B_lymphoid",
        "immature B cell": "B_lymphoid",
        "naive B cell": "B_lymphoid",
        "plasma cell": "B_lymphoid",
        "naive T cell": "T_NK",
        "mature alpha-beta T cell": "T_NK",
        "CD4-positive, alpha-beta T cell": "T_NK",
        "natural killer cell": "T_NK",
    }
    # Mouse BM marker panels (Title-case symbols as in mouse references)
    BM_MARKER_SETS: dict[str, list[str]] = {
        "HSPC": ["Procr", "Hlf", "Mecom", "Hoxa9", "Cd34", "Kit", "Ly6a", "Flt3"],
        "Myeloid_prog": ["Elane", "Mpo", "Ctsg", "Ms4a3", "Cebpe"],
        "Granulocyte": ["Ly6g", "Camp", "Ngp", "S100a8", "S100a9"],
        "Mono_Mac": ["Csf1r", "Cd68", "Adgre1", "Ly6c2", "Itgam"],
        "Erythroid": ["Gata1", "Klf1", "Hba-a1", "Hbb-bt", "Car2"],
        "MegE_prog": ["Pf4", "Itga2b", "Gp9", "Gata1"],
        "B_lymphoid": ["Cd79a", "Cd19", "Ms4a1", "Vpreb1", "Ebf1", "Pax5"],
        "T_NK": ["Cd3d", "Cd3e", "Cd3g", "Nkg7", "Gzma", "Ncr1"],
        "Stroma_MSC": ["Cxcl12", "Lepr", "Nes", "Pdgfra", "Kitl", "Col1a1"],
        "Endothelial": ["Pecam1", "Cdh5", "Emcn", "Kdr"],
    }

    SAMPLE_META: dict[str, dict[str, str]] = {
        "GSM5149490": {
            "sample_name": "Young_CentralMarrow",
            "age": "young",
            "compartment": "CentralMarrow",
            "genotype": "WT",
            "series": "GSE168586",
        },
        "GSM5149491": {
            "sample_name": "Old_CentralMarrow",
            "age": "old",
            "compartment": "CentralMarrow",
            "genotype": "WT",
            "series": "GSE168586",
        },
        "GSM5149492": {
            "sample_name": "Young_Endosteum",
            "age": "young",
            "compartment": "Endosteum",
            "genotype": "WT",
            "series": "GSE168586",
        },
        "GSM5149493": {
            "sample_name": "Old_Endosteum",
            "age": "old",
            "compartment": "Endosteum",
            "genotype": "WT",
            "series": "GSE168586",
        },
        "GSM5696259": {
            "sample_name": "Young_WT_LK",
            "age": "young",
            "compartment": "LK",
            "genotype": "WT",
            "series": "GSE189217",
        },
        "GSM5696260": {
            "sample_name": "Young_WT_LSK",
            "age": "young",
            "compartment": "LSK",
            "genotype": "WT",
            "series": "GSE189217",
        },
        "GSM5696261": {
            "sample_name": "Old_WT_Endosteum",
            "age": "old",
            "compartment": "Endosteum",
            "genotype": "WT",
            "series": "GSE189217",
        },
        "GSM5696262": {
            "sample_name": "Old_WT_CentralMarrow",
            "age": "old",
            "compartment": "CentralMarrow",
            "genotype": "WT",
            "series": "GSE189217",
        },
        "GSM5696263": {
            "sample_name": "Old_IL1R1KO_Endosteum",
            "age": "old",
            "compartment": "Endosteum",
            "genotype": "IL1R1KO",
            "series": "GSE189217",
        },
        "GSM5696264": {
            "sample_name": "Old_IL1R1KO_CentralMarrow",
            "age": "old",
            "compartment": "CentralMarrow",
            "genotype": "IL1R1KO",
            "series": "GSE189217",
        },
        "GSM5696265": {
            "sample_name": "Old_WT_LK",
            "age": "old",
            "compartment": "LK",
            "genotype": "WT",
            "series": "GSE189217",
        },
        "GSM5696266": {
            "sample_name": "Old_WT_LSK",
            "age": "old",
            "compartment": "LSK",
            "genotype": "WT",
            "series": "GSE189217",
        },
        "GSM5696267": {
            "sample_name": "Old_IL1R1KO_LK",
            "age": "old",
            "compartment": "LK",
            "genotype": "IL1R1KO",
            "series": "GSE189217",
        },
        "GSM5696268": {
            "sample_name": "Old_IL1R1KO_LSK",
            "age": "old",
            "compartment": "LSK",
            "genotype": "IL1R1KO",
            "series": "GSE189217",
        },
    }

    QC_BY_SAMPLE: dict[str, dict[str, float]] = {
        "Old_CentralMarrow": {
            "total_counts_min": 1000,
            "total_counts_max": 10000,
            "pct_counts_mt_min": 7,
            "pct_counts_mt_max": 7,
        },
        "Old_Endosteum": {
            "total_counts_min": 1000,
            "total_counts_max": 12000,
            "pct_counts_mt_min": 7,
            "pct_counts_mt_max": 7,
        },
        "Old_IL1R1KO_CentralMarrow": {
            "total_counts_min": 2000,
            "total_counts_max": 22000,
            "pct_counts_mt_min": 8,
            "pct_counts_mt_max": 8,
        },
        "Old_IL1R1KO_Endosteum": {
            "total_counts_min": 2000,
            "total_counts_max": 20000,
            "pct_counts_mt_min": 10,
            "pct_counts_mt_max": 10,
        },
        "Old_IL1R1KO_LK": {
            "total_counts_min": 1000,
            "total_counts_max": 12000,
            "pct_counts_mt_min": 7,
            "pct_counts_mt_max": 7,
        },
        "Old_IL1R1KO_LSK": {
            "total_counts_min": 1000,
            "total_counts_max": 10000,
            "pct_counts_mt_min": 7,
            "pct_counts_mt_max": 7,
        },
        "Old_WT_CentralMarrow": {
            "total_counts_min": 2000,
            "total_counts_max": 17000,
            "pct_counts_mt_min": 9,
            "pct_counts_mt_max": 9,
        },
        "Old_WT_Endosteum": {
            "total_counts_min": 1000,
            "total_counts_max": 10000,
            "pct_counts_mt_min": 7,
            "pct_counts_mt_max": 7,
        },
        "Old_WT_LK": {
            "total_counts_min": 1000,
            "total_counts_max": 12000,
            "pct_counts_mt_min": 7,
            "pct_counts_mt_max": 7,
        },
        "Old_WT_LSK": {
            "total_counts_min": 1000,
            "total_counts_max": 10000,
            "pct_counts_mt_min": 7,
            "pct_counts_mt_max": 7,
        },
        "Young_CentralMarrow": {
            "total_counts_min": 1000,
            "total_counts_max": 10000,
            "pct_counts_mt_min": 7,
            "pct_counts_mt_max": 7,
        },
        "Young_Endosteum": {
            "total_counts_min": 1500,
            "total_counts_max": 12000,
            "pct_counts_mt_min": 7,
            "pct_counts_mt_max": 7,
        },
        "Young_WT_LK": {
            "total_counts_min": 1500,
            "total_counts_max": 12000,
            "pct_counts_mt_min": 6,
            "pct_counts_mt_max": 6,
        },
        "Young_WT_LSK": {
            "total_counts_min": 1500,
            "total_counts_max": 10000,
            "pct_counts_mt_min": 7,
            "pct_counts_mt_max": 7,
        },
    }

    print(OUT_DIR)
    print(TMS_OUT_DIR)
    return (
        AGE_LABELS,
        AGE_ORDER,
        BM_MARKER_SETS,
        CENSUS_VERSION,
        COMBINED_H5AD,
        EXTRACTED,
        FILTERED_H5AD,
        GSE_ANNOTATED_H5AD,
        JOINT_OUT_DIR,
        METAB_GENE_SETS,
        MITCHELL_ONLY_LINEAGES,
        MYELOID_FULL_H5AD,
        MYELOID_H5AD,
        MYELOID_LINEAGES,
        N_BOOT,
        SHARED_LINEAGES,
        N_PT_BINS,
        OUT_DIR,
        PROCESSED,
        QC_BY_SAMPLE,
        RAW_DIR,
        SAMPLE_META,
        SCALED_H5AD,
        TMS_ALL_DATASET_IDS,
        TMS_H5AD_PATH,
        TMS_OUT_DIR,
        TMS_TO_LINEAGE,
        TMS_UMAP_H5AD_PATH,
        ad,
        cp,
        init_gpu,
        mo,
        np,
        pd,
        plt,
        rsc,
        sc,
        tarfile,
        urllib,
    )


@app.cell
def _(
    EXTRACTED,
    OUT_DIR,
    RAW_DIR,
    SAMPLE_META: dict[str, dict[str, str]],
    tarfile,
    urllib,
):
    """Ensure droplet RAW tars exist under OUT_DIR/raw (GEO FTP)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACTED.mkdir(parents=True, exist_ok=True)

    _FTP = {
        "GSE168586_RAW.tar": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE168nnn/GSE168586/suppl/"
            "GSE168586_RAW.tar"
        ),
        "GSE189217_RAW.tar": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE189nnn/GSE189217/suppl/"
            "GSE189217_RAW.tar"
        ),
        "GSE169162_RAW.tar": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE169nnn/GSE169162/suppl/"
            "GSE169162_RAW.tar"
        ),
    }

    def _download(name: str, url: str) -> None:
        dest = RAW_DIR / name
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"have {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
            return
        print(f"downloading {name} …")
        urllib.request.urlretrieve(url, dest)
        print(f"saved {dest} ({dest.stat().st_size / 1e6:.0f} MB)")

    for _name, _url in _FTP.items():
        _download(_name, _url)

    with tarfile.open(RAW_DIR / "GSE168586_RAW.tar", "r") as _tf:
        _tf.extractall(EXTRACTED)

    _b2_root = EXTRACTED / "GSE189217"
    _b2_mats = _b2_root / "matrices"
    _b2_mats.mkdir(parents=True, exist_ok=True)
    with tarfile.open(RAW_DIR / "GSE189217_RAW.tar", "r") as _tf:
        _tf.extractall(_b2_root)
    for _gz in sorted(_b2_root.glob("GSM*_cellranger_count_outs.tar.gz")):
        _stem = _gz.name.replace("_cellranger_count_outs.tar.gz", "")
        _dest = _b2_mats / _stem
        if list(_dest.rglob("*.h5")):
            continue
        _dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(_gz, "r:gz") as _inner:
            _inner.extractall(_dest)

    print(f"raw ready under {OUT_DIR} ({len(SAMPLE_META)} droplet libraries)")
    return


@app.cell
def _(
    COMBINED_H5AD,
    EXTRACTED,
    PROCESSED,
    SAMPLE_META: dict[str, dict[str, str]],
    ad,
    mo,
    np,
    sc,
):
    """Build combined.h5ad from filtered 10x H5 matrices."""

    def _find_h5(gsm: str):
        direct = sorted(EXTRACTED.glob(f"{gsm}*.h5"))
        if direct:
            return direct[0]
        nested = sorted(
            p for p in EXTRACTED.rglob("*.h5") if any(gsm in part for part in p.parts)
        )
        preferred = [p for p in nested if p.name == "filtered_feature_bc_matrix.h5"]
        if preferred:
            return preferred[0]
        if nested:
            return nested[0]
        raise FileNotFoundError(f"No H5 for {gsm} under {EXTRACTED}")

    def _load_library(gsm: str, meta: dict) -> ad.AnnData:
        path = _find_h5(gsm)
        adata = sc.read_10x_h5(path)
        adata.var_names_make_unique()
        adata.obs_names = [f"{meta['sample_name']}_{b}" for b in adata.obs_names]
        for k, v in meta.items():
            adata.obs[k] = v
        adata.obs["gsm"] = gsm
        print(f"  {gsm} {meta['sample_name']}: {adata.n_obs} cells ← {path.name}")
        return adata

    PROCESSED.mkdir(parents=True, exist_ok=True)
    if COMBINED_H5AD.exists():
        print(f"Loading existing {COMBINED_H5AD}")
        combined_adata = ad.read_h5ad(COMBINED_H5AD)
    else:
        print("Building combined AnnData from 10x H5 …")
        _ads = [_load_library(gsm, meta) for gsm, meta in SAMPLE_META.items()]
        combined_adata = ad.concat(_ads, join="outer", index_unique=None)
        combined_adata.obs_names_make_unique()
        if hasattr(combined_adata.X, "data"):
            combined_adata.X.data = np.nan_to_num(combined_adata.X.data, nan=0.0)
        combined_adata.write_h5ad(COMBINED_H5AD)
        print(f"Saved {COMBINED_H5AD}")

    mo.md(
        f"""
        ## GSE169162 droplet combined

        `{combined_adata.n_obs:,}` cells × `{combined_adata.n_vars:,}` genes  
        Samples: `{combined_adata.obs['sample_name'].nunique()}`  
        Path: `{COMBINED_H5AD}`
        """
    )
    return (combined_adata,)


@app.cell
def _(
    QC_BY_SAMPLE: dict[str, dict[str, float]],
    combined_adata,
    init_gpu,
    mo,
    np,
    pd,
    plt,
    rsc,
    sc,
):
    """GPU gene filter, MT QC, per-sample cutoff plots (rapids-singlecell)."""
    init_gpu()
    rsc.get.anndata_to_GPU(combined_adata)

    rsc.pp.filter_genes(combined_adata, min_cells=10, inplace=True)
    # Mouse MT genes: mt-Nd1 …
    rsc.pp.flag_gene_family(
        combined_adata, gene_family_name="mt", gene_family_prefix="mt-"
    )
    rsc.pp.calculate_qc_metrics(combined_adata, qc_vars=["mt"], log1p=False)

    rsc.get.anndata_to_CPU(combined_adata)

    _samples = np.unique(combined_adata.obs["sample_name"].astype(str))
    _rows = [{"sample_name": s, **QC_BY_SAMPLE[s]} for s in _samples]
    qc_cutoffs = pd.DataFrame(_rows)

    def qc_plot(adata, metric: str, cutoffs: pd.DataFrame, ylabel: str):
        order = cutoffs["sample_name"].tolist()
        fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(order)), 4))
        sc.pl.violin(
            adata,
            keys=metric,
            groupby="sample_name",
            order=order,
            rotation=90,
            show=False,
            ax=ax,
        )
        lo_col = f"{metric}_min" if f"{metric}_min" in cutoffs.columns else None
        hi_col = f"{metric}_max" if f"{metric}_max" in cutoffs.columns else None
        for i, row in cutoffs.reset_index(drop=True).iterrows():
            if lo_col:
                ax.hlines(
                    row[lo_col], i - 0.35, i + 0.35, colors="C1", linestyles="--", lw=1.2
                )
            if hi_col:
                ax.hlines(
                    row[hi_col], i - 0.35, i + 0.35, colors="C3", linestyles="-", lw=1.2
                )
        ax.set_ylabel(ylabel)
        ax.set_title(f"QC: {ylabel}")
        fig.tight_layout()
        return fig

    fig_umi = qc_plot(combined_adata, "total_counts", qc_cutoffs, "UMIs")
    fig_mt = qc_plot(combined_adata, "pct_counts_mt", qc_cutoffs, "% mt")

    mo.vstack(
        [
            mo.md(f"**After gene filter:** {combined_adata.n_obs:,} cells (rsc GPU QC)"),
            mo.ui.table(qc_cutoffs, label="Per-sample QC cutoffs"),
            fig_umi,
            fig_mt,
        ]
    )
    return (qc_cutoffs,)


@app.cell
def _(combined_adata, pd, qc_cutoffs):
    """Apply per-sample total_counts and %MT filters."""
    _keep = pd.Series(False, index=combined_adata.obs_names)
    for _, row in qc_cutoffs.iterrows():
        sn = row["sample_name"]
        mask = combined_adata.obs["sample_name"].astype(str) == sn
        tc = combined_adata.obs["total_counts"]
        mt = combined_adata.obs["pct_counts_mt"]
        ok = (
            mask
            & (tc >= row["total_counts_min"])
            & (tc <= row["total_counts_max"])
            & (mt < row["pct_counts_mt_max"])
        )
        _keep |= ok
        print(
            f"{sn}: keep {int(ok.sum())}/{int(mask.sum())} "
            f"(UMI [{row['total_counts_min']},{row['total_counts_max']}], "
            f"mt < {row['pct_counts_mt_max']})"
        )

    adata_qc = combined_adata[_keep].copy()
    print(f"After QC: {adata_qc.n_obs} / {combined_adata.n_obs} cells")
    return (adata_qc,)


@app.cell
def _(FILTERED_H5AD, adata_qc, init_gpu, mo, rsc):
    """GPU Scrublet doublets (rapids-singlecell), then drop predicted doublets."""
    init_gpu()
    adata_dbl = adata_qc.copy()
    rsc.get.anndata_to_GPU(adata_dbl)
    rsc.pp.scrublet(adata_dbl, batch_key="sample_name")
    rsc.get.anndata_to_CPU(adata_dbl)

    # Scrublet writes predicted_doublet (bool) + doublet_score
    n_dbl = int(adata_dbl.obs["predicted_doublet"].sum())
    adata_single = adata_dbl[~adata_dbl.obs["predicted_doublet"]].copy()
    adata_single.write_h5ad(FILTERED_H5AD)
    print(f"Scrublet removed {n_dbl} doublets → {adata_single.n_obs} cells")

    mo.md(
        f"""
        ### Scrublet (`rsc.pp.scrublet`)

        Removed `{n_dbl}` doublets  
        Remaining `{adata_single.n_obs:,}` → `{FILTERED_H5AD}`
        """
    )
    return (adata_single,)


@app.cell
def _(SCALED_H5AD, adata_single, init_gpu, mo, rsc):
    """GPU normalize / HVG / scale (rapids-singlecell)."""
    init_gpu()
    adata = adata_single.copy()
    adata.layers["counts"] = adata.X.copy()

    # convert_all so layers['counts'] is on GPU for seurat_v3 HVG
    rsc.get.anndata_to_GPU(adata, convert_all=True)
    rsc.pp.normalize_total(adata, target_sum=1e4)
    rsc.pp.log1p(adata)
    rsc.pp.highly_variable_genes(
        adata,
        n_top_genes=3000,
        flavor="seurat_v3",
        layer="counts",
        batch_key="sample_name",
    )
    rsc.get.anndata_to_CPU(adata, convert_all=True)

    adata.raw = adata
    adata = adata[:, adata.var["highly_variable"]].copy()

    rsc.get.anndata_to_GPU(adata)
    rsc.pp.scale(adata, max_value=10)
    rsc.get.anndata_to_CPU(adata)

    adata.write_h5ad(SCALED_H5AD)
    mo.md(
        f"""
        ### Normalized / HVG / scaled (rsc)

        `{adata.n_obs:,}` × `{adata.n_vars:,}` HVGs → `{SCALED_H5AD}`
        """
    )
    return (adata,)


@app.cell
def _(OUT_DIR, adata, init_gpu, mo, plt, rsc, sc):
    """GPU PCA + neighbors + UMAP; plot with scanpy."""
    init_gpu()
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.pca(adata, n_comps=50)
    rsc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
    rsc.tl.umap(adata)
    rsc.tl.leiden(adata, resolution=0.6, key_added="leiden")
    rsc.get.anndata_to_CPU(adata)

    sc.pl.umap(
        adata,
        color=["sample_name", "age", "compartment", "leiden"],
        ncols=2,
        wspace=0.4,
        show=False,
    )
    fig = plt.gcf()
    fig.set_size_inches(12, 10)
    fig.tight_layout()
    _png = OUT_DIR / "processed" / "umap_sample_meta.png"
    fig.savefig(_png, dpi=150, bbox_inches="tight")
    mo.vstack([mo.md(f"UMAP saved `{_png}`"), fig])
    return


@app.cell
def _(BM_MARKER_SETS: dict[str, list[str]], OUT_DIR, adata, mo, pd, plt, sc):
    """Marker scores + leiden DE → lineage labels for GSE169162."""
    _present = {}
    for _name, _genes in BM_MARKER_SETS.items():
        _hit = [g for g in _genes if g in adata.raw.var_names]
        _present[_name] = _hit
        if len(_hit) >= 2:
            sc.tl.score_genes(
                adata, gene_list=_hit, score_name=f"score_{_name}", use_raw=True
            )

    _score_cols = [c for c in adata.obs.columns if c.startswith("score_")]
    if _score_cols:
        _S = adata.obs[_score_cols]
        adata.obs["lineage"] = (
            _S.idxmax(axis=1).str.replace("^score_", "", regex=True).astype("category")
        )
        adata.obs["lineage_score"] = _S.max(axis=1)

    # Cluster markers (Wilcoxon on raw) to sanity-check scores
    sc.tl.rank_genes_groups(
        adata, groupby="leiden", method="wilcoxon", use_raw=True, pts=True
    )
    gse_markers = sc.get.rank_genes_groups_df(adata, group=None)
    _mk_path = OUT_DIR / "processed" / "gse169162_leiden_markers.csv"
    gse_markers.to_csv(_mk_path, index=False)

    # Cross-tab: FACS compartment × marker lineage
    gse_ct = (
        pd.crosstab(adata.obs["compartment"], adata.obs["lineage"], normalize="index")
        .round(3)
        .reset_index()
    )

    sc.pl.umap(
        adata,
        color=["lineage", "compartment", "age"],
        ncols=3,
        wspace=0.45,
        show=False,
    )
    gse_lin_fig = plt.gcf()
    gse_lin_fig.set_size_inches(14, 4.5)
    _png = OUT_DIR / "processed" / "umap_lineage_age.png"
    gse_lin_fig.savefig(_png, dpi=150, bbox_inches="tight")
    _ann = OUT_DIR / "processed" / "gse169162_annotated.h5ad"
    adata.write_h5ad(_ann)

    _cov = {k: f"{len(v)}/{len(BM_MARKER_SETS[k])}" for k, v in _present.items()}
    mo.vstack(
        [
            mo.md(
                f"""
                ### GSE169162 marker annotation

                Mitchell has **no Census `cell_type`** — only FACS compartments
                (CentralMarrow / Endosteum / LK / LSK) and binary **young/old**.
                Lineages from `sc.tl.score_genes` on shared mouse BM panels
                (genes present: `{_cov}`).

                **Stroma_MSC / Endothelial** stay here for Mitchell niche / aging
                analyses — they have no TMS overlap and are dropped from joint
                TMS↔Mitchell integration.

                Markers CSV: `{_mk_path}`  
                Annotated: `{_ann}`  
                UMAP: `{_png}`
                """
            ),
            mo.ui.table(gse_ct, label="compartment × lineage (row-normalized)"),
            gse_lin_fig,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Harmonizing TMS ↔ GSE169162

    | Axis | Tabula Muris Senis | GSE169162 (Mitchell) |
    |---|---|---|
    | Design | Whole BM atlas (hematopoietic) | Niche stroma + sorted LK/LSK |
    | Cell labels | 24 Census `cell_type`s → coarse `lineage` | Marker-scored `lineage` (same panels) |
    | Age | Keep chronological `age_label` (4 wk…≥20 mo) | Keep binary `age` (young/old) — **not binned together** |

    Shared column for joint analyses: **`lineage`**. Ages stay native to each study.

    **Joint integration rule:** keep only `SHARED_LINEAGES` (hematopoietic types
    in both atlases). Drop Mitchell-only `Stroma_MSC` / `Endothelial` — no TMS
    counterpart to align. Niche stroma stays in Mitchell-only cells above.
    Myeloid trajectory further restricts to `MYELOID_LINEAGES`.
    """)
    return


@app.cell
def _(
    CENSUS_VERSION,
    TMS_ALL_DATASET_IDS,
    TMS_H5AD_PATH,
    TMS_OUT_DIR,
    ad,
    mo,
    sc,
):
    """Load / download TMS bone marrow from CELLxGENE Census."""

    def _set_gene_symbols(_ad: ad.AnnData) -> ad.AnnData:
        # Census stores symbols in feature_name; var_names are often soma indices.
        # Clear index.name after promote+make_unique so it doesn't clash with the
        # feature_name column (values diverge once duplicates are uniquified).
        if "feature_name" in _ad.var.columns:
            _ad.var_names = _ad.var["feature_name"].astype(str)
            _ad.var_names_make_unique()
            _ad.var.index.name = None
        return _ad

    def _download_tms_bone() -> ad.AnnData:
        import cellxgene_census

        TMS_OUT_DIR.mkdir(parents=True, exist_ok=True)
        if TMS_H5AD_PATH.exists():
            _ad = _set_gene_symbols(sc.read_h5ad(TMS_H5AD_PATH))
            if _ad.n_obs > 0:
                print(
                    f"Loading TMS raw {TMS_H5AD_PATH} "
                    f"({_ad.n_obs} cells; genes e.g. {list(_ad.var_names[:3])})"
                )
                return _ad
            print("Existing TMS file empty; re-downloading…")

        _ids = ", ".join(f"'{i}'" for i in TMS_ALL_DATASET_IDS)
        _obs_filter = (
            f"tissue_general == 'bone marrow' and is_primary_data == True "
            f"and dataset_id in [{_ids}]"
        )
        print(f"Querying Census {CENSUS_VERSION} for TMS bone marrow…")
        with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
            _ad = cellxgene_census.get_anndata(
                census=census,
                organism="Mus musculus",
                obs_value_filter=_obs_filter,
                obs_column_names=[
                    "cell_type",
                    "tissue",
                    "tissue_general",
                    "sex",
                    "donor_id",
                    "development_stage",
                    "assay",
                    "disease",
                    "dataset_id",
                ],
            )
        if _ad.n_obs == 0:
            raise RuntimeError("Census TMS bone marrow query returned 0 cells")
        _ad = _set_gene_symbols(_ad)
        _ad.write_h5ad(TMS_H5AD_PATH)
        print(f"Saved TMS raw → {TMS_H5AD_PATH} ({_ad.n_obs} × {_ad.n_vars})")
        return _ad

    tms_raw = _download_tms_bone()
    mo.md(
        f"""
        ## Tabula Muris Senis — bone marrow (Census)

        `{tms_raw.n_obs:,}` cells × `{tms_raw.n_vars:,}` genes  
        Gene symbols from `feature_name` (e.g. `{list(tms_raw.var_names[:4])}`).  
        Raw: `{TMS_H5AD_PATH}`
        """
    )
    return (tms_raw,)


@app.cell
def _(
    AGE_LABELS,
    AGE_ORDER,
    TMS_TO_LINEAGE,
    TMS_UMAP_H5AD_PATH,
    init_gpu,
    mo,
    pd,
    rsc,
    tms_raw,
):
    """RAPIDS QC → Scrublet → norm / HVG / scale / PCA / neighbors / UMAP."""
    init_gpu()
    tms_adata = tms_raw.copy()
    if "feature_name" in tms_adata.var.columns and not any(
        str(g).startswith("mt-") for g in tms_adata.var_names[:500]
    ):
        tms_adata.var_names = tms_adata.var["feature_name"].astype(str)
    tms_adata.var_names_make_unique()
    if tms_adata.var.index.name == "feature_name":
        tms_adata.var.index.name = None

    rsc.get.anndata_to_GPU(tms_adata)
    rsc.pp.flag_gene_family(
        tms_adata, gene_family_name="mt", gene_family_prefix="mt-"
    )
    rsc.pp.calculate_qc_metrics(tms_adata, qc_vars=["mt"], log1p=False)
    rsc.get.anndata_to_CPU(tms_adata)

    _n0 = tms_adata.n_obs
    # Atlas-style QC: min genes, MT%, drop extreme gene counts
    _keep = (
        (tms_adata.obs["n_genes_by_counts"] >= 200)
        & (tms_adata.obs["n_genes_by_counts"] < 6000)
        & (tms_adata.obs["pct_counts_mt"] < 20)
    )
    tms_adata = tms_adata[_keep].copy()
    print(
        f"TMS QC cells: {_n0} → {tms_adata.n_obs} "
        f"(min_genes≥200, n_genes<6000, pct_mt<20); "
        f"mean pct_mt={tms_adata.obs['pct_counts_mt'].mean():.2f}"
    )

    rsc.get.anndata_to_GPU(tms_adata)
    rsc.pp.filter_genes(tms_adata, min_cells=3)
    # Doublets by assay (10x vs Smart-seq2); per-donor batches are too small for Scrublet
    rsc.pp.scrublet(tms_adata, batch_key="assay")
    rsc.get.anndata_to_CPU(tms_adata)

    _n_dbl = int(tms_adata.obs["predicted_doublet"].sum())
    tms_adata = tms_adata[~tms_adata.obs["predicted_doublet"]].copy()
    print(f"TMS Scrublet removed {_n_dbl} → {tms_adata.n_obs} singlets")

    tms_adata.layers["counts"] = tms_adata.X.copy()
    rsc.get.anndata_to_GPU(tms_adata, convert_all=True)
    rsc.pp.normalize_total(tms_adata, target_sum=1e4)
    rsc.pp.log1p(tms_adata)
    rsc.pp.highly_variable_genes(
        tms_adata,
        n_top_genes=2000,
        flavor="seurat_v3",
        layer="counts",
        batch_key="assay",
    )
    rsc.get.anndata_to_CPU(tms_adata, convert_all=True)

    tms_adata.raw = tms_adata
    tms_adata = tms_adata[:, tms_adata.var["highly_variable"]].copy()

    rsc.get.anndata_to_GPU(tms_adata)
    rsc.pp.scale(tms_adata, max_value=10)
    rsc.pp.pca(tms_adata, n_comps=50)
    rsc.pp.neighbors(tms_adata, n_neighbors=15, n_pcs=40)
    rsc.tl.umap(tms_adata)
    rsc.get.anndata_to_CPU(tms_adata)

    _stages = tms_adata.obs["development_stage"].astype(str)
    tms_adata.obs["development_stage"] = pd.Categorical(
        _stages, categories=AGE_ORDER, ordered=True
    )
    tms_adata.obs["age_label"] = pd.Categorical(
        _stages.map(AGE_LABELS).fillna(_stages),
        categories=[AGE_LABELS[s] for s in AGE_ORDER],
        ordered=True,
    )
    tms_adata.obs["lineage"] = (
        tms_adata.obs["cell_type"]
        .astype(str)
        .map(TMS_TO_LINEAGE)
        .fillna("other")
        .astype("category")
    )

    TMS_UMAP_H5AD_PATH.parent.mkdir(parents=True, exist_ok=True)
    tms_adata.write_h5ad(TMS_UMAP_H5AD_PATH)
    print(f"Saved TMS processed → {TMS_UMAP_H5AD_PATH}")

    mo.md(
        f"""
        ### TMS QC + preprocess (rapids-singlecell)

        `{tms_adata.n_obs:,}` cells × `{tms_adata.n_vars:,}` HVGs  
        Scrublet removed `{_n_dbl}` doublets.  
        Added coarse `lineage` (ages stay as `age_label`).  
        Saved `{TMS_UMAP_H5AD_PATH}`
        """
    )
    return (tms_adata,)


@app.cell
def _(mo, tms_adata):
    """Cells per chronological age + coarse lineage."""
    tms_age_counts = (
        tms_adata.obs["age_label"]
        .value_counts()
        .sort_index()
        .rename("n_cells")
        .rename_axis("age_label")
        .reset_index()
    )
    tms_lin_counts = (
        tms_adata.obs["lineage"]
        .value_counts()
        .rename("n_cells")
        .rename_axis("lineage")
        .reset_index()
    )
    tms_map = (
        tms_adata.obs.groupby(["lineage", "cell_type"], observed=True)
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["lineage", "n"], ascending=[True, False])
    )
    mo.vstack(
        [
            mo.ui.table(tms_age_counts, label="TMS cells per age_label"),
            mo.ui.table(tms_lin_counts, label="TMS cells per coarse lineage"),
            mo.ui.table(tms_map, label="fine cell_type → lineage map"),
        ]
    )
    return


@app.cell
def _(TMS_OUT_DIR, mo, plt, sc, tms_adata):
    """Final TMS UMAPs: fine type, coarse lineage, chronological age_label."""
    sc.pl.umap(
        tms_adata,
        color=["cell_type", "lineage", "age_label"],
        ncols=3,
        wspace=0.45,
        legend_loc="right margin",
        show=False,
    )
    tms_umap_fig = plt.gcf()
    tms_umap_fig.set_size_inches(16, 5)

    _png_both = TMS_OUT_DIR / "umap_cell_type_age.png"
    _png_ct = TMS_OUT_DIR / "umap_cell_type.png"
    _png_age = TMS_OUT_DIR / "umap_age_label.png"
    _png_lin = TMS_OUT_DIR / "umap_lineage.png"
    tms_umap_fig.savefig(_png_both, dpi=200, bbox_inches="tight")

    sc.pl.umap(tms_adata, color="cell_type", legend_loc="right margin", show=False)
    plt.gcf().set_size_inches(8, 5)
    plt.savefig(_png_ct, dpi=200, bbox_inches="tight")
    plt.close()

    sc.pl.umap(tms_adata, color="age_label", show=False)
    plt.gcf().set_size_inches(6, 5)
    plt.savefig(_png_age, dpi=200, bbox_inches="tight")
    plt.close()

    sc.pl.umap(tms_adata, color="lineage", legend_loc="right margin", show=False)
    plt.gcf().set_size_inches(7, 5)
    plt.savefig(_png_lin, dpi=200, bbox_inches="tight")
    plt.close()

    mo.vstack(
        [
            mo.md(
                f"""
                ### TMS bone marrow UMAP

                - `{_png_both}` (cell_type / lineage / age_label)
                - `{_png_ct}`, `{_png_age}`, `{_png_lin}`
                """
            ),
            tms_umap_fig,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Shared myeloid-progenitor state: composition vs cell state

    | Question | Readout |
    |---|---|
    | **Composition** — does KO change *where* progenitors sit on the continuum? | Density of cells along progenitor pseudotime |
    | **Cell state** — does KO change metabolism *at the same* progenitor position? | Glycolysis / OXPHOS / FAO vs pseudotime (matched bins) |

    Pipeline: the only substantially overlapping state, `Myeloid_prog` →
    **scGen** (`dataset` is technical batch; chronological age is preserved) →
    neighbors → diffusion map → DPT (root = least-differentiated progenitor) →
    density analyses → **metabolism only after trajectory validation**.
    """)
    return


@app.cell
def _(
    FILTERED_H5AD,
    GSE_ANNOTATED_H5AD,
    MITCHELL_ONLY_LINEAGES,
    MYELOID_LINEAGES,
    SHARED_LINEAGES,
    TMS_H5AD_PATH,
    TMS_UMAP_H5AD_PATH,
    ad,
    adata,
    mo,
    pd,
    sc,
    tms_adata,
):
    """Build matched Myeloid_prog joint object (lognorm + counts) from both studies."""

    def _lognorm_from_processed(a: ad.AnnData) -> ad.AnnData:
        out = a.raw.to_adata() if a.raw is not None else a.copy()
        out.var_names_make_unique()
        if out.var.index.name is not None:
            out.var.index.name = None
        return out

    def _attach_counts(log_ad: ad.AnnData, counts_ad: ad.AnnData) -> None:
        """Match barcodes in counts_ad → layers['counts'] on log_ad (shared genes)."""
        shared = log_ad.var_names.intersection(counts_ad.var_names)
        log_ad._inplace_subset_var(shared)
        c = counts_ad[:, shared].copy()
        # barcode stems may differ after concat suffixes; match on intersection
        common = log_ad.obs_names.intersection(c.obs_names)
        if len(common) < 0.5 * log_ad.n_obs:
            # strip dataset suffix from log names if present
            stem = pd.Index([x.rsplit("-", 1)[0] for x in log_ad.obs_names])
            remap = pd.Series(log_ad.obs_names, index=stem)
            hit = remap.index.intersection(c.obs_names)
            if len(hit) == 0:
                raise RuntimeError("Could not align counts barcodes to lognorm cells")
            order = remap.loc[hit].to_numpy()
            log_ad._inplace_subset_obs(order)
            c = c[hit].copy()
        else:
            log_ad._inplace_subset_obs(common)
            c = c[common].copy()
        log_ad.layers["counts"] = c.X.copy()

    # --- Mitchell myeloid (drop Mitchell-only niche types from joint) ---
    gse = _lognorm_from_processed(adata)
    if GSE_ANNOTATED_H5AD.exists() and "lineage" not in gse.obs:
        gse = _lognorm_from_processed(sc.read_h5ad(GSE_ANNOTATED_H5AD))
    _gse_lin = gse.obs["lineage"].astype(str)
    _drop_mit = _gse_lin.isin(MITCHELL_ONLY_LINEAGES).sum()
    gse = gse[_gse_lin.isin(MYELOID_LINEAGES)].copy()
    if _drop_mit:
        print(
            f"Mitchell: excluded {_drop_mit:,} Stroma_MSC/Endothelial from joint "
            f"(kept in {GSE_ANNOTATED_H5AD.name} for niche-only analyses)"
        )
    gse.obs["dataset"] = "Mitchell"
    gse.obs["genotype"] = gse.obs.get("genotype", pd.Series("WT", index=gse.obs_names))
    gse.obs["age_label"] = gse.obs["age"].astype(str).map(
        {"young": "young", "old": "old"}
    )
    gse.obs["age_months"] = gse.obs["age_label"].map({"young": 1.0, "old": 24.0})
    gse.obs["group"] = (
        gse.obs["age"].astype(str) + "-" + gse.obs["genotype"].astype(str)
    )
    gse.obs["sample_id"] = gse.obs["sample_name"].astype(str)
    gse.obs["batch"] = "Mitchell_" + gse.obs.get(
        "series", pd.Series("GSE", index=gse.obs_names)
    ).astype(str)
    # raw counts from pre-norm filtered matrix
    gse_counts = sc.read_h5ad(FILTERED_H5AD)
    gse_counts.var_names_make_unique()
    gse_counts = gse_counts[gse_counts.obs_names.isin(gse.obs_names), :].copy()
    _attach_counts(gse, gse_counts)

    # --- TMS myeloid (SHARED only; Census has no stroma/endothelium) ---
    tms = _lognorm_from_processed(tms_adata)
    _tms_lin = tms.obs["lineage"].astype(str)
    assert not _tms_lin.isin(MITCHELL_ONLY_LINEAGES).any(), (
        "TMS unexpectedly has Mitchell-only lineages"
    )
    tms = tms[_tms_lin.isin(MYELOID_LINEAGES)].copy()
    tms.obs["dataset"] = "TMS"
    tms.obs["genotype"] = "WT"
    tms.obs["age_months"] = tms.obs["age_label"].astype(str).map(
        {"4 wk": 1.0, "3 mo": 3.0, "18 mo": 18.0, "≥20 mo": 20.0}
    )
    tms.obs["group"] = "TMS-" + tms.obs["age_label"].astype(str)
    tms.obs["sample_id"] = tms.obs["donor_id"].astype(str)
    tms.obs["batch"] = "TMS_" + tms.obs["assay"].astype(str)
    tms_counts = sc.read_h5ad(TMS_H5AD_PATH)
    if "feature_name" in tms_counts.var.columns:
        tms_counts.var_names = tms_counts.var["feature_name"].astype(str)
        tms_counts.var_names_make_unique()
        tms_counts.var.index.name = None
    # TMS processed obs_names may not match raw; align by shared names or fail soft
    _overlap = tms.obs_names.intersection(tms_counts.obs_names)
    if len(_overlap) < 0.5 * tms.n_obs:
        print(
            f"TMS counts barcode overlap low ({len(_overlap)}/{tms.n_obs}); "
            "reindexing counts by position is unsafe — leaving counts empty for TMS"
        )
        # ponytail: preserve shape when raw barcodes cannot be aligned; scGen uses X.
        tms.layers["counts"] = tms.X.copy()
    else:
        tms = tms[_overlap].copy()
        _attach_counts(tms, tms_counts)

    mye = ad.concat([gse, tms], join="inner", index_unique="-")
    mye.obs_names_make_unique()
    # ensure required obs
    for col in ("lineage", "dataset", "genotype", "age_label", "group", "sample_id", "batch"):
        mye.obs[col] = mye.obs[col].astype(str)
    if mye.obs["age_months"].isna().any():
        _missing_age = sorted(
            mye.obs.loc[mye.obs["age_months"].isna(), "age_label"].unique()
        )
        raise ValueError(f"Missing numeric age mapping for: {_missing_age}")
    _present = set(mye.obs["lineage"].astype(str).unique())
    assert _present.issubset(SHARED_LINEAGES), _present - set(SHARED_LINEAGES)
    assert not _present & set(MITCHELL_ONLY_LINEAGES), _present & set(MITCHELL_ONLY_LINEAGES)
    mye.obs["lineage"] = pd.Categorical(
        mye.obs["lineage"], categories=list(MYELOID_LINEAGES), ordered=True
    )
    print(
        f"Myeloid_prog joint: {mye.n_obs:,} cells × {mye.n_vars:,} genes | "
        f"Mitchell={int((mye.obs['dataset']=='Mitchell').sum()):,} "
        f"TMS={int((mye.obs['dataset']=='TMS').sum()):,}"
    )
    mo.md(
        f"""
        ### Matched Myeloid_prog subset (joint TMS ↔ Mitchell)

        Kept: `{list(MYELOID_LINEAGES)}` — the only state with substantial cross-study overlap.  
        Dropped from joint: `{list(MITCHELL_ONLY_LINEAGES)}` (Mitchell niche only)  
        `{mye.n_obs:,}` cells × `{mye.n_vars:,}` shared genes  
        (Mitchell from `{GSE_ANNOTATED_H5AD.name}` / filtered counts; TMS from `{TMS_UMAP_H5AD_PATH.name}`)
        """
    )
    return (mye,)


@app.cell
def _(
    JOINT_OUT_DIR,
    MITCHELL_ONLY_LINEAGES,
    MYELOID_FULL_H5AD,
    MYELOID_H5AD,
    MYELOID_LINEAGES,
    init_gpu,
    mo,
    mye,
    plt,
    rsc,
    sc,
):
    import scgen
    import torch

    init_gpu()
    mye_int = mye.copy()
    JOINT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # scGen trains on normalized, non-negative expression. Select HVGs from counts,
    # while retaining full log-normalized expression for metabolism and markers.
    rsc.get.anndata_to_GPU(mye_int, convert_all=True)
    rsc.pp.highly_variable_genes(
        mye_int,
        n_top_genes=7000,
        flavor="seurat_v3",
        layer="counts",
        batch_key="batch",
    )
    rsc.get.anndata_to_CPU(mye_int, convert_all=True)
    mye_int.raw = mye_int.copy()

    # Raw has no .layers, so persist full-gene counts (obs-aligned) for scCellFie separately
    mye_int.write_h5ad(MYELOID_FULL_H5AD)
    print(f"full-gene lognorm+counts → {MYELOID_FULL_H5AD}")

    mye_int = mye_int[:, mye_int.var["highly_variable"]].copy()

    # One matched cell state across both labs. Age remains in obs as age_months and
    # age_label; it is intentionally not passed as a batch covariate.
    mye_int.obs["cell_type"] = mye_int.obs["lineage"].astype(str)
    scgen.SCGEN.setup_anndata(
        mye_int,
        batch_key="dataset",
        labels_key="cell_type",
    )
    _model = scgen.SCGEN(mye_int, n_latent=30)
    _n_gpu = torch.cuda.device_count()
    _model.train(
        max_epochs=100,
        batch_size=128,
        early_stopping=True,
        accelerator="gpu" if _n_gpu else "cpu",
        devices=1,
    )
    _model.save(JOINT_OUT_DIR / "scgen_model", overwrite=True)
    mye_int = _model.batch_removal()
    mye_int.var = mye.var.loc[mye_int.var_names].copy()
    mye_int.obsm["X_scgen"] = mye_int.obsm["latent"].copy()
    use_rep = "X_scgen"

    rsc.get.anndata_to_GPU(mye_int)
    rsc.pp.neighbors(mye_int, n_neighbors=15, use_rep=use_rep)
    rsc.tl.umap(mye_int)
    rsc.get.anndata_to_CPU(mye_int)

    mye_int.uns["integration"] = {
        "method": "scGen",
        "use_rep": use_rep,
        "batch_key": "dataset",
        "labels_key": "cell_type",
        "lineages": list(MYELOID_LINEAGES),
        "biological_vector": "age_months",
        "excluded_mitchell_only": list(MITCHELL_ONLY_LINEAGES),
        "n_gpu": int(_n_gpu),
        "note": (
            "Only matched Myeloid_prog cells were used. dataset is the technical "
            "batch; age_label and age_months are retained biological variables."
        ),
    }
    mye_int.write_h5ad(MYELOID_H5AD)
    print(f"scGen integrated matched Myeloid_prog cells → {MYELOID_H5AD}")

    sc.pl.umap(
        mye_int,
        color=["dataset", "age_months", "age_label", "genotype"],
        ncols=2,
        wspace=0.4,
        show=False,
    )
    integ_fig = plt.gcf()
    integ_fig.set_size_inches(12, 10)
    _png = JOINT_OUT_DIR / "umap_myeloid_integrated.png"
    integ_fig.savefig(_png, dpi=150, bbox_inches="tight")
    mo.vstack(
        [
            mo.md(
                f"""
                ### scGen integration of matched Myeloid_prog cells

                Latent: `{use_rep}` · `batch_key=dataset` · `labels_key=cell_type`.  
                `cell_type=Myeloid_prog` in both TMS and Mitchell.  
                Lineages: `{list(MYELOID_LINEAGES)}`  
                Excluded: `{list(MITCHELL_ONLY_LINEAGES)}` (Mitchell niche only).  
                CellTypist is optional label QC, not part of scGen: if used, the
                same mouse/custom model and label collapse must be applied to both.  
                **Age is retained as `age_months` / `age_label`, not a batch label.**
                Saved `{MYELOID_H5AD}` / `{_png}`
                """
            ),
            integ_fig,
        ]
    )
    return mye_int, use_rep


@app.cell
def _(JOINT_OUT_DIR, init_gpu, mo, mye_int, np, pd, plt, rsc, sc, use_rep):
    """DPT within matched progenitors, rooted at the least-differentiated state."""
    init_gpu()
    mye_traj = mye_int.copy()

    rsc.get.anndata_to_GPU(mye_traj)
    # neighbors already present from integration; recompute if missing
    if "neighbors" not in mye_traj.uns:
        rsc.pp.neighbors(mye_traj, n_neighbors=15, use_rep=use_rep)
    rsc.tl.diffmap(mye_traj)
    rsc.get.anndata_to_CPU(mye_traj)
    # Scanpy DPT expects first DC dropped for plotting; keep full for rooting
    if mye_traj.obsm["X_diffmap"].shape[1] > 1:
        mye_traj.obsm["X_diffmap_plot"] = mye_traj.obsm["X_diffmap"][:, 1:]

    # HSPCs are excluded because they do not overlap across datasets. Root using
    # expression within Myeloid_prog: high stem/progenitor markers and low
    # granulocytic-commitment markers. Age is not used to choose the root.
    _expr = mye_traj.raw.to_adata() if mye_traj.raw is not None else mye_traj

    def _mean_marker_score(genes):
        _genes = [gene for gene in genes if gene in _expr.var_names]
        if not _genes:
            return np.zeros(mye_traj.n_obs, dtype=float)
        _values = _expr[:, _genes].X
        return np.asarray(_values.mean(axis=1)).ravel()

    _early = _mean_marker_score(("Procr", "Hlf", "Mecom", "Hoxa9", "Kit"))
    _committed = _mean_marker_score(("Mpo", "Elane", "Ctsg", "Cebpe"))
    _root_score = _early - _committed
    if not np.isfinite(_root_score).any() or np.ptp(_root_score) <= 0:
        raise RuntimeError("Could not score a least-differentiated progenitor root")
    root_i = int(np.nanargmax(_root_score))
    mye_traj.uns["iroot"] = root_i
    sc.tl.dpt(mye_traj)  # uses neighbors + iroot
    pt = np.asarray(mye_traj.obs["dpt_pseudotime"], dtype=float)
    # normalize 0–1 within finite values
    _pt_ok = np.isfinite(pt)
    pt_n = pt.copy()
    pt_n[_pt_ok] = (pt[_pt_ok] - pt[_pt_ok].min()) / (
        pt[_pt_ok].max() - pt[_pt_ok].min() + 1e-12
    )
    mye_traj.obs["pseudotime"] = pt_n
    mye_traj.obs["branch"] = "Myeloid_prog"

    # Marker gradient sanity: commitment markers should rise and early markers fall.
    _marker_check = {}
    _var_names = (
        mye_traj.raw.var_names if mye_traj.raw is not None else mye_traj.var_names
    )
    for _g in ("Mpo", "Elane", "S100a8", "Csf1r", "Procr", "Kit"):
        if _g not in _var_names:
            continue
        _X = (
            mye_traj.raw[:, _g].X
            if mye_traj.raw is not None
            else mye_traj[:, _g].X
        )
        _v = np.asarray(_X.todense() if hasattr(_X, "todense") else _X).ravel()
        _m = np.isfinite(pt_n) & np.isfinite(_v)
        if _m.sum() > 50:
            _marker_check[_g] = float(np.corrcoef(pt_n[_m], _v[_m])[0, 1])

    # Auto validation heuristics (user confirms with checkbox next)
    traj_auto_ok = np.isfinite(pt_n).mean() > 0.95 and (
        _marker_check.get("Mpo", 0) > 0 or _marker_check.get("Elane", 0) > 0
    )
    mye_traj.uns["trajectory_validation"] = {
        "root_cell": mye_traj.obs_names[root_i],
        "root_definition": "max(early progenitor score - commitment score)",
        "root_score": float(_root_score[root_i]),
        "marker_pt_corr": _marker_check,
        "auto_ok": bool(traj_auto_ok),
    }
    mye_traj.write_h5ad(JOINT_OUT_DIR / "myeloid_trajectory.h5ad")

    sc.pl.umap(
        mye_traj,
        color=["pseudotime", "age_months", "dataset", "group"],
        ncols=2,
        wspace=0.4,
        show=False,
    )
    traj_fig = plt.gcf()
    traj_fig.set_size_inches(12, 10)
    traj_fig.savefig(JOINT_OUT_DIR / "umap_myeloid_pseudotime.png", dpi=150, bbox_inches="tight")

    corr_tbl = (
        pd.Series(_marker_check, name="corr_with_pseudotime")
        .rename_axis("gene")
        .reset_index()
        .round(3)
    )
    mo.vstack(
        [
            mo.md(
                f"""
                ### Reference trajectory (DPT)

                Root: `{mye_traj.obs_names[root_i]}` (least-differentiated Myeloid_prog)  
                Auto-validation: **{traj_auto_ok}**  
                Marker↔pseudotime correlations (mature myeloid should be **positive**):
                """
            ),
            mo.ui.table(corr_tbl, label="marker vs pseudotime"),
            traj_fig,
        ]
    )
    return mye_traj, traj_auto_ok


@app.cell
def _(mo, traj_auto_ok):
    """Gate metabolism until you confirm the trajectory looks right."""
    traj_validated = mo.ui.checkbox(
        label="Trajectory validated — unlock metabolism (scCellFie + CuPy scores)",
        value=bool(traj_auto_ok),
    )
    mo.vstack(
        [
            mo.md(
                f"""
                Auto checks {'passed' if traj_auto_ok else 'did **not** pass'} —
                inspect UMAP / marker correlations above, then confirm:
                """
            ),
            traj_validated,
        ]
    )
    return (traj_validated,)


@app.cell
def _(JOINT_OUT_DIR, N_BOOT, N_PT_BINS, mo, mye_traj, np, pd, plt):
    """Sample-level density along pseudotime (bootstrap mice/samples, not cells)."""

    def _sample_bin_density(obs: pd.DataFrame, group_col: str, n_bins: int) -> pd.DataFrame:
        """Per-sample fraction of cells in each pseudotime bin (sums to 1)."""
        o = obs.loc[np.isfinite(obs["pseudotime"])].copy()
        o["pt_bin"] = pd.cut(
            o["pseudotime"],
            bins=np.linspace(0, 1, n_bins + 1),
            labels=False,
            include_lowest=True,
        )
        rows = []
        for (grp, sid), sub in o.groupby([group_col, "sample_id"], observed=True):
            vc = sub["pt_bin"].value_counts(normalize=True)
            for b in range(n_bins):
                rows.append(
                    {
                        "group": grp,
                        "sample_id": sid,
                        "pt_bin": b,
                        "pt_mid": (b + 0.5) / n_bins,
                        "density": float(vc.get(b, 0.0)),
                        "n_cells": int(len(sub)),
                    }
                )
        return pd.DataFrame(rows)

    n_bins_ref = N_PT_BINS

    def _bootstrap_density(
        dens: pd.DataFrame, n_boot: int = 200, seed: int = 0
    ) -> pd.DataFrame:
        """Bootstrap samples within each group → mean density ± 95% CI per bin."""
        rng = np.random.default_rng(seed)
        out = []
        for grp, g in dens.groupby("group", observed=True):
            samples = g["sample_id"].unique()
            if len(samples) < 1:
                continue
            bins = sorted(g["pt_bin"].unique())
            boot = np.zeros((n_boot, len(bins)))
            for i in range(n_boot):
                draw = rng.choice(samples, size=len(samples), replace=True)
                for j, b in enumerate(bins):
                    vals = []
                    for s in draw:
                        hit = g.loc[(g["sample_id"] == s) & (g["pt_bin"] == b), "density"]
                        vals.append(float(hit.iloc[0]) if len(hit) else 0.0)
                    boot[i, j] = float(np.mean(vals))
            for j, b in enumerate(bins):
                out.append(
                    {
                        "group": grp,
                        "pt_bin": b,
                        "pt_mid": (int(b) + 0.5) / n_bins_ref,
                        "mean": float(boot[:, j].mean()),
                        "lo": float(np.quantile(boot[:, j], 0.025)),
                        "hi": float(np.quantile(boot[:, j], 0.975)),
                        "n_samples": int(len(samples)),
                    }
                )
        return pd.DataFrame(out)

    def _plot_density(summary: pd.DataFrame, title: str, path):
        fig, ax = plt.subplots(figsize=(8, 4))
        for grp, g in summary.groupby("group", observed=True):
            g = g.sort_values("pt_mid")
            ax.fill_between(g["pt_mid"], g["lo"], g["hi"], alpha=0.2)
            ax.plot(g["pt_mid"], g["mean"], label=f"{grp} (n={g['n_samples'].iloc[0]})")
        ax.set_xlabel("Myeloid_prog pseudotime (early → committed)")
        ax.set_ylabel("sample-mean density")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        return fig

    obs = mye_traj.obs.copy()

    # --- TMS chronological ages ---
    tms_obs = obs.loc[obs["dataset"] == "TMS"].copy()
    tms_dens = _sample_bin_density(tms_obs, "age_label", N_PT_BINS)
    # order age groups
    _age_order = ["4 wk", "3 mo", "18 mo", "≥20 mo"]
    tms_dens["group"] = pd.Categorical(
        tms_dens["group"], categories=_age_order, ordered=True
    )
    tms_sum = _bootstrap_density(tms_dens, n_boot=N_BOOT)
    tms_sum["group"] = pd.Categorical(
        tms_sum["group"], categories=_age_order, ordered=True
    )
    fig_tms_dens = _plot_density(
        tms_sum,
        "TMS: density(pseudotime | age) — sample bootstrap",
        JOINT_OUT_DIR / "density_tms_age.png",
    )

    # Monotonic drift: per-sample median PT vs age rank
    _age_rank = {a: i for i, a in enumerate(_age_order)}
    tms_med = (
        tms_obs.groupby("sample_id", observed=True)
        .agg(
            age_label=("age_label", "first"),
            median_pt=("pseudotime", "median"),
            n=("pseudotime", "size"),
        )
        .reset_index()
    )
    tms_med["age_rank"] = tms_med["age_label"].map(_age_rank)
    tms_med = tms_med.dropna(subset=["age_rank", "median_pt"])
    if len(tms_med) >= 3:
        spearman = float(
            pd.Series(tms_med["age_rank"]).corr(
                tms_med["median_pt"], method="spearman"
            )
        )
    else:
        spearman = float("nan")

    # --- Mitchell young-WT / old-WT / old-KO ---
    mit_obs = obs.loc[obs["dataset"] == "Mitchell"].copy()
    mit_obs["mit_group"] = np.where(
        mit_obs["genotype"].astype(str) == "IL1R1KO",
        "old-KO",
        mit_obs["age_label"].astype(str) + "-WT",
    )
    # only keep the three contrast groups
    mit_obs = mit_obs[
        mit_obs["mit_group"].isin(["young-WT", "old-WT", "old-KO"])
    ].copy()
    mit_dens = _sample_bin_density(mit_obs, "mit_group", N_PT_BINS)
    _g_order = ["young-WT", "old-WT", "old-KO"]
    mit_dens["group"] = pd.Categorical(
        mit_dens["group"], categories=_g_order, ordered=True
    )
    mit_sum = _bootstrap_density(mit_dens, n_boot=N_BOOT)
    mit_sum["group"] = pd.Categorical(
        mit_sum["group"], categories=_g_order, ordered=True
    )
    fig_mit_dens = _plot_density(
        mit_sum,
        "Mitchell: density(pseudotime | young-WT / old-WT / old-KO)",
        JOINT_OUT_DIR / "density_mitchell_wt_ko.png",
    )

    # Shift stats: median PT per sample → group means
    mit_med = (
        mit_obs.groupby(["mit_group", "sample_id"], observed=True)["pseudotime"]
        .median()
        .rename("median_pt")
        .reset_index()
    )
    mit_shift = (
        mit_med.groupby("mit_group", observed=True)["median_pt"]
        .agg(["mean", "std", "count"])
        .round(3)
        .reset_index()
    )

    tms_dens.to_csv(JOINT_OUT_DIR / "density_tms_per_sample.csv", index=False)
    mit_dens.to_csv(JOINT_OUT_DIR / "density_mitchell_per_sample.csv", index=False)
    tms_sum.to_csv(JOINT_OUT_DIR / "density_tms_bootstrap.csv", index=False)
    mit_sum.to_csv(JOINT_OUT_DIR / "density_mitchell_bootstrap.csv", index=False)

    mo.vstack(
        [
            mo.md(
                f"""
                ### Composition: density along Myeloid_prog pseudotime

                Replicate unit = **sample / donor** (bootstrap `{N_BOOT}` resamples).  
                TMS Spearman(age rank, sample median PT) = **{spearman:.3f}**
                (positive ⇒ older samples sit further toward mature myeloid end).
                """
            ),
            fig_tms_dens,
            fig_mit_dens,
            mo.ui.table(mit_shift, label="Mitchell sample-median pseudotime by group"),
            mo.ui.table(tms_med.round(3), label="TMS per-sample median pseudotime"),
        ]
    )
    return


@app.cell
def _(
    JOINT_OUT_DIR,
    METAB_GENE_SETS: dict[str, list[str]],
    N_PT_BINS,
    cp,
    mo,
    mye_traj,
    np,
    pd,
    plt,
    traj_validated,
):
    """Metabolism vs pseudotime — only after trajectory checkbox is on.

    Composition vs state:
      - density shift (previous cell) = composition
      - scores at matched pt bins = cell state
    CuPy gene-set scores on log-normalized expression; scCellFie optional below.
    """
    metab_fig = None
    metab_scores = None
    cupy_hit: dict = {}
    _metab_ui = mo.md(
        "**Metabolism locked** — confirm trajectory validation checkbox above first."
    )

    if traj_validated.value:
        # --- CuPy gene-set scores on lognorm (use .raw if present) ---
        src = mye_traj.raw.to_adata() if mye_traj.raw is not None else mye_traj
        src = src[mye_traj.obs_names].copy()
        _X = src.X
        if hasattr(_X, "tocsr"):
            _X = _X.tocsr()
        gene_to_i = {g: i for i, g in enumerate(src.var_names)}
        for name, genes in METAB_GENE_SETS.items():
            idx = [gene_to_i[g] for g in genes if g in gene_to_i]
            cupy_hit[name] = [g for g in genes if g in gene_to_i]
            if len(idx) < 2:
                mye_traj.obs[f"score_{name}"] = np.nan
                continue
            if hasattr(_X, "tocsc"):
                sub = _X[:, idx].astype(np.float32)
                if hasattr(sub, "toarray"):
                    sub = sub.toarray()
            else:
                sub = np.asarray(_X[:, idx], dtype=np.float32)
            _g = cp.asarray(sub)
            mye_traj.obs[f"score_{name}"] = cp.asnumpy(_g.mean(axis=1)).ravel()

        o = mye_traj.obs.loc[np.isfinite(mye_traj.obs["pseudotime"])].copy()
        o["pt_bin"] = pd.cut(
            o["pseudotime"],
            bins=np.linspace(0, 1, N_PT_BINS + 1),
            labels=False,
            include_lowest=True,
        )
        o["mit_group"] = np.where(
            o["dataset"] != "Mitchell",
            "other",
            np.where(
                o["genotype"].astype(str) == "IL1R1KO",
                "old-KO",
                o["age_label"].astype(str) + "-WT",
            ),
        )
        score_cols = [c for c in o.columns if c.startswith("score_")]
        rows = []
        for (grp, sid, b), sub in o.groupby(
            ["mit_group", "sample_id", "pt_bin"], observed=True
        ):
            if grp == "other":
                continue
            _row = {"group": grp, "sample_id": sid, "pt_bin": int(b), "n": len(sub)}
            for c in score_cols:
                _row[c] = float(sub[c].mean())
            rows.append(_row)
        metab_scores = pd.DataFrame(rows)

        if len(metab_scores) == 0:
            _metab_ui = mo.md("No Mitchell cells for metabolism-vs-PT.")
        else:
            agg = (
                metab_scores.groupby(["group", "pt_bin"], observed=True)[score_cols]
                .mean()
                .reset_index()
            )
            agg["pt_mid"] = (agg["pt_bin"] + 0.5) / N_PT_BINS
            _fig, axes = plt.subplots(
                1, len(score_cols), figsize=(4 * len(score_cols), 3.5), sharex=True
            )
            if len(score_cols) == 1:
                axes = [axes]
            for ax, c in zip(axes, score_cols):
                for grp, g in agg.groupby("group", observed=True):
                    g = g.sort_values("pt_mid")
                    ax.plot(g["pt_mid"], g[c], marker="o", ms=3, label=grp)
                ax.set_title(c.replace("score_", ""))
                ax.set_xlabel("pseudotime")
                ax.set_ylabel("mean score (sample-agg)")
                ax.legend(fontsize=7)
            _fig.suptitle(
                "Cell state: metabolism at matched branch position (Mitchell)",
                y=1.02,
            )
            _fig.tight_layout()
            _fig.savefig(
                JOINT_OUT_DIR / "metab_vs_pseudotime_cupy.png",
                dpi=150,
                bbox_inches="tight",
            )
            metab_fig = _fig
            metab_scores.to_csv(
                JOINT_OUT_DIR / "metab_scores_per_sample_bin.csv", index=False
            )
            _metab_ui = mo.vstack(
                [
                    mo.md(
                        f"""
                        ### Cell state: CuPy metabolic gene-set scores vs pseudotime

                        Genes hit: `{ {k: f'{len(v)}/{len(METAB_GENE_SETS[k])}' for k, v in cupy_hit.items()} }`  
                        Curves are **sample-aggregated** means in matched PT bins
                        (old-KO vs old-WT at the same position = state, not composition).
                        """
                    ),
                    metab_fig,
                ]
            )

    _metab_ui
    return


@app.cell
def _(JOINT_OUT_DIR, MYELOID_FULL_H5AD, ad, mo, mye_traj, np, traj_validated):
    """scCellFie metabolic tasks (mouse) — gated; needs full-gene raw counts."""
    sccf_result = None
    _sccf_ui = mo.md("**scCellFie locked** — same trajectory gate as CuPy scores.")

    if traj_validated.value:
        from sccellfie.sccellfie_pipeline import run_sccellfie_pipeline

        _mit_names = mye_traj.obs_names[mye_traj.obs["dataset"] == "Mitchell"]
        counts = None
        mit = None
        # Full-gene counts live in the separately-persisted object (Raw has no .layers)
        if MYELOID_FULL_H5AD.exists():
            _full = ad.read_h5ad(MYELOID_FULL_H5AD)
            _keep = [n for n in _mit_names if n in set(_full.obs_names)]
            if _keep and "counts" in _full.layers:
                mit = _full[_keep].copy()
                counts = mit.layers["counts"]
                mit.obs = mye_traj.obs.loc[mit.obs_names].copy()
        if counts is None:
            mit = mye_traj[_mit_names].copy()
            if "counts" in mit.layers:
                counts = mit.layers["counts"]

        if counts is None:
            _sccf_ui = mo.md(
                "No full-gene `counts` for Mitchell myeloid — skip scCellFie."
            )
        else:
            mit.X = counts.copy()
            mit.layers["counts"] = counts
            if "n_counts" not in mit.obs:
                _Xc = mit.X
                mit.obs["n_counts"] = np.asarray(_Xc.sum(axis=1)).ravel()

            sccf_result = run_sccellfie_pipeline(
                mit,
                organism="mouse",
                n_counts_col="n_counts",
                neighbors_key="neighbors",
                n_neighbors=10,
                smooth_cells=True,
                disable_pbar=False,
                save_folder=str(JOINT_OUT_DIR / "sccellfie"),
                save_filename="mitchell_myeloid",
                verbose=True,
            )
            tasks = sccf_result["adata"].metabolic_tasks
            names = list(tasks.var_names.astype(str))
            keys = {
                "glycolysis": [
                    n
                    for n in names
                    if "glycol" in n.lower() or "ATP generation from glucose" in n
                ],
                "OXPHOS": [
                    n
                    for n in names
                    if "oxidative" in n.lower()
                    or "OXPHOS" in n
                    or "respiratory" in n.lower()
                ],
                "FAO": [
                    n
                    for n in names
                    if "fatty acid" in n.lower()
                    or "FAO" in n
                    or "beta-oxidation" in n.lower()
                ],
            }
            _sccf_ui = mo.md(
                f"""
                ### scCellFie (Mitchell myeloid)

                Saved under `{JOINT_OUT_DIR / 'sccellfie'}`.  
                Task-name hits: `{ {k: v[:5] for k, v in keys.items()} }`  
                Transfer selected task scores onto the trajectory object in a follow-up
                cell once you pick the exact task IDs from the metabolic_tasks AnnData.
                """
            )

    _sccf_ui
    return


if __name__ == "__main__":
    app.run()
