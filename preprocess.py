#!/usr/bin/env python3
"""Shared scRNA QC → filter → doublets → normalize for aging-backbone cohorts.

Datasets (counts already deposited — **no Cell Ranger**):
  su2024      Smart-seq2 CD49b± HSC (Zenodo 13378750 / ENA PRJEB55627)
  gse70657    Grover C1/Fluidigm young/old LT-HSC RefSeq **read counts** (GSE70657)
  gse169162   Mitchell 10x niche+HSPC (GEO → combined.h5ad)
  gse310923   White Lin−BM 10x (GEO h5ad `.raw` integer UMIs)
  gse169608   Yang WT BM 1/6/20 mo whole-BM 10x (GSE169608)
  gse147729   Hérault young/old HSPC 10x (GSE147729)
  gse246464   Elias/van den Brink young/old HSC multiome **RNA only** (GSE246464)
  gse59114    Kowalczyk SMART-seq young/old LT/ST/MPP (**log-norm only** — not scGen)

Same QC contract as validate_hspc_bridge._gpu_qc_preprocess so outputs can be
concatenated later (harmonized obs: dataset, technical_batch, age_label, lineage).

Usage:
  source .venv/bin/activate
  python preprocess.py --dataset gse169608 --annotate
  python preprocess.py --dataset age_core --annotate
  python preprocess.py --dataset gse70657 --annotate --force
  python preprocess.py --dataset gse59114 --annotate --force

Cell Ranger: not required. Su/Grover are plate-seq counts; Mitchell GEO ships filtered 10x
H5; White GEO ships Seurat h5ad with UMI counts in `.raw`; Yang/Hérault/Elias ship MTX.
GSE59114 GEO ships log-scaled expression (no integer counts) — reference only.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import warnings
from pathlib import Path

import anndata as ad
import cupy as cp
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
import rmm
import scanpy as sc
import torch
from rmm.allocators.cupy import rmm_cupy_allocator
from scipy import sparse

warnings.filterwarnings("ignore", category=FutureWarning)
sc.settings.verbosity = 1

BONE = Path("/cis/net/r41/data/iessien1/bone")

PATHS = {
    "su2024": {
        "source": BONE / "Su2024_CD49b_HSC" / "processed" / "su2024_scrna_filtered.h5ad",
        "out": BONE / "Su2024_CD49b_HSC" / "processed" / "su2024_qc_preprocessed.h5ad",
    },
    "gse169162": {
        "source": BONE / "GSE169162" / "processed" / "combined.h5ad",
        "ann": BONE / "GSE169162" / "processed" / "gse169162_annotated.h5ad",
        "out": BONE / "GSE169162" / "processed" / "gse169162_qc_preprocessed.h5ad",
    },
    "gse310923": {
        "source": BONE / "GSE310923" / "GSE310923_bone_marrow.h5ad",
        "out": BONE / "GSE310923" / "processed" / "white_linbm_reprocessed.h5ad",
    },
    "gse169608": {
        "raw": BONE / "GSE169608" / "raw" / "extracted",
        "source": BONE / "GSE169608" / "processed" / "combined_counts.h5ad",
        "out": BONE / "GSE169608" / "processed" / "gse169608_qc_preprocessed.h5ad",
    },
    "gse147729": {
        "raw": BONE / "GSE147729" / "raw" / "extracted",
        "source": BONE / "GSE147729" / "processed" / "combined_counts.h5ad",
        "out": BONE / "GSE147729" / "processed" / "gse147729_qc_preprocessed.h5ad",
    },
    "gse246464": {
        "raw": BONE / "GSE246464" / "raw" / "rna",
        "source": BONE / "GSE246464" / "processed" / "combined_counts_rna.h5ad",
        "out": BONE / "GSE246464" / "processed" / "gse246464_qc_preprocessed.h5ad",
    },
    "gse70657": {
        "raw": BONE / "GSE70657" / "raw" / "GSE70657_Grover.A_et.al_RefSeq.Read.Count.txt.gz",
        "source": BONE / "GSE70657" / "processed" / "combined_counts.h5ad",
        "out": BONE / "GSE70657" / "processed" / "gse70657_qc_preprocessed.h5ad",
    },
    "gse59114": {
        "raw": BONE / "GSE59114" / "raw" / "GSE59114_C57BL6_GEO_all.xlsx",
        "source": BONE / "GSE59114" / "processed" / "combined_lognorm.h5ad",
        "out": BONE / "GSE59114" / "processed" / "gse59114_qc_preprocessed.h5ad",
    },
}

AGE_CORE_10X = ["gse169162", "gse310923", "gse169608", "gse147729", "gse246464"]
# Integer plate-seq in the scGen join. Grover/Kowalczyk are young↔old only (no adult
# midpoint) — kept as optional loaders, not age_core.
AGE_CORE_PLATESEQ = ["su2024"]
OPTIONAL_PLATESEQ = ["gse70657", "gse59114"]
ALL_DATASETS = [*AGE_CORE_PLATESEQ, *AGE_CORE_10X, *OPTIONAL_PLATESEQ]

# Shared mouse BM panels (same as explore.py / validate_hspc_bridge)
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

WHITE_HSPC = {"HSC", "HSC|CMP", "MPP"}
WHITE_MYE = {"GMP 1", "GMP 2", "CMP|GMP"}

SU_LINEAGE = {
    "CD49b-": "HSPC",
    "CD49b+": "HSPC",
    "LMPP": "HSPC",
    "GMP": "Myeloid_prog",
}

GSE169608_SAMPLES = {
    "GSM5210632_WT-1m": {"age_months": 1.0, "age_label": "1mo", "age_group": "young"},
    "GSM5210633_WT-6m": {"age_months": 6.0, "age_label": "6mo", "age_group": "adult"},
    "GSM5210634_WT-20m": {"age_months": 20.0, "age_label": "20mo", "age_group": "old"},
}

GSE147729_SAMPLES = {
    "GSM4443875_young_A": {"age_months": 2.0, "age_label": "young", "age_group": "young"},
    "GSM4443876_young_B": {"age_months": 2.0, "age_label": "young", "age_group": "young"},
    "GSM4443877_old_A": {"age_months": 18.0, "age_label": "old", "age_group": "old"},
    "GSM4443878_old_B": {"age_months": 18.0, "age_label": "old", "age_group": "old"},
}

GSE246464_SAMPLES = {
    "GSM7869307_3575_HA-1536_Young_mice_filtered": {
        "age_months": 2.0,
        "age_label": "young",
        "age_group": "young",
        "rep": "1",
    },
    "GSM7869308_3574_HA-1536_Old_mice_filtered": {
        "age_months": 22.5,
        "age_label": "old",
        "age_group": "old",
        "rep": "1",
    },
    "GSM7869309_3647_HA-1570_Young_mice_filtered": {
        "age_months": 2.0,
        "age_label": "young",
        "age_group": "young",
        "rep": "2",
    },
    "GSM7869310_3648_HA-1570_Old_mice_filtered": {
        "age_months": 22.5,
        "age_label": "old",
        "age_group": "old",
        "rep": "2",
    },
}


def _as_csr(X):
    if sparse.issparse(X):
        return X.tocsr()
    return sparse.csr_matrix(X)


def assign_age_bin(a: ad.AnnData, *, months_key: str = "age_months") -> ad.AnnData:
    """Calendar bins from ``age_months`` (author ``age_label`` / ``age_group`` unchanged).

    - early: ≤ 2.5 mo
    - mid:   (2.5, 8] mo  (covers Su ~3 mo, White ~3–4 mo, Yang 6 mo)
    - late:  ≥ 18 mo
    - unassigned: missing months or (8, 18)
    """

    def _bin(m) -> str:
        if m is None or (isinstance(m, float) and np.isnan(m)):
            return "unassigned"
        try:
            x = float(m)
        except (TypeError, ValueError):
            return "unassigned"
        if x <= 2.5:
            return "early"
        if x <= 8.0:
            return "mid"
        if x >= 18.0:
            return "late"
        return "unassigned"

    if months_key not in a.obs:
        a.obs["age_bin"] = "unassigned"
    else:
        a.obs["age_bin"] = a.obs[months_key].map(_bin).astype(str)
    a.uns["age_bin_definition"] = {
        "early": "age_months <= 2.5",
        "mid": "2.5 < age_months <= 8",
        "late": "age_months >= 18",
        "unassigned": "missing or 8 < age_months < 18",
        "note": (
            "age_label/age_group keep author young/adult/old; age_bin is the "
            "calendar continuum for occupancy / mid-age GMP analyses."
        ),
    }
    return a


def _init_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required (rapids-singlecell QC / Scrublet).")
    rmm.reinitialize(managed_memory=False, pool_allocator=False, devices=0)
    cp.cuda.set_allocator(rmm_cupy_allocator)


def _sanitize_var(a: ad.AnnData) -> None:
    a.var.index.name = None
    a.var.drop(columns=["_index"], errors="ignore", inplace=True)
    if a.raw is not None:
        a.raw.var.index.name = None
        a.raw.var.drop(columns=["_index"], errors="ignore", inplace=True)


def _restore_gene_symbols(a: ad.AnnData) -> None:
    if "_index" in a.var:
        a.var_names = a.var["_index"].astype(str).to_numpy()
    elif "feature_name" in a.var:
        a.var_names = a.var["feature_name"].astype(str).to_numpy()
    elif "gene_symbol" in a.var and not str(a.var_names[0]).startswith("ENSMUSG"):
        a.var_names = a.var["gene_symbol"].astype(str).to_numpy()
    a.var_names_make_unique()
    _sanitize_var(a)


def _su_ensembl_to_symbol(a: ad.AnnData) -> None:
    """Map Su Zenodo matrices to mouse gene symbols.

    Upstream dump sometimes stores ``ENSMUSG…\\tSymbol\\tSymbol`` in both
    ``var_names`` and ``gene_symbol`` — strip to the Title-case symbol.
    """

    def _symbol(x: object) -> str:
        s = str(x)
        if "\t" in s:
            parts = [p for p in s.split("\t") if p]
            # Prefer non-Ensembl token
            for p in parts:
                if not p.startswith("ENSMUSG"):
                    return p
            return parts[-1]
        return s

    raw = (
        a.var["gene_symbol"].astype(str)
        if "gene_symbol" in a.var.columns
        else pd.Series(a.var_names.astype(str), index=a.var_names)
    )
    ids = raw.map(lambda s: str(s).split("\t")[0] if "\t" in str(s) else str(s))
    syms = raw.map(_symbol)
    a.var["gene_id"] = ids.to_numpy()
    a.var["gene_symbol"] = syms.to_numpy()
    a.var_names = syms.to_numpy()
    a.var_names_make_unique()
    _sanitize_var(a)
    n_ok = sum(1 for g in ("Kit", "Procr", "Cd34", "Mpo") if g in a.var_names)
    print(f"  Su gene symbols: {n_ok}/4 axis markers present (e.g. Kit/Procr)")


def _stage_geo_mtx_dir(
    raw_dir: Path,
    sample_key: str,
    *,
    gene_glob: str = "genes",
) -> Path:
    """Symlink GEO-prefixed MTX triples into a scanpy-readable 10x folder.

    GEO naming varies: ``{key}_barcodes.tsv.gz`` vs ``{key}.barcodes.tsv.gz``.
    Always stage as barcodes/genes|features/matrix so ``read_10x_mtx`` works.
    """
    staged = raw_dir / "_staged_10x" / sample_key
    staged.mkdir(parents=True, exist_ok=True)

    def _find(candidates: list[str]) -> Path:
        for name in candidates:
            p = raw_dir / name
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Missing MTX component for {sample_key} under {raw_dir}; tried {candidates}"
        )

    src_bc = _find(
        [f"{sample_key}_barcodes.tsv.gz", f"{sample_key}.barcodes.tsv.gz"]
    )
    src_genes = _find(
        [
            f"{sample_key}_{gene_glob}.tsv.gz",
            f"{sample_key}.{gene_glob}.tsv.gz",
            f"{sample_key}_genes.tsv.gz",
            f"{sample_key}.genes.tsv.gz",
            f"{sample_key}_features.tsv.gz",
            f"{sample_key}.features.tsv.gz",
        ]
    )
    src_mtx = _find(
        [f"{sample_key}_matrix.mtx.gz", f"{sample_key}.matrix.mtx.gz"]
    )

    # Old Cell Ranger: 2-col genes.tsv. scanpy ≥1.10 looks for features.tsv.gz
    # (is_legacy only if uncompressed genes.tsv exists) and needs a 3rd column.
    with gzip.open(src_genes, "rt") as fh:
        ncols = len(fh.readline().rstrip("\n").split("\t"))

    mapping = {
        "barcodes.tsv.gz": src_bc,
        "genes.tsv.gz": src_genes,
        "matrix.mtx.gz": src_mtx,
    }
    for dst_name, src in mapping.items():
        dst = staged / dst_name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())

    feat = staged / "features.tsv.gz"
    if feat.exists() or feat.is_symlink():
        feat.unlink()
    if ncols >= 3:
        feat.symlink_to(src_genes.resolve())
    else:
        with gzip.open(src_genes, "rt") as fin, gzip.open(feat, "wt") as fout:
            for line in fin:
                fout.write(line.rstrip("\n") + "\tGene Expression\n")
    return staged


def _concat_samples(parts: list[ad.AnnData]) -> ad.AnnData:
    a = ad.concat(parts, join="outer", merge="same", index_unique=None)
    a.obs_names_make_unique()
    a.var_names_make_unique()
    return a


def _ribo_frac(a: ad.AnnData) -> np.ndarray:
    """Fraction of counts in Rpl/Rps genes (ambient / low-complexity junk)."""
    X = _as_csr(a.X)
    totals = np.asarray(X.sum(axis=1)).ravel().astype(float)
    genes = np.asarray(a.var_names.astype(str))
    ribo = np.fromiter(
        (g.lower().startswith(("rpl", "rps")) for g in genes),
        dtype=bool,
        count=len(genes),
    )
    if not ribo.any():
        return np.zeros(a.n_obs, dtype=float)
    return np.asarray(X[:, ribo].sum(axis=1)).ravel() / np.maximum(totals, 1.0)


def gpu_qc_preprocess(
    a: ad.AnnData,
    qc_batch_key: str,
    scrublet_batch_key: str | None,
    *,
    min_genes: int = 200,
    max_mt: float = 20.0,
    max_ribo: float | None = None,
    min_cells_gene: int = 10,
    gene_mad: float = 5.0,
    run_scrublet: bool = True,
) -> ad.AnnData:
    """GPU gene filter, MAD depth+complexity QC within batch, optional Scrublet, log1p.

    Depth MAD is per technical batch so Smart-seq2 and 10x are not forced onto
    one UMI cap (same contract as validate_hspc_bridge). Complexity MAD on
    ``n_genes_by_counts`` drops empty/stressed droplets that pass depth alone.
    """
    _init_gpu()
    a = a.copy()
    _sanitize_var(a)
    a.X = _as_csr(a.X)

    rsc.get.anndata_to_GPU(a)
    rsc.pp.filter_genes(a, min_cells=min_cells_gene)
    rsc.pp.flag_gene_family(a, gene_family_name="mt", gene_family_prefix="mt-")
    rsc.pp.calculate_qc_metrics(a, qc_vars=["mt"], log1p=False)
    rsc.get.anndata_to_CPU(a)

    a.obs["ribo_frac"] = _ribo_frac(a)

    n_before = a.n_obs
    keep = pd.Series(False, index=a.obs_names)
    for _, idx in a.obs.groupby(qc_batch_key, observed=True).groups.items():
        obs = a.obs.loc[idx]
        log_umi = np.log1p(obs["total_counts"].astype(float))
        median = float(log_umi.median())
        mad = max(float((log_umi - median).abs().median()), 0.25)
        depth_ok = log_umi.between(median - 5 * mad, median + 5 * mad)
        log_genes = np.log1p(obs["n_genes_by_counts"].astype(float))
        g_med = float(log_genes.median())
        g_mad = max(float((log_genes - g_med).abs().median()), 0.15)
        genes_ok = log_genes.between(g_med - gene_mad * g_mad, g_med + gene_mad * g_mad)
        ok = (
            depth_ok
            & genes_ok
            & (obs["n_genes_by_counts"].astype(float) >= min_genes)
            & (obs["pct_counts_mt"].astype(float) < max_mt)
        )
        if max_ribo is not None:
            ok = ok & (obs["ribo_frac"].astype(float) < max_ribo)
        keep.loc[idx] = ok
    a = a[keep].copy()
    ribo_msg = f" + ribo<{max_ribo}" if max_ribo is not None else ""
    print(
        f"  QC filter: {a.n_obs:,} / {n_before:,} cells "
        f"(MAD depth/genes + genes≥{min_genes} + mt<{max_mt}{ribo_msg})"
    )

    n_doublets = 0
    if run_scrublet:
        if scrublet_batch_key is None:
            raise ValueError("scrublet_batch_key required when run_scrublet=True")
        # ponytail: Scrublet on Smart-seq2 plates is noisy when n_cells/plate is tiny;
        # prefer --no-scrublet for Su if predicted_doublet rates look pathological.
        rsc.get.anndata_to_GPU(a)
        rsc.pp.scrublet(a, batch_key=scrublet_batch_key)
        rsc.get.anndata_to_CPU(a)
        n_doublets = int(a.obs["predicted_doublet"].sum())
        a = a[~a.obs["predicted_doublet"]].copy()
        print(f"  Scrublet: removed {n_doublets:,} → {a.n_obs:,} cells")
    else:
        a.obs["predicted_doublet"] = False
        a.obs["doublet_score"] = np.nan
        print("  Scrublet: skipped")

    a.layers["counts"] = a.X.copy()
    rsc.get.anndata_to_GPU(a, convert_all=True)
    rsc.pp.normalize_total(a, target_sum=1e4)
    rsc.pp.log1p(a)
    rsc.get.anndata_to_CPU(a, convert_all=True)
    _sanitize_var(a)
    a.uns["preprocess"] = {
        "qc_batch_key": qc_batch_key,
        "scrublet_batch_key": scrublet_batch_key,
        "run_scrublet": run_scrublet,
        "n_doublets_removed": n_doublets,
        "min_genes": min_genes,
        "max_mt": max_mt,
        "max_ribo": max_ribo,
        "gene_mad": gene_mad,
    }
    return a


def score_markers(a: ad.AnnData, *, overwrite_lineage: bool = False) -> None:
    """score_genes on BM panels; optionally set lineage = argmax score."""
    present: dict[str, list[str]] = {}
    for name, genes in BM_MARKER_SETS.items():
        hit = [g for g in genes if g in a.var_names]
        present[name] = hit
        if len(hit) >= 2:
            sc.tl.score_genes(a, gene_list=hit, score_name=f"score_{name}")
    score_cols = [c for c in a.obs.columns if c.startswith("score_")]
    if score_cols:
        S = a.obs[score_cols]
        pred = S.idxmax(axis=1).str.replace("^score_", "", regex=True)
        a.obs["marker_lineage"] = pred.astype("category")
        a.obs["marker_lineage_score"] = S.max(axis=1)
        if overwrite_lineage or "lineage" not in a.obs:
            a.obs["lineage"] = a.obs["marker_lineage"].astype(str)
    cov = {k: f"{len(v)}/{len(BM_MARKER_SETS[k])}" for k, v in present.items()}
    print(f"  marker gene coverage: {cov}")


def load_su2024_counts() -> ad.AnnData:
    p = PATHS["su2024"]["source"]
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p}. Build from Zenodo first (su2024_scrna_filtered.h5ad)."
        )
    a = sc.read_h5ad(p)
    _su_ensembl_to_symbol(a)
    # Counts are integer-like Smart-seq2 read counts (author-filtered cells).
    a.obs["sample_name"] = a.obs["plate"].astype(str)
    a.obs["dataset"] = "Su2024"
    a.obs["technical_batch"] = "Su2024_Smartseq2"
    a.obs["assay"] = "Smart-seq2"
    a.obs["age_label"] = a.obs["Age"].astype(str).str.lower()
    a.obs["age_group"] = a.obs["age_label"].map(
        {"juvenile": "young", "adult": "adult", "old": "old"}
    )
    a.obs["age_months"] = a.obs["age_label"].map(
        {"juvenile": 1.0, "adult": 3.0, "old": 20.0}
    )
    a.obs["cell_type"] = a.obs["Cell_type"].astype(str)
    a.obs["lineage"] = a.obs["Cell_type"].astype(str).map(SU_LINEAGE).fillna("Other")
    a.obs["genotype"] = "WT"
    return a


def load_mitchell_counts() -> ad.AnnData:
    p = PATHS["gse169162"]["source"]
    a = sc.read_h5ad(p)
    a.var_names_make_unique()
    a.obs["dataset"] = "Mitchell"
    a.obs["technical_batch"] = "Mitchell_10x"
    a.obs["assay"] = "10x"
    if "age" in a.obs:
        a.obs["age_label"] = a.obs["age"].astype(str)
        a.obs["age_group"] = a.obs["age_label"]
        a.obs["age_months"] = a.obs["age_label"].map({"young": 2.0, "old": 24.0})
    if "genotype" not in a.obs:
        a.obs["genotype"] = "WT"
    # Transfer marker lineages from annotated object when present
    ann_path = PATHS["gse169162"]["ann"]
    if ann_path.exists():
        ann = sc.read_h5ad(ann_path, backed="r")
        common = a.obs_names.intersection(ann.obs_names)
        if "lineage" in ann.obs and len(common):
            a.obs["lineage"] = "Other"
            a.obs.loc[common, "lineage"] = ann.obs.loc[common, "lineage"].astype(str).values
        del ann
    elif "lineage" not in a.obs:
        a.obs["lineage"] = "Other"
    return a


def load_white_counts() -> ad.AnnData:
    p = PATHS["gse310923"]["source"]
    w = sc.read_h5ad(p)
    assert w.raw is not None, "expected .raw UMI counts in GSE310923 h5ad"
    raw = w.raw.to_adata()
    raw.obs = w.obs.copy()
    raw.obs_names = w.obs_names
    _restore_gene_symbols(raw)
    raw.obs["sample_name"] = raw.obs["sample_id"].astype(str)
    raw.obs["dataset"] = "White"
    raw.obs["technical_batch"] = "White_10x"
    raw.obs["assay"] = "10x"
    raw.obs["age_label"] = raw.obs["Age"].astype(str).str.lower()
    raw.obs["age_group"] = raw.obs["age_label"]
    raw.obs["age_months"] = raw.obs["age_label"].map({"young": 4.0, "old": 24.0})
    # White2026: paper states 4-mo (young) vs 24-mo (old); age_bin mid/late from months
    raw.obs["genotype"] = "WT"
    ct = raw.obs["cell_type"].astype(str)
    raw.obs["lineage"] = "Other"
    raw.obs.loc[ct.isin(WHITE_HSPC), "lineage"] = "HSPC"
    raw.obs.loc[ct.isin(WHITE_MYE), "lineage"] = "Myeloid_prog"
    return raw


def build_gse169608_counts(*, force: bool = False) -> Path:
    cfg = PATHS["gse169608"]
    out: Path = cfg["source"]
    if out.exists() and not force:
        return out
    raw: Path = cfg["raw"]
    parts: list[ad.AnnData] = []
    for key, meta in GSE169608_SAMPLES.items():
        staged = _stage_geo_mtx_dir(raw, key, gene_glob="genes")
        a = sc.read_10x_mtx(staged, var_names="gene_symbols", cache=False)
        a.var_names_make_unique()
        a.obs_names = [f"{key}_{b}" for b in a.obs_names]
        a.obs["sample_name"] = key
        a.obs["gsm"] = key.split("_")[0]
        a.obs["dataset"] = "Yang2022"
        a.obs["technical_batch"] = "Yang_GSE169608_10x"
        a.obs["assay"] = "10x"
        a.obs["genotype"] = "WT"
        a.obs["lineage"] = "Other"
        for k, v in meta.items():
            a.obs[k] = v
        print(f"  {key}: {a.n_obs:,} × {a.n_vars:,}")
        parts.append(a)
    combined = _concat_samples(parts)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.h5ad")
    combined.write_h5ad(tmp, compression="gzip")
    tmp.replace(out)
    print(f"  wrote counts {out} ({combined.n_obs:,} × {combined.n_vars:,})")
    return out


def build_gse147729_counts(*, force: bool = False) -> Path:
    cfg = PATHS["gse147729"]
    out: Path = cfg["source"]
    if out.exists() and not force:
        return out
    raw: Path = cfg["raw"]
    parts: list[ad.AnnData] = []
    for key, meta in GSE147729_SAMPLES.items():
        staged = _stage_geo_mtx_dir(raw, key, gene_glob="genes")
        a = sc.read_10x_mtx(staged, var_names="gene_symbols", cache=False)
        a.var_names_make_unique()
        a.obs_names = [f"{key}_{b}" for b in a.obs_names]
        a.obs["sample_name"] = key
        a.obs["gsm"] = key.split("_")[0]
        a.obs["dataset"] = "Herault2021"
        a.obs["technical_batch"] = "Herault_GSE147729_10x"
        a.obs["assay"] = "10x"
        a.obs["genotype"] = "WT"
        a.obs["lineage"] = "HSPC"
        a.obs["cell_type"] = "HSPC"
        for k, v in meta.items():
            a.obs[k] = v
        print(f"  {key}: {a.n_obs:,} × {a.n_vars:,}")
        parts.append(a)
    combined = _concat_samples(parts)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.h5ad")
    combined.write_h5ad(tmp, compression="gzip")
    tmp.replace(out)
    print(f"  wrote counts {out} ({combined.n_obs:,} × {combined.n_vars:,})")
    return out


def build_gse246464_counts(*, force: bool = False) -> Path:
    """RNA-only multiome filtered MTX (ATAC fragments intentionally ignored)."""
    cfg = PATHS["gse246464"]
    out: Path = cfg["source"]
    if out.exists() and not force:
        return out
    raw: Path = cfg["raw"]
    parts: list[ad.AnnData] = []
    for key, meta in GSE246464_SAMPLES.items():
        staged = _stage_geo_mtx_dir(raw, key, gene_glob="features")
        a = sc.read_10x_mtx(staged, var_names="gene_symbols", cache=False, gex_only=True)
        a.var_names_make_unique()
        a.obs_names = [f"{key}_{b}" for b in a.obs_names]
        a.obs["sample_name"] = key
        a.obs["gsm"] = key.split("_")[0]
        a.obs["dataset"] = "Elias2025"
        a.obs["technical_batch"] = "Elias_GSE246464_10x"
        a.obs["assay"] = "10x_multiome_RNA"
        a.obs["genotype"] = "WT"
        a.obs["lineage"] = "HSPC"
        a.obs["cell_type"] = "HSC"
        for k, v in meta.items():
            a.obs[k] = v
        print(f"  {key}: {a.n_obs:,} × {a.n_vars:,}")
        parts.append(a)
    combined = _concat_samples(parts)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.h5ad")
    combined.write_h5ad(tmp, compression="gzip")
    tmp.replace(out)
    print(f"  wrote counts {out} ({combined.n_obs:,} × {combined.n_vars:,})")
    return out


def load_gse169608_counts(*, force: bool = False) -> ad.AnnData:
    p = build_gse169608_counts(force=force)
    return sc.read_h5ad(p)


def load_gse147729_counts(*, force: bool = False) -> ad.AnnData:
    p = build_gse147729_counts(force=force)
    return sc.read_h5ad(p)


def load_gse246464_counts(*, force: bool = False) -> ad.AnnData:
    p = build_gse246464_counts(force=force)
    return sc.read_h5ad(p)


def build_gse70657_counts(*, force: bool = False) -> Path:
    """Grover GSE70657: genes × cells RefSeq integer read counts (C1/Fluidigm)."""
    cfg = PATHS["gse70657"]
    out: Path = cfg["source"]
    if out.exists() and not force:
        return out
    raw: Path = cfg["raw"]
    if not raw.exists():
        raise FileNotFoundError(
            f"Missing {raw}. Download GEO suppl "
            "GSE70657_Grover.A_et.al_RefSeq.Read.Count.txt.gz"
        )
    mat = pd.read_csv(raw, sep="\t", compression="gzip", index_col=0)
    # columns like A10_young / H5_old
    ages = pd.Index(mat.columns.astype(str)).str.extract(r"_(young|old)$", expand=False)
    if ages.isna().any():
        bad = mat.columns[ages.isna()].tolist()[:5]
        raise ValueError(f"GSE70657 columns missing _young/_old suffix: {bad}")
    X = sparse.csr_matrix(mat.to_numpy(dtype=np.float32).T)
    a = ad.AnnData(X)
    a.obs_names = mat.columns.astype(str).tolist()
    a.var_names = mat.index.astype(str).tolist()
    a.var_names_make_unique()
    a.obs["sample_name"] = a.obs_names.str.replace(r"_(young|old)$", "", regex=True)
    a.obs["age_label"] = ages.astype(str).values
    a.obs["age_group"] = a.obs["age_label"]
    a.obs["age_months"] = a.obs["age_label"].map({"young": 2.5, "old": 22.5})
    a.obs["dataset"] = "Grover2016"
    a.obs["technical_batch"] = "Grover_GSE70657_C1"
    a.obs["assay"] = "Fluidigm_C1"
    a.obs["genotype"] = "WT"
    a.obs["cell_type"] = "LT_HSC"
    a.obs["lineage"] = "HSPC"
    a.obs["facs_subset"] = "LSK_CD150+_CD48-"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.h5ad")
    a.write_h5ad(tmp, compression="gzip")
    tmp.replace(out)
    print(f"  wrote counts {out} ({a.n_obs:,} × {a.n_vars:,})")
    print(f"  ages: {a.obs['age_label'].value_counts().to_dict()}")
    return out


def load_gse70657_counts(*, force: bool = False) -> ad.AnnData:
    p = build_gse70657_counts(force=force)
    return sc.read_h5ad(p)


def _parse_kowalczyk_cell(name: str) -> dict:
    """young_LT_HSC_2 / old_ST_HSC_biol_replicate_12 → age + FACS subset."""
    n = name.strip().strip("'")
    parts = n.split("_")
    age = parts[0].lower()
    if age not in {"young", "old"}:
        raise ValueError(f"bad Kowalczyk cell name: {name}")
    rest = parts[1:]
    if rest and rest[-1].isdigit():
        rest = rest[:-1]
    facs = "_".join(rest) if rest else "HSC"
    return {
        "age_label": age,
        "age_group": age,
        "age_months": 2.5 if age == "young" else 22.0,
        "facs_subset": facs,
        "cell_type": facs,
        "lineage": "HSPC",
    }


def build_gse59114_lognorm(*, force: bool = False) -> Path:
    """Kowalczyk GSE59114 C57BL/6: GEO Excel is log-scaled SMART-seq (not counts).

    Last 6 columns are population averages — dropped. Gene symbols arrive quoted.
    """
    cfg = PATHS["gse59114"]
    out: Path = cfg["source"]
    if out.exists() and not force:
        return out
    raw: Path = cfg["raw"]
    if not raw.exists():
        raise FileNotFoundError(
            f"Missing {raw}. Download GEO suppl GSE59114_C57BL6_GEO_all.xlsx"
        )
    print(f"  reading {raw} (slow; ~100MB xlsx)…")
    df = pd.read_excel(raw, sheet_name="Sheet1", header=1)
    # columns: Gene Symbol | UCSC transcripts | cells… | optional avg cols
    gene_col = df.columns[0]
    genes = (
        df[gene_col]
        .astype(str)
        .str.strip()
        .str.strip("'\"")
        .str.replace(r"^'|'$", "", regex=True)
    )
    cell_cols = []
    for c in df.columns[2:]:
        cs = str(c)
        # drop population-average columns and unnamed junk
        if cs.startswith("Unnamed"):
            continue
        if "'" in cs or " " in cs:
            continue
        if cs.lower().startswith(("young_", "old_")):
            cell_cols.append(c)
    if not cell_cols:
        raise RuntimeError("GSE59114: no young_/old_ single-cell columns found")
    mat = df[cell_cols].to_numpy(dtype=np.float32)
    # genes × cells → cells × genes
    X = sparse.csr_matrix(np.nan_to_num(mat.T, nan=0.0))
    a = ad.AnnData(X)
    a.obs_names = [str(c) for c in cell_cols]
    a.var_names = genes.tolist()
    # drop empty gene symbols / duplicates
    keep_g = np.array([g not in {"", "nan", "None"} for g in a.var_names])
    a = a[:, keep_g].copy()
    a.var_names_make_unique()
    meta = [_parse_kowalczyk_cell(n) for n in a.obs_names]
    for k in meta[0]:
        a.obs[k] = [m[k] for m in meta]
    a.obs["sample_name"] = a.obs["facs_subset"].astype(str) + "_" + a.obs["age_label"].astype(str)
    a.obs["dataset"] = "Kowalczyk2015"
    a.obs["technical_batch"] = "Kowalczyk_GSE59114_Smartseq"
    a.obs["assay"] = "SMART-seq_lognorm"
    a.obs["genotype"] = "WT"
    a.uns["expression_note"] = (
        "GEO GSE59114_C57BL6_GEO_all.xlsx deposits log-scaled SMART-seq values "
        "(max≈16.7), not integer counts. Not suitable for seurat_v3/scGen counts HVG."
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.h5ad")
    a.write_h5ad(tmp, compression="gzip")
    tmp.replace(out)
    print(f"  wrote lognorm {out} ({a.n_obs:,} × {a.n_vars:,})")
    print(f"  ages: {a.obs['age_label'].value_counts().to_dict()}")
    print(f"  facs: {a.obs['facs_subset'].value_counts().to_dict()}")
    return out


def load_gse59114_lognorm(*, force: bool = False) -> ad.AnnData:
    p = build_gse59114_lognorm(force=force)
    return sc.read_h5ad(p)


def gpu_qc_lognorm_plate(
    a: ad.AnnData,
    qc_batch_key: str,
    *,
    min_genes: int = 1000,
    min_cells_gene: int = 3,
    gene_mad: float = 4.0,
) -> ad.AnnData:
    """Light GPU QC for already log-normalized plate-seq (no Scrublet / no counts)."""
    _init_gpu()
    a = a.copy()
    _sanitize_var(a)
    a.X = _as_csr(a.X)
    rsc.get.anndata_to_GPU(a)
    rsc.pp.filter_genes(a, min_cells=min_cells_gene)
    rsc.pp.flag_gene_family(a, gene_family_name="mt", gene_family_prefix="mt-")
    rsc.pp.calculate_qc_metrics(a, qc_vars=["mt"], log1p=False)
    rsc.get.anndata_to_CPU(a)
    a.obs["ribo_frac"] = _ribo_frac(a)
    n_before = a.n_obs
    keep = pd.Series(False, index=a.obs_names)
    for _, idx in a.obs.groupby(qc_batch_key, observed=True).groups.items():
        obs = a.obs.loc[idx]
        log_genes = np.log1p(obs["n_genes_by_counts"].astype(float))
        g_med = float(log_genes.median())
        g_mad = max(float((log_genes - g_med).abs().median()), 0.15)
        genes_ok = log_genes.between(g_med - gene_mad * g_mad, g_med + gene_mad * g_mad)
        ok = genes_ok & (obs["n_genes_by_counts"].astype(float) >= min_genes)
        keep.loc[idx] = ok
    a = a[keep].copy()
    print(f"  QC filter (lognorm plate): {a.n_obs:,} / {n_before:,} cells")
    # X already log-like — do not re-normalize; no layers['counts']
    a.layers["lognorm"] = a.X.copy()
    a.uns["qc"] = {
        "mode": "lognorm_plate",
        "qc_batch_key": qc_batch_key,
        "min_genes": min_genes,
        "min_cells_gene": min_cells_gene,
        "gene_mad": gene_mad,
        "has_counts_layer": False,
    }
    return a


def process_dataset(
    name: str,
    *,
    force: bool = False,
    run_scrublet: bool = True,
    scrublet_smartseq2: bool = False,
    annotate: bool = False,
    hspc_only: bool = False,
) -> Path:
    cfg = PATHS[name]
    out: Path = cfg["out"]
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and not force:
        print(f"{name}: cached {out} (use --force to redo)")
        return out

    print(f"\n=== {name} ===")
    if name == "su2024":
        a = load_su2024_counts()
        # Scrublet on ~100–200-cell Smart-seq2 plates is unreliable (pathological
        # estimated doublet rates). Off unless scrublet_smartseq2=True.
        a = gpu_qc_preprocess(
            a,
            qc_batch_key="sample_name",
            scrublet_batch_key="sample_name",
            min_genes=2000,
            max_mt=5.0,
            max_ribo=0.15,
            min_cells_gene=3,
            gene_mad=4.0,
            run_scrublet=run_scrublet and scrublet_smartseq2,
        )
    elif name == "gse70657":
        # C1 Fluidigm integer read counts (~135 LT-HSCs). Same plate-seq QC as Su.
        a = load_gse70657_counts(force=force)
        a = gpu_qc_preprocess(
            a,
            qc_batch_key="age_label",
            scrublet_batch_key="age_label",
            min_genes=1000,
            max_mt=10.0,
            max_ribo=0.25,
            min_cells_gene=3,
            gene_mad=4.0,
            run_scrublet=run_scrublet and scrublet_smartseq2,
        )
    elif name == "gse59114":
        # Log-normalized SMART-seq only — reference / marker QC, not scGen counts.
        a = load_gse59114_lognorm(force=force)
        a = gpu_qc_lognorm_plate(
            a,
            qc_batch_key="sample_name",
            min_genes=1000,
            min_cells_gene=3,
            gene_mad=4.0,
        )
    elif name == "gse169162":
        a = load_mitchell_counts()
        a = gpu_qc_preprocess(
            a,
            qc_batch_key="sample_name",
            scrublet_batch_key="sample_name",
            run_scrublet=run_scrublet,
        )
    elif name == "gse310923":
        a = load_white_counts()
        a = gpu_qc_preprocess(
            a,
            qc_batch_key="sample_name",
            scrublet_batch_key="sample_name",
            run_scrublet=run_scrublet,
        )
    elif name == "gse169608":
        a = load_gse169608_counts(force=force)
        a = gpu_qc_preprocess(
            a,
            qc_batch_key="sample_name",
            scrublet_batch_key="sample_name",
            run_scrublet=run_scrublet,
        )
    elif name == "gse147729":
        a = load_gse147729_counts(force=force)
        a = gpu_qc_preprocess(
            a,
            qc_batch_key="sample_name",
            scrublet_batch_key="sample_name",
            run_scrublet=run_scrublet,
        )
    elif name == "gse246464":
        # Multiome RNA: higher ambient / non-HSPC leakage in FACS HSC gates
        a = load_gse246464_counts(force=force)
        a = gpu_qc_preprocess(
            a,
            qc_batch_key="sample_name",
            scrublet_batch_key="sample_name",
            min_genes=2000,
            max_mt=10.0,
            max_ribo=0.25,
            gene_mad=4.0,
            run_scrublet=run_scrublet,
        )
    else:
        raise ValueError(name)

    if annotate:
        # Su / White / Grover / Kowalczyk: FACS/author labels trusted — markers cross-check.
        # Elias multiome HSC sort: FACS gate leaks MegE/Endo/Ery → overwrite lineage.
        # Yang / Mitchell: marker argmax when unlabeled or mostly Other.
        if name == "gse246464" and "lineage" in a.obs:
            a.obs["facs_lineage"] = a.obs["lineage"].astype(str)
        overwrite = name in {"gse169162", "gse169608", "gse246464"} and (
            name == "gse246464"
            or "lineage" not in a.obs
            or (a.obs["lineage"] == "Other").mean() > 0.5
        )
        score_markers(a, overwrite_lineage=overwrite)
        if name == "gse246464" and "score_HSPC" in a.obs:
            # Drop weak HSPC calls that still win argmax on noise
            weak = (
                (a.obs["lineage"].astype(str) == "HSPC")
                & (a.obs["score_HSPC"].astype(float) < 0.2)
            )
            n_weak = int(weak.sum())
            if n_weak:
                a = a[~weak].copy()
                print(f"  Elias: dropped {n_weak:,} weak HSPC (score_HSPC<0.2)")

    if hspc_only:
        keep = a.obs["lineage"].astype(str).isin(["HSPC", "Myeloid_prog"])
        if name == "su2024" and "is_HSC" in a.obs:
            keep = keep | a.obs["is_HSC"].astype(bool)
        a = a[keep].copy()
        print(f"  HSPC/Myeloid_prog subset: {a.n_obs:,}")

    for col in (
        "dataset",
        "technical_batch",
        "assay",
        "age_label",
        "age_group",
        "age_months",
        "age_bin",
        "lineage",
        "genotype",
        "sample_name",
    ):
        if col not in a.obs:
            a.obs[col] = np.nan

    assign_age_bin(a)

    tmp = out.with_suffix(".tmp.h5ad")
    a.write_h5ad(tmp, compression="gzip")
    tmp.replace(out)
    print(f"  wrote {out} ({a.n_obs:,} × {a.n_vars:,})")
    print(a.obs.groupby(["age_bin", "age_label", "lineage"], observed=True).size().to_string())
    return out


def write_joint_manifest(paths: dict[str, Path], dest: Path) -> None:
    rows = []
    for name, p in paths.items():
        a = sc.read_h5ad(p, backed="r")
        rows.append(
            {
                "dataset": name,
                "path": str(p),
                "n_obs": int(a.n_obs),
                "n_vars": int(a.n_vars),
                "has_counts_layer": "counts" in a.layers,
                "lineages": a.obs["lineage"].astype(str).value_counts().to_dict()
                if "lineage" in a.obs
                else {},
                "ages": a.obs["age_label"].astype(str).value_counts().to_dict()
                if "age_label" in a.obs
                else {},
                "technical_batch": a.obs["technical_batch"].astype(str).iloc[0]
                if "technical_batch" in a.obs
                else None,
            }
        )
        del a
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rows, indent=2))
    print(f"\nmanifest → {dest}")


def patch_age_bin_cached() -> None:
    """Add ``age_bin`` to existing QC outputs + joint scGen object (no re-QC)."""
    targets: list[Path] = []
    for cfg in PATHS.values():
        p = cfg.get("out")
        if p is not None and Path(p).exists():
            targets.append(Path(p))
    joint = BONE / "results" / "joint_hsc_aging" / "age_core_scgen.h5ad"
    hold = BONE / "results" / "joint_hsc_aging" / "su_holdout_adult_juvenile_sharedgenes.h5ad"
    for p in (joint, hold):
        if p.exists():
            targets.append(p)

    for path in targets:
        a = sc.read_h5ad(path)
        assign_age_bin(a)
        tmp = path.with_suffix(".tmp.h5ad")
        a.write_h5ad(tmp, compression="gzip")
        tmp.replace(path)
        tab = (
            a.obs.groupby(["dataset", "age_bin", "age_label"], observed=True)
            .size()
            .rename("n")
            .reset_index()
            if "dataset" in a.obs
            else a.obs.groupby(["age_bin", "age_label"], observed=True)
            .size()
            .rename("n")
            .reset_index()
        )
        print(f"patched {path.name}")
        print(tab.to_string(index=False))
        del a


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--dataset",
        choices=[*ALL_DATASETS, "age_core", "all"],
        default="gse169608",
    )
    ap.add_argument("--force", action="store_true", help="Ignore cached outputs")
    ap.add_argument(
        "--no-scrublet",
        action="store_true",
        help="Skip doublet detection on 10x datasets",
    )
    ap.add_argument(
        "--scrublet-smartseq2",
        action="store_true",
        help="Force Scrublet on plate-seq (Su/Grover; off by default; often unreliable)",
    )
    ap.add_argument(
        "--annotate",
        action="store_true",
        help="Add BM marker scores (and marker_lineage); overwrites Other on Yang/Mitchell",
    )
    ap.add_argument(
        "--hspc-only",
        action="store_true",
        help="Keep HSPC + Myeloid_prog lineages only",
    )
    ap.add_argument(
        "--assign-age-bin",
        action="store_true",
        help=(
            "Patch age_bin onto cached QC / joint h5ads without re-running QC "
            "(early≤2.5, mid≤8, late≥18 mo)"
        ),
    )
    args = ap.parse_args()

    if args.assign_age_bin:
        patch_age_bin_cached()
        return

    if args.dataset == "all":
        names = ALL_DATASETS
    elif args.dataset == "age_core":
        # scGen-ready integer counts only (excludes GSE59114 log-norm reference)
        names = [*AGE_CORE_PLATESEQ, *AGE_CORE_10X]
    else:
        names = [args.dataset]

    outs: dict[str, Path] = {}
    for name in names:
        outs[name] = process_dataset(
            name,
            force=args.force,
            run_scrublet=not args.no_scrublet,
            scrublet_smartseq2=args.scrublet_smartseq2,
            annotate=args.annotate,
            hspc_only=args.hspc_only,
        )

    if len(outs) > 1:
        write_joint_manifest(
            outs, BONE / "results" / "joint_hsc_aging" / "preprocess_manifest.json"
        )
        print(
            "\nJoint next step: concat on shared genes with batch_key=technical_batch, "
            "then scGen batch_removal + young↔old transfer; Yang lineage via markers/scGen."
        )
        print(
            "Optional (not in age_core): gse70657 young/old counts; "
            "gse59114 log2(TPM+1) reference — neither adds an adult midpoint."
        )


if __name__ == "__main__":
    main()

# Minimal self-check (import-safe): path table is complete
assert set(PATHS) == set(ALL_DATASETS)
assert "HSPC" in BM_MARKER_SETS and "Myeloid_prog" in BM_MARKER_SETS
# silence unused import if someone greps for cleanup hooks
_ = shutil
_ = gzip