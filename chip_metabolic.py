#!/usr/bin/env python3
"""Tet2 × IL-1 deep factor graph: McClatchy (Young) + Caiado (Old) on one shared graph."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import types
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

BONE = Path("/cis/net/r41/data/iessien1/bone")
RESULTS = Path("/cis/net/r41/data/iessien1/bone_marrow_results")
MOUSE_DB = BONE / "sccellfie" / "mus_musculus"
OUT = RESULTS / "chip_metabolic_graph"
QC = BONE / "GSE209994" / "processed" / "gse209994_qc_preprocessed.h5ad"
IL1B = OUT / "gse209994_il1b_with_tasks_full.h5ad"
VEHICLE = OUT / "gse209994_vehicle_with_tasks.h5ad"
FACTORIAL = OUT / "gse209994_2x2_with_tasks.h5ad"
CAIADO_QC = BONE / "PRJEB56666" / "processed" / "prjeb56666_qc_preprocessed.h5ad"
CAIADO_SCORED = OUT / "prjeb56666_old_with_tasks.h5ad"
YOUNG_OLD = OUT / "young_old_factorial_with_tasks.h5ad"

MYELOID = ("HSPC", "Myeloid_prog", "Mono_Mac", "Granulocyte")
# age_cohort × genotype × treatment (IL1a/IL1b → IL1; PBS → vehicle; Tet2_* → Tet2)
CLASSES = [
    "Young_WT_vehicle",
    "Young_WT_IL1",
    "Young_Tet2_vehicle",
    "Young_Tet2_IL1",
    "Old_WT_vehicle",
    "Old_WT_IL1",
    "Old_Tet2_vehicle",
    "Old_Tet2_IL1",
]
YOUNG_ONLY_CLASSES = [
    "Young_WT_vehicle",
    "Young_WT_IL1",
    "Young_Tet2_vehicle",
    "Young_Tet2_IL1",
]
EPOCHS = 100
HIDDEN = 96
STEPS = 10
BATCH_PER_GPU = 1024

AXIS_TASKS: dict[str, list[str]] = {
    "N_glycosylation": [
        "N-linked glycosylation",
        "N-glycan processing (ER)",
        "Branching (N-acetylglucosaminyltransferases)",
        "Mannose trimming (mannosidase)",
        "Galactosylation (addition of galactose)",
        "Fucosylation (addition of fucose)",
        "Sialylation (addition of sialic acid)",
        "Biosynthesis of g3m8masn",
        "Biosynthesis of m4mpdol_U",
        "Degradation of n2m2nmasn",
        "Degradation of s2l2fn2m2masn",
        "Keratan sulfate biosynthesis from N-glycan",
    ],
    "glycolysis": [
        "ATP generation from glucose (hypoxic conditions) - glycolysis",
        "Glycogen degradation",
        "Glycogen biosynthesis",
        "Mannose degradation (to fructose-6-phosphate)",
        "Fructose degradation (to glucose-3-phosphate)",
        "Synthesis of fructose-6-phosphate from erythrose-4-phosphate (HMP shunt)",
        "UDP-glucose synthesis",
    ],
    "OXPHOS_TCA": [
        "Oxidative phosphorylation via NADH-coenzyme Q oxidoreductase (COMPLEX I)",
        "Oxidative phosphorylation via succinate-coenzyme Q oxidoreductase (COMPLEX II)",
        "Krebs cycle - NADH generation",
        "Krebs cycle - oxidative decarboxylation of pyruvate",
        "Reactive oxygen species detoxification (H2O2 to H2O)",
    ],
}


def _sccellfie_root() -> Path:
    local = (
        Path(__file__).resolve().parent / ".venv/lib/python3.10/site-packages/sccellfie"
    )
    return (
        local
        if local.exists()
        else Path(sys.prefix) / "lib/python3.10/site-packages/sccellfie"
    )


def _install_sccellfie_without_spatial() -> None:
    """Pipeline needs scCellFie submodules, not spatial; squidpy→xarray_dataclasses is broken in this env."""
    for key in list(sys.modules):
        if key == "sccellfie" or key.startswith("sccellfie."):
            del sys.modules[key]
    root = _sccellfie_root()
    pkg = types.ModuleType("sccellfie")
    pkg.__path__ = [str(root)]
    pkg.__file__ = str(root / "__init__.py")
    pkg.__version__ = "0.6.0"
    sys.modules["sccellfie"] = pkg
    spatial = types.ModuleType("sccellfie.spatial")
    spatial.__path__ = [str(root / "spatial")]
    sys.modules["sccellfie.spatial"] = spatial
    pkg.spatial = spatial


def _import_sccellfie_pipeline():
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache_bm")
    try:
        from sccellfie.sccellfie_pipeline import run_sccellfie_pipeline

        return run_sccellfie_pipeline
    except ImportError:
        _install_sccellfie_without_spatial()
        return importlib.import_module(
            "sccellfie.sccellfie_pipeline"
        ).run_sccellfie_pipeline


def _ensure_counts(adata: ad.AnnData) -> ad.AnnData:
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    if "n_counts" not in adata.obs:
        X = adata.X
        adata.obs["n_counts"] = (
            np.asarray(X.sum(axis=1)).ravel() if sparse.issparse(X) else X.sum(axis=1)
        )
    return adata


def _attach_metabolic_tasks(out: ad.AnnData, out_stem: str) -> ad.AnnData:
    mt = getattr(out, "metabolic_tasks", None)
    if mt is None:
        mt = ad.read_h5ad(OUT / f"{out_stem}_metabolic_tasks.h5ad")
    out.uns["metabolic_task_names"] = list(mt.var_names.astype(str))
    out.obsm["metabolic_tasks"] = (
        mt.X.toarray() if sparse.issparse(mt.X) else np.asarray(mt.X)
    )
    return out


def _run_sccellfie(adata: ad.AnnData, out_stem: str) -> ad.AnnData:
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=False)
    sc.pp.pca(adata, n_comps=30)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    adata = _ensure_counts(adata)
    db = _import_sccellfie_pipeline()(
        adata,
        organism="mouse",
        sccellfie_data_folder=str(MOUSE_DB),
        n_counts_col="n_counts",
        neighbors_key="neighbors",
        n_neighbors=15,
        smooth_cells=True,
        alpha=0.33,
        chunk_size=4000,
        save_folder=str(OUT),
        save_filename=out_stem,
        compute_ablation_impact=False,
        verbose=True,
    )
    return _attach_metabolic_tasks(db["adata"], out_stem)


def _score_arm(treatment: str, out_stem: str) -> ad.AnnData:
    raw = ad.read_h5ad(QC)
    m = (
        raw.obs["treatment"].astype(str).eq(treatment)
        & raw.obs["lineage"].isin(MYELOID)
        & raw.obs["genotype"].isin(["WT", "Tet2_KO"])
    )
    adata = _ensure_counts(raw[m].copy())
    print(
        f"{treatment}: n={adata.n_obs} {adata.obs['genotype'].value_counts().to_dict()}",
        flush=True,
    )
    return _run_sccellfie(adata, out_stem)


def _align_concat(a: ad.AnnData, b: ad.AnnData) -> ad.AnnData:
    genes = a.var_names.intersection(b.var_names)
    a, b = a[:, genes].copy(), b[:, genes].copy()
    ta, tb = list(a.uns["metabolic_task_names"]), list(b.uns["metabolic_task_names"])
    tasks = [t for t in ta if t in set(tb)]
    a.obsm["metabolic_tasks"] = np.asarray(a.obsm["metabolic_tasks"])[
        :, [ta.index(t) for t in tasks]
    ]
    b.obsm["metabolic_tasks"] = np.asarray(b.obsm["metabolic_tasks"])[
        :, [tb.index(t) for t in tasks]
    ]
    a.uns["metabolic_task_names"] = b.uns["metabolic_task_names"] = tasks
    for x in (a, b):
        x.obs["condition"] = (
            x.obs["genotype"].astype(str) + "_" + x.obs["treatment"].astype(str)
        )
    out = ad.concat(
        [a, b], join="inner", label="arm", keys=["a", "b"], index_unique="-"
    )
    out.uns["metabolic_task_names"] = tasks
    return out


def load_or_build_factorial() -> ad.AnnData:
    OUT.mkdir(parents=True, exist_ok=True)
    if FACTORIAL.exists():
        print(f"reusing {FACTORIAL}", flush=True)
        return ad.read_h5ad(FACTORIAL)
    if IL1B.exists():
        il1b = ad.read_h5ad(IL1B)
    else:
        il1b = _score_arm("IL1b", "gse209994_il1b_sccellfie_full")
        il1b.write_h5ad(IL1B)
    if VEHICLE.exists():
        veh = ad.read_h5ad(VEHICLE)
    else:
        veh = _score_arm("vehicle", "gse209994_vehicle_sccellfie")
        veh.write_h5ad(VEHICLE)
    adata = _align_concat(il1b, veh)
    adata.write_h5ad(FACTORIAL)
    print(f"wrote {FACTORIAL} n={adata.n_obs}", flush=True)
    return adata


def _to_dense(X) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray()
    if hasattr(X, "toarray"):
        return np.asarray(X.toarray())
    return np.asarray(X)


def _remap_genotype(s: pd.Series) -> pd.Series:
    m = {"WT": "WT", "Tet2_KO": "Tet2", "Tet2_het": "Tet2", "Tet2": "Tet2"}
    out = s.astype(str).map(m)
    bad = out.isna()
    if bad.any():
        raise SystemExit(f"unmapped genotype: {s[bad].unique().tolist()}")
    return out


def _remap_treatment(s: pd.Series) -> pd.Series:
    m = {
        "vehicle": "vehicle",
        "PBS": "vehicle",
        "IL1b": "IL1",
        "IL1a": "IL1",
        "IL1": "IL1",
    }
    out = s.astype(str).map(m)
    bad = out.isna()
    if bad.any():
        raise SystemExit(f"unmapped treatment: {s[bad].unique().tolist()}")
    return out


def _stamp_age_condition(adata: ad.AnnData, age_cohort: str) -> ad.AnnData:
    """Harmonize genotype/treatment and set 8-way condition with age_cohort."""
    a = adata.copy()
    a.obs["genotype_raw"] = a.obs["genotype"].astype(str)
    a.obs["treatment_raw"] = a.obs["treatment"].astype(str)
    a.obs["genotype"] = _remap_genotype(a.obs["genotype"])
    a.obs["treatment"] = _remap_treatment(a.obs["treatment"])
    a.obs["age_cohort"] = age_cohort
    a.obs["condition"] = (
        a.obs["age_cohort"].astype(str)
        + "_"
        + a.obs["genotype"].astype(str)
        + "_"
        + a.obs["treatment"].astype(str)
    )
    if "sample_name" not in a.obs.columns or a.obs["sample_name"].isna().all():
        a.obs["sample_name"] = a.obs_names.astype(str)
    else:
        sn = a.obs["sample_name"].astype(str)
        a.obs["sample_name"] = sn.where(
            ~sn.isin(["nan", "None", ""]), a.obs_names.astype(str)
        )
    return a


def _task_means_from_genes(adata: ad.AnnData, tbg: pd.DataFrame) -> ad.AnnData:
    """Fill gene_scores + metabolic_tasks from expression matrix (bulk path)."""
    Xd = _to_dense(adata.X).astype(np.float32)
    adata.layers["gene_scores"] = Xd
    g_idx = {g: i for i, g in enumerate(adata.var_names.astype(str))}
    cols: list[np.ndarray] = []
    kept: list[str] = []
    for t in tbg.index.astype(str):
        row = tbg.loc[t]
        idxs = [g_idx[str(g)] for g, v in row.items() if v > 0 and str(g) in g_idx]
        if len(idxs) < 2:
            continue
        cols.append(Xd[:, idxs].mean(1))
        kept.append(t)
    adata.uns["metabolic_task_names"] = kept
    adata.obsm["metabolic_tasks"] = (
        np.column_stack(cols) if cols else np.zeros((adata.n_obs, 0), dtype=np.float32)
    )
    return adata


def load_or_build_caiado_old(tbg: pd.DataFrame) -> ad.AnnData:
    """PRJEB56666 bulk HSC → Old cohort with task scores."""
    OUT.mkdir(parents=True, exist_ok=True)
    if CAIADO_SCORED.exists():
        print(f"reusing {CAIADO_SCORED}", flush=True)
        return ad.read_h5ad(CAIADO_SCORED)
    if not CAIADO_QC.exists():
        raise SystemExit(
            f"missing {CAIADO_QC}\n"
            "Run: python preprocess.py --dataset prjeb56666 --annotate --force"
        )
    raw = ad.read_h5ad(CAIADO_QC)
    adata = _stamp_age_condition(raw, "Old")
    adata.obs["dataset"] = "Caiado2023"
    adata = _task_means_from_genes(adata, tbg)
    adata.write_h5ad(CAIADO_SCORED)
    print(
        f"wrote {CAIADO_SCORED} n={adata.n_obs} "
        f"{adata.obs['condition'].value_counts().to_dict()} "
        f"tasks={len(adata.uns['metabolic_task_names'])}",
        flush=True,
    )
    return adata


def _stamp_mcclatchy_young(adata: ad.AnnData) -> ad.AnnData:
    a = _stamp_age_condition(adata, "Young")
    if "dataset" not in a.obs.columns:
        a.obs["dataset"] = "McClatchy2023"
    else:
        ds = a.obs["dataset"].astype(str)
        a.obs["dataset"] = ds.where(~ds.isin(["", "nan", "None"]), "McClatchy2023")
    return a


def load_or_build_young_old_factorial(tbg: pd.DataFrame) -> ad.AnnData:
    """McClatchy Young + Caiado Old on shared genes/tasks for one factor graph."""
    OUT.mkdir(parents=True, exist_ok=True)
    if YOUNG_OLD.exists():
        print(f"reusing {YOUNG_OLD}", flush=True)
        return ad.read_h5ad(YOUNG_OLD)
    young = _stamp_mcclatchy_young(load_or_build_factorial())
    old = load_or_build_caiado_old(tbg)
    genes = young.var_names.intersection(old.var_names)
    if len(genes) < 100:
        raise SystemExit(f"too few shared genes Young∩Old: {len(genes)}")
    young, old = young[:, genes].copy(), old[:, genes].copy()
    # Align gene_scores columns to shared genes
    for a in (young, old):
        if "gene_scores" not in a.layers:
            a.layers["gene_scores"] = _to_dense(a.X).astype(np.float32)
        else:
            gs = _to_dense(a.layers["gene_scores"]).astype(np.float32)
            # factorial may already be gene-subset; reindex if needed
            if gs.shape[1] != a.n_vars:
                raise SystemExit("gene_scores width mismatch after gene subset")
            a.layers["gene_scores"] = gs
    ty = list(young.uns["metabolic_task_names"])
    to = list(old.uns["metabolic_task_names"])
    tasks = [t for t in ty if t in set(to)]
    if len(tasks) < 10:
        raise SystemExit(f"too few shared metabolic tasks: {len(tasks)}")
    young.obsm["metabolic_tasks"] = np.asarray(young.obsm["metabolic_tasks"])[
        :, [ty.index(t) for t in tasks]
    ]
    old.obsm["metabolic_tasks"] = np.asarray(old.obsm["metabolic_tasks"])[
        :, [to.index(t) for t in tasks]
    ]
    young.uns["metabolic_task_names"] = old.uns["metabolic_task_names"] = tasks
    out = ad.concat(
        [young, old],
        join="inner",
        label="age_source",
        keys=["Young", "Old"],
        index_unique="-",
    )
    out.uns["metabolic_task_names"] = tasks
    out.uns["age_cohort_note"] = (
        "Young=GSE209994 McClatchy; Old=PRJEB56666 Caiado (~6–9 mo adult "
        "IL-1α challenge proxy, not chronological 18–24 mo aging RNA)"
    )
    out.write_h5ad(YOUNG_OLD)
    print(
        f"wrote {YOUNG_OLD} n={out.n_obs} genes={out.n_vars} tasks={len(tasks)}\n"
        f"  conditions: {out.obs['condition'].value_counts().to_dict()}",
        flush=True,
    )
    return out


def young_old_task_prior(
    adata: ad.AnnData, source: str = "Young_vs_Old", out_csv: Path | None = None
) -> pd.DataFrame:
    """Young vs Old task Cohen's d from age_cohort (for optional edge prior)."""
    tasks = list(adata.uns["metabolic_task_names"])
    M = np.asarray(adata.obsm["metabolic_tasks"], dtype=np.float64)
    age = adata.obs["age_cohort"].astype(str)
    samples = adata.obs["sample_name"].astype(str)
    pb = (
        pd.DataFrame(M, index=adata.obs_names, columns=tasks)
        .assign(sample_name=samples.values, age_cohort=age.values)
        .groupby(["sample_name", "age_cohort"], observed=True)
        .mean(numeric_only=True)
        .reset_index()
    )
    young = pb[pb["age_cohort"] == "Young"][tasks]
    old = pb[pb["age_cohort"] == "Old"][tasks]
    rows = []
    for t in tasks:
        y, o = young[t].to_numpy(), old[t].to_numpy()
        if len(y) < 1 or len(o) < 1:
            continue
        delta = float(o.mean() - y.mean())
        pooled = np.sqrt(
            (
                (y.var(ddof=1) if len(y) > 1 else 0.0)
                + (o.var(ddof=1) if len(o) > 1 else 0.0)
            )
            / 2
        )
        d = delta / pooled if pooled > 0 else 0.0
        rows.append(
            {
                "task": t,
                "source": source,
                "young_mean": float(y.mean()),
                "old_mean": float(o.mean()),
                "delta_old_minus_young": delta,
                "cohens_d": float(d),
                "prior_weight": 1.0 + min(abs(d), 1.0),
            }
        )
    tab = pd.DataFrame(rows)
    if out_csv is not None:
        tab.to_csv(out_csv, index=False)
    print(
        f"age task prior [{source}]: n_tasks={len(tab)} "
        f"mean_|d|={tab['cohens_d'].abs().mean():.3f}",
        flush=True,
    )
    return tab


def apply_task_prior_to_graph(graph: dict, prior: pd.DataFrame | None) -> dict:
    """Scale gene→task edge weights by Young/Old age prior."""
    if prior is None or prior.empty:
        return graph
    wmap = dict(zip(prior["task"].astype(str), prior["prior_weight"].astype(float)))
    weights = torch.tensor(
        [float(wmap.get(t, 1.0)) for t in graph["tasks"]], dtype=torch.float32
    )
    out = dict(graph)
    out["tg_w"] = graph["tg_w"] * weights[graph["tg_dst"]]
    tw = out["tg_w"]
    denom = torch.zeros(len(graph["tasks"]), dtype=torch.float32)
    denom.scatter_add_(0, graph["tg_dst"], tw)
    out["tg_w"] = tw / denom[graph["tg_dst"]].clamp_min(1e-6)
    out["task_prior_weights"] = weights
    return out


def _rownorm(A: np.ndarray) -> np.ndarray:
    return A / np.clip(A.sum(1, keepdims=True), 1, None)


def _coo(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dst, src = (A > 0).nonzero(as_tuple=True)
    return dst.long(), src.long(), A[dst, src].float()


def build_deep_factor_graph(
    gene_names: list[str], task_names: list[str], tbg: pd.DataFrame
) -> dict:
    info = pd.read_csv(MOUSE_DB / "Task-Info.csv")
    info["Task"] = info["Task"].astype(str)
    info["Subsystem"] = info["Subsystem"].astype(str)
    info["System"] = info["System"].astype(str)
    task_to_sub = dict(zip(info["Task"], info["Subsystem"]))
    task_to_sys = dict(zip(info["Task"], info["System"]))
    sub_to_sys = dict(zip(info["Subsystem"], info["System"]))

    g_idx = {g: i for i, g in enumerate(gene_names)}
    tasks = [t for t in task_names if t in tbg.index]
    A_tg = np.zeros((len(tasks), len(gene_names)), dtype=np.float32)
    for ti, t in enumerate(tasks):
        row = tbg.loc[t]
        for g, v in row.items():
            if v > 0 and str(g) in g_idx:
                A_tg[ti, g_idx[str(g)]] = 1.0
    keep = A_tg.sum(1) > 0
    A_tg = A_tg[keep]
    tasks = [t for t, k in zip(tasks, keep) if k]

    subs = sorted({task_to_sub.get(t, "UNKNOWN") for t in tasks})
    s_idx = {s: i for i, s in enumerate(subs)}
    A_st = np.zeros((len(subs), len(tasks)), dtype=np.float32)
    for ti, t in enumerate(tasks):
        A_st[s_idx[task_to_sub.get(t, "UNKNOWN")], ti] = 1.0

    systems = sorted({sub_to_sys.get(s, "UNKNOWN") for s in subs})
    y_idx = {y: i for i, y in enumerate(systems)}
    A_ys = np.zeros((len(systems), len(subs)), dtype=np.float32)
    for s in subs:
        A_ys[y_idx[sub_to_sys.get(s, "UNKNOWN")], s_idx[s]] = 1.0

    A_tg_t = torch.tensor(_rownorm(A_tg), dtype=torch.float32)
    tg_dst, tg_src, tg_w = _coo(A_tg_t)
    graph = {
        "tasks": tasks,
        "subsystems": subs,
        "systems": systems,
        "n_genes": len(gene_names),
        "tg_dst": tg_dst,
        "tg_src": tg_src,
        "tg_w": tg_w,
        "A_sub_task": torch.tensor(_rownorm(A_st), dtype=torch.float32),
        "A_sys_sub": torch.tensor(_rownorm(A_ys), dtype=torch.float32),
    }
    torch.save({**graph, "gene_names": gene_names}, OUT / "deep_factor_graph.pt")
    pd.DataFrame(
        {
            "task": tasks,
            "subsystem": [task_to_sub.get(t, "UNKNOWN") for t in tasks],
            "system": [task_to_sys.get(t, "UNKNOWN") for t in tasks],
        }
    ).to_csv(OUT / "hierarchy_task_map.csv", index=False)
    print(
        f"deep hierarchy: genes={len(gene_names)} → tasks={len(tasks)} "
        f"→ subsystems={len(subs)} → systems={len(systems)} "
        f"(gene↔task edges={tg_w.numel()})",
        flush=True,
    )
    return graph


def _gated_msg_dense(
    A: torch.Tensor, h_src: torch.Tensor, h_dst: torch.Tensor
) -> torch.Tensor:
    """Dense gate for small bipartite levels (sub↔task, sys↔sub)."""
    k = F.normalize(h_src, dim=-1)
    q = F.normalize(h_dst, dim=-1)
    G = torch.sigmoid(torch.einsum("bdh,bsh->bds", q, k))
    A_dyn = A.unsqueeze(0) * G
    A_dyn = A_dyn / A_dyn.sum(-1, keepdim=True).clamp_min(1e-6)
    return torch.einsum("bds,bsh->bdh", A_dyn, h_src)


def _gated_msg_coo(
    dst: torch.Tensor,
    src: torch.Tensor,
    w: torch.Tensor,
    h_src: torch.Tensor,
    h_dst: torch.Tensor,
) -> torch.Tensor:
    """Sparse edge gate: only nonzero prior edges, O(B·E·H) not O(B·N_dst·N_src·H)."""
    B, n_dst, H = h_dst.shape
    k = F.normalize(h_src, dim=-1)
    q = F.normalize(h_dst, dim=-1)
    gate = torch.sigmoid((q[:, dst] * k[:, src]).sum(-1))
    ew = w * gate
    denom = h_dst.new_zeros(B, n_dst)
    denom.scatter_add_(1, dst.expand(B, -1), ew)
    ew = ew / denom.gather(1, dst.expand(B, -1)).clamp_min(1e-6)
    msg = h_dst.new_zeros(B, n_dst, H)
    msg.scatter_add_(
        1, dst.view(1, -1, 1).expand(B, -1, H), ew.unsqueeze(-1) * h_src[:, src]
    )
    return msg


def _scatter_coo(
    dst: torch.Tensor,
    src: torch.Tensor,
    w: torch.Tensor,
    h_src: torch.Tensor,
    n_dst: int,
) -> torch.Tensor:
    B, _, H = h_src.shape
    out = h_src.new_zeros(B, n_dst, H)
    out.scatter_add_(
        1, dst.view(1, -1, 1).expand(B, -1, H), w.view(1, -1, 1) * h_src[:, src]
    )
    return out


def _self_check_sparse_gate() -> None:
    torch.manual_seed(0)
    A = torch.zeros(3, 5)
    A[0, [0, 1]] = 0.5
    A[1, [1, 2, 3]] = 1.0 / 3
    A[2, 4] = 1.0
    dst, src, w = _coo(A)
    h_src = torch.randn(4, 5, 8)
    h_dst = torch.randn(4, 3, 8)
    dense = _gated_msg_dense(A, h_src, h_dst)
    sparse_m = _gated_msg_coo(dst, src, w, h_src, h_dst)
    assert torch.allclose(dense, sparse_m, atol=1e-5, rtol=1e-5), (
        (dense - sparse_m).abs().max()
    )
    td_dense = torch.einsum("tg,bth->bgh", A, h_dst)
    td_sparse = _scatter_coo(src, dst, w, h_dst, n_dst=5)
    assert torch.allclose(td_dense, td_sparse, atol=1e-5, rtol=1e-5)
    # prior reweights edges; stay row-normalized per task
    g = {
        "tasks": ["t0", "t1", "t2"],
        "tg_dst": dst,
        "tg_src": src,
        "tg_w": w.clone(),
    }
    prior = pd.DataFrame({"task": ["t0", "t1", "t2"], "prior_weight": [2.0, 1.0, 1.5]})
    gp = apply_task_prior_to_graph(g, prior)
    for ti in range(3):
        m = gp["tg_dst"] == ti
        if m.any():
            assert torch.isclose(gp["tg_w"][m].sum(), torch.tensor(1.0), atol=1e-5)
    m = DeepFactorGraph(
        {
            "tasks": ["t0", "t1", "t2"],
            "subsystems": ["s0"],
            "systems": ["y0"],
            "n_genes": 5,
            "tg_dst": dst,
            "tg_src": src,
            "tg_w": w,
            "A_sub_task": torch.ones(1, 3) / 3,
            "A_sys_sub": torch.ones(1, 1),
        },
        n_classes=2,
        hidden=8,
        steps=1,
    )
    w0 = m.edge_weights().detach().clone()
    for ti in range(3):
        mask = m.tg_dst == ti
        if mask.any():
            assert torch.isclose(w0[mask].sum(), torch.tensor(1.0), atol=1e-5)
    with torch.no_grad():
        m.tg_logit[0] += 2.0
    w1 = m.edge_weights()
    assert not torch.allclose(w0, w1)
    assert torch.isclose(w1[m.tg_dst == 0].sum(), torch.tensor(1.0), atol=1e-5)
    m.reset_head(4)
    assert m.head[-1].out_features == 4


class DeepFactorGraph(nn.Module):
    """gene ↔ task ↔ subsystem ↔ system; learnable gene→task edge weights + gates."""

    def __init__(
        self,
        graph: dict,
        hidden: int = HIDDEN,
        steps: int = STEPS,
        n_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.steps = steps
        self.register_buffer("tg_dst", graph["tg_dst"].long())
        self.register_buffer("tg_src", graph["tg_src"].long())
        # Task_by_Gene mask prior (row-normalized); init for learnable logits
        prior = graph["tg_w"].float().clamp_min(1e-6)
        self.register_buffer("tg_w_prior", prior)
        self.tg_logit = nn.Parameter(torch.log(prior))
        self.register_buffer("A_st", graph["A_sub_task"])
        self.register_buffer("A_ys", graph["A_sys_sub"])
        n_t, n_s, n_y = (
            len(graph["tasks"]),
            len(graph["subsystems"]),
            len(graph["systems"]),
        )
        n_g = int(graph.get("n_genes", int(graph["tg_src"].max().item()) + 1))

        self.gene_in = nn.Linear(1, hidden)
        # per-gene input scale; exp(0)=1 at init
        self.gene_log_scale = nn.Parameter(torch.zeros(n_g))
        self.task_init = nn.Parameter(torch.zeros(1, n_t, hidden))
        self.sub_init = nn.Parameter(torch.zeros(1, n_s, hidden))
        self.sys_init = nn.Parameter(torch.zeros(1, n_y, hidden))

        self.upd_g = nn.GRUCell(hidden, hidden)
        self.upd_t = nn.GRUCell(hidden, hidden)
        self.upd_s = nn.GRUCell(hidden, hidden)
        self.upd_y = nn.GRUCell(hidden, hidden)

        self.attn_t = nn.Linear(hidden, 1)
        self.attn_s = nn.Linear(hidden, 1)
        self.attn_y = nn.Linear(hidden, 1)
        self.head = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
        self._hidden = hidden
        self._dropout = dropout
        self._n_tasks = n_t

    def edge_weights(self) -> torch.Tensor:
        """Learnable gene→task weights; softplus + renorm per task."""
        w = F.softplus(self.tg_logit)
        denom = w.new_zeros(self._n_tasks)
        denom.scatter_add_(0, self.tg_dst, w)
        return w / denom[self.tg_dst].clamp_min(1e-6)

    def reset_head(self, n_classes: int) -> None:
        h, d = self._hidden, self._dropout
        self.head = nn.Sequential(
            nn.Linear(3 * h, h),
            nn.ReLU(),
            nn.Dropout(d),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Dropout(d),
            nn.Linear(h, n_classes),
        )

    def forward(self, x_genes: torch.Tensor, return_attn: bool = False):
        B = x_genes.size(0)
        scale = torch.exp(self.gene_log_scale)
        h_g = self.gene_in((x_genes * scale).unsqueeze(-1))
        h_t = self.task_init.expand(B, -1, -1).contiguous()
        h_s = self.sub_init.expand(B, -1, -1).contiguous()
        h_y = self.sys_init.expand(B, -1, -1).contiguous()
        tg_w = self.edge_weights()

        for _ in range(self.steps):
            msg_t = _gated_msg_coo(self.tg_dst, self.tg_src, tg_w, h_g, h_t)
            h_t = self.upd_t(
                msg_t.reshape(-1, msg_t.size(-1)), h_t.reshape(-1, h_t.size(-1))
            ).view_as(h_t)

            msg_s = _gated_msg_dense(self.A_st, h_t, h_s)
            h_s = self.upd_s(
                msg_s.reshape(-1, msg_s.size(-1)), h_s.reshape(-1, h_s.size(-1))
            ).view_as(h_s)

            msg_y = _gated_msg_dense(self.A_ys, h_s, h_y)
            h_y = self.upd_y(
                msg_y.reshape(-1, msg_y.size(-1)), h_y.reshape(-1, h_y.size(-1))
            ).view_as(h_y)

            h_s = h_s + 0.5 * torch.einsum("ys,byh->bsh", self.A_ys, h_y)
            h_t = h_t + 0.5 * torch.einsum("st,bsh->bth", self.A_st, h_s)
            msg_g = _scatter_coo(
                self.tg_src, self.tg_dst, tg_w, h_t, n_dst=h_g.size(1)
            )
            h_g = self.upd_g(
                msg_g.reshape(-1, msg_g.size(-1)), h_g.reshape(-1, h_g.size(-1))
            ).view_as(h_g)

        a_t = torch.softmax(self.attn_t(h_t).squeeze(-1), dim=-1)
        a_s = torch.softmax(self.attn_s(h_s).squeeze(-1), dim=-1)
        a_y = torch.softmax(self.attn_y(h_y).squeeze(-1), dim=-1)
        pooled = torch.cat(
            [
                torch.einsum("bt,bth->bh", a_t, h_t),
                torch.einsum("bs,bsh->bh", a_s, h_s),
                torch.einsum("by,byh->bh", a_y, h_y),
            ],
            dim=-1,
        )
        logits = self.head(pooled)
        if return_attn:
            return logits, a_t
        return logits


def _export_learned_gene_weights(
    raw: DeepFactorGraph, graph: dict, gene_names: list[str], out_csv: Path
) -> None:
    """Write learned gene→task edge weights (+ per-gene scale)."""
    w = raw.edge_weights().detach().cpu().numpy()
    prior = raw.tg_w_prior.detach().cpu().numpy()
    scale = torch.exp(raw.gene_log_scale).detach().cpu().numpy()
    rows = []
    for e, (ti, gi) in enumerate(
        zip(raw.tg_dst.cpu().numpy(), raw.tg_src.cpu().numpy())
    ):
        rows.append(
            {
                "task": graph["tasks"][int(ti)],
                "gene": gene_names[int(gi)],
                "weight": float(w[e]),
                "prior_weight": float(prior[e]),
                "gene_scale": float(scale[int(gi)]),
                "delta_vs_prior": float(w[e] - prior[e]),
            }
        )
    pd.DataFrame(rows).sort_values(
        "delta_vs_prior", key=lambda s: s.abs(), ascending=False
    ).to_csv(out_csv, index=False)


def factorial_contrasts(adata: ad.AnnData) -> pd.DataFrame:
    """Tet2 × IL1 axis contrasts; stratified by age_cohort when present."""
    tasks = pd.DataFrame(
        np.asarray(adata.obsm["metabolic_tasks"], dtype=np.float64),
        index=adata.obs_names,
        columns=list(adata.uns["metabolic_task_names"]),
    )
    axes = {ax: [t for t in ts if t in tasks.columns] for ax, ts in AXIS_TASKS.items()}
    axis_df = pd.DataFrame(
        {ax: tasks[ts].mean(axis=1) for ax, ts in axes.items() if ts}
    )
    axis_df["glycolysis_minus_OXPHOS"] = axis_df["glycolysis"] - axis_df["OXPHOS_TCA"]
    has_age = "age_cohort" in adata.obs.columns
    meta_cols = ["sample_name", "genotype", "treatment"] + (
        ["age_cohort"] if has_age else []
    )
    meta = adata.obs[meta_cols].copy()
    for c in meta_cols:
        meta[c] = meta[c].astype(str)
    gcols = ["sample_name", "genotype", "treatment"] + (
        ["age_cohort"] if has_age else []
    )
    pb = (
        axis_df.join(meta)
        .groupby(gcols, observed=True)
        .mean(numeric_only=True)
        .reset_index()
    )
    strata = (
        [("pooled", pb)]
        + (
            [(age, pb[pb["age_cohort"] == age]) for age in sorted(pb["age_cohort"].unique())]
            if has_age
            else []
        )
    )
    rows = []
    for age_label, sub in strata:
        if sub.empty:
            continue
        for feat in axis_df.columns:

            def arm(g, t, feat=feat, sub=sub):
                return sub[(sub["genotype"] == g) & (sub["treatment"] == t)][
                    feat
                ].to_numpy()

            wt_v, wt_i = arm("WT", "vehicle"), arm("WT", "IL1")
            ko_v, ko_i = arm("Tet2", "vehicle"), arm("Tet2", "IL1")
            if min(map(len, (wt_v, wt_i, ko_v, ko_i))) < 1:
                continue
            d_il1 = float(ko_i.mean() - wt_i.mean())
            d_veh = float(ko_v.mean() - wt_v.mean())
            rows.append(
                {
                    "age_cohort": age_label,
                    "feature": feat,
                    "WT_vehicle": float(wt_v.mean()),
                    "WT_IL1": float(wt_i.mean()),
                    "Tet2_vehicle": float(ko_v.mean()),
                    "Tet2_IL1": float(ko_i.mean()),
                    "Tet2_effect_vehicle": d_veh,
                    "Tet2_effect_IL1": d_il1,
                    "IL1_effect_WT": float(wt_i.mean() - wt_v.mean()),
                    "IL1_effect_Tet2": float(ko_i.mean() - ko_v.mean()),
                    "interaction_Tet2xIL1": d_il1 - d_veh,
                }
            )
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "factorial_axis_interaction.csv", index=False)
    return tab


def gene_axis_hits(adata: ad.AnnData, tbg: pd.DataFrame, top: int = 10) -> dict:
    if "gene_scores" not in adata.layers:
        return {}
    gs = adata.layers["gene_scores"]
    G = pd.DataFrame(
        gs.toarray() if sparse.issparse(gs) else np.asarray(gs),
        index=adata.obs_names,
        columns=adata.var_names.astype(str),
    )
    sub = adata.obs["treatment"].astype(str).eq("IL1")
    out = {}
    for ax, ts in AXIS_TASKS.items():
        genes: set[str] = set()
        for t in ts:
            if t in tbg.index:
                row = tbg.loc[t]
                genes.update(row[row > 0].index.astype(str))
        present = [g for g in genes if g in G.columns]
        if not present:
            continue
        meta = adata.obs.loc[sub, ["sample_name", "genotype"]]
        pb = (
            G.loc[sub, present]
            .join(meta)
            .groupby(["sample_name", "genotype"], observed=True)
            .mean()
            .reset_index()
        )
        geno = pb["genotype"].astype(str)
        wt_mask, ko_mask = geno.eq("WT"), geno.eq("Tet2")
        rows = []
        for g in present:
            wt, ko = pb.loc[wt_mask, g].to_numpy(), pb.loc[ko_mask, g].to_numpy()
            if len(wt) < 2 or len(ko) < 2:
                continue
            delta = float(ko.mean() - wt.mean())
            pooled = np.sqrt((wt.var(ddof=1) + ko.var(ddof=1)) / 2)
            d = delta / pooled if pooled > 0 else np.nan
            rows.append({"gene": g, "delta": delta, "cohens_d": d})
        out[ax] = sorted(rows, key=lambda r: abs(r["cohens_d"] or 0), reverse=True)[
            :top
        ]
    pd.DataFrame([{**r, "axis": ax} for ax, rows in out.items() for r in rows]).to_csv(
        OUT / "axis_gene_hits_il1.csv", index=False
    )
    return out


def _labels_from_condition(obs: pd.DataFrame, classes: list[str]) -> np.ndarray:
    if "condition" in obs:
        cond = obs["condition"].astype(str)
    else:
        cond = obs["genotype"].astype(str) + "_" + obs["treatment"].astype(str)
    y = cond.map({c: i for i, c in enumerate(classes)}).to_numpy()
    if np.any(pd.isna(y)):
        bad = cond[pd.isna(y)].unique().tolist()
        raise SystemExit(f"unknown condition labels for {classes}: {bad[:8]}")
    return y.astype(int)


def _train_loop(
    model: nn.Module,
    raw: DeepFactorGraph,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    classes: list[str],
    epochs: int,
    device: torch.device,
    device_ids: list[int],
    tag: str,
) -> dict:
    n_cls = len(classes)
    n_gpu = max(len(device_ids), 1)
    tr, te = next(
        GroupShuffleSplit(1, test_size=0.25, random_state=0).split(X, y, groups)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    # Host pin + per-batch H2D so DataParallel can scatter across all GPUs
    Xt = torch.from_numpy(np.ascontiguousarray(X)).pin_memory()
    yt = torch.from_numpy(np.ascontiguousarray(y.astype(np.int64))).pin_memory()
    bs = BATCH_PER_GPU * n_gpu
    best, best_state, history = -1.0, None, []

    def _batch_to_device(idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        xb = Xt[idx].to(device, non_blocking=True)
        yb = yt[idx].to(device, non_blocking=True)
        return xb, yb

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(tr)
        loss_sum = 0.0
        n_seen = 0
        for i in range(0, len(perm), bs):
            idx = perm[i : i + bs]
            # DataParallel needs ≥1 sample per GPU
            if len(idx) < n_gpu:
                continue
            xb, yb = _batch_to_device(idx)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach()) * len(idx)
            n_seen += len(idx)
        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                probs = []
                for i in range(0, len(te), bs):
                    idx = te[i : i + bs]
                    if len(idx) < n_gpu:
                        # evaluate remainder on primary GPU without DP scatter
                        xb = Xt[idx].to(device, non_blocking=True)
                        logits = raw(xb)
                    else:
                        xb, _ = _batch_to_device(idx)
                        logits = model(xb)
                    probs.append(logits.softmax(1).cpu().numpy())
                P = np.vstack(probs)
                yt_te = y[te]
                aucs = [
                    roc_auc_score((yt_te == c).astype(int), P[:, c])
                    for c in range(n_cls)
                    if len(np.unique((yt_te == c).astype(int))) > 1
                ]
                auc = float(np.mean(aucs)) if aucs else float("nan")
                acc = float((P.argmax(1) == yt_te).mean())
            train_n = max(n_seen, 1)
            history.append(
                {
                    "epoch": epoch,
                    "loss": loss_sum / train_n,
                    "macro_auroc": auc,
                    "acc": acc,
                }
            )
            print(
                f"[{tag}] epoch {epoch:03d} loss={loss_sum / train_n:.4f} "
                f"macroAUROC={auc:.4f} acc={acc:.4f} bs={bs} gpus={n_gpu}",
                flush=True,
            )
            if auc == auc and auc > best:
                best = auc
                best_state = {
                    k: v.detach().cpu().clone() for k, v in raw.state_dict().items()
                }

    if best_state:
        raw.load_state_dict(best_state)
    return {
        "best_macro_auroc": best,
        "history": history,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "test_idx": te,
        "best_state": best_state,
        "batch_size": bs,
        "n_gpu": n_gpu,
    }


def _cargo_attention_report(
    raw: DeepFactorGraph,
    graph: dict,
    X: np.ndarray,
    y: np.ndarray,
    te: np.ndarray,
    classes: list[str],
    device: torch.device,
    out_csv: Path,
) -> dict:
    Xt = torch.from_numpy(np.ascontiguousarray(X)).pin_memory()
    bs = BATCH_PER_GPU
    attn_by = {c: [] for c in classes}
    raw.eval()
    with torch.no_grad():
        for i in range(0, len(te), bs):
            idx = te[i : i + bs]
            xb = Xt[idx].to(device, non_blocking=True)
            _, a_t = raw(xb, return_attn=True)
            a_t = a_t.cpu().numpy()
            for j, ci in enumerate(y[idx]):
                attn_by[classes[ci]].append(a_t[j])
    mean_a = {c: np.mean(v, 0) for c, v in attn_by.items() if v}

    def _inter(prefix: str) -> np.ndarray | None:
        keys = [
            f"{prefix}_WT_vehicle",
            f"{prefix}_WT_IL1",
            f"{prefix}_Tet2_vehicle",
            f"{prefix}_Tet2_IL1",
        ]
        if not all(k in mean_a for k in keys):
            return None
        return (mean_a[keys[3]] - mean_a[keys[1]]) - (
            mean_a[keys[2]] - mean_a[keys[0]]
        )

    young_inter = _inter("Young")
    old_inter = _inter("Old")

    if young_inter is not None and old_inter is not None:
        inter = 0.5 * (young_inter + old_inter)
        inter_name = "interaction_attn_mean_YoungOld"
    elif young_inter is not None:
        inter = young_inter
        inter_name = "interaction_attn_Young"
    elif old_inter is not None:
        inter = old_inter
        inter_name = "interaction_attn_Old"
    else:
        # fall back: any available class means difference is undefined → zeros
        inter = np.zeros(len(graph["tasks"]), dtype=np.float64)
        inter_name = "interaction_attn"

    cols = {"factor": graph["tasks"], inter_name: inter}
    if young_inter is not None:
        cols["interaction_attn_Young"] = young_inter
    if old_inter is not None:
        cols["interaction_attn_Old"] = old_inter
    for c, v in mean_a.items():
        cols[f"attn_{c}"] = v
    factor_imp = pd.DataFrame(cols).sort_values(
        inter_name, key=lambda s: s.abs(), ascending=False
    )
    factor_imp.to_csv(out_csv, index=False)
    axis_attn = {
        ax: float(factor_imp.loc[factor_imp["factor"].isin(ts), inter_name].sum())
        for ax, ts in AXIS_TASKS.items()
    }
    return {
        "interaction_column": inter_name,
        "axis_interaction_attention_sum": axis_attn,
        "top_interaction_factors": factor_imp.head(15).to_dict(orient="records"),
    }


_CUDA_SETUP: tuple[torch.device, list[int]] | None = None


def _cuda_devices() -> tuple[torch.device, list[int]]:
    """Require CUDA; return primary device + all visible GPU ids (cached)."""
    global _CUDA_SETUP
    if _CUDA_SETUP is not None:
        return _CUDA_SETUP
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise SystemExit(
            "CUDA required for chip_metabolic training. "
            "Check drivers / unset CUDA_VISIBLE_DEVICES to expose all GPUs."
        )
    ids = list(range(torch.cuda.device_count()))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    names = [torch.cuda.get_device_name(i) for i in ids]
    print(
        f"CUDA devices ({len(ids)}): "
        + ", ".join(f"{i}:{n}" for i, n in zip(ids, names)),
        flush=True,
    )
    _CUDA_SETUP = (torch.device("cuda:0"), ids)
    return _CUDA_SETUP


def _wrap_data_parallel(model: DeepFactorGraph, device_ids: list[int]) -> nn.Module:
    model = model.cuda(device_ids[0])
    if len(device_ids) > 1:
        return nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
    return model


def _raw_module(model: nn.Module) -> DeepFactorGraph:
    raw = model.module if isinstance(model, nn.DataParallel) else model
    assert isinstance(raw, DeepFactorGraph)
    return raw


def train_factorial(
    adata: ad.AnnData,
    tbg: pd.DataFrame,
    classes: list[str] | None = None,
    use_age_prior: bool = False,
    tag: str = "young_old",
) -> dict:
    """Train DeepFactorGraph on graph-aligned factorial (8-class or young-only)."""
    classes = list(classes or CLASSES)
    if "gene_scores" not in adata.layers:
        raise SystemExit("gene_scores missing")
    X = _to_dense(adata.layers["gene_scores"]).astype(np.float32)
    if "age_cohort" in adata.obs.columns and (
        adata.obs["age_cohort"].astype(str) == "Young"
    ).any():
        ym = adata.obs["age_cohort"].astype(str).eq("Young").to_numpy()
        mu = X[ym].mean(0, keepdims=True)
        sd = X[ym].std(0, keepdims=True) + 1e-6
    else:
        mu = X.mean(0, keepdims=True)
        sd = X.std(0, keepdims=True) + 1e-6
    X = (X - mu) / sd
    gene_names = list(adata.var_names.astype(str))
    graph = build_deep_factor_graph(
        gene_names, list(adata.uns["metabolic_task_names"]), tbg
    )
    if use_age_prior and "age_cohort" in adata.obs.columns:
        prior = young_old_task_prior(
            adata, "Young_vs_Old", OUT / "young_vs_old_task_prior.csv"
        )
        graph = apply_task_prior_to_graph(graph, prior)
        print("applied Young/Old task-edge prior", flush=True)

    device, device_ids = _cuda_devices()
    y = _labels_from_condition(adata.obs, classes)
    groups = adata.obs["sample_name"].astype(str).to_numpy()
    model = _wrap_data_parallel(
        DeepFactorGraph(graph, n_classes=len(classes)), device_ids
    )
    raw = _raw_module(model)
    metrics = _train_loop(
        model,
        raw,
        X,
        y,
        groups,
        classes,
        EPOCHS,
        device,
        device_ids,
        tag=tag,
    )
    attn = _cargo_attention_report(
        raw,
        graph,
        X,
        y,
        metrics["test_idx"],
        classes,
        device,
        OUT / f"deep_factor_interaction_attention_{tag}.csv",
    )
    w_csv = OUT / f"learned_gene_task_weights_{tag}.csv"
    _export_learned_gene_weights(raw, graph, gene_names, w_csv)
    torch.save(
        {
            "tag": tag,
            "classes": classes,
            "use_age_prior": use_age_prior,
            "learnable_gene_weights": True,
            "state": metrics["best_state"],
            "n_gpu": len(device_ids),
            "device_ids": device_ids,
            "hierarchy": {
                "n_genes": len(gene_names),
                "n_tasks": len(graph["tasks"]),
                "n_subs": len(graph["subsystems"]),
                "n_systems": len(graph["systems"]),
                "systems": graph["systems"],
            },
            "steps": STEPS,
            "mu": mu,
            "sd": sd,
            "gene_names": gene_names,
        },
        OUT / f"deep_fgnn_best_{tag}.pt",
    )
    return {
        "tag": tag,
        "classes": classes,
        "use_age_prior": use_age_prior,
        "learnable_gene_weights": True,
        "learned_weights_csv": str(w_csv),
        "n_gpu": len(device_ids),
        "best_macro_auroc": metrics["best_macro_auroc"],
        "history": metrics["history"],
        "n_train": metrics["n_train"],
        "n_test": metrics["n_test"],
        "batch_size": metrics["batch_size"],
        "depth_steps": STEPS,
        "hierarchy": ["gene", "task", "subsystem", "system"],
        "n_systems": len(graph["systems"]),
        "n_subsystems": len(graph["subsystems"]),
        "n_tasks": len(graph["tasks"]),
        "n_rows": int(adata.n_obs),
        "n_genes": len(gene_names),
        **attn,
    }


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--young-only",
        action="store_true",
        help="McClatchy Young 4-class ablation (no Caiado Old).",
    )
    p.add_argument(
        "--age-prior",
        action="store_true",
        help="Scale gene→task edges by Young vs Old task Cohen's d prior.",
    )
    args = p.parse_args(argv)

    _self_check_sparse_gate()
    tbg = pd.read_csv(MOUSE_DB / "Task_by_Gene.csv", index_col=0)

    if args.young_only:
        adata = _stamp_mcclatchy_young(load_or_build_factorial())
        classes = YOUNG_ONLY_CLASSES
        tag = "young_only"
    else:
        adata = load_or_build_young_old_factorial(tbg)
        classes = CLASSES
        tag = "young_old"

    print(
        adata.obs.groupby(
            ["age_cohort", "treatment", "genotype"], observed=True
        ).size(),
        flush=True,
    )
    print("conditions:", adata.obs["condition"].value_counts().to_dict(), flush=True)

    contrasts = factorial_contrasts(adata)
    print("\n=== module-axis interaction (by age_cohort) ===", flush=True)
    print(contrasts.to_string(index=False))

    hits = gene_axis_hits(adata, tbg)
    use_prior = bool(args.age_prior) and (not args.young_only)
    model = train_factorial(
        adata, tbg, classes=classes, use_age_prior=use_prior, tag=tag
    )

    payload = {
        "pipeline": "chip_metabolic.py",
        "question": "Tet2 × IL-1 × age_cohort on glycosylation / glycolysis / OXPHOS",
        "age_context": {
            "Young": "GSE209994 McClatchy scRNA Tet2×IL-1β",
            "Old": (
                "PRJEB56666 Caiado bulk HSC Tet2×IL-1α (~6–9 mo adult proxy; "
                "not chronological 18–24 mo aging RNA)"
            ),
            "alignment": "shared deep factor graph (intersect genes + Task_by_Gene)",
            "young_only": bool(args.young_only),
            "age_prior": use_prior,
            "learnable_gene_weights": True,
        },
        "classes": classes,
        "n_rows": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "factorial_contrasts": contrasts.to_dict(orient="records"),
        "gene_hits_il1": hits,
        "deep_fgnn": model,
    }
    (OUT / "chip_metabolic_summary.json").write_text(
        json.dumps(payload, indent=2, default=str)
    )
    print(json.dumps({"deep_fgnn": model}, indent=2, default=str))


if __name__ == "__main__":
    main()
