#!/usr/bin/env python3
from __future__ import annotations

import importlib
import os
import sys
import types
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache_bm")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from torch_geometric.utils import scatter
import rapids_singlecell as rsc


BONE = Path("/cis/net/r41/data/iessien1/bone")
RESULTS = Path("/cis/net/r41/data/iessien1/bone_marrow_results")
MOUSE_DB = BONE / "sccellfie" / "mus_musculus"
HUMAN_DB = BONE / "sccellfie" / "homo_sapiens"
OUT = RESULTS / "chip_metabolic_graph"
QC = Path(
    "/cis/home/iessien1/Documents/bone_marrow/data/GSE209994/processed/"
    "gse209994_qc_preprocessed.h5ad"
)
HUMAN_QC = Path(
    "/cis/home/iessien1/Documents/bone_marrow/data/GSE285379/processed/"
    "gse285379_qc_preprocessed.h5ad"
)

LINEAGE = "HSPC"
KEEP_GENOTYPE = ("WT", "Tet2_KO")
KEEP_TREATMENT = ("vehicle", "IL1b")
GENOTYPES = ("WT", "Tet2")
TREATMENTS = ("vehicle", "IL1")
HUMAN_TREATMENTS = ("CTRL", "LPS")
ARMS = tuple(f"{g}_{t}" for g in GENOTYPES for t in TREATMENTS)
MOUSE_TMAP = {"vehicle": "vehicle", "IL1b": "IL1"}
HUMAN_TMAP = {"CTRL": "CTRL", "LPS": "LPS"}


LEVELS = ("task", "subsystem", "system")
EPOCHS = 20
HIDDEN = 96
STEPS = 10
BATCH_SIZE = 256

AXIS_TASKS: dict[str, list[str]] = {
    "glycolysis": [
        "ATP generation from glucose (hypoxic conditions) - glycolysis",
    ],
    "OXPHOS_TCA": [
        "Oxidative phosphorylation via NADH-coenzyme Q oxidoreductase (COMPLEX I)",
        "Oxidative phosphorylation via succinate-coenzyme Q oxidoreductase (COMPLEX II)",
        "Krebs cycle - NADH generation",
        "Krebs cycle - oxidative decarboxylation of pyruvate",
    ],
    "PPP": [
        "Synthesis of fructose-6-phosphate from erythrose-4-phosphate (HMP shunt)",
        "Synthesis of ribose-5-phosphate",
    ],
}
HYPOTHESIS_TASKS = tuple(t for ts in AXIS_TASKS.values() for t in ts)
AXIS_TITLES = {
    "glycolysis": "Glycolysis",
    "OXPHOS_TCA": "OXPHOS/TCA",
    "PPP": "PPP",
}
TASK_SHORT = {
    "ATP generation from glucose (hypoxic conditions) - glycolysis": "Glycolysis",
    "Oxidative phosphorylation via NADH-coenzyme Q oxidoreductase (COMPLEX I)": "Complex I",
    "Oxidative phosphorylation via succinate-coenzyme Q oxidoreductase (COMPLEX II)": "Complex II",
    "Krebs cycle - NADH generation": "TCA NADH",
    "Krebs cycle - oxidative decarboxylation of pyruvate": "Pyruvate",
    "Synthesis of fructose-6-phosphate from erythrose-4-phosphate (HMP shunt)": "HMP / F6P",
    "Synthesis of ribose-5-phosphate": "Ribose-5-P",
}
AXIS_READOUT = {
    "glycolysis": "lactate",
    "OXPHOS_TCA": "succinate, α-KG",
    "PPP": "NADPH, pentose",
}
AXIS_LABELS = {
    "glycolysis": {"pkm", "gapdh", "gapdhs", "hk1", "pfkl", "aldoa", "eno1"},
    "OXPHOS_TCA": {"sdha", "ogdh", "idh3a", "suclg1", "cs", "fh1", "fh"},
    "PPP": {"g6pdx", "g6pd", "pgd", "tkt", "rpia"},
}
_MT_ND = {"nd1", "nd2", "nd3", "nd4", "nd4l", "nd5", "nd6"}

_CUDA: torch.device | None = None


def _sccellfie_root():
    local = (
        Path(__file__).resolve().parent / ".venv/lib/python3.10/site-packages/sccellfie"
    )
    return (
        local
        if local.exists()
        else Path(sys.prefix) / "lib/python3.10/site-packages/sccellfie"
    )


def _install_sccellfie_without_spatial():
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
    try:
        from sccellfie.sccellfie_pipeline import run_sccellfie_pipeline
        from sccellfie.preprocessing.prepare_inputs import CORRECT_GENES
    except ImportError:
        _install_sccellfie_without_spatial()
        run_sccellfie_pipeline = importlib.import_module(
            "sccellfie.sccellfie_pipeline"
        ).run_sccellfie_pipeline
        CORRECT_GENES = importlib.import_module(
            "sccellfie.preprocessing.prepare_inputs"
        ).CORRECT_GENES
    return run_sccellfie_pipeline, CORRECT_GENES


def _to_dense(X):
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _collapse_corrected_genes(adata: ad.AnnData, mapping: dict[str, str]):
    names = np.asarray(adata.var_names.astype(str), dtype=object)
    if mapping:
        names = np.array([mapping.get(n, n) for n in names], dtype=object)
    uniq, first, inv = np.unique(names, return_index=True, return_inverse=True)
    if len(uniq) == len(names):
        adata.var_names = names.astype(str)
        return adata
    n_g, n_u = len(names), len(uniq)
    G = sparse.csc_matrix(
        (np.ones(n_g, dtype=np.float64), (np.arange(n_g), inv)),
        shape=(n_g, n_u),
    )

    def coll(M):
        S = M if sparse.issparse(M) else sparse.csr_matrix(M)
        return S @ G

    var = adata.var.iloc[first].copy()
    var.index = pd.Index(uniq.astype(str), name=adata.var.index.name)
    out = ad.AnnData(
        X=coll(adata.X),
        obs=adata.obs.copy(),
        var=var,
        layers={k: coll(v) for k, v in adata.layers.items()},
        uns=adata.uns.copy(),
        obsm=adata.obsm.copy(),
        obsp=adata.obsp.copy(),
    )
    print(f"collapsed {n_g - n_u} duplicate genes after alias rename", flush=True)
    return out


def _ensure_counts(adata: ad.AnnData):
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    if "n_counts" not in adata.obs:
        adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
    return adata


def _positive_genes(tbg: pd.DataFrame, task: str) -> list[str]:
    row = tbg.loc[task]
    return [str(g) for g in row[row > 0].index]


def _run_sccellfie(adata: ad.AnnData, *, organism: str, db: Path):
    adata = adata.copy()
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.normalize_total(adata, target_sum=1e4)
    rsc.pp.log1p(adata)
    rsc.pp.highly_variable_genes(adata, n_top_genes=2000)
    rsc.pp.pca(adata, n_comps=30)
    rsc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    rsc.get.anndata_to_CPU(adata)
    adata = _ensure_counts(adata)
    run, correct = _import_sccellfie_pipeline()
    adata = _collapse_corrected_genes(adata, correct.get(organism.lower(), {}))
    out = run(
        adata,
        organism=organism,
        sccellfie_data_folder=str(db),
        n_counts_col="n_counts",
        neighbors_key="neighbors",
        n_neighbors=15,
        smooth_cells=True,
        alpha=0.33,
        chunk_size=4000,
        save_folder=None,
        compute_ablation_impact=False,
        verbose=True,
    )
    adata = out["adata"]
    adata.layers["gene_scores"] = (
        adata.layers["gene_scores"] if "gene_scores" in adata.layers else adata.X
    )
    return adata


def _remap_obs(s: pd.Series, mapping: dict[str, str], label: str):
    out = s.astype(str).map(mapping)
    if out.isna().any():
        bad = sorted(set(s.astype(str)[out.isna()]))
        raise ValueError(f"unmapped {label} {bad}")
    return out


def _stamp_arms(adata: ad.AnnData, tmap: dict[str, str]):
    adata.obs["genotype"] = _remap_obs(
        adata.obs["genotype"],
        {"WT": "WT", "Tet2_KO": "Tet2"},
        "genotype",
    )
    adata.obs["treatment"] = _remap_obs(
        adata.obs["treatment"],
        tmap,
        "treatment",
    )
    adata.obs["arm"] = (
        adata.obs["genotype"].astype(str) + "_" + adata.obs["treatment"].astype(str)
    )
    return adata


def _fix_sample_names(adata: ad.AnnData):
    sn = (
        adata.obs["sample_name"].astype(str)
        if "sample_name" in adata.obs
        else adata.obs_names.astype(str)
    )
    adata.obs["sample_name"] = np.where(
        sn.isin(["nan", "None", ""]), adata.obs_names.astype(str), sn
    )
    return adata


def load_cells():
    return _sccellfie_hspc(human=False)


def load_human_hspc():
    return _sccellfie_hspc(human=True)


def _sccellfie_hspc(*, human: bool) -> ad.AnnData:
    adata = _load_hspc_counts(human=human)
    tag = "human HSPC" if human else "cells"
    nkey = "libraries" if human else "mice"
    print(
        f"QC n={adata.n_obs} "
        f"{adata.obs['genotype'].value_counts().to_dict()} "
        f"{adata.obs['treatment'].value_counts().to_dict()}",
        flush=True,
    )
    if human:
        adata = _run_sccellfie(adata, organism="human", db=HUMAN_DB)
    else:
        adata = _run_sccellfie(adata, organism="mouse", db=MOUSE_DB)
    print(
        f"{tag} n={adata.n_obs} {nkey}={adata.obs['sample_name'].nunique()} "
        f"arms={adata.obs['arm'].value_counts().to_dict()}",
        flush=True,
    )
    return adata


def _rownorm(A: np.ndarray):
    return A / np.clip(A.sum(1, keepdims=True), 1, None)


def _parent_adj(children: list[str], parent_of: dict[str, str]):
    parents = sorted({parent_of.get(c, "UNKNOWN") for c in children})
    p_idx = {p: i for i, p in enumerate(parents)}
    A = np.zeros((len(parents), len(children)), dtype=np.float32)
    for i, c in enumerate(children):
        A[p_idx[parent_of.get(c, "UNKNOWN")], i] = 1.0
    return parents, torch.tensor(_rownorm(A), dtype=torch.float32)


@dataclass(frozen=True)
class HypothesisGraph:
    tasks: tuple[str, ...]
    subsystems: tuple[str, ...]
    systems: tuple[str, ...]
    n_genes: int
    A_tg: torch.Tensor
    A_ts: torch.Tensor
    A_sy: torch.Tensor


def build_hypothesis_graph(
    gene_names: list[str], tbg: pd.DataFrame, db: Path = MOUSE_DB
) -> HypothesisGraph:
    info = pd.read_csv(db / "Task-Info.csv")
    info["Task"] = info["Task"].astype(str)
    task_to_sub = dict(zip(info["Task"], info["Subsystem"].astype(str)))
    sub_to_sys = dict(zip(info["Subsystem"].astype(str), info["System"].astype(str)))
    g_idx = {g: i for i, g in enumerate(gene_names)}
    edges: list[tuple[int, int]] = []
    kept: list[str] = []
    for t in HYPOTHESIS_TASKS:
        if t not in tbg.index:
            continue
        pairs = [(len(kept), g_idx[g]) for g in _positive_genes(tbg, t) if g in g_idx]
        if not pairs:
            continue
        edges.extend(pairs)
        kept.append(t)
    if not edges:
        return None
    n_g = len(gene_names)
    A_tg = np.zeros((len(kept), n_g), dtype=np.float32)
    for ti, gi in edges:
        A_tg[ti, gi] = 1.0
    A_tg = _rownorm(A_tg)
    subs, A_ts = _parent_adj(kept, task_to_sub)
    systems, A_sy = _parent_adj(subs, sub_to_sys)
    print(
        f"VNN graph: genes={n_g} tasks={len(kept)} "
        f"subs={len(subs)} systems={len(systems)}",
        flush=True,
    )
    return HypothesisGraph(
        tasks=tuple(kept),
        subsystems=tuple(subs),
        systems=tuple(systems),
        n_genes=n_g,
        A_tg=torch.tensor(A_tg, dtype=torch.float32),
        A_ts=A_ts,
        A_sy=A_sy,
    )


def _gated(x_src, x_dst, A):
    k = F.normalize(x_src, dim=-1)
    q = F.normalize(x_dst, dim=-1)
    raw = (
        A.unsqueeze(0) * torch.sigmoid(torch.einsum("bdh,bsh->bds", q, k))
    ).clamp_min(1e-8)
    logits = raw.log().masked_fill(A.unsqueeze(0) == 0, torch.finfo(raw.dtype).min)
    return torch.einsum("bds,bsh->bdh", torch.softmax(logits, dim=-1), x_src)


def _weighted(x_src, A, scale: float = 1.0):
    return scale * torch.einsum("ds,bsh->bdh", A, x_src)


def _gru(cell, msg, h):
    return cell(msg.reshape(-1, msg.size(-1)), h.reshape(-1, h.size(-1))).view_as(h)


class MetabolicVNN(nn.Module):
    def __init__(
        self, graph: HypothesisGraph, hidden: int = HIDDEN, steps: int = STEPS
    ):
        super().__init__()
        self.steps = steps
        self.register_buffer("mask_tg", (graph.A_tg > 0).to(torch.float32))
        self.register_buffer("A_ts", graph.A_ts.float())
        self.register_buffer("A_sy", graph.A_sy.float())
        prior = graph.A_tg.float().clamp_min(1e-6)
        logit = torch.log(prior).masked_fill(self.mask_tg == 0, 0.0)
        self.tg_logit = nn.Parameter(logit)
        self.n_g = int(graph.n_genes)
        self.n_t = len(graph.tasks)
        self.n_s = len(graph.subsystems)
        self.n_y = len(graph.systems)
        self.gene_in = nn.Linear(1, hidden)
        self.gene_log_scale = nn.Parameter(torch.zeros(self.n_g))
        self.task_init = nn.Parameter(torch.zeros(1, self.n_t, hidden))
        self.sub_init = nn.Parameter(torch.zeros(1, self.n_s, hidden))
        self.sys_init = nn.Parameter(torch.zeros(1, self.n_y, hidden))
        self.upd_g = nn.GRUCell(hidden, hidden)
        self.upd_t = nn.GRUCell(hidden, hidden)
        self.upd_s = nn.GRUCell(hidden, hidden)
        self.upd_y = nn.GRUCell(hidden, hidden)
        self.attn = nn.ModuleDict({lv: nn.Linear(hidden, 1) for lv in LEVELS})
        self.recon = nn.Linear(3 * hidden, self.n_g)

    def edge_weights_dense(self):
        w = F.softplus(self.tg_logit) * self.mask_tg
        return w / w.sum(-1, keepdim=True).clamp_min(1e-6)

    def encode(self, x_genes: torch.Tensor):
        B = x_genes.size(0)
        scale = torch.exp(self.gene_log_scale)
        h_g = self.gene_in((x_genes * scale).unsqueeze(-1))
        h_t = self.task_init.expand(B, -1, -1).contiguous()
        h_s = self.sub_init.expand(B, -1, -1).contiguous()
        h_y = self.sys_init.expand(B, -1, -1).contiguous()
        A_tg = self.edge_weights_dense()
        for _ in range(self.steps):
            h_t = _gru(self.upd_t, _gated(h_g, h_t, A_tg), h_t)
            h_s = _gru(self.upd_s, _gated(h_t, h_s, self.A_ts), h_s)
            h_y = _gru(self.upd_y, _gated(h_s, h_y, self.A_sy), h_y)
            h_s = h_s + _weighted(h_y, self.A_sy.T, scale=0.5)
            h_t = h_t + _weighted(h_s, self.A_ts.T, scale=0.5)
            h_g = _gru(self.upd_g, _weighted(h_t, A_tg.T), h_g)
        hs = {"task": h_t, "subsystem": h_s, "system": h_y}
        attn = {
            lv: torch.softmax(self.attn[lv](hs[lv]).squeeze(-1), dim=-1)
            for lv in LEVELS
        }
        pooled = torch.cat(
            [torch.einsum("bn,bnh->bh", attn[lv], hs[lv]) for lv in LEVELS],
            dim=-1,
        )
        return pooled, attn, h_t, h_g

    def forward(self, x_genes: torch.Tensor):
        pooled, attn, h_t, _h_g = self.encode(x_genes)
        return self.recon(pooled), attn, h_t


def _fit_epochs(model, Xt, device):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    n = Xt.size(0)
    bs = min(BATCH_SIZE, n)
    model.train()
    for _ in range(EPOCHS):
        perm = np.random.permutation(n)
        for i in range(0, n, bs):
            idx = torch.as_tensor(perm[i : i + bs], device=device)
            xb = Xt.index_select(0, idx)
            opt.zero_grad(set_to_none=True)
            xhat, _, _ = model(xb)
            F.mse_loss(xhat, xb).backward()
            opt.step()


def _train_recon(graph, X, device):
    print(
        f"recon cells={X.shape[0]} epochs={EPOCHS} batch={BATCH_SIZE}",
        flush=True,
    )
    Xt = torch.from_numpy(np.ascontiguousarray(X)).to(device)
    model = MetabolicVNN(graph).to(device)
    _fit_epochs(model, Xt, device)
    return model, Xt


def _encode_cells(model, Xt):
    model.eval()
    hs = []
    gs = []
    at = []
    with torch.no_grad():
        for i in range(0, Xt.size(0), BATCH_SIZE):
            _, attn, h_t, h_g = model.encode(Xt[i : i + BATCH_SIZE])
            hs.append(h_t.mean(-1))
            gs.append(h_g.mean(-1))
            at.append(attn["task"])
    return (
        torch.cat(hs, dim=0),
        torch.cat(gs, dim=0),
        {"task": torch.cat(at, dim=0).cpu().numpy()},
    )


def _interaction_2x2(wv, wi, tv, ti):
    return (ti - wi) - (tv - wv)


def _interaction(mean_a: dict[str, np.ndarray]):
    missing = [a for a in ARMS if a not in mean_a]
    if missing:
        raise ValueError(f"missing arm attention {missing}")
    return _interaction_2x2(*(mean_a[a] for a in ARMS))


def _mouse_treatment_combos(g_mouse: np.ndarray, t_mouse: np.ndarray):
    g_mouse = np.asarray(g_mouse)
    t_mouse = np.asarray(t_mouse)
    groups = []
    for gi in np.unique(g_mouse):
        idx = np.where(g_mouse == gi)[0]
        n_il1 = int((t_mouse[idx] == 1).sum())
        groups.append((idx, n_il1))
    for picked in product(*(combinations(idx, n_il1) for idx, n_il1 in groups)):
        t_p = np.zeros_like(t_mouse)
        for il1_idx in picked:
            t_p[list(il1_idx)] = 1
        yield t_p


def _codes(values, levels, label="value"):
    c = pd.Categorical(np.asarray(values).astype(str), categories=list(levels))
    out = c.codes.astype(np.int64)
    if (out < 0).any():
        raise ValueError(f"unmapped {label}")
    return out


def _perm_matrix(
    y, genotype, treatment, mouse, genotypes=GENOTYPES, treatments=TREATMENTS
):
    device = _device()
    y = torch.as_tensor(y, device=device, dtype=torch.float64)
    if y.ndim == 1:
        y = y.unsqueeze(1)
    arms = tuple(f"{g}_{t}" for g in genotypes for t in treatments)
    g = torch.as_tensor(_codes(genotype, genotypes, "genotype"), device=device)
    t = torch.as_tensor(_codes(treatment, treatments, "treatment"), device=device)
    _, inv_np = np.unique(np.asarray(mouse, dtype=str), return_inverse=True)
    inv = torch.as_tensor(inv_np, device=device, dtype=torch.int64)
    n_mice = int(inv.max().item() + 1)
    g_m = torch.empty(n_mice, dtype=g.dtype, device=device)
    t_m = torch.empty(n_mice, dtype=t.dtype, device=device)
    g_m[inv] = g
    t_m[inv] = t
    if not torch.equal(g_m[inv], g):
        raise ValueError("mixed genotype within mouse")
    if not torch.equal(t_m[inv], t):
        raise ValueError("mixed treatment within mouse")
    arm = 2 * g + t
    n_arm = torch.bincount(arm, minlength=len(arms)).tolist()
    if min(n_arm) == 0:
        raise ValueError("empty arm")
    means = scatter(y, arm, dim=0, dim_size=len(arms), reduce="mean")
    inter = _interaction_2x2(*means)
    tet2 = means[2] - means[0]
    il1 = means[1] - means[0]
    combos = list(
        _mouse_treatment_combos(g_m.detach().cpu().numpy(), t_m.detach().cpu().numpy())
    )
    n_combos = len(combos)
    T = torch.as_tensor(np.stack(combos), device=device, dtype=torch.int64)
    arm_p = 2 * g.unsqueeze(0) + T[:, inv]
    combo_means = []
    for a in range(len(arms)):
        m = arm_p == a
        den = m.sum(1).clamp_min(1).to(y.dtype).unsqueeze(1)
        combo_means.append(m.to(y.dtype) @ y / den)
    combo_inter = _interaction_2x2(*torch.stack(combo_means))
    geq = (combo_inter.abs() >= inter.abs()).sum(0)
    p = geq.to(y.dtype) / n_combos
    y_m = scatter(y, inv, dim=0, dim_size=n_mice, reduce="mean")
    print(f"perm mice={n_mice} combos={n_combos} cells={int(y.size(0))}", flush=True)
    return {
        "tet2": tet2,
        "il1": il1,
        "interaction": inter,
        "p_interaction_perm": p,
        "k_interaction_perm": geq,
        "n_cells": int(y.size(0)),
        "n_mice": n_mice,
        "n_combos": int(n_combos),
        "n_arm": n_arm,
        "means": means,
        "combo_inter": combo_inter,
        "mouse_y": y_m,
        "mouse_g": g_m,
        "mouse_t": t_m,
        "arms": arms,
        "genotypes": genotypes,
        "treatments": treatments,
    }


def _arm_mean_grid(row: pd.Series, genotypes, treatments) -> np.ndarray:
    return np.array(
        [[row[f"mean_{g}_{t}"] for t in treatments] for g in genotypes],
        dtype=float,
    )


def _figure_title(stem: str) -> str:
    s = stem.lower()
    if s.startswith("human") or "gse285379" in s:
        return "Human"
    return "Mice"


def _public_stem(stem: str) -> str:
    return "human" if _figure_title(stem) == "Human" else "mice"


def _task_title(name: str) -> str:
    key = str(name)
    if key in TASK_SHORT:
        return TASK_SHORT[key]
    if key.startswith("axis:"):
        return AXIS_TITLES.get(key[5:], key[5:])
    return AXIS_TITLES.get(key, key)


def _task_axis(name: str) -> str:
    for ax, ts in AXIS_TASKS.items():
        if name in ts:
            return ax
    return ""


def _task_groups(names) -> list[tuple[str, list[str]]]:
    have = set(map(str, names))
    groups = []
    for ax, ts in AXIS_TASKS.items():
        kept = [t for t in ts if t in have]
        if kept:
            groups.append((ax, kept))
    return groups


def _is_mt_nd(name: str) -> bool:
    gl = name.lower()
    if gl.startswith("mt-"):
        gl = gl[3:]
    return gl in _MT_ND


def _collapse_genes(genes: list[str]) -> list[str]:
    out: list[str] = []
    saw_nduf = False
    saw_mt = False
    for g in genes:
        gl = g.lower()
        if gl.startswith("nduf"):
            saw_nduf = True
            continue
        if _is_mt_nd(g):
            saw_mt = True
            continue
        out.append(g)
    prefix: list[str] = []
    if saw_mt:
        prefix.append("mt-Nd")
    if saw_nduf:
        prefix.append("Nduf")
    return prefix + out


def _color_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=float)
    g = g - np.nanmean(g)
    m = float(np.nanmax(np.abs(g)))
    if m > 0:
        return np.clip(g / m, -1.0, 1.0)
    return np.zeros_like(g)


def _unique(seq: list[str]) -> list[str]:
    out: list[str] = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


def _load_hspc_counts(*, human: bool) -> ad.AnnData:
    if human:
        raw = ad.read_h5ad(HUMAN_QC)
        adata = _ensure_counts(raw[raw.obs["compartment"].astype(str).eq("HSPC")].copy())
        return _fix_sample_names(_stamp_arms(adata, HUMAN_TMAP))
    raw = ad.read_h5ad(QC)
    m = (
        raw.obs["lineage"].astype(str).eq(LINEAGE)
        & raw.obs["genotype"].isin(KEEP_GENOTYPE)
        & raw.obs["treatment"].astype(str).isin(KEEP_TREATMENT)
    )
    adata = _ensure_counts(raw[m].copy())
    return _fix_sample_names(_stamp_arms(adata, MOUSE_TMAP))


def _feature_arm_effects(
    X,
    names: list[str],
    obs: pd.DataFrame,
    genotypes=GENOTYPES,
    treatments=TREATMENTS,
) -> pd.DataFrame:
    X = np.asarray(
        X.detach().cpu().numpy() if torch.is_tensor(X) else X, dtype=np.float64
    )
    sample = obs["sample_name"].astype(str).to_numpy()
    g_cell = obs["genotype"].astype(str).to_numpy()
    t_cell = obs["treatment"].astype(str).to_numpy()
    mice = _unique(list(sample))
    M = np.vstack([X[sample == s].mean(0) for s in mice])
    g_m = np.array([g_cell[sample == s][0] for s in mice])
    t_m = np.array([t_cell[sample == s][0] for s in mice])
    arm: dict[str, np.ndarray] = {}
    for gi in genotypes:
        for ti in treatments:
            mask = (g_m == gi) & (t_m == ti)
            if not mask.any():
                raise ValueError(f"empty arm {gi}_{ti}")
            arm[f"{gi}_{ti}"] = M[mask].mean(0)
    g0, g1 = genotypes
    t0, t1 = treatments
    wv, wi = arm[f"{g0}_{t0}"], arm[f"{g0}_{t1}"]
    tv, ti = arm[f"{g1}_{t0}"], arm[f"{g1}_{t1}"]
    il1 = 0.5 * (wi + ti) - 0.5 * (wv + tv)
    tet2 = 0.5 * (tv + ti) - 0.5 * (wv + wi)
    inter = _interaction_2x2(wv, wi, tv, ti)
    return pd.DataFrame(
        {"gene": list(names), "il1": il1, "tet2": tet2, "interaction": inter}
    )


def _annotate_gene(ax, x, y, name: str):
    ax.annotate(
        name,
        (x, y),
        textcoords="offset points",
        xytext=(4, 4),
        fontsize=7.5,
        color="0.15",
        ha="left",
        va="bottom",
    )


def _task_subplots(groups, *, w: float, h: float, share: bool):
    nrows = len(groups)
    ncols = max(len(ts) for _, ts in groups)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(w * ncols, h * nrows),
        squeeze=False,
        sharex=share,
        sharey=share,
    )
    return fig, axs, nrows, ncols


def _plot_gene_cloud(tab: pd.DataFrame, stem: str, challenge: str):
    groups = _task_groups(tab["task"])
    if not groups:
        return
    fig, axs, nrows, ncols = _task_subplots(groups, w=3.2, h=3.2, share=True)
    xmax = float(np.nanmax(np.abs(tab["il1"]))) if len(tab) else 1.0
    ymax = float(np.nanmax(np.abs(tab["interaction"]))) if len(tab) else 1.0
    limx = 1.15 * xmax if xmax > 0 else 1.0
    limy = 1.15 * ymax if ymax > 0 else 1.0
    for i, (axis, tasks) in enumerate(groups):
        want = AXIS_LABELS.get(axis, set())
        for j in range(ncols):
            ax = axs[i][j]
            if j >= len(tasks):
                ax.axis("off")
                continue
            sub = tab.loc[tab["task"] == tasks[j]]
            ax.axhline(0, color="0.85", lw=0.6, zorder=0)
            ax.axvline(0, color="0.85", lw=0.6, zorder=0)
            ax.scatter(
                sub["il1"],
                sub["interaction"],
                s=22,
                c="0.35",
                linewidths=0,
                zorder=2,
            )
            labeled = set()
            nd = sub[sub["gene"].str.lower().str.startswith("nduf")]
            if len(nd):
                _annotate_gene(
                    ax,
                    float(nd["il1"].mean()),
                    float(nd["interaction"].mean()),
                    "Nduf",
                )
                labeled.update(nd["gene"])
            mt = sub[sub["gene"].map(_is_mt_nd)]
            if len(mt):
                _annotate_gene(
                    ax,
                    float(mt["il1"].mean()),
                    float(mt["interaction"].mean()),
                    "mt-Nd",
                )
                labeled.update(mt["gene"])
            for _, r in sub.iterrows():
                if r["gene"] in labeled:
                    continue
                if str(r["gene"]).lower() not in want:
                    continue
                x, y = float(r["il1"]), float(r["interaction"])
                if (abs(x) / limx) ** 2 + (abs(y) / limy) ** 2 < 0.08:
                    continue
                _annotate_gene(ax, x, y, str(r["gene"]))
            ax.set_xlim(-limx, limx)
            ax.set_ylim(-limy, limy)
            ax.set_title(_task_title(tasks[j]), fontweight="bold", fontsize=11)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(length=3, labelsize=8)
            ax.text(
                0.5,
                -0.28,
                f"confirm {AXIS_READOUT.get(axis, '')}",
                ha="center",
                va="top",
                fontsize=8,
                fontstyle="italic",
                color="0.25",
                transform=ax.transAxes,
            )
            if j == 0:
                ax.set_ylabel(f"Tet2 × {challenge}")
            if i == nrows - 1:
                ax.set_xlabel(f"{challenge} main")
    fig.suptitle(_figure_title(stem), fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / f"{_public_stem(stem)}_genes.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def _write_vnn_gene_cloud(
    gene_h,
    gene_names: list[str],
    graph: HypothesisGraph,
    adata: ad.AnnData,
    stem: str,
    genotypes=GENOTYPES,
    treatments=TREATMENTS,
    treat_labels=("vehicle", "IL-1"),
):
    eff = _feature_arm_effects(
        gene_h, gene_names, adata.obs, genotypes, treatments
    )
    A = graph.A_tg.detach().cpu().numpy()
    rows = []
    for ti, task in enumerate(graph.tasks):
        gidx = np.flatnonzero(A[ti] > 0)
        if gidx.size == 0:
            continue
        sub = eff.iloc[gidx].copy()
        sub["task"] = task
        sub["axis"] = _task_axis(task)
        rows.append(sub)
        print(f"{stem} {_task_title(task)} genes={len(sub)}", flush=True)
    if not rows:
        return
    tab = pd.concat(rows, ignore_index=True)
    pub = _public_stem(stem)
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / f"{pub}_genes.csv", index=False)
    _plot_gene_cloud(tab, pub, treat_labels[-1])


def _plot_task_2x2(
    axes: pd.DataFrame,
    stem: str,
    genotypes=GENOTYPES,
    treatments=TREATMENTS,
    treat_labels=("vehicle", "IL-1"),
):
    tab = axes.loc[~axes["task"].astype(str).str.startswith("axis:")].copy()
    groups = _task_groups(tab["task"])
    if not groups:
        return
    fig, axs, _nrows, ncols = _task_subplots(groups, w=3.3, h=3.1, share=False)
    cmap = plt.cm.RdBu_r
    im = None
    used = []
    for i, (axis, tasks) in enumerate(groups):
        for j in range(ncols):
            ax = axs[i][j]
            if j >= len(tasks):
                ax.axis("off")
                continue
            row = tab.loc[tab["task"] == tasks[j]].iloc[0]
            grid = _arm_mean_grid(row, genotypes, treatments)
            color = _color_grid(grid)
            im = ax.imshow(
                color, cmap=cmap, vmin=-1.0, vmax=1.0, origin="upper", aspect="equal"
            )
            used.append(ax)
            k = int(row["k_interaction_perm"])
            n_c = int(row["n_combos"])
            ax.set_title(_task_title(tasks[j]), fontweight="bold", fontsize=11, pad=8)
            ax.set_xlabel(
                f"interaction $\\Delta$ = {float(row['interaction']):.3g}   $p$ = {k}/{n_c}",
                fontsize=8,
            )
            ax.set_xticks([0, 1])
            ax.set_xticklabels(list(treat_labels))
            ax.set_yticks([0, 1])
            ax.set_yticklabels(list(genotypes))
            if j == 0:
                ax.set_ylabel(AXIS_TITLES.get(axis, axis), fontweight="bold")
            for ii, _g in enumerate(genotypes):
                for jj, _tr in enumerate(treatments):
                    val = float(color[ii, jj])
                    ax.text(
                        jj,
                        ii,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        color="white" if abs(val) > 0.55 else "black",
                        fontsize=9,
                        fontweight="bold",
                    )
    cbar = fig.colorbar(im, ax=used, fraction=0.02, pad=0.04)
    cbar.set_ticks([-1.0, 1.0])
    cbar.set_ticklabels(["−1", "1"])
    cbar.set_label("Normalized")
    fig.suptitle(_figure_title(stem), fontweight="bold", fontsize=14, y=0.98)
    fig.savefig(OUT / f"{_public_stem(stem)}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def perm_task_contrasts(
    states,
    names: list[str],
    adata: ad.AnnData,
    stem: str,
    genotypes=GENOTYPES,
    treatments=TREATMENTS,
    treat_labels=("vehicle", "IL-1"),
    write: bool = True,
):
    challenge = treat_labels[-1]
    Y = states if torch.is_tensor(states) else torch.as_tensor(states)
    Y = Y.to(dtype=torch.float64)
    col_names = list(names)
    hyp = set(HYPOTHESIS_TASKS)
    obs = _perm_matrix(
        Y,
        adata.obs["genotype"].to_numpy(),
        adata.obs["treatment"].to_numpy(),
        adata.obs["sample_name"].astype(str).to_numpy(),
        genotypes=genotypes,
        treatments=treatments,
    )
    means = obs["means"].detach().cpu().numpy()
    k_perm = obs["k_interaction_perm"].detach().cpu().numpy()
    arms = obs["arms"]
    rows = []
    for j, name in enumerate(col_names):
        rows.append(
            {
                "task": name,
                "on_axis": name in hyp,
                "tet2": float(obs["tet2"][j]),
                "il1": float(obs["il1"][j]),
                "interaction": float(obs["interaction"][j]),
                "p_interaction_perm": float(obs["p_interaction_perm"][j]),
                "k_interaction_perm": int(k_perm[j]),
                "n_cells": obs["n_cells"],
                "n_mice": obs["n_mice"],
                "n_combos": obs["n_combos"],
                **{f"mean_{a}": float(means[i, j]) for i, a in enumerate(arms)},
                **{f"n_{a}": obs["n_arm"][i] for i, a in enumerate(arms)},
            }
        )
    tab = pd.DataFrame(rows)
    if write:
        pub = _public_stem(stem)
        OUT.mkdir(parents=True, exist_ok=True)
        tab.to_csv(OUT / f"{pub}.csv", index=False)
        if not tab.empty:
            _plot_task_2x2(
                tab,
                pub,
                genotypes=genotypes,
                treatments=treatments,
                treat_labels=treat_labels,
            )
    for _, r in tab.iterrows():
        print(
            f"{stem} {_task_title(r['task'])}  Δ={r['interaction']:.4g} p={int(r['k_interaction_perm'])}/{int(r['n_combos'])}"
            f"  {challenge}={r['il1']:.4g}  Tet2={r['tet2']:.4g}",
            flush=True,
        )
    return tab

def _device():
    global _CUDA
    if _CUDA is not None:
        return _CUDA
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n < 1:
        raise SystemExit("CUDA required for metabolic VNN training and permutation.")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    _CUDA = torch.device("cuda:0")
    return _CUDA


def train_vnn(
    adata: ad.AnnData,
    tbg: pd.DataFrame,
    *,
    stem: str,
    db: Path = MOUSE_DB,
    genotypes=GENOTYPES,
    treatments=TREATMENTS,
    treat_labels=("vehicle", "IL-1"),
):
    hyp = {
        g for t in HYPOTHESIS_TASKS if t in tbg.index for g in _positive_genes(tbg, t)
    }
    names = list(adata.var_names.astype(str))
    keep = [g for g in names if g in hyp]
    if len(keep) < 10:
        raise ValueError(f"only {len(keep)} hypothesis-task genes")
    idx = {g: i for i, g in enumerate(names)}
    cols = [idx[g] for g in keep]
    G = adata.layers["gene_scores"]
    X = _to_dense(G[:, cols]).astype(np.float32)
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-6
    X = (X - mu) / sd
    graph = build_hypothesis_graph(keep, tbg, db=db)
    if graph is None:
        raise ValueError("empty hypothesis graph")
    device = _device()
    model, Xt = _train_recon(graph, X, device)
    states, gene_h, _attn = _encode_cells(model, Xt)
    print(f"{stem} 2x2 from VNN task encodings, gene clouds from gene embeddings", flush=True)
    tab = perm_task_contrasts(
        states,
        list(graph.tasks),
        adata,
        stem=stem,
        genotypes=genotypes,
        treatments=treatments,
        treat_labels=treat_labels,
        write=True,
    )
    _write_vnn_gene_cloud(
        gene_h,
        keep,
        graph,
        adata,
        stem,
        genotypes=genotypes,
        treatments=treatments,
        treat_labels=treat_labels,
    )
    return tab


def _self_check(tbg: pd.DataFrame):
    dup = ad.AnnData(
        X=np.array([[1.0, 2.0, 3.0]]),
        var=pd.DataFrame(index=["A", "A", "B"]),
    )
    col = _collapse_corrected_genes(dup, {})
    assert list(col.var_names) == ["A", "B"]
    assert np.allclose(_to_dense(col.X), [[3.0, 3.0]])
    genes: list[str] = []
    for t in HYPOTHESIS_TASKS:
        if t not in tbg.index:
            continue
        genes.extend(g for g in _positive_genes(tbg, t) if g not in genes)
        if len(genes) >= 8:
            break
    genes = genes[:8]
    X = np.random.default_rng(0).standard_normal((8, len(genes))).astype(np.float32)
    cells = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {
                "genotype": ["WT"] * 4 + ["Tet2_KO"] * 4,
                "treatment": ["vehicle", "vehicle", "IL1b", "IL1b"] * 2,
                "sample_name": list("abcdefgh"),
                "lineage": [LINEAGE] * 8,
            },
            index=[f"c{i}" for i in range(8)],
        ),
    )
    cells.var_names = genes
    cells.layers["gene_scores"] = X
    cells = _stamp_arms(_fix_sample_names(cells), MOUSE_TMAP)
    graph = build_hypothesis_graph(genes, tbg)
    device = _device()
    model, Xt = _train_recon(graph, X, device)
    states, gene_h, attn = _encode_cells(model, Xt)
    n_t = len(graph.tasks)
    assert states.shape == (8, n_t)
    assert gene_h.shape == (8, len(genes))
    assert attn["task"].shape == (8, n_t)
    g = cells.obs["genotype"].to_numpy()
    t = cells.obs["treatment"].to_numpy()
    mouse = cells.obs["sample_name"].astype(str).to_numpy()
    y = (
        1.0
        + 2.0 * (g == "Tet2")
        + 3.0 * (t == "IL1")
        + 4.0 * ((g == "Tet2") & (t == "IL1"))
    )
    algebra = _perm_matrix(y, g, t, mouse)
    mean_a = {a: algebra["means"][i].detach().cpu().numpy() for i, a in enumerate(ARMS)}
    assert abs(float(_interaction(mean_a)[0]) - 4) < 1e-8
    assert abs(float(algebra["interaction"][0]) - 4) < 1e-8
    obs = _perm_matrix(states, g, t, mouse)
    assert int(obs["n_combos"]) == 36
    assert tuple(obs["combo_inter"].shape) == (36, n_t)
    human = _perm_matrix(
        np.arange(4, dtype=float),
        np.array(["WT", "WT", "Tet2", "Tet2"]),
        np.array(["CTRL", "LPS", "CTRL", "LPS"]),
        np.array(list("abcd")),
        treatments=HUMAN_TREATMENTS,
    )
    assert int(human["n_combos"]) == 4
    t_mix = np.array(t, dtype=object)
    t_mix[1] = "IL1"
    mouse_mix = np.array(mouse, dtype=object)
    mouse_mix[1] = mouse_mix[0]
    try:
        _perm_matrix(y, g, t_mix, mouse_mix)
        raise AssertionError("mixed treatment should fail")
    except ValueError:
        pass
    scaled = _color_grid(np.array([[1.23, 1.21], [1.28, 2.64]]))
    assert float(np.nanmax(np.abs(scaled))) <= 1.0 + 1e-12
    collapsed = _collapse_genes(["Ndufa1", "Nd1", "mt-Nd1", "Sdha", "Ogdh"])
    assert collapsed == ["mt-Nd", "Nduf", "Sdha", "Ogdh"]
    wv, wi, tv, ti = 1.0, 4.0, 3.0, 10.0
    assert abs(float(_interaction_2x2(wv, wi, tv, ti)) - 4.0) < 1e-12
    assert abs((0.5 * (wi + ti) - 0.5 * (wv + tv)) - 5.0) < 1e-12
    assert [ax for ax, _ in _task_groups(HYPOTHESIS_TASKS)] == [
        "glycolysis",
        "OXPHOS_TCA",
        "PPP",
    ]
    print("self-check ok", flush=True)


KEEP = {
    "mice.png",
    "mice.csv",
    "mice_genes.png",
    "mice_genes.csv",
    "human.png",
    "human.csv",
    "human_genes.png",
    "human_genes.csv",
}


def _cleanup_results():
    for p in OUT.iterdir():
        if p.is_file() and p.name not in KEEP:
            p.unlink()


def _first_existing(names: tuple[str, ...]) -> Path:
    for name in names:
        path = OUT / name
        if path.exists():
            return path
    raise FileNotFoundError(names)


def replot_saved_axes():
    jobs = (
        (
            "mice",
            ("mice.csv",),
            GENOTYPES,
            TREATMENTS,
            ("vehicle", "IL-1"),
        ),
        (
            "human",
            ("human.csv",),
            GENOTYPES,
            HUMAN_TREATMENTS,
            ("CTRL", "LPS"),
        ),
    )
    for pub, csvs, genotypes, treatments, labels in jobs:
        axes = pd.read_csv(_first_existing(csvs))
        axes.to_csv(OUT / f"{pub}.csv", index=False)
        _plot_task_2x2(
            axes,
            pub,
            genotypes=genotypes,
            treatments=treatments,
            treat_labels=labels,
        )
        genes = OUT / f"{pub}_genes.csv"
        if genes.exists():
            _plot_gene_cloud(pd.read_csv(genes), pub, labels[-1])
        print(f"replotted {pub}", flush=True)
    _cleanup_results()


def main():
    if sys.argv[1:] == ["replot"]:
        replot_saved_axes()
        return
    tbg = pd.read_csv(MOUSE_DB / "Task_by_Gene.csv", index_col=0)
    _self_check(tbg)
    cells = load_cells()
    print(cells.obs["arm"].value_counts().to_dict(), flush=True)
    out = train_vnn(cells, tbg, stem="mice")
    print(out, flush=True)
    human_tbg = pd.read_csv(HUMAN_DB / "Task_by_Gene.csv", index_col=0)
    human_cells = load_human_hspc()
    human = train_vnn(
        human_cells,
        human_tbg,
        stem="human",
        db=HUMAN_DB,
        treatments=HUMAN_TREATMENTS,
        treat_labels=("CTRL", "LPS"),
    )
    print(human, flush=True)
    _cleanup_results()


if __name__ == "__main__":
    main()
