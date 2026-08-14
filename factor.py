#!/usr/bin/env python3
from __future__ import annotations

import importlib
import os
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from itertools import permutations, product
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache_bm")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter, softmax

BONE = Path("/cis/net/r41/data/iessien1/bone")
RESULTS = Path("/cis/net/r41/data/iessien1/bone_marrow_results")
MOUSE_DB = BONE / "sccellfie" / "mus_musculus"
OUT = RESULTS / "chip_metabolic_graph"
QC = Path(
    "/cis/home/iessien1/Documents/bone_marrow/data/GSE209994/processed/"
    "gse209994_qc_preprocessed.h5ad"
)

LINEAGE = "HSPC"
GENOTYPES = ("WT", "Tet2")
TREATMENTS = ("vehicle", "IL1")
ARMS = tuple(f"{g}_{t}" for g in GENOTYPES for t in TREATMENTS)
EPOCHS = 20
HIDDEN = 96
STEPS = 10
BATCH_SIZE = 256
N_PERM = 10000

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

        return run_sccellfie_pipeline
    except ImportError:
        _install_sccellfie_without_spatial()
        return importlib.import_module(
            "sccellfie.sccellfie_pipeline"
        ).run_sccellfie_pipeline


def _to_dense(X):
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _ensure_counts(adata: ad.AnnData):
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    if "n_counts" not in adata.obs:
        X = adata.X
        adata.obs["n_counts"] = (
            np.asarray(X.sum(axis=1)).ravel() if sparse.issparse(X) else X.sum(axis=1)
        )
    return adata


def _run_sccellfie(adata: ad.AnnData):
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
        save_folder=None,
        compute_ablation_impact=False,
        verbose=True,
    )
    return db["adata"]


def _remap_obs(s: pd.Series, mapping: dict[str, str], label: str):
    out = s.astype(str).map(mapping)
    bad = out.isna()
    if bad.any():
        raise SystemExit(f"unmapped {label}: {s[bad].unique().tolist()}")
    return out


def _stamp_arms(adata: ad.AnnData):
    adata.obs["genotype"] = _remap_obs(
        adata.obs["genotype"],
        {"WT": "WT", "Tet2_KO": "Tet2", "Tet2_het": "Tet2", "Tet2": "Tet2"},
        "genotype",
    )
    adata.obs["treatment"] = _remap_obs(
        adata.obs["treatment"],
        {
            "vehicle": "vehicle",
            "PBS": "vehicle",
            "IL1b": "IL1",
            "IL1a": "IL1",
            "IL1": "IL1",
        },
        "treatment",
    )
    adata.obs["arm"] = (
        adata.obs["genotype"].astype(str) + "_" + adata.obs["treatment"].astype(str)
    )
    bad = ~adata.obs["arm"].isin(ARMS)
    if bad.any():
        raise SystemExit(
            f"unexpected arms {adata.obs.loc[bad, 'arm'].unique().tolist()}"
        )
    return adata


def load_cells():
    raw = ad.read_h5ad(QC)
    m = (
        raw.obs["lineage"].astype(str).eq(LINEAGE)
        & raw.obs["genotype"].isin(["WT", "Tet2_KO"])
        & raw.obs["treatment"].astype(str).isin(["vehicle", "IL1b"])
    )
    adata = _ensure_counts(raw[m].copy())
    print(
        f"QC {QC} n={adata.n_obs} "
        f"{adata.obs['genotype'].value_counts().to_dict()} "
        f"{adata.obs['treatment'].value_counts().to_dict()}",
        flush=True,
    )
    adata = _run_sccellfie(adata)
    adata.layers["gene_scores"] = _to_dense(
        adata.layers["gene_scores"] if "gene_scores" in adata.layers else adata.X
    ).astype(np.float32)
    adata = _stamp_arms(adata)
    if "sample_name" not in adata.obs.columns:
        adata.obs["sample_name"] = adata.obs_names.astype(str)
    sn = adata.obs["sample_name"].astype(str)
    adata.obs["sample_name"] = sn.where(
        ~sn.isin(["nan", "None", ""]), adata.obs_names.astype(str)
    )
    print(
        f"cells n={adata.n_obs} mice={adata.obs['sample_name'].nunique()} "
        f"arms={adata.obs['arm'].value_counts().to_dict()}",
        flush=True,
    )
    return adata


def _rownorm(A: np.ndarray):
    return A / np.clip(A.sum(1, keepdims=True), 1, None)


def _adj_edges(A: np.ndarray):
    dst, src = np.nonzero(A)
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    edge_weight = torch.tensor(A[dst, src], dtype=torch.float32)
    return edge_index, edge_weight


def _repeat_edges(edge_index: torch.Tensor, n_src: int, n_dst: int, batch: int):
    src, dst = edge_index
    e = src.numel()
    b = torch.arange(batch, device=src.device)
    src = src.repeat(batch) + b.repeat_interleave(e) * n_src
    dst = dst.repeat(batch) + b.repeat_interleave(e) * n_dst
    return torch.stack([src, dst], 0)


def build_hypothesis_graph(gene_names: list[str], tbg: pd.DataFrame):
    info = pd.read_csv(MOUSE_DB / "Task-Info.csv")
    info["Task"] = info["Task"].astype(str)
    task_to_sub = dict(zip(info["Task"], info["Subsystem"].astype(str)))
    sub_to_sys = dict(zip(info["Subsystem"].astype(str), info["System"].astype(str)))
    g_idx = {g: i for i, g in enumerate(gene_names)}
    tasks = [t for t in HYPOTHESIS_TASKS if t in tbg.index]
    edges: list[tuple[int, int]] = []
    kept: list[str] = []
    for t in tasks:
        row = tbg.loc[t]
        pairs = [
            (len(kept), g_idx[str(g)]) for g in row[row > 0].index if str(g) in g_idx
        ]
        if not pairs:
            continue
        edges.extend(pairs)
        kept.append(t)
    if not edges:
        raise SystemExit("no gene→task edges for glycolysis/TCA/PPP")
    n_g = len(gene_names)
    A_tg = np.zeros((len(kept), n_g), dtype=np.float32)
    for ti, gi in edges:
        A_tg[ti, gi] = 1.0
    subs = sorted({task_to_sub.get(t, "UNKNOWN") for t in kept})
    s_idx = {s: i for i, s in enumerate(subs)}
    A_st = np.zeros((len(subs), len(kept)), dtype=np.float32)
    for ti, t in enumerate(kept):
        A_st[s_idx[task_to_sub.get(t, "UNKNOWN")], ti] = 1.0
    systems = sorted({sub_to_sys.get(s, "UNKNOWN") for s in subs})
    y_idx = {y: i for i, y in enumerate(systems)}
    A_ys = np.zeros((len(systems), len(subs)), dtype=np.float32)
    for s in subs:
        A_ys[y_idx[sub_to_sys.get(s, "UNKNOWN")], s_idx[s]] = 1.0
    gene_task, gene_task_w = _adj_edges(_rownorm(A_tg))
    task_sub, task_sub_w = _adj_edges(_rownorm(A_st))
    sub_sys, sub_sys_w = _adj_edges(_rownorm(A_ys))
    hetero = HeteroData()
    hetero["gene"].num_nodes = n_g
    hetero["task"].num_nodes = len(kept)
    hetero["subsystem"].num_nodes = len(subs)
    hetero["system"].num_nodes = len(systems)
    hetero["gene", "to", "task"].edge_index = gene_task
    hetero["gene", "to", "task"].edge_weight = gene_task_w
    hetero["task", "to", "subsystem"].edge_index = task_sub
    hetero["task", "to", "subsystem"].edge_weight = task_sub_w
    hetero["subsystem", "to", "system"].edge_index = sub_sys
    hetero["subsystem", "to", "system"].edge_weight = sub_sys_w
    print(
        f"VNN graph: genes={n_g} tasks={len(kept)} "
        f"subs={len(subs)} systems={len(systems)} edges={int(gene_task.size(1))}",
        flush=True,
    )
    return {
        "tasks": kept,
        "subsystems": subs,
        "systems": systems,
        "n_genes": n_g,
        "hetero": hetero,
    }


class GatedPriorConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add", flow="source_to_target")

    def forward(self, x_src, x_dst, edge_index, edge_weight):
        B, n_src, H = x_src.shape
        n_dst = x_dst.size(1)
        ei = _repeat_edges(edge_index, n_src, n_dst, B)
        src = x_src.reshape(B * n_src, H)
        dst = x_dst.reshape(B * n_dst, H)
        out = self.propagate(
            ei,
            x=src,
            k=F.normalize(src, dim=-1),
            q=F.normalize(dst, dim=-1),
            edge_weight=edge_weight.repeat(B),
            size=(B * n_src, B * n_dst),
        )
        return out.view(B, n_dst, H)

    def message(self, x_j, k_j, q_i, edge_weight, index, ptr, size_i):
        alpha = (edge_weight * torch.sigmoid((q_i * k_j).sum(-1))).clamp_min(1e-8)
        alpha = softmax(alpha.log(), index, ptr, size_i)
        return alpha.unsqueeze(-1) * x_j


class WeightedConv(MessagePassing):
    def __init__(self, scale: float = 1.0):
        super().__init__(aggr="add", flow="source_to_target")
        self.scale = scale

    def forward(self, x_src, n_dst, edge_index, edge_weight):
        B, n_src, H = x_src.shape
        ei = _repeat_edges(edge_index, n_src, n_dst, B)
        out = self.propagate(
            ei,
            x=x_src.reshape(B * n_src, H),
            edge_weight=edge_weight.repeat(B),
            size=(B * n_src, B * n_dst),
        )
        return out.view(B, n_dst, H)

    def message(self, x_j, edge_weight):
        return self.scale * edge_weight.view(-1, 1) * x_j


def _gru(cell, msg, h):
    return cell(msg.reshape(-1, msg.size(-1)), h.reshape(-1, h.size(-1))).view_as(h)


class MetabolicVNN(nn.Module):
    def __init__(self, graph: dict, hidden: int = HIDDEN, steps: int = STEPS):
        super().__init__()
        self.steps = steps
        h = graph["hetero"]
        self.register_buffer("gene_task", h["gene", "to", "task"].edge_index.long())
        prior = h["gene", "to", "task"].edge_weight.float().clamp_min(1e-6)
        self.tg_logit = nn.Parameter(torch.log(prior))
        self.register_buffer("task_sub", h["task", "to", "subsystem"].edge_index.long())
        self.register_buffer(
            "task_sub_w", h["task", "to", "subsystem"].edge_weight.float()
        )
        self.register_buffer(
            "sub_sys", h["subsystem", "to", "system"].edge_index.long()
        )
        self.register_buffer(
            "sub_sys_w", h["subsystem", "to", "system"].edge_weight.float()
        )
        n_t = len(graph["tasks"])
        n_s = len(graph["subsystems"])
        n_y = len(graph["systems"])
        n_g = int(graph["n_genes"])
        self.n_g = n_g
        self.n_t = n_t
        self.n_s = n_s
        self.n_y = n_y
        self.gene_in = nn.Linear(1, hidden)
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
        self.recon = nn.Linear(3 * hidden, n_g)
        self.gated = GatedPriorConv()
        self.rev = WeightedConv(scale=0.5)
        self.back = WeightedConv(scale=1.0)

    def edge_weights(self):
        w = F.softplus(self.tg_logit)
        dst = self.gene_task[1]
        denom = scatter(w, dst, dim=0, dim_size=self.n_t, reduce="sum")
        return w / denom[dst].clamp_min(1e-6)

    def encode(self, x_genes: torch.Tensor):
        B = x_genes.size(0)
        scale = torch.exp(self.gene_log_scale)
        h_g = self.gene_in((x_genes * scale).unsqueeze(-1))
        h_t = self.task_init.expand(B, -1, -1).contiguous()
        h_s = self.sub_init.expand(B, -1, -1).contiguous()
        h_y = self.sys_init.expand(B, -1, -1).contiguous()
        tg_w = self.edge_weights()
        gt, tg = self.gene_task, self.gene_task.flip(0)
        ts, st = self.task_sub, self.task_sub.flip(0)
        sy, ys = self.sub_sys, self.sub_sys.flip(0)
        for _ in range(self.steps):
            h_t = _gru(self.upd_t, self.gated(h_g, h_t, gt, tg_w), h_t)
            h_s = _gru(self.upd_s, self.gated(h_t, h_s, ts, self.task_sub_w), h_s)
            h_y = _gru(self.upd_y, self.gated(h_s, h_y, sy, self.sub_sys_w), h_y)
            h_s = h_s + self.rev(h_y, self.n_s, ys, self.sub_sys_w)
            h_t = h_t + self.rev(h_s, self.n_t, st, self.task_sub_w)
            h_g = _gru(self.upd_g, self.back(h_t, self.n_g, tg, tg_w), h_g)
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
        return pooled, {"task": a_t, "subsystem": a_s, "system": a_y}, h_t

    def forward(self, x_genes: torch.Tensor):
        pooled, attn, h_t = self.encode(x_genes)
        return self.recon(pooled), attn, h_t


def _fit_epochs(model, Xt, tr, epochs, device):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    bs = min(BATCH_SIZE, len(tr))
    for _ in range(epochs):
        model.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), bs):
            idx = perm[i : i + bs]
            xb = Xt[idx].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            xhat, _, _ = model(xb)
            F.mse_loss(xhat, xb).backward()
            opt.step()


def _train_recon(graph, X, device, tag: str):
    n = X.shape[0]
    Xt = torch.from_numpy(np.ascontiguousarray(X)).pin_memory()
    print(f"[{tag}] recon cells={n} epochs={EPOCHS} batch={BATCH_SIZE}", flush=True)
    model = MetabolicVNN(graph).to(device)
    _fit_epochs(model, Xt, np.arange(n, dtype=np.int64), EPOCHS, device)
    return model


def _encode_cells(model, X, device):
    model.eval()
    hs = []
    at = {k: [] for k in ("task", "subsystem", "system")}
    with torch.no_grad():
        for i in range(0, X.shape[0], BATCH_SIZE):
            xb = torch.from_numpy(np.ascontiguousarray(X[i : i + BATCH_SIZE])).to(
                device, non_blocking=True
            )
            _, attn, h_t = model(xb)
            hs.append(h_t.mean(-1))
            for k in at:
                at[k].append(attn[k].cpu())
    states = torch.cat(hs, dim=0)
    attn = {k: torch.cat(v, dim=0).numpy() for k, v in at.items()}
    return states, attn


def _interaction(mean_a: dict[str, np.ndarray]):
    missing = [a for a in ARMS if a not in mean_a]
    if missing:
        raise SystemExit(f"missing arm attention {missing}")
    return (mean_a[ARMS[3]] - mean_a[ARMS[1]]) - (mean_a[ARMS[2]] - mean_a[ARMS[0]])


def _mouse_treatment_combos(g_mouse: np.ndarray, t_mouse: np.ndarray):
    groups = []
    for gi in np.unique(g_mouse):
        idx = np.where(g_mouse == gi)[0]
        uniq = sorted(set(permutations(t_mouse[idx].tolist())))
        groups.append((idx, uniq))
    for choice in product(*(u for _, u in groups)):
        t_p = t_mouse.copy()
        for (idx, _), perm in zip(groups, choice):
            t_p[list(idx)] = perm
        yield t_p


def _codes(values, levels):
    s = np.asarray(values).astype(str)
    out = np.full(len(s), -1, dtype=np.int64)
    for i, lv in enumerate(levels):
        out[s == lv] = i
    bad = out < 0
    if np.any(bad):
        raise SystemExit(f"unmapped {np.unique(s[bad]).tolist()}")
    return out


def _tuple_chunk(y, arm, n, seed, device):
    torch.cuda.set_device(device)
    y = y.to(device, non_blocking=True)
    arm = arm.to(device, non_blocking=True)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    picked = []
    for a in range(len(ARMS)):
        loc = torch.nonzero(arm == a, as_tuple=False).view(-1)
        if loc.numel() == 0:
            raise SystemExit(f"missing arm {ARMS[a]}")
        sel = loc[torch.randint(0, loc.numel(), (n,), generator=gen, device=device)]
        picked.append(y.index_select(0, sel))
    wv, wi, tv, ti = picked
    return (ti - wi) - (tv - wv)


def _tuple_gpu(y, arm, n, seed, devices):
    sizes = [
        n // len(devices) + (1 if i < n % len(devices) else 0)
        for i in range(len(devices))
    ]
    jobs = [(sz, seed + 10007 * i, dev) for i, (sz, dev) in enumerate(zip(sizes, devices)) if sz]
    if len(jobs) == 1:
        sz, sd, dev = jobs[0]
        return _tuple_chunk(y, arm, sz, sd, dev)
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        parts = list(
            pool.map(
                lambda job: _tuple_chunk(y, arm, job[0], job[1], job[2]),
                jobs,
            )
        )
    return torch.cat([p.to(devices[0], non_blocking=True) for p in parts], dim=0)


def _perm_matrix(y, genotype, treatment, mouse, n_perm: int = N_PERM, seed: int = 0):
    devices = _cuda_devices()
    device = devices[0]
    y = torch.as_tensor(y, device=device, dtype=torch.float64)
    if y.ndim == 1:
        y = y.unsqueeze(1)
    g = torch.as_tensor(_codes(genotype, GENOTYPES), device=device)
    t = torch.as_tensor(_codes(treatment, TREATMENTS), device=device)
    _, inv_np = np.unique(np.asarray(mouse, dtype=str), return_inverse=True)
    inv = torch.as_tensor(inv_np, device=device, dtype=torch.int64)
    n_mice = int(inv.max().item() + 1)
    g_m = torch.empty(n_mice, dtype=g.dtype, device=device)
    t_m = torch.empty(n_mice, dtype=t.dtype, device=device)
    g_m[inv] = g
    t_m[inv] = t
    arm = 2 * g + t
    n_arm = [(arm == a).sum().item() for a in range(len(ARMS))]
    if min(n_arm) == 0:
        raise SystemExit(f"missing arm {ARMS[n_arm.index(0)]}")
    means = torch.stack([y[arm == a].mean(0) for a in range(len(ARMS))])
    wv, wi, tv, ti = means
    tet2 = tv - wv
    il1 = wi - wv
    inter = (ti - wi) - (tv - wv)
    tuples = _tuple_gpu(y, arm, n_perm, seed, devices)
    g_m_np = g_m.detach().cpu().numpy()
    t_m_np = t_m.detach().cpu().numpy()
    combos = [t_p for t_p in _mouse_treatment_combos(g_m_np, t_m_np) if not np.array_equal(t_p, t_m_np)]
    if len(combos) > n_perm:
        rng = np.random.default_rng(seed)
        rng.shuffle(combos)
        combos = combos[:n_perm]
    n_null = len(combos)
    if n_null:
        T = torch.as_tensor(np.stack(combos), device=device, dtype=torch.int64)
        arm_p = 2 * g.unsqueeze(0) + T[:, inv]
        null_means = []
        for a in range(len(ARMS)):
            m = arm_p == a
            den = m.sum(1).clamp_min(1).to(y.dtype).unsqueeze(1)
            null_means.append(m.to(y.dtype) @ y / den)
        nw, ni, ntv, nti = torch.stack(null_means)
        geq = (((nti - ni) - (ntv - nw)).abs() >= inter.abs()).sum(0)
        p = (1.0 + geq.to(y.dtype)) / (n_null + 1)
    else:
        p = torch.full_like(inter, float("nan"))
    print(f"perm gpus={len(devices)} tuples={int(tuples.size(0))} combos={n_null + 1}", flush=True)
    return {
        "tet2": tet2,
        "il1": il1,
        "interaction": inter,
        "combo_mean": tuples.mean(0),
        "combo_frac_pos": (tuples > 0).to(y.dtype).mean(0),
        "combo_q05": torch.quantile(tuples, 0.05, dim=0),
        "combo_q95": torch.quantile(tuples, 0.95, dim=0),
        "p_interaction_perm": p,
        "n_cells": int(y.size(0)),
        "n_mice": n_mice,
        "n_combos": int(n_null + 1),
        "n_tuples": int(tuples.size(0)),
        "n_arm": n_arm,
        "means": means,
    }, tuples


def perm_task_contrasts(states, names: list[str], adata: ad.AnnData, tag: str):
    device = _cuda_devices()[0]
    Y = states if torch.is_tensor(states) else torch.as_tensor(states, device=device)
    Y = Y.to(device=device, dtype=torch.float64)
    col_names = list(names)
    for ax, ts in AXIS_TASKS.items():
        idx = [i for i, n in enumerate(names) if n in ts]
        if not idx:
            continue
        Y = torch.cat([Y, Y[:, idx].mean(1, keepdim=True)], dim=1)
        col_names.append(f"axis:{ax}")
    hyp = set(HYPOTHESIS_TASKS)
    obs, tuples = _perm_matrix(
        Y,
        adata.obs["genotype"].to_numpy(),
        adata.obs["treatment"].to_numpy(),
        adata.obs["sample_name"].astype(str).to_numpy(),
    )
    tuples_np = tuples.detach().cpu().numpy()
    means = obs["means"].detach().cpu().numpy()
    rows = []
    combo_parts = []
    for j, name in enumerate(col_names):
        row = {
            "task": name,
            "on_axis": name in hyp or name.startswith("axis:"),
            "tet2": float(obs["tet2"][j]),
            "il1": float(obs["il1"][j]),
            "interaction": float(obs["interaction"][j]),
            "combo_mean": float(obs["combo_mean"][j]),
            "combo_frac_pos": float(obs["combo_frac_pos"][j]),
            "combo_q05": float(obs["combo_q05"][j]),
            "combo_q95": float(obs["combo_q95"][j]),
            "p_interaction_perm": float(obs["p_interaction_perm"][j]),
            "n_cells": obs["n_cells"],
            "n_mice": obs["n_mice"],
            "n_combos": obs["n_combos"],
            "n_tuples": obs["n_tuples"],
            **{f"mean_{a}": float(means[i, j]) for i, a in enumerate(ARMS)},
            **{f"n_{a}": obs["n_arm"][i] for i, a in enumerate(ARMS)},
        }
        rows.append(row)
        if name.startswith("axis:"):
            combo_parts.append(pd.DataFrame({"task": name, "interaction": tuples_np[:, j]}))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / f"vnn_task_state_2x2_{tag}.csv", index=False)
    axes = tab[tab["task"].str.startswith("axis:")].copy()
    axes.to_csv(OUT / f"vnn_axis_2x2_{tag}.csv", index=False)
    pd.concat(combo_parts, ignore_index=True).to_csv(
        OUT / f"vnn_cell_combo_interaction_{tag}.csv", index=False
    )
    print(tab.to_string(index=False), flush=True)
    return tab


def write_attention_report(oof_attn: dict, graph: dict, arms, tag: str):
    levels = {
        "task": graph["tasks"],
        "subsystem": graph["subsystems"],
        "system": graph["systems"],
    }
    arms = np.asarray(arms)
    task_tab = None
    for lv, names in levels.items():
        a = np.asarray(oof_attn[lv], dtype=np.float64)
        mean_a = {c: a[arms == c].mean(0) for c in ARMS if (arms == c).any()}
        inter = _interaction(mean_a)
        cols = {"factor": names, "level": lv, "interaction_attn": inter}
        for c, v in mean_a.items():
            cols[f"attn_{c}"] = v
        tab = pd.DataFrame(cols).sort_values(
            "interaction_attn", key=lambda s: s.abs(), ascending=False
        )
        suffix = "" if lv == "task" else f"_{lv}"
        tab.to_csv(OUT / f"vnn_interaction_attention_{tag}{suffix}.csv", index=False)
        if lv == "task":
            task_tab = tab
    axis_attn = {
        ax: float(task_tab.loc[task_tab["factor"].isin(ts), "interaction_attn"].sum())
        for ax, ts in AXIS_TASKS.items()
    }
    attn_cols = [f"attn_{a}" for a in ARMS]
    mat = task_tab.set_index("factor")[attn_cols]
    mat.columns = list(ARMS)
    fig, ax = plt.subplots(
        figsize=(max(6.0, 0.9 * mat.shape[1] + 2), max(3.5, 0.35 * mat.shape[0] + 1.2))
    )
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(list(mat.columns), rotation=25, ha="right")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(list(mat.index))
    ax.set_title("Cell task attention")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="attention")
    fig.tight_layout()
    png = OUT / f"vnn_attention_heatmap_{tag}.png"
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    mat.reset_index().to_csv(OUT / f"vnn_attention_heatmap_{tag}.csv", index=False)
    return {
        "png": str(png),
        "axis_interaction_attention_sum": axis_attn,
        "top_interaction_factors": task_tab.head(10).to_dict(orient="records"),
    }


def write_edge_weights(model, graph, gene_names: list[str], tag: str):
    w = model.edge_weights().detach().cpu().numpy()
    src, dst = model.gene_task.cpu().numpy()
    tab = pd.DataFrame(
        {
            "task": [graph["tasks"][i] for i in dst],
            "gene": [gene_names[i] for i in src],
            "weight": w,
        }
    ).sort_values("weight", ascending=False)
    path = OUT / f"vnn_gene_task_weights_{tag}.csv"
    tab.to_csv(path, index=False)
    return str(path)


def _cuda_devices():
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n < 1:
        raise SystemExit("CUDA required for metabolic VNN training and permutation.")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return [torch.device(f"cuda:{i}") for i in range(n)]


def _cuda_device():
    return _cuda_devices()[0]


def train_vnn(adata: ad.AnnData, tbg: pd.DataFrame, tag: str = "tet2_il1"):
    OUT.mkdir(parents=True, exist_ok=True)
    X = _to_dense(adata.layers["gene_scores"]).astype(np.float32)
    hyp_genes = []
    for t in HYPOTHESIS_TASKS:
        if t not in tbg.index:
            continue
        row = tbg.loc[t]
        hyp_genes.extend(str(g) for g in row[row > 0].index)
    keep = [g for g in adata.var_names.astype(str) if g in set(hyp_genes)]
    if len(keep) < 10:
        raise SystemExit(f"only {len(keep)} hypothesis-task genes")
    names = list(adata.var_names.astype(str))
    X = X[:, [names.index(g) for g in keep]]
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-6
    X = (X - mu) / sd
    graph = build_hypothesis_graph(keep, tbg)
    device = _cuda_device()
    model = _train_recon(graph, X, device, tag=tag)
    states, attn = _encode_cells(model, X, device)
    contrasts = perm_task_contrasts(states, graph["tasks"], adata, tag)
    attn_rep = write_attention_report(
        attn, graph, adata.obs["arm"].astype(str).to_numpy(), tag
    )
    edges = write_edge_weights(model, graph, keep, tag)
    torch.save(
        {
            "tag": tag,
            "arms": list(ARMS),
            "state": {k: v.cpu() for k, v in model.state_dict().items()},
            "tasks": graph["tasks"],
            "mu": mu,
            "sd": sd,
            "gene_names": keep,
        },
        OUT / f"vnn_best_{tag}.pt",
    )
    return {
        "n_cells": int(adata.n_obs),
        "n_mice": int(adata.obs["sample_name"].nunique()),
        "gene_task_weights": edges,
        "task_contrasts": contrasts.to_dict(orient="records"),
        **attn_rep,
    }


def _self_check(tbg: pd.DataFrame):
    genes: list[str] = []
    for t in HYPOTHESIS_TASKS:
        if t not in tbg.index:
            continue
        row = tbg.loc[t]
        genes.extend(str(g) for g in row[row > 0].index if str(g) not in genes)
        if len(genes) >= 8:
            break
    genes = genes[:8]
    graph = build_hypothesis_graph(genes, tbg)
    device = _cuda_device()
    X = np.random.default_rng(0).standard_normal((8, len(genes))).astype(np.float32)
    model = _train_recon(graph, X, device, tag="self_check")
    states, attn = _encode_cells(model, X, device)
    n_t = len(graph["tasks"])
    assert states.shape == (8, n_t)
    assert attn["task"].shape == (8, n_t)
    g = np.array(["WT"] * 4 + ["Tet2"] * 4)
    t = np.array(["vehicle", "vehicle", "IL1", "IL1"] * 2)
    mouse = np.array(list("abcdefgh"))
    y = (
        1.0
        + 2.0 * (g == "Tet2")
        + 3.0 * (t == "IL1")
        + 4.0 * ((g == "Tet2") & (t == "IL1"))
    )
    obs, tuples = _perm_matrix(y, g, t, mouse, n_perm=64)
    mean_a = {a: obs["means"][i].detach().cpu().numpy() for i, a in enumerate(ARMS)}
    assert abs(float(_interaction(mean_a)[0]) - 4) < 1e-8
    assert abs(float(obs["interaction"][0]) - 4) < 1e-8
    assert abs(float(obs["combo_mean"][0]) - 4) < 1e-8
    assert abs(float(obs["combo_frac_pos"][0]) - 1.0) < 1e-8
    assert int(tuples.size(0)) == 64
    print("self-check ok", flush=True)


def main():
    tbg = pd.read_csv(MOUSE_DB / "Task_by_Gene.csv", index_col=0)
    _self_check(tbg)
    cells = load_cells()
    print(cells.obs["arm"].value_counts().to_dict(), flush=True)
    out = train_vnn(cells, tbg)
    print(out, flush=True)


if __name__ == "__main__":
    main()
