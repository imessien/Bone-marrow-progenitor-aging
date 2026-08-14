#!/usr/bin/env python3
from __future__ import annotations

import importlib
import os
import sys
import types
from concurrent.futures import ThreadPoolExecutor
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
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter, softmax
import rapids_singlecell as rsc

BONE = Path("/cis/net/r41/data/iessien1/bone")
RESULTS = Path("/cis/net/r41/data/iessien1/bone_marrow_results")
MOUSE_DB = BONE / "sccellfie" / "mus_musculus"
OUT = RESULTS / "chip_metabolic_graph"
QC = Path(
    "/cis/home/iessien1/Documents/bone_marrow/data/GSE209994/processed/"
    "gse209994_qc_preprocessed.h5ad"
)

LINEAGE = "HSPC"
KEEP_GENOTYPE = ("WT", "Tet2_KO")
KEEP_TREATMENT = ("vehicle", "IL1b")
GENOTYPES = ("WT", "Tet2")
TREATMENTS = ("vehicle", "IL1")
ARMS = tuple(f"{g}_{t}" for g in GENOTYPES for t in TREATMENTS)
LEVELS = ("task", "subsystem", "system")
EPOCHS = 20
HIDDEN = 96
STEPS = 10
BATCH_SIZE = 256
N_PERM = 10_000

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

_CUDA: list[torch.device] | None = None


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
        adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
    return adata


def _positive_genes(tbg: pd.DataFrame, task: str) -> list[str]:
    row = tbg.loc[task]
    return [str(g) for g in row[row > 0].index]


def _run_sccellfie(adata: ad.AnnData):
    adata = adata.copy()
    if hasattr(adata, "X") and hasattr(adata.X, "to"):
        adata.X = adata.X.to("cuda")
    rsc.pp.normalize_total(adata, target_sum=1e4)
    rsc.pp.log1p(adata)
    rsc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=False)
    rsc.pp.pca(adata, n_comps=30)
    rsc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    adata = _ensure_counts(adata)
    if hasattr(adata, "X") and hasattr(adata.X, "to"):
        adata.X = adata.X.to("cpu")
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
        {"WT": "WT", "Tet2_KO": "Tet2"},
        "genotype",
    )
    adata.obs["treatment"] = _remap_obs(
        adata.obs["treatment"],
        {"vehicle": "vehicle", "IL1b": "IL1"},
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
    raw = ad.read_h5ad(QC)
    m = (
        raw.obs["lineage"].astype(str).eq(LINEAGE)
        & raw.obs["genotype"].isin(KEEP_GENOTYPE)
        & raw.obs["treatment"].astype(str).isin(KEEP_TREATMENT)
    )
    adata = _ensure_counts(raw[m].copy())
    print(
        f"QC {QC} n={adata.n_obs} "
        f"{adata.obs['genotype'].value_counts().to_dict()} "
        f"{adata.obs['treatment'].value_counts().to_dict()}",
        flush=True,
    )
    adata = _run_sccellfie(adata)
    scores = adata.layers["gene_scores"] if "gene_scores" in adata.layers else adata.X
    adata.layers["gene_scores"] = _to_dense(scores).astype(np.float32)
    adata = _stamp_arms(adata)
    adata = _fix_sample_names(adata)
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


def _parent_adj(children: list[str], parent_of: dict[str, str]):
    parents = sorted({parent_of.get(c, "UNKNOWN") for c in children})
    p_idx = {p: i for i, p in enumerate(parents)}
    A = np.zeros((len(parents), len(children)), dtype=np.float32)
    for i, c in enumerate(children):
        A[p_idx[parent_of.get(c, "UNKNOWN")], i] = 1.0
    ei, ew = _adj_edges(_rownorm(A))
    return parents, ei, ew


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
        raise SystemExit("no gene→task edges for glycolysis/TCA/PPP")
    n_g = len(gene_names)
    A_tg = np.zeros((len(kept), n_g), dtype=np.float32)
    for ti, gi in edges:
        A_tg[ti, gi] = 1.0
    gene_task, gene_task_w = _adj_edges(_rownorm(A_tg))
    subs, task_sub, task_sub_w = _parent_adj(kept, task_to_sub)
    systems, sub_sys, sub_sys_w = _parent_adj(subs, sub_to_sys)
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
        "gene_task": gene_task,
        "gene_task_w": gene_task_w,
        "task_sub": task_sub,
        "task_sub_w": task_sub_w,
        "sub_sys": sub_sys,
        "sub_sys_w": sub_sys_w,
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
        gt = graph["gene_task"].long()
        ts = graph["task_sub"].long()
        sy = graph["sub_sys"].long()
        self.register_buffer("gene_task", gt)
        self.register_buffer("task_gene", gt.flip(0).contiguous())
        self.register_buffer("task_sub", ts)
        self.register_buffer("sub_task", ts.flip(0).contiguous())
        self.register_buffer("sub_sys", sy)
        self.register_buffer("sys_sub", sy.flip(0).contiguous())
        prior = graph["gene_task_w"].float().clamp_min(1e-6)
        self.tg_logit = nn.Parameter(torch.log(prior))
        self.register_buffer("task_sub_w", graph["task_sub_w"].float())
        self.register_buffer("sub_sys_w", graph["sub_sys_w"].float())
        self.n_g = int(graph["n_genes"])
        self.n_t = len(graph["tasks"])
        self.n_s = len(graph["subsystems"])
        self.n_y = len(graph["systems"])
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
        for _ in range(self.steps):
            h_t = _gru(self.upd_t, self.gated(h_g, h_t, self.gene_task, tg_w), h_t)
            h_s = _gru(
                self.upd_s, self.gated(h_t, h_s, self.task_sub, self.task_sub_w), h_s
            )
            h_y = _gru(
                self.upd_y, self.gated(h_s, h_y, self.sub_sys, self.sub_sys_w), h_y
            )
            h_s = h_s + self.rev(h_y, self.n_s, self.sys_sub, self.sub_sys_w)
            h_t = h_t + self.rev(h_s, self.n_t, self.sub_task, self.task_sub_w)
            h_g = _gru(self.upd_g, self.back(h_t, self.n_g, self.task_gene, tg_w), h_g)
        hs = {"task": h_t, "subsystem": h_s, "system": h_y}
        attn = {
            lv: torch.softmax(self.attn[lv](hs[lv]).squeeze(-1), dim=-1)
            for lv in LEVELS
        }
        pooled = torch.cat(
            [torch.einsum("bn,bnh->bh", attn[lv], hs[lv]) for lv in LEVELS],
            dim=-1,
        )
        return pooled, attn, h_t

    def forward(self, x_genes: torch.Tensor):
        pooled, attn, h_t = self.encode(x_genes)
        return self.recon(pooled), attn, h_t


def _fit_epochs(model, Xt, device):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    n = Xt.size(0)
    bs = min(BATCH_SIZE, n)
    for _ in range(EPOCHS):
        model.train()
        perm = np.random.permutation(n)
        for i in range(0, n, bs):
            idx = torch.as_tensor(perm[i : i + bs], device=device)
            xb = Xt.index_select(0, idx)
            opt.zero_grad(set_to_none=True)
            xhat, _, _ = model(xb)
            F.mse_loss(xhat, xb).backward()
            opt.step()


def _train_recon(graph, X, device, tag: str):
    print(
        f"[{tag}] recon cells={X.shape[0]} epochs={EPOCHS} batch={BATCH_SIZE}",
        flush=True,
    )
    Xt = torch.from_numpy(np.ascontiguousarray(X)).to(device)
    model = MetabolicVNN(graph).to(device)
    _fit_epochs(model, Xt, device)
    return model, Xt


def _encode_cells(model, Xt):
    model.eval()
    hs = []
    at = {k: [] for k in LEVELS}
    with torch.no_grad():
        for i in range(0, Xt.size(0), BATCH_SIZE):
            _, attn, h_t = model.encode(Xt[i : i + BATCH_SIZE])
            hs.append(h_t.mean(-1))
            for k in LEVELS:
                at[k].append(attn[k])
    states = torch.cat(hs, dim=0)
    attn = {k: torch.cat(v, dim=0).cpu().numpy() for k, v in at.items()}
    return states, attn


def _interaction_2x2(wv, wi, tv, ti):
    return (ti - wi) - (tv - wv)


def _interaction(mean_a: dict[str, np.ndarray]):
    missing = [a for a in ARMS if a not in mean_a]
    if missing:
        raise SystemExit(f"missing arm attention {missing}")
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


def _codes(values, levels):
    c = pd.Categorical(np.asarray(values).astype(str), categories=list(levels))
    out = c.codes.astype(np.int64)
    if (out < 0).any():
        bad = np.asarray(values).astype(str)[out < 0]
        raise SystemExit(f"unmapped {np.unique(bad).tolist()}")
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
    return _interaction_2x2(*picked)


def _tuple_gpu(y, arm, n, seed, devices):
    sizes = [
        n // len(devices) + (1 if i < n % len(devices) else 0)
        for i in range(len(devices))
    ]
    jobs = [
        (sz, seed + 10007 * i, dev)
        for i, (sz, dev) in enumerate(zip(sizes, devices))
        if sz
    ]
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
    n_arm = torch.bincount(arm, minlength=len(ARMS)).tolist()
    if min(n_arm) == 0:
        raise SystemExit(f"missing arm {ARMS[n_arm.index(0)]}")
    means = scatter(y, arm, dim=0, dim_size=len(ARMS), reduce="mean")
    inter = _interaction_2x2(*means)
    tet2 = means[2] - means[0]
    il1 = means[1] - means[0]
    tuples = _tuple_gpu(y, arm, n_perm, seed, devices)
    g_m_np = g_m.detach().cpu().numpy()
    t_m_np = t_m.detach().cpu().numpy()
    combos = list(_mouse_treatment_combos(g_m_np, t_m_np))
    if len(combos) > n_perm:
        rng = np.random.default_rng(seed)
        rng.shuffle(combos)
        combos = combos[:n_perm]
    n_combos = len(combos)
    T = torch.as_tensor(np.stack(combos), device=device, dtype=torch.int64)
    arm_p = 2 * g.unsqueeze(0) + T[:, inv]
    combo_means = []
    for a in range(len(ARMS)):
        m = arm_p == a
        den = m.sum(1).clamp_min(1).to(y.dtype).unsqueeze(1)
        combo_means.append(m.to(y.dtype) @ y / den)
    combo_inter = _interaction_2x2(*torch.stack(combo_means))
    geq = (combo_inter.abs() >= inter.abs()).sum(0)
    p = geq.to(y.dtype) / n_combos
    print(
        f"perm gpus={len(devices)} tuples={int(tuples.size(0))} combos={n_combos}",
        flush=True,
    )
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
        "n_combos": int(n_combos),
        "n_tuples": int(tuples.size(0)),
        "n_arm": n_arm,
        "means": means,
        "combo_inter": combo_inter,
    }, tuples


def _plot_combo_hist(combo: pd.DataFrame, axes: pd.DataFrame):
    names = [n.replace("axis:", "") for n in axes["task"]]
    n = len(names)
    fig, axs = plt.subplots(n, 1, figsize=(6.2, 0.95 * n + 0.85), sharex=True)
    if n == 1:
        axs = [axs]
    obs = dict(zip(axes["task"], axes["interaction"].to_numpy(dtype=float)))
    rows = []
    for name in names:
        key = f"axis:{name}"
        vals = combo.loc[combo["task"] == key, "interaction"].to_numpy(dtype=float)
        ux, cnt = np.unique(np.round(vals, 12), return_counts=True)
        rows.append((name, key, vals, ux, cnt))
    all_x = np.concatenate([r[2] for r in rows]) if rows else np.array([0.0])
    xmin = float(min(np.nanmin(all_x), 0.0))
    xmax = float(max(np.nanmax(all_x), 0.0))
    if xmin == xmax:
        xmin, xmax = xmin - 1.0, xmax + 1.0
    stack_top = max((int(c.max()) if c.size else 1) for *_, c in rows)
    for i, (ax, (name, key, vals, ux, cnt)) in enumerate(zip(axs, rows)):
        xs, ys = [], []
        for x, c in zip(ux, cnt):
            for h in range(int(c)):
                xs.append(float(x))
                ys.append(h)
        ax.scatter(xs, ys, marker="|", s=36, c="0.2", linewidths=0.7, zorder=3)
        ax.axvline(0.0, color="0.75", lw=0.6, zorder=1)
        ax.plot(
            [obs[key], obs[key]],
            [0.0, 1.25],
            color="#c44e52",
            lw=1.4,
            solid_capstyle="butt",
            zorder=4,
        )
        k = int(np.sum(np.abs(vals) >= np.abs(obs[key]) - 1e-12))
        ax.set_ylabel(
            f"{name}\np={k}/{len(vals)}",
            rotation=0,
            ha="right",
            va="center",
        )
        ax.set_yticks([])
        ax.set_ylim(-0.45, max(stack_top, 2) + 0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(i == n - 1)
        if i == n - 1:
            ax.spines["bottom"].set_bounds(xmin, xmax)
            ticks = [xmin, xmax]
            if xmin < 0.0 < xmax:
                ticks = [xmin, 0.0, xmax]
            ax.set_xticks(ticks)
    axs[-1].set_xlim(xmin, xmax)
    axs[-1].set_xlabel("Tet2 × IL-1")
    fig.tight_layout()
    fig.savefig(OUT / "vnn_axis_summary.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def perm_task_contrasts(states, names: list[str], adata: ad.AnnData, tag: str):
    Y = states if torch.is_tensor(states) else torch.as_tensor(states)
    Y = Y.to(dtype=torch.float64)
    col_names = list(names)
    for ax, ts in AXIS_TASKS.items():
        idx = [i for i, n in enumerate(names) if n in ts]
        if not idx:
            continue
        Y = torch.cat([Y, Y[:, idx].mean(1, keepdim=True)], dim=1)
        col_names.append(f"axis:{ax}")
    hyp = set(HYPOTHESIS_TASKS)
    is_axis = [n.startswith("axis:") for n in col_names]
    obs, tuples = _perm_matrix(
        Y,
        adata.obs["genotype"].to_numpy(),
        adata.obs["treatment"].to_numpy(),
        adata.obs["sample_name"].astype(str).to_numpy(),
    )
    axis_idx = [j for j, flag in enumerate(is_axis) if flag]
    tuples_np = (
        tuples[:, axis_idx].detach().cpu().numpy() if axis_idx else np.empty((0, 0))
    )
    combo_np = obs["combo_inter"].detach().cpu().numpy()
    means = obs["means"].detach().cpu().numpy()
    rows = []
    combo_parts = []
    for j, name in enumerate(col_names):
        row = {
            "task": name,
            "on_axis": name in hyp or is_axis[j],
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
        if is_axis[j]:
            combo_parts.append(
                pd.DataFrame(
                    {"task": name, "interaction": tuples_np[:, axis_idx.index(j)]}
                )
            )
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / f"vnn_task_state_2x2_{tag}.csv", index=False)
    axes = tab.loc[is_axis]
    axes.to_csv(OUT / f"vnn_axis_2x2_{tag}.csv", index=False)
    if combo_parts:
        pd.concat(combo_parts, ignore_index=True).to_csv(
            OUT / f"vnn_cell_combo_interaction_{tag}.csv", index=False
        )
    if axis_idx:
        mouse_combo = pd.concat(
            [
                pd.DataFrame(
                    {
                        "task": col_names[j],
                        "combo": np.arange(combo_np.shape[0]),
                        "interaction": combo_np[:, j],
                    }
                )
                for j in axis_idx
            ],
            ignore_index=True,
        )
        mouse_combo.to_csv(OUT / f"vnn_mouse_combo_interaction_{tag}.csv", index=False)
        _plot_combo_hist(mouse_combo, axes)
    print(tab.to_string(index=False), flush=True)
    return tab


def write_attention_report(attn: dict, graph: dict, arms, tag: str):
    levels = {
        "task": graph["tasks"],
        "subsystem": graph["subsystems"],
        "system": graph["systems"],
    }
    arms = np.asarray(arms)
    task_tab = None
    for lv, names in levels.items():
        a = np.asarray(attn[lv], dtype=np.float64)
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
    attn_cols = [f"attn_{a}" for a in ARMS]
    mat = task_tab.set_index("factor")[attn_cols]
    mat.columns = list(ARMS)
    order = [t for ts in AXIS_TASKS.values() for t in ts if t in mat.index]
    order += [t for t in mat.index if t not in order]
    mat = mat.loc[order]
    fig, ax = plt.subplots(
        figsize=(max(6.0, 0.9 * mat.shape[1] + 2), max(3.5, 0.35 * mat.shape[0] + 1.2))
    )
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(list(mat.columns), rotation=25, ha="right")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(list(mat.index))
    ax.set_title("task attention (interpretability)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="attention")
    fig.tight_layout()
    png = OUT / "vnn_attention_heatmap.png"
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    mat.reset_index().to_csv(OUT / "vnn_attention_heatmap.csv", index=False)
    return {"png": str(png)}


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
    global _CUDA
    if _CUDA is not None:
        return _CUDA
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n < 1:
        raise SystemExit("CUDA required for metabolic VNN training and permutation.")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    _CUDA = [torch.device(f"cuda:{i}") for i in range(n)]
    return _CUDA


def _downstream(
    model, Xt, adata: ad.AnnData, graph: dict, gene_names: list[str], tag: str
):
    model.eval()
    states, attn = _encode_cells(model, Xt)
    pd.DataFrame(
        states.detach().cpu().numpy(),
        columns=graph["tasks"],
        index=np.asarray(adata.obs_names.astype(str)),
    ).to_csv(OUT / f"vnn_cell_task_states_{tag}.csv")
    contrasts = perm_task_contrasts(states, graph["tasks"], adata, tag)
    attn_rep = write_attention_report(
        attn, graph, adata.obs["arm"].astype(str).to_numpy(), tag
    )
    edges = write_edge_weights(model, graph, gene_names, tag)
    return {
        "n_cells": int(adata.n_obs),
        "n_mice": int(adata.obs["sample_name"].nunique()),
        "gene_task_weights": edges,
        "task_contrasts": contrasts.to_dict(orient="records"),
        "attention_heatmap": attn_rep["png"],
    }


def train_vnn(adata: ad.AnnData, tbg: pd.DataFrame, tag: str = "tet2_il1"):
    OUT.mkdir(parents=True, exist_ok=True)
    hyp = {
        g for t in HYPOTHESIS_TASKS if t in tbg.index for g in _positive_genes(tbg, t)
    }
    names = list(adata.var_names.astype(str))
    keep = [g for g in names if g in hyp]
    if len(keep) < 10:
        raise SystemExit(f"only {len(keep)} hypothesis-task genes")
    idx = {g: i for i, g in enumerate(names)}
    cols = [idx[g] for g in keep]
    G = adata.layers["gene_scores"]
    X = (G[:, cols].toarray() if sparse.issparse(G) else np.asarray(G)[:, cols]).astype(
        np.float32
    )
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-6
    X = (X - mu) / sd
    graph = build_hypothesis_graph(keep, tbg)
    device = _cuda_devices()[0]
    model, Xt = _train_recon(graph, X, device, tag=tag)
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
    return _downstream(model, Xt, adata, graph, keep, tag)


def _self_check(tbg: pd.DataFrame):
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
    cells = _stamp_arms(_fix_sample_names(cells))
    graph = build_hypothesis_graph(genes, tbg)
    assert "hetero" not in graph
    device = _cuda_devices()[0]
    model, Xt = _train_recon(graph, X, device, tag="self_check")
    states, attn = _encode_cells(model, Xt)
    n_t = len(graph["tasks"])
    assert states.shape == (8, n_t)
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
    algebra, _ = _perm_matrix(y, g, t, mouse, n_perm=64)
    mean_a = {a: algebra["means"][i].detach().cpu().numpy() for i, a in enumerate(ARMS)}
    assert abs(float(_interaction(mean_a)[0]) - 4) < 1e-8
    assert abs(float(algebra["interaction"][0]) - 4) < 1e-8
    assert abs(float(algebra["combo_mean"][0]) - 4) < 1e-8
    obs, tuples = _perm_matrix(states, g, t, mouse, n_perm=64)
    assert int(tuples.size(0)) == 64
    assert int(obs["n_combos"]) == 36
    assert tuple(obs["combo_inter"].shape) == (36, n_t)
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
