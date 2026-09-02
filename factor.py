#!/usr/bin/env python3
from __future__ import annotations

import importlib
import os
import sys
import types
from dataclasses import dataclass, replace
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache_bm")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylimma
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
MOUSE_TMAP = {"vehicle": "vehicle", "IL1b": "IL1"}
HUMAN_TMAP = {"CTRL": "CTRL", "LPS": "LPS"}

LEVELS = ("task", "subsystem", "system")
EPOCHS = 20
HIDDEN = 96
STEPS = 10
BATCH_SIZE = 256
N_PERM = 50

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
TASK_READOUT = {
    "ATP generation from glucose (hypoxic conditions) - glycolysis": "lactate",
    "Oxidative phosphorylation via NADH-coenzyme Q oxidoreductase (COMPLEX I)": "NADH",
    "Oxidative phosphorylation via succinate-coenzyme Q oxidoreductase (COMPLEX II)": "succinate",
    "Krebs cycle - NADH generation": "α-KG",
    "Krebs cycle - oxidative decarboxylation of pyruvate": "acetyl-CoA",
    "Synthesis of fructose-6-phosphate from erythrose-4-phosphate (HMP shunt)": "F6P",
    "Synthesis of ribose-5-phosphate": "ribose-5-P",
}
_MT_ND = {"nd1", "nd2", "nd3", "nd4", "nd4l", "nd5", "nd6"}
TOP_GENES = 10
P_SIG = 0.05
_LABEL_OFF = (
    (5, 5),
    (5, -10),
    (-32, 5),
    (-32, -10),
    (8, 12),
    (-36, 12),
    (8, -16),
    (-36, -16),
    (12, 0),
    (-40, 0),
)


def _import_sccellfie():
    try:
        from sccellfie.sccellfie_pipeline import run_sccellfie_pipeline
        from sccellfie.preprocessing.prepare_inputs import CORRECT_GENES
        return run_sccellfie_pipeline, CORRECT_GENES
    except ImportError:
        pass
    for key in list(sys.modules):
        if key == "sccellfie" or key.startswith("sccellfie."):
            del sys.modules[key]
    local = (
        Path(__file__).resolve().parent / ".venv/lib/python3.10/site-packages/sccellfie"
    )
    root = (
        local
        if local.exists()
        else Path(sys.prefix) / "lib/python3.10/site-packages/sccellfie"
    )
    pkg = types.ModuleType("sccellfie")
    pkg.__path__ = [str(root)]
    pkg.__file__ = str(root / "__init__.py")
    pkg.__version__ = "0.6.0"
    sys.modules["sccellfie"] = pkg
    spatial = types.ModuleType("sccellfie.spatial")
    spatial.__path__ = [str(root / "spatial")]
    sys.modules["sccellfie.spatial"] = spatial
    pkg.spatial = spatial
    return (
        importlib.import_module("sccellfie.sccellfie_pipeline").run_sccellfie_pipeline,
        importlib.import_module("sccellfie.preprocessing.prepare_inputs").CORRECT_GENES,
    )


def _to_dense(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


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
    return ad.AnnData(
        X=coll(adata.X),
        obs=adata.obs.copy(),
        var=var,
        layers={k: coll(v) for k, v in adata.layers.items()},
        uns=adata.uns.copy(),
        obsm=adata.obsm.copy(),
        obsp=adata.obsp.copy(),
    )


def _sccellfie(*, human: bool) -> ad.AnnData:
    if human:
        raw = ad.read_h5ad(HUMAN_QC)
        adata = raw[raw.obs["compartment"].astype(str).eq("HSPC")].copy()
        tmap, organism, db = HUMAN_TMAP, "human", HUMAN_DB
    else:
        raw = ad.read_h5ad(QC)
        m = (
            raw.obs["lineage"].astype(str).eq(LINEAGE)
            & raw.obs["genotype"].isin(KEEP_GENOTYPE)
            & raw.obs["treatment"].astype(str).isin(KEEP_TREATMENT)
        )
        adata = raw[m].copy()
        tmap, organism, db = MOUSE_TMAP, "mouse", MOUSE_DB
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    if "n_counts" not in adata.obs:
        adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
    adata.obs["genotype"] = adata.obs["genotype"].astype(str).map(
        {"WT": "WT", "Tet2_KO": "Tet2"}
    )
    adata.obs["treatment"] = adata.obs["treatment"].astype(str).map(tmap)
    adata.obs["arm"] = (
        adata.obs["genotype"].astype(str) + "_" + adata.obs["treatment"].astype(str)
    )
    sn = (
        adata.obs["sample_name"].astype(str)
        if "sample_name" in adata.obs
        else adata.obs_names.astype(str)
    )
    adata.obs["sample_name"] = np.where(
        sn.isin(["nan", "None", ""]), adata.obs_names.astype(str), sn
    )
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.normalize_total(adata, target_sum=1e4)
    rsc.pp.log1p(adata)
    rsc.pp.highly_variable_genes(adata, n_top_genes=2000)
    rsc.pp.pca(adata, n_comps=30)
    rsc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    rsc.get.anndata_to_CPU(adata)
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    run, correct = _import_sccellfie()
    adata = _collapse_corrected_genes(adata, correct.get(organism, {}))
    adata = run(
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
    )["adata"]
    adata.layers["gene_scores"] = adata.layers.get("gene_scores", adata.X)
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
        row = tbg.loc[t]
        pairs = [
            (len(kept), g_idx[g])
            for g in map(str, row[row > 0].index)
            if g in g_idx
        ]
        if not pairs:
            continue
        edges.extend(pairs)
        kept.append(t)
    n_g = len(gene_names)
    A_tg = np.zeros((len(kept), n_g), dtype=np.float32)
    for ti, gi in edges:
        A_tg[ti, gi] = 1.0
    A_tg = _rownorm(A_tg)
    subs, A_ts = _parent_adj(kept, task_to_sub)
    systems, A_sy = _parent_adj(subs, sub_to_sys)
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
        return pooled, h_t, h_g

    def forward(self, x_genes: torch.Tensor):
        pooled, _, _ = self.encode(x_genes)
        return self.recon(pooled)


def _train_recon(graph, X, device):
    Xt = torch.from_numpy(np.ascontiguousarray(X)).to(device)
    model = MetabolicVNN(graph).to(device)
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
            F.mse_loss(model(xb), xb).backward()
            opt.step()
    return model, Xt


def _encode_cells(model, Xt):
    model.eval()
    hs, gs = [], []
    with torch.no_grad():
        for i in range(0, Xt.size(0), BATCH_SIZE):
            _, h_t, h_g = model.encode(Xt[i : i + BATCH_SIZE])
            hs.append(h_t.mean(-1))
            gs.append(h_g.mean(-1))
    return torch.cat(hs, dim=0), torch.cat(gs, dim=0)


def _interaction_2x2(wv, wi, tv, ti):
    return (ti - wi) - (tv - wv)


def _mouse_means(X, obs: pd.DataFrame):
    sample = obs["sample_name"].astype(str).to_numpy()
    mice, inv = np.unique(sample, return_inverse=True)
    X = np.asarray(X, dtype=np.float64)
    Y = np.zeros((len(mice), X.shape[1]), dtype=np.float64)
    np.add.at(Y, inv, X)
    Y /= np.bincount(inv).astype(np.float64)[:, None]
    _, first = np.unique(inv, return_index=True)
    g = obs["genotype"].astype(str).to_numpy()[first]
    t = obs["treatment"].astype(str).to_numpy()[first]
    return Y, g, t, list(mice)


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.2f}"


def _codes(values, levels):
    return pd.Categorical(
        np.asarray(values).astype(str), categories=list(levels)
    ).codes.astype(np.int64)


def _arm_stats(y, genotype, treatment, genotypes, treatments, device):
    y = torch.as_tensor(y, device=device, dtype=torch.float64)
    if y.ndim == 1:
        y = y.unsqueeze(1)
    arms = tuple(f"{g}_{t}" for g in genotypes for t in treatments)
    g = torch.as_tensor(_codes(genotype, genotypes), device=device)
    t = torch.as_tensor(_codes(treatment, treatments), device=device)
    arm = 2 * g + t
    means = scatter(y, arm, dim=0, dim_size=len(arms), reduce="mean")
    return {
        "tet2": means[2] - means[0],
        "il1": means[1] - means[0],
        "interaction": _interaction_2x2(*means),
        "means": means,
        "arms": arms,
    }


def _task_title(name: str) -> str:
    key = str(name)
    return TASK_SHORT.get(key, AXIS_TITLES.get(key, key))


def _readable_gene(name: str) -> str:
    g = str(name)
    gl = g.lower()
    if gl.startswith("nduf"):
        return "Nduf"
    if _is_mt_nd(g):
        return "mt-Nd"
    alpha = "".join(c for c in g if c.isalpha())
    if alpha.isupper():
        out = []
        saw = False
        for c in g:
            if c.isalpha():
                out.append(c.upper() if not saw else c.lower())
                saw = True
            else:
                out.append(c)
        return "".join(out)
    return g


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


def _rank_gene_labels(genes, xn, yn, n: int = TOP_GENES):
    genes = np.asarray(genes)
    xn = np.asarray(xn, dtype=float)
    yn = np.asarray(yn, dtype=float)
    score = xn * xn + yn * yn
    order = np.argsort(-score)
    nduf = np.array([str(g).lower().startswith("nduf") for g in genes])
    mt = np.array([_is_mt_nd(str(g)) for g in genes])
    out = []
    used = set()
    for i in order:
        if len(out) >= n:
            break
        if nduf[i]:
            if "Nduf" in used:
                continue
            used.add("Nduf")
            out.append((float(xn[nduf].mean()), float(yn[nduf].mean()), "Nduf"))
            continue
        if mt[i]:
            if "mt-Nd" in used:
                continue
            used.add("mt-Nd")
            out.append((float(xn[mt].mean()), float(yn[mt].mean()), "mt-Nd"))
            continue
        g = _readable_gene(str(genes[i]))
        if g in used:
            continue
        used.add(g)
        out.append((float(xn[i]), float(yn[i]), g))
    return out


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
    fig.subplots_adjust(
        left=0.10,
        right=0.88,
        top=0.90,
        bottom=0.16,
        wspace=0.55,
        hspace=1.05,
    )
    return fig, axs, ncols


def _plot_gene_cloud(tab: pd.DataFrame, stem: str, challenge: str):
    groups = _task_groups(tab["task"])
    if not groups:
        return
    fig, axs, ncols = _task_subplots(groups, w=3.8, h=4.2, share=True)
    for i, (_axis, tasks) in enumerate(groups):
        for j in range(ncols):
            ax = axs[i][j]
            if j >= len(tasks):
                ax.axis("off")
                continue
            sub = tab.loc[tab["task"] == tasks[j]]
            x = sub["il1"].to_numpy(dtype=float)
            y = sub["interaction"].to_numpy(dtype=float)
            m = max(float(np.nanmax(np.abs(x))), float(np.nanmax(np.abs(y))), 1e-12)
            xn, yn = x / m, y / m
            ax.axhline(0, color="0.85", lw=0.6, zorder=0)
            ax.axvline(0, color="0.85", lw=0.6, zorder=0)
            ax.scatter(xn, yn, s=22, c="0.35", linewidths=0, zorder=2)
            for rank, (gx, gy, name) in enumerate(
                _rank_gene_labels(sub["gene"].to_numpy(), xn, yn)
            ):
                dx, dy = _LABEL_OFF[rank % len(_LABEL_OFF)]
                ax.annotate(
                    name,
                    (gx, gy),
                    textcoords="offset points",
                    xytext=(dx, dy),
                    fontsize=7,
                    color="0.15",
                    ha="left",
                    va="bottom",
                )
            ax.set_xlim(-1.15, 1.15)
            ax.set_ylim(-1.15, 1.15)
            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([-1, 0, 1])
            ax.set_title(_task_title(tasks[j]), fontweight="bold", fontsize=11, pad=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(length=3, labelsize=8)
            ax.set_xlabel(
                f"{challenge} main\n{TASK_READOUT.get(tasks[j], _task_title(tasks[j]))}",
                fontsize=8,
                labelpad=8,
            )
            if j == 0:
                ax.set_ylabel("interaction", labelpad=10)
    fig.suptitle(
        "Human" if "human" in stem.lower() else "Mice",
        fontweight="bold",
        fontsize=14,
        y=0.98,
    )
    fig.savefig(
        OUT / f"{stem}_genes.png",
        bbox_inches="tight",
        pad_inches=0.4,
        dpi=200,
    )
    plt.close(fig)


def _plot_task_2x2(
    axes: pd.DataFrame,
    stem: str,
    genotypes=GENOTYPES,
    treatments=TREATMENTS,
    treat_labels=("vehicle", "IL-1"),
):
    groups = _task_groups(axes["task"])
    if not groups:
        return
    fig, axs, ncols = _task_subplots(groups, w=3.8, h=3.9, share=False)
    cmap = plt.cm.RdBu_r
    im = None
    used = []
    for i, (axis, tasks) in enumerate(groups):
        for j in range(ncols):
            ax = axs[i][j]
            if j >= len(tasks):
                ax.axis("off")
                continue
            row = axes.loc[axes["task"] == tasks[j]].iloc[0]
            grid = np.array(
                [[row[f"mean_{g}_{t}"] for t in treatments] for g in genotypes],
                dtype=float,
            )
            grid = grid - np.nanmean(grid)
            m = float(np.nanmax(np.abs(grid)))
            color = np.clip(grid / m, -1.0, 1.0) if m > 0 else np.zeros_like(grid)
            im = ax.imshow(
                color, cmap=cmap, vmin=-1.0, vmax=1.0, origin="upper", aspect="equal"
            )
            used.append(ax)
            p = float(row["p_camera"])
            sig = np.isfinite(p) and p < P_SIG
            ax.set_title(
                _task_title(tasks[j]),
                fontweight="bold",
                fontsize=12 if sig else 11,
                pad=12,
                color="0.0" if sig else "0.15",
            )
            ax.set_xlabel(
                f"{TASK_READOUT.get(tasks[j], _task_title(tasks[j]))}\n$p$ = {_fmt_p(p)}",
                fontsize=8,
                fontweight="bold" if sig else "normal",
                labelpad=10,
            )
            ax.set_xticks([0, 1])
            ax.set_xticklabels(list(treat_labels))
            ax.set_yticks([0, 1])
            ax.set_yticklabels(list(genotypes))
            if j == 0:
                ax.set_ylabel(
                    AXIS_TITLES.get(axis, axis), fontweight="bold", labelpad=12
                )
            for ii in range(len(genotypes)):
                for jj in range(len(treatments)):
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
    cbar = fig.colorbar(im, ax=used, fraction=0.025, pad=0.06)
    cbar.set_ticks([-1.0, 1.0])
    cbar.set_ticklabels(["−1", "1"])
    cbar.set_label("Relative mean")
    fig.suptitle(
        "Human" if "human" in stem.lower() else "Mice",
        fontweight="bold",
        fontsize=14,
        y=0.98,
    )
    fig.savefig(
        OUT / f"{stem}.png",
        bbox_inches="tight",
        pad_inches=0.4,
        dpi=200,
    )
    plt.close(fig)


def _device():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for metabolic VNN training and permutation.")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch.device("cuda:0")


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
        g
        for t in HYPOTHESIS_TASKS
        if t in tbg.index
        for g in map(str, tbg.loc[t][tbg.loc[t] > 0].index)
    }
    names = list(adata.var_names.astype(str))
    keep = [g for g in names if g in hyp]
    idx = {g: i for i, g in enumerate(names)}
    X = _to_dense(adata.layers["gene_scores"][:, [idx[g] for g in keep]]).astype(
        np.float32
    )
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)
    graph = build_hypothesis_graph(keep, tbg, db=db)
    device = _device()
    model, Xt = _train_recon(graph, X, device)
    states, gene_h = _encode_cells(model, Xt)
    obs = _arm_stats(
        states,
        adata.obs["genotype"].to_numpy(),
        adata.obs["treatment"].to_numpy(),
        genotypes,
        treatments,
        device,
    )
    abs_obs = np.abs(obs["interaction"].detach().cpu().numpy())
    g_cell = adata.obs["genotype"].to_numpy()
    t_cell = adata.obs["treatment"].to_numpy()
    rng = np.random.default_rng(0)
    geq = np.zeros(abs_obs.shape[0], dtype=np.int64)
    print(f"graph perm n={N_PERM} tasks={abs_obs.shape[0]}", flush=True)
    for _ in range(N_PERM):
        A = graph.A_tg.detach().cpu().numpy()[:, rng.permutation(graph.n_genes)]
        g_perm = replace(graph, A_tg=torch.tensor(_rownorm(A), dtype=torch.float32))
        m_p, Xt_p = _train_recon(g_perm, X, device)
        st_p, _ = _encode_cells(m_p, Xt_p)
        inter = (
            _arm_stats(st_p, g_cell, t_cell, genotypes, treatments, device)[
                "interaction"
            ]
            .detach()
            .cpu()
            .numpy()
        )
        geq += np.abs(inter) >= abs_obs
    p_perm = (geq + 1) / (N_PERM + 1)
    means = obs["means"].detach().cpu().numpy()
    arms = obs["arms"]
    tab = pd.DataFrame(
        [
            {
                "task": name,
                "tet2": float(obs["tet2"][j]),
                "il1": float(obs["il1"][j]),
                "interaction": float(obs["interaction"][j]),
                "p_interaction_perm": float(p_perm[j]),
                **{f"mean_{a}": float(means[i, j]) for i, a in enumerate(arms)},
            }
            for j, name in enumerate(graph.tasks)
        ]
    )
    vidx = {g: i for i, g in enumerate(map(str, adata.var_names))}
    kept = [g for g in keep if g in vidx]
    Y, g_raw, t_raw, mice = _mouse_means(
        _to_dense(adata.layers["gene_scores"][:, [vidx[g] for g in kept]]),
        adata.obs,
    )
    g = _codes(g_raw, genotypes)
    t = _codes(t_raw, treatments)
    design = np.column_stack([np.ones(len(g)), g, t, g * t])
    A = graph.A_tg.detach().cpu().numpy() > 0
    gene_sets = {
        task: [keep[j] for j in np.flatnonzero(A[ti]) if keep[j] in vidx]
        for ti, task in enumerate(graph.tasks)
    }
    index = pylimma.ids2indices(gene_sets, kept)
    pb = ad.AnnData(
        X=Y,
        obs=pd.DataFrame(
            {"genotype": g_raw, "treatment": t_raw},
            index=pd.Index(np.asarray(mice, dtype=str), name="sample"),
        ),
        var=pd.DataFrame(index=kept),
    )
    try:
        cam = pylimma.camera(
            pb,
            index,
            design=design,
            contrast=design.shape[1] - 1,
            inter_gene_cor=None,
            sort=False,
        )
    except ValueError:
        # ponytail: camera_pr when residual df < 1 (human 4 libraries)
        beta, *_ = np.linalg.lstsq(design, Y, rcond=None)
        cam = pylimma.camera_pr(beta[3], index, sort=False)
    cam = cam.copy()
    cam["task"] = cam.index.astype(str)
    tab = tab.merge(
        cam[["task", "p_value"]].rename(columns={"p_value": "p_camera"}),
        on="task",
        how="left",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / f"{stem}.csv", index=False)
    gene_h_np = gene_h.detach().cpu().numpy() if torch.is_tensor(gene_h) else gene_h
    M, g_m, t_m, _ = _mouse_means(np.asarray(gene_h_np, dtype=np.float64), adata.obs)
    arm = {
        f"{gi}_{ti}": M[(g_m == gi) & (t_m == ti)].mean(0)
        for gi in genotypes
        for ti in treatments
    }
    g0, g1 = genotypes
    t0, t1 = treatments
    wv, wi = arm[f"{g0}_{t0}"], arm[f"{g0}_{t1}"]
    tv, ti = arm[f"{g1}_{t0}"], arm[f"{g1}_{t1}"]
    eff = pd.DataFrame(
        {
            "gene": keep,
            "il1": 0.5 * (wi + ti) - 0.5 * (wv + tv),
            "interaction": _interaction_2x2(wv, wi, tv, ti),
        }
    )
    A_tg = graph.A_tg.detach().cpu().numpy()
    rows = []
    for ti, task in enumerate(graph.tasks):
        gidx = np.flatnonzero(A_tg[ti] > 0)
        if gidx.size == 0:
            continue
        sub = eff.iloc[gidx].copy()
        sub["task"] = task
        rows.append(sub)
    if rows:
        genes = pd.concat(rows, ignore_index=True)
        genes.to_csv(OUT / f"{stem}_genes.csv", index=False)
        _plot_gene_cloud(genes, stem, treat_labels[-1])
    _plot_task_2x2(
        tab,
        stem,
        genotypes=genotypes,
        treatments=treatments,
        treat_labels=treat_labels,
    )


def main():
    train_vnn(
        _sccellfie(human=False),
        pd.read_csv(MOUSE_DB / "Task_by_Gene.csv", index_col=0),
        stem="mice",
    )
    train_vnn(
        _sccellfie(human=True),
        pd.read_csv(HUMAN_DB / "Task_by_Gene.csv", index_col=0),
        stem="human",
        db=HUMAN_DB,
        treatments=HUMAN_TREATMENTS,
        treat_labels=("CTRL", "LPS"),
    )


if __name__ == "__main__":
    main()
