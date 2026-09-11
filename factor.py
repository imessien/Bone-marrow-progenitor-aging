#!/usr/bin/env python3
from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache_bm")

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy import sparse
from sklearn.mixture import GaussianMixture
from torch_geometric.utils import scatter

try:
    import cupy as cp
except ImportError:
    cp = None


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
EPOCHS = 100
HIDDEN = 96
STEPS = 10
BATCH_SIZE = 2048
N_PERM = 9_999
PERM_BATCH = 20_000

KEEP_SUBSYSTEMS = (
    "KREBS CYCLE",
    "OXYDATIVE PHOSPHORYLATION",
    "PENTOSE PHOSPHATE PATHWAY",
    "ATP GENERATION",
)
SPECIES = (
    {
        "stem": "mice",
        "qc": QC,
        "db": MOUSE_DB,
        "organism": "mouse",
        "tmap": MOUSE_TMAP,
        "treatments": TREATMENTS,
        "treat_labels": ("vehicle", "IL-1"),
    },
    {
        "stem": "human",
        "qc": HUMAN_QC,
        "db": HUMAN_DB,
        "organism": "human",
        "tmap": HUMAN_TMAP,
        "treatments": HUMAN_TREATMENTS,
        "treat_labels": ("CTRL", "LPS"),
    },
)
MIX_LOW = "#4C72B0"
MIX_HIGH = "#C44E52"
GMM_MIN_N = 8
LOCAL_OUT = Path(__file__).resolve().parent / "results" / "chip_metabolic_graph"
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


def _sccellfie(spec):
    raw = ad.read_h5ad(spec["qc"])
    if spec["stem"] == "human":
        adata = raw[raw.obs["compartment"].astype(str).eq("HSPC")].copy()
    else:
        m = (
            raw.obs["lineage"].astype(str).eq(LINEAGE)
            & raw.obs["genotype"].isin(KEEP_GENOTYPE)
            & raw.obs["treatment"].astype(str).isin(KEEP_TREATMENT)
        )
        adata = raw[m].copy()
    tmap, organism, db = spec["tmap"], spec["organism"], spec["db"]
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    if "n_counts" not in adata.obs:
        adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
    adata.obs["genotype"] = (
        adata.obs["genotype"].astype(str).map({"WT": "WT", "Tet2_KO": "Tet2"})
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
    import rapids_singlecell as rsc

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
):
    info = _task_info(db)
    task_to_sub = info["Subsystem"].astype(str).to_dict()
    sub_to_sys = dict(zip(info["Subsystem"].astype(str), info["System"].astype(str)))
    g_idx = {g: i for i, g in enumerate(gene_names)}
    edges: list[tuple[int, int]] = []
    kept: list[str] = []
    for t in (x for ts in _axis_tasks(db).values() for x in ts):
        if t not in tbg.index:
            continue
        row = tbg.loc[t]
        pairs = [
            (len(kept), g_idx[g]) for g in map(str, row[row > 0].index) if g in g_idx
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


def _recon_mse(model, Xt, idx, *, use_amp):
    model.eval()
    tot = 0.0
    n = 0
    bs = min(BATCH_SIZE, max(int(idx.numel()), 1))
    with torch.no_grad():
        for i in range(0, idx.numel(), bs):
            xb = Xt.index_select(0, idx[i : i + bs])
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                y = model(xb)
            tot += F.mse_loss(y.float(), xb.float(), reduction="sum").item()
            n += xb.numel()
    return tot / max(n, 1)


def _plot_loss(hist, stem):
    ep = np.arange(1, len(hist) + 1)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(ep, hist, color="0.2")
    ax.set_xlabel("epoch")
    ax.set_ylabel("reconstruction MSE")
    ax.set_title("Mice" if stem == "mice" else "Human", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _out(f"{stem}_loss.png", fig)
    plt.close(fig)
    _out(f"{stem}_loss.csv", pd.DataFrame({"epoch": ep, "train": hist}))


def _fit(model, Xt, device, *, stem, epochs: int = EPOCHS):
    if device.type != "cuda":
        raise SystemExit(f"{stem}: VNN must train on CUDA, got {device}")
    kw = {"lr": 1e-3, "weight_decay": 1e-3}
    try:
        opt = torch.optim.AdamW(model.parameters(), fused=True, **kw)
    except (TypeError, RuntimeError):
        opt = torch.optim.AdamW(model.parameters(), **kw)
    n = Xt.size(0)
    idx = torch.arange(n, device=device)
    bs = min(BATCH_SIZE, n)
    scaler = torch.cuda.amp.GradScaler()
    hist = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            xb = Xt.index_select(0, perm[i : i + bs])
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = F.mse_loss(model(xb), xb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        hist.append(_recon_mse(model, Xt, idx, use_amp=True))
    _plot_loss(hist, stem)
    print(f"{stem}: epoch {epochs} train MSE {hist[-1]:.5g} on {device}", flush=True)


def _encode_cells(model, Xt):
    model.eval()
    hs, gs = [], []
    bs = min(BATCH_SIZE, Xt.size(0))
    with torch.no_grad():
        for i in range(0, Xt.size(0), bs):
            _, h_t, h_g = model.encode(Xt[i : i + bs])
            hs.append(h_t.mean(-1))
            gs.append(h_g.mean(-1))
    return torch.cat(hs, dim=0), torch.cat(gs, dim=0)


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


def _fmt_p(p):
    if not np.isfinite(p):
        return "NA"
    return f"{p:.4f}" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def _p_label(p, *, bold: bool):
    s = _fmt_p(p)
    if bold:
        return rf"$\mathbf{{p = {s}}}$"
    return f"$p$ = {s}"


_TASK_INFO: dict[str, pd.DataFrame] = {}


def _task_info(db: Path):
    key = str(db)
    if key not in _TASK_INFO:
        info = pd.read_csv(db / "Task-Info.csv")
        info["Task"] = info["Task"].astype(str)
        _TASK_INFO[key] = info.set_index("Task")
    return _TASK_INFO[key]


def _axis_tasks(db: Path):
    info = _task_info(db)
    groups: dict[str, list[str]] = {}
    for task, sub in info["Subsystem"].astype(str).items():
        sub = str(sub)
        if sub not in KEEP_SUBSYSTEMS:
            continue
        groups.setdefault(sub, []).append(str(task))
    return groups


def _out(name, obj):
    saved, err = False, None
    for dest in (OUT, LOCAL_OUT):
        try:
            dest.mkdir(parents=True, exist_ok=True)
            path = dest / name
            if hasattr(obj, "savefig"):
                obj.savefig(path, bbox_inches="tight", pad_inches=0.4, dpi=200)
            else:
                obj.to_csv(path, index=False)
            saved = True
        except OSError as e:
            err = e
    if not saved and err is not None:
        raise err


def _read_table(name: str):
    for dest in (LOCAL_OUT, OUT):
        try:
            return pd.read_csv(dest / name)
        except FileNotFoundError:
            continue
    return None


def _gmm_fit(X, *, seed: int = 0):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X = X[np.isfinite(X).all(axis=1)]
    if X.shape[0] < GMM_MIN_N:
        return None
    model = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=seed,
        n_init=3,
        max_iter=200,
    )
    model.fit(X)
    if X.shape[1] == 1:
        order = np.argsort(model.means_.ravel())
    else:
        order = np.argsort(np.linalg.norm(model.means_, axis=1))
    return model, order


def _cov_ellipse(ax, mean, cov, nstd: float = 1.5, **kw):
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (2, 2):
        return
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-12, None)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * nstd * np.sqrt(vals)
    ax.add_patch(Ellipse(xy=mean, width=w, height=h, angle=theta, **kw))


def _codes(values, levels):
    return pd.Categorical(
        np.asarray(values).astype(str), categories=list(levels)
    ).codes.astype(np.int64)


def _arm_stats(y, genotype, treatment, genotypes, treatments, device):
    y = torch.as_tensor(y, device=device, dtype=torch.float64)
    if y.ndim == 1:
        y = y.unsqueeze(1)
    n_t = len(treatments)
    arms = tuple(f"{g}_{t}" for g in genotypes for t in treatments)
    g = torch.as_tensor(_codes(genotype, genotypes), device=device)
    t = torch.as_tensor(_codes(treatment, treatments), device=device)
    arm = g * n_t + t
    counts = torch.bincount(arm, minlength=len(arms))
    if int(counts.min().item()) < 1:
        missing = [arms[i] for i, c in enumerate(counts.tolist()) if c < 1]
        raise ValueError(f"empty arm {missing}")
    means = scatter(y, arm, dim=0, dim_size=len(arms), reduce="mean")
    # arms layout: (g0,t0), (g0,t1), (g1,t0), (g1,t1)
    return {
        "tet2": means[2] - means[0],
        "il1": means[1] - means[0],
        "interaction": (means[3] - means[1]) - (means[2] - means[0]),
        "means": means,
        "arms": arms,
    }


def _task_title(name: str):
    t = str(name)
    if " - " in t:
        t = t.split(" - ")[-1].strip()
    elif t.endswith(")") and "(" in t:
        t = t[t.rfind("(") + 1 : t.rfind(")")].strip()
    return t[:1].upper() + t[1:] if t else t


def _readable_gene(name: str):
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


def _task_groups(names, db: Path):
    axis_of = {t: axis for axis, ts in _axis_tasks(db).items() for t in ts}
    groups = {axis: [] for axis in KEEP_SUBSYSTEMS}
    seen = set()
    for n in map(str, names):
        if n in seen:
            continue
        seen.add(n)
        axis = axis_of.get(n)
        if axis is not None:
            groups[axis].append(n)
    return [(axis, groups[axis]) for axis in KEEP_SUBSYSTEMS if groups[axis]]


def _is_mt_nd(name: str):
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
        left=0.16,
        right=0.88,
        top=0.90,
        bottom=0.16,
        wspace=0.55,
        hspace=0.70,
    )
    return fig, axs, ncols


def _row_cols(n, ncols):
    if n <= 1:
        return [0]
    if n >= ncols:
        return list(range(n))
    return [round(i * (ncols - 1) / (n - 1)) for i in range(n)]


def _arm_grid(row, spec):
    g, t = GENOTYPES, spec["treatments"]
    grid = np.array([[row[f"mean_{gi}_{ti}"] for ti in t] for gi in g], dtype=float)
    grid = grid - np.nanmean(grid)
    m = float(np.nanmax(np.abs(grid)))
    return np.clip(grid / m, -1.0, 1.0) if m > 0 else np.zeros_like(grid)


def _paint_2x2(ax, grid, spec, *, cmap, vmin, vmax, xlab, ylab=None, title=None):
    im = ax.imshow(
        grid, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", aspect="equal"
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(list(spec["treat_labels"]))
    ax.set_yticks([0, 1])
    ax.set_yticklabels(list(GENOTYPES))
    if title is not None:
        ax.set_title(title, fontweight="bold", fontsize=12, pad=10)
    ax.set_xlabel(xlab, fontsize=8, labelpad=8)
    if ylab is not None:
        ax.set_ylabel(ylab, fontweight="bold", fontsize=9, labelpad=12)
    for ii in range(grid.shape[0]):
        for jj in range(grid.shape[1]):
            ax.text(
                jj,
                ii,
                f"{float(grid[ii, jj]):.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )
    return im


def _plot_gene_cloud(tab, spec):
    stem, challenge = spec["stem"], spec["treat_labels"][-1]
    groups = _task_groups(tab["task"], spec["db"])
    if not groups:
        return
    fig, axs, ncols = _task_subplots(groups, w=6.0, h=5.2, share=True)
    fig.subplots_adjust(top=0.88, bottom=0.16, hspace=0.80)
    drew_mix = False
    for i, (axis, tasks) in enumerate(groups):
        slot = dict(zip(_row_cols(len(tasks), ncols), tasks))
        for j in range(ncols):
            ax = axs[i][j]
            if j not in slot:
                ax.axis("off")
                continue
            task = slot[j]
            sub = tab.loc[tab["task"] == task]
            x = sub["il1"].to_numpy(dtype=float)
            y = sub["interaction"].to_numpy(dtype=float)
            m = max(float(np.nanmax(np.abs(x))), float(np.nanmax(np.abs(y))), 1e-12)
            xn, yn = x / m, y / m
            xy = np.column_stack([xn, yn])
            ax.axhline(0, color="0.85", lw=0.6, zorder=0)
            ax.axvline(0, color="0.85", lw=0.6, zorder=0)
            fit = _gmm_fit(xy)
            if fit is None:
                ax.scatter(xn, yn, s=22, c="0.35", linewidths=0, zorder=2)
            else:
                model, order = fit
                lab = model.predict(xy)
                remap = {int(order[0]): 0, int(order[1]): 1}
                comp = np.array([remap[int(v)] for v in lab])
                ax.scatter(
                    xn,
                    yn,
                    s=22,
                    c=np.where(comp == 1, MIX_HIGH, MIX_LOW),
                    linewidths=0,
                    zorder=2,
                )
                for k, name in zip(order, ("low", "high")):
                    mean = model.means_[int(k)]
                    color = MIX_LOW if name == "low" else MIX_HIGH
                    _cov_ellipse(
                        ax,
                        mean,
                        model.covariances_[int(k)],
                        facecolor="none",
                        edgecolor=color,
                        lw=1.1,
                        zorder=1,
                    )
                    ax.annotate(
                        name,
                        mean,
                        fontsize=8,
                        fontweight="bold",
                        color=color,
                        ha="center",
                        va="bottom",
                        xytext=(0, 6),
                        textcoords="offset points",
                    )
                drew_mix = True
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
            p = float(sub["p_perm"].iloc[0]) if "p_perm" in sub.columns else np.nan
            pred = _task_title(task) if np.isfinite(p) and p < P_SIG else None
            ax.set_title(
                _task_title(task),
                fontweight="bold" if pred else "normal",
                fontsize=11,
                pad=12,
                color="0.0" if pred else "0.15",
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(length=3, labelsize=8)
            ax.set_xlabel(f"{challenge} main", fontsize=8, labelpad=8)
            if j == 0:
                ax.set_ylabel("interaction", labelpad=10)
        used = sorted(slot)
        pos0 = axs[i][used[0]].get_position()
        pos1 = axs[i][used[-1]].get_position()
        fig.text(
            0.5 * (pos0.x0 + pos1.x1),
            pos0.y1 + 0.028,
            axis,
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )
    if drew_mix:
        fig.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=c,
                    markersize=7,
                    label=lab,
                )
                for c, lab in ((MIX_LOW, "low state"), (MIX_HIGH, "high state"))
            ],
            loc="lower center",
            ncol=2,
            frameon=False,
            fontsize=9,
            bbox_to_anchor=(0.5, 0.02),
        )
    fig.suptitle(
        "Human" if stem == "human" else "Mice",
        fontweight="bold",
        fontsize=14,
        y=0.98,
    )
    _out(f"{stem}_genes.png", fig)
    plt.close(fig)


def _plot_task_2x2(tab, spec):
    groups = _task_groups(tab["task"], spec["db"])
    if not groups:
        return
    fig, axs, ncols = _task_subplots(groups, w=6.0, h=5.2, share=False)
    im, used = None, []
    pmap = tab.set_index("task")["p_perm"]
    for i, (axis, tasks) in enumerate(groups):
        slot = dict(zip(_row_cols(len(tasks), ncols), tasks))
        for j in range(ncols):
            ax = axs[i][j]
            if j not in slot:
                ax.axis("off")
                continue
            task = slot[j]
            row = tab.loc[tab["task"] == task].iloc[0]
            p = float(pmap[task]) if task in pmap.index else np.nan
            pred = np.isfinite(p) and p < P_SIG
            xlab = _p_label(p, bold=pred)
            im = _paint_2x2(
                ax,
                _arm_grid(row, spec),
                spec,
                cmap=plt.cm.RdBu_r,
                vmin=-1.0,
                vmax=1.0,
                xlab=xlab,
                ylab=axis if j == 0 else None,
            )
            ax.set_xlabel(
                xlab, fontsize=8, fontweight="bold" if pred else "normal", labelpad=10
            )
            ax.set_title(
                _task_title(task),
                fontweight="bold" if pred else "normal",
                fontsize=12 if pred else 11,
                pad=12,
                color="0.0" if pred else "0.15",
            )
            used.append(ax)
    cbar = fig.colorbar(im, ax=used, fraction=0.025, pad=0.06)
    cbar.set_ticks([-1.0, 1.0])
    cbar.set_ticklabels(["−1", "1"])
    cbar.set_label("Relative mean")
    fig.suptitle(
        "Human" if spec["stem"] == "human" else "Mice",
        fontweight="bold",
        fontsize=14,
        y=0.98,
    )
    _out(f"{spec['stem']}.png", fig)
    plt.close(fig)


def _set_mean_perm(
    M, genotype, treatment, genotypes, treatments, gene_sets, genes, device
):
    # Competitive mean: |task-set mean − universe mean| on the frozen
    # gene-embedding interaction, then N_PERM gene shuffles. Smallest p is
    # 1/(N_PERM+1).
    inter = (
        _arm_stats(M, genotype, treatment, genotypes, treatments, device)["interaction"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    gix = {g: i for i, g in enumerate(genes)}
    names, sets_idx = [], []
    for name, members in gene_sets.items():
        idx = np.array([gix[g] for g in members if g in gix], dtype=np.int64)
        if idx.size == 0:
            continue
        names.append(name)
        sets_idx.append(idx)
    mean_all = float(inter.mean())
    obs = np.array([abs(inter[idx].mean() - mean_all) for idx in sets_idx])
    rng = np.random.default_rng(0)
    geq = np.zeros(len(sets_idx), dtype=np.int64)
    G = inter.size
    for start in range(0, N_PERM, PERM_BATCH):
        b = min(PERM_BATCH, N_PERM - start)
        shuf = rng.permuted(np.broadcast_to(inter, (b, G)).copy(), axis=1)
        for j, idx in enumerate(sets_idx):
            null = np.abs(shuf[:, idx].mean(axis=1) - mean_all)
            geq[j] += int(np.sum(null >= obs[j]))
    p = (geq + 1) / (N_PERM + 1)
    return pd.DataFrame({"task": names, "p_value": p})


def _device(index=0):
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for metabolic VNN training.")
    if index < 0 or index >= torch.cuda.device_count():
        raise SystemExit(
            f"CUDA device {index} unavailable (count={torch.cuda.device_count()})."
        )
    torch.cuda.set_device(index)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch.device(f"cuda:{index}")


def train_vnn(adata, tbg, spec, device):
    stem, db = spec["stem"], spec["db"]
    genotypes, treatments = GENOTYPES, spec["treatments"]
    hyp = {
        g
        for ts in _axis_tasks(db).values()
        for t in ts
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
    if cp is not None:
        cp.get_default_memory_pool().free_all_blocks()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    Xt = torch.from_numpy(np.ascontiguousarray(X)).to(device)
    model = MetabolicVNN(graph).to(device)
    print(f"{stem}: VNN on {device}, n={Xt.size(0)} genes={Xt.size(1)}", flush=True)
    _fit(model, Xt, device, stem=stem)
    states, gene_h = _encode_cells(model, Xt)
    cells = pd.DataFrame(
        {
            "sample_name": adata.obs["sample_name"].astype(str).to_numpy(),
            "genotype": adata.obs["genotype"].astype(str).to_numpy(),
            "treatment": adata.obs["treatment"].astype(str).to_numpy(),
        }
    )
    st = states.detach().cpu().numpy()
    for j, name in enumerate(graph.tasks):
        cells[name] = st[:, j]
    _out(f"{stem}_cells.csv", cells)
    Y_m, g_m, t_m, _mice = _mouse_means(st, adata.obs)
    obs = _arm_stats(Y_m, g_m, t_m, genotypes, treatments, device)
    means = obs["means"].detach().cpu().numpy()
    arms = obs["arms"]
    tab = pd.DataFrame(
        [
            {
                "task": name,
                "tet2": float(obs["tet2"][j]),
                "il1": float(obs["il1"][j]),
                "interaction": float(obs["interaction"][j]),
                **{f"mean_{a}": float(means[i, j]) for i, a in enumerate(arms)},
            }
            for j, name in enumerate(graph.tasks)
        ]
    )
    gene_h_np = np.asarray(
        gene_h.detach().cpu().numpy() if torch.is_tensor(gene_h) else gene_h,
        dtype=np.float64,
    )
    M, g_raw, t_raw, _ = _mouse_means(gene_h_np, adata.obs)
    A = graph.A_tg.detach().cpu().numpy() > 0
    gene_sets = {
        task: [keep[j] for j in np.flatnonzero(A[ti])]
        for ti, task in enumerate(graph.tasks)
    }
    perm = _set_mean_perm(
        M, g_raw, t_raw, genotypes, treatments, gene_sets, keep, device
    )
    tab = tab.merge(
        perm[["task", "p_value"]].rename(columns={"p_value": "p_perm"}),
        on="task",
        how="left",
    )
    tab["n_perm"] = N_PERM
    tab["epochs"] = EPOCHS
    _out(f"{stem}.csv", tab)
    arm = {
        f"{gi}_{ti}": M[(g_raw == gi) & (t_raw == ti)].mean(0)
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
            "interaction": (ti - wi) - (tv - wv),
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
    if not rows:
        raise ValueError(f"{stem}: no task gene edges for gene cloud")
    genes = pd.concat(rows, ignore_index=True)
    genes["p_perm"] = genes["task"].map(tab.set_index("task")["p_perm"])
    _out(f"{stem}_genes.csv", genes)
    _plot_gene_cloud(genes, spec)
    _plot_task_2x2(tab, spec)


def _run_species(spec, gpu):
    print(f"{spec['stem']}: GPU {gpu}", flush=True)
    if cp is not None:
        cp.cuda.Device(gpu).use()
    train_vnn(
        _sccellfie(spec),
        pd.read_csv(spec["db"] / "Task_by_Gene.csv", index_col=0),
        spec,
        _device(gpu),
    )


def replot_saved():
    for spec in SPECIES:
        stem = spec["stem"]
        tab, genes = _read_table(f"{stem}.csv"), _read_table(f"{stem}_genes.csv")
        if tab is None or genes is None:
            raise SystemExit(f"missing {stem}.csv or {stem}_genes.csv")
        if "p_perm" not in genes.columns:
            genes = genes.copy()
            genes["p_perm"] = genes["task"].map(tab.set_index("task")["p_perm"])
        _plot_gene_cloud(genes, spec)
        _plot_task_2x2(tab, spec)


def _gpu_free_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    return [int(x.strip()) for x in out.splitlines() if x.strip()]


def main():
    tabs = [_read_table(f"{s['stem']}.csv") for s in SPECIES]
    genes = [_read_table(f"{s['stem']}_genes.csv") for s in SPECIES]
    fresh = (
        all(t is not None and g is not None for t, g in zip(tabs, genes))
        and all("p_perm" in t.columns for t in tabs)
        and all("patience" not in t.columns for t in tabs)
        and all(
            "n_perm" in t.columns
            and int(t["n_perm"].iloc[0]) == N_PERM
            and "epochs" in t.columns
            and int(t["epochs"].iloc[0]) == EPOCHS
            for t in tabs
        )
    )
    if fresh:
        replot_saved()
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise SystemExit("CUDA required for metabolic VNN training.")
    free = _gpu_free_mib()
    gpus = [i for i, mib in enumerate(free) if mib >= 4000]
    if not gpus:
        raise SystemExit(f"no GPU with 4 GiB free (free MiB={free})")
    if len(gpus) >= 2:
        jobs = ((SPECIES[0], gpus[0]), (SPECIES[1], gpus[1]))
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=_run_species, args=job) for job in jobs]
        for p in procs:
            p.start()
        bad = []
        for p, (spec, gpu) in zip(procs, jobs):
            p.join()
            if p.exitcode:
                bad.append((spec["stem"], gpu, p.exitcode))
        if bad:
            raise SystemExit(f"species worker failed: {bad}")
        return
    for spec in SPECIES:
        _run_species(spec, gpus[0])


if __name__ == "__main__":
    main()
