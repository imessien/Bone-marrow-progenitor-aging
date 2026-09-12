#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
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
GENOTYPES = ("WT", "Tet2")
KEEP_GENOTYPE = ("WT", "Tet2_KO")

SPECIES = tuple(
    {
        "stem": stem,
        "qc": qc,
        "db": db,
        "organism": organism,
        "tmap": tmap,
        "keep_treatment": tuple(tmap),
        "treatments": (tx := tuple(tmap.values())),
        "treat_labels": treat_labels,
        "root_arm": f"{GENOTYPES[0]}_{tx[0]}",
        "arm_order": tuple(f"{g}_{t}" for g in GENOTYPES for t in tx),
        "arm_labels": tuple(f"{g} · {lab}" for g in GENOTYPES for lab in treat_labels),
    }
    for stem, qc, db, organism, tmap, treat_labels in (
        (
            "mice",
            QC,
            MOUSE_DB,
            "mouse",
            {"vehicle": "vehicle", "IL1b": "IL1"},
            ("vehicle", "IL-1"),
        ),
        (
            "human",
            HUMAN_QC,
            HUMAN_DB,
            "human",
            {"CTRL": "CTRL", "LPS": "LPS"},
            ("CTRL", "LPS"),
        ),
    )
)

LEVELS = ("task", "subsystem", "system")
EPOCHS = 100
HIDDEN = 96
STEPS = 10
BATCH_SIZE = 2048
N_PERM = 9_999
PERM_BATCH = 20_000

SUBSYSTEMS = (
    "KREBS CYCLE",
    "OXYDATIVE PHOSPHORYLATION",
    "PENTOSE PHOSPHATE PATHWAY",
    "ATP GENERATION",
)
MIX_LOW = "#4C72B0"
MIX_HIGH = "#C44E52"
GMM_MIN_N = 8
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
            & raw.obs["treatment"].astype(str).isin(spec["keep_treatment"])
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
    adata.uns.pop("log1p", None)
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
        self,
        graph: HypothesisGraph,
        hidden: int = HIDDEN,
        steps: int = STEPS,
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
            sel = idx[i : i + bs]
            xb = Xt.index_select(0, sel)
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
    ax.set_ylabel("train loss")
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
            sel = perm[i : i + bs]
            xb = Xt.index_select(0, sel)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = F.mse_loss(model(xb), xb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        hist.append(_recon_mse(model, Xt, idx, use_amp=True))
    _plot_loss(hist, stem)
    print(f"{stem}: epoch {epochs} train loss {hist[-1]:.5g} on {device}", flush=True)


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
        if sub not in SUBSYSTEMS:
            continue
        groups.setdefault(sub, []).append(str(task))
    return groups


def _out(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    if hasattr(obj, "savefig"):
        obj.savefig(path, bbox_inches="tight", pad_inches=0.4, dpi=200)
    else:
        obj.to_csv(path, index=False)


def _read_table(name: str):
    path = OUT / name
    if not path.exists():
        return None
    return pd.read_csv(path)


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
    by = {a: means[i] for i, a in enumerate(arms)}
    g0, g1 = genotypes
    t0, t1 = treatments
    return {
        "tet2": by[f"{g1}_{t0}"] - by[f"{g0}_{t0}"],
        "il1": by[f"{g0}_{t1}"] - by[f"{g0}_{t0}"],
        "interaction": (by[f"{g1}_{t1}"] - by[f"{g0}_{t1}"])
        - (by[f"{g1}_{t0}"] - by[f"{g0}_{t0}"]),
        "means": means,
        "arms": arms,
    }


def _title(name):
    t = str(name)
    if " - " in t:
        t = t.split(" - ")[-1].strip()
    elif t.endswith(")") and "(" in t:
        t = t[t.rfind("(") + 1 : t.rfind(")")].strip()
    if len(t) > 40 and "(" in t:
        t = t[: t.find("(")].strip()
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
    groups = {axis: [] for axis in SUBSYSTEMS}
    seen = set()
    for n in map(str, names):
        if n in seen:
            continue
        seen.add(n)
        axis = axis_of.get(n)
        if axis is not None:
            groups[axis].append(n)
    return [(axis, groups[axis]) for axis in SUBSYSTEMS if groups[axis]]


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
            pred = _title(task) if np.isfinite(p) and p < P_SIG else None
            ax.set_title(
                _title(task),
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
                _title(task),
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


def _panel_rxns(db):
    info = pd.read_csv(db / "Task-Info.csv")
    tbr = pd.read_csv(db / "Task_by_Rxn.csv", index_col=0)
    out = {}
    for sub in SUBSYSTEMS:
        tasks = info.loc[info["Subsystem"].astype(str).eq(sub), "Task"].astype(str)
        for t in tasks:
            if t not in tbr.index:
                continue
            row = tbr.loc[t]
            for r in row.index[np.asarray(row) > 0]:
                out.setdefault(str(r), set()).add(sub)
    return {r: frozenset(s) for r, s in out.items()}


def _rxn_proxy(adata, db):
    if not hasattr(adata, "reactions"):
        raise ValueError("adata.reactions required (run scCellFie first)")
    panel = _panel_rxns(db)
    R = _to_dense(adata.reactions.X).astype(np.float64)
    r_idx = {str(r): i for i, r in enumerate(adata.reactions.var_names)}
    cols, labels = [], []
    for sub in SUBSYSTEMS:
        ix = [r_idx[r] for r, subs in panel.items() if sub in subs and r in r_idx]
        if not ix:
            continue
        v = R[:, ix].mean(1)
        v = (v - v.mean()) / (v.std() + 1e-6)
        cols.append(v.astype(np.float32))
        labels.append(sub)
    if not cols:
        raise ValueError("no subsystem reaction proxies")
    return np.column_stack(cols), labels


BRANCHES = ("stem", "myeloid", "MegE", "lymphoid")
BRANCH_COLORS = {
    "stem": "#1f77b4",
    "myeloid": "#2ca02c",
    "MegE": "#d62728",
    "lymphoid": "#9467bd",
}
CELLTYPIST_MODEL = "Immune_All_Low.pkl"


def _ct_branch(label):
    t = str(label).lower()
    if any(
        k in t
        for k in (
            "megakaryocyte",
            "platelet",
            "erythroid",
            "erythrocyte",
            "early mk",
            "megakaryocyte-erythroid",
            "memp",
        )
    ):
        return "MegE"
    if any(
        k in t
        for k in (
            "b cell",
            "t cell",
            "t lymphoid",
            "nk cell",
            "ilc",
            "plasma",
            "lymphocyte",
            "clp",
            "elp",
            "etp",
            "thymocyte",
            "mait",
            "gamma-delta",
            "cd8",
            "cd4",
            "early lymphoid",
            "follicular",
            "germinal",
        )
    ):
        return "lymphoid"
    if t == "nk":
        return "lymphoid"
    if any(
        k in t
        for k in (
            "monocyte",
            "macrophage",
            "neutrophil",
            "granulocyte",
            "dendritic",
            "dc1",
            "dc2",
            "dc3",
            "pdc",
            "gmp",
            "cmp",
            "mdp",
            "mnp",
            "myelocyte",
            "promyelocyte",
            "basophil",
            "eosinophil",
            "mast cell",
            "mono-mac",
            "neutrophil-myeloid",
            "cycling dc",
            "migratory dc",
            "transitional dc",
            "dc precursor",
            " dc",
        )
    ):
        return "myeloid"
    if t == "dc" or t.startswith("dc"):
        return "myeloid"
    return "stem"


def _load_hspc(spec):
    raw = ad.read_h5ad(spec["qc"])
    if spec["stem"] == "human":
        key = "compartment" if "compartment" in raw.obs else "lineage"
        adata = raw[raw.obs[key].astype(str).eq("HSPC")].copy()
    else:
        m = (
            raw.obs["lineage"].astype(str).eq(LINEAGE)
            & raw.obs["genotype"].isin(KEEP_GENOTYPE)
            & raw.obs["treatment"].astype(str).isin(spec["keep_treatment"])
        )
        adata = raw[m].copy()
    adata.obs["genotype"] = (
        adata.obs["genotype"]
        .astype(str)
        .map({"WT": "WT", "Tet2_KO": "Tet2", "TET2_KO": "Tet2"})
    )
    adata.obs["treatment"] = adata.obs["treatment"].astype(str).map(spec["tmap"])
    adata.obs["arm"] = (
        adata.obs["genotype"].astype(str) + "_" + adata.obs["treatment"].astype(str)
    )
    adata = adata[adata.obs["arm"].isin(spec["arm_order"])].copy()
    if "sample_name" not in adata.obs:
        adata.obs["sample_name"] = adata.obs_names.astype(str)
    return adata


def _branch_occ(bdata, adata, spec):
    import scanpy as sc
    import celltypist
    from celltypist import models
    from sklearn.neighbors import NearestNeighbors

    scor = adata.copy()
    if "counts" in scor.layers:
        scor.X = scor.layers["counts"].copy()
        scor.uns.pop("log1p", None)
        sc.pp.normalize_total(scor, target_sum=1e4)
        sc.pp.log1p(scor)
    elif "gene_scores" in scor.layers:
        scor.X = scor.layers["gene_scores"].copy()
    else:
        xmax = float(np.max(_to_dense(scor.X)))
        if xmax > 20:
            scor.uns.pop("log1p", None)
            sc.pp.normalize_total(scor, target_sum=1e4)
            sc.pp.log1p(scor)
    if spec["stem"] == "mice":
        scor.var_names = pd.Index(scor.var_names.astype(str).str.upper())
        scor = scor[:, ~scor.var_names.duplicated()].copy()

    pred = celltypist.annotate(
        scor,
        model=models.Model.load(model=CELLTYPIST_MODEL),
        majority_voting=False,
    )
    prob = pred.probability_matrix
    occ = np.zeros((scor.n_obs, len(BRANCHES)), dtype=np.float64)
    for j, col in enumerate(prob.columns.astype(str)):
        occ[:, BRANCHES.index(_ct_branch(col))] += prob.iloc[:, j].to_numpy(dtype=float)
    occ = occ / np.clip(occ.sum(axis=1, keepdims=True), 1e-12, None)

    Z = bdata.obsm["X_vnn"]
    k = min(30, max(5, bdata.n_obs // 50))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(Z)
    occ = occ[nn.kneighbors(Z, return_distance=False)].mean(axis=1)

    hard = [BRANCHES[i] for i in occ.argmax(axis=1)]
    br_order = [c for c in BRANCHES if c in set(hard)]
    bdata.obs["branch_pred"] = pd.Categorical(hard, categories=br_order)
    bdata.obs["celltypist"] = (
        pred.predicted_labels["predicted_labels"].astype(str).to_numpy()
    )
    for j, lab in enumerate(BRANCHES):
        bdata.obs[f"occupancy:{lab}"] = occ[:, j]
    bdata.obsm["X_occupancy"] = occ.astype(np.float32)


def _em(adata, cells, tab, spec):
    import scanpy as sc
    from pygam import LinearGAM, s
    from sklearn.preprocessing import StandardScaler
    from statsmodels.stats.multitest import multipletests

    stem = spec["stem"]
    if len(cells) != adata.n_obs:
        raise ValueError(f"{stem}: cells n={len(cells)} adata n={adata.n_obs}")
    meta = {"sample_name", "genotype", "treatment"}
    task_cols = [c for c in cells.columns if c not in meta and c not in SUBSYSTEMS]
    X_task = cells[task_cols].to_numpy(dtype=np.float64)
    if hasattr(adata, "reactions"):
        X_rate, rate_names = _rxn_proxy(adata, spec["db"])
    else:
        rate_names = [s for s in SUBSYSTEMS if s in cells.columns]
        if not rate_names:
            raise ValueError(
                f"{stem}: need adata.reactions (scCellFie) or subsystem columns in cells"
            )
        X_rate = cells[rate_names].to_numpy(dtype=np.float64)
    X = np.hstack([X_task, X_rate])
    Z = StandardScaler().fit_transform(np.nan_to_num(X, nan=0.0)).astype(np.float32)
    bdata = ad.AnnData(X=sparse.csr_matrix(Z))
    for c in ("genotype", "treatment", "arm", "sample_name"):
        if c in adata.obs:
            bdata.obs[c] = adata.obs[c].astype(str).to_numpy()
    if "arm" not in bdata.obs:
        bdata.obs["arm"] = (
            bdata.obs["genotype"].astype(str) + "_" + bdata.obs["treatment"].astype(str)
        )
    bdata.obsm["X_vnn"] = Z
    for j, name in enumerate(task_cols):
        bdata.obs[name] = X_task[:, j]
    for j, name in enumerate(rate_names):
        bdata.obs[name] = X_rate[:, j]

    _branch_occ(bdata, adata, spec)
    sc.pp.neighbors(bdata, use_rep="X_vnn", n_neighbors=15, metric="euclidean")
    sc.tl.umap(bdata, min_dist=0.3)
    sc.tl.diffmap(bdata, n_comps=15)
    root_arm = spec["root_arm"]
    mask = bdata.obs["arm"].astype(str).eq(root_arm).to_numpy()
    if not mask.any():
        raise ValueError(f"{stem}: no cells in root arm {root_arm}")
    cent = Z[mask].mean(0)
    root = int(np.argmin(((Z - cent) ** 2).sum(1)))
    bdata.uns["iroot"] = root
    sc.tl.dpt(bdata)
    pt = bdata.obs["dpt_pseudotime"].to_numpy(dtype=float)
    flip_key = (
        "OXYDATIVE PHOSPHORYLATION"
        if "OXYDATIVE PHOSPHORYLATION" in bdata.obs.columns
        else rate_names[0]
    )
    flip = np.asarray(bdata.obs[flip_key], dtype=float)
    if np.corrcoef(pt, flip)[0, 1] < 0:
        pt = float(np.nanmax(pt)) - pt
    bdata.obs["pseudotime_pred"] = pt
    um = bdata.obsm["X_umap"]

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    sca = ax.scatter(
        um[:, 0], um[:, 1], c=pt, s=5, cmap="gnuplot2", linewidths=0, rasterized=True
    )
    ax.scatter(um[root, 0], um[root, 1], c="cyan", s=40, linewidths=0.5, edgecolors="k")
    ax.set_title(f"{stem} · metabolic pseudotime", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.02)
    _out(f"{stem}_umap_lineage_pseudotime.png", fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    br = bdata.obs["branch_pred"].astype(str)
    for cat in br.unique():
        m = br.eq(cat).to_numpy()
        ax.scatter(
            um[m, 0],
            um[m, 1],
            c=BRANCH_COLORS.get(cat, "0.7"),
            s=5,
            linewidths=0,
            label=cat,
            rasterized=True,
        )
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.set_title(f"{stem} · branch occupancy", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    _out(f"{stem}_umap_lineage_branch.png", fig)
    plt.close(fig)

    occ_tab = (
        pd.crosstab(
            bdata.obs["arm"].astype(str),
            bdata.obs["branch_pred"].astype(str),
            normalize="index",
        )
        .reindex(index=list(spec["arm_order"]))
        .fillna(0.0)
    )
    _out(f"{stem}_lineage_occupancy.csv", occ_tab.reset_index())
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    bottom = np.zeros(len(occ_tab))
    x = np.arange(len(occ_tab))
    for col in [c for c in BRANCHES if c in occ_tab.columns]:
        ax.bar(
            x,
            occ_tab[col].to_numpy(),
            bottom=bottom,
            color=BRANCH_COLORS[col],
            width=0.7,
            label=col,
        )
        bottom = bottom + occ_tab[col].to_numpy()
    ax.set_xticks(x)
    ax.set_xticklabels(list(spec["arm_labels"]))
    ax.set_ylabel("Branch occupancy")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.set_title(f"{stem} · arm × branch", fontweight="bold")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    _out(f"{stem}_umap_lineage_occupancy.png", fig)
    plt.close(fig)

    gam_rows = []
    for arm in spec["arm_order"]:
        m = bdata.obs["arm"].astype(str).eq(arm).to_numpy()
        if m.sum() < 30:
            continue
        t = pt[m]
        for task in task_cols:
            y = bdata.obs[task].to_numpy(dtype=float)[m]
            gam = LinearGAM(s(0, n_splines=8)).fit(t.reshape(-1, 1), y)
            pvals = gam.statistics_.get("p_values", [np.nan])
            p = float(pvals[0]) if len(pvals) else np.nan
            r2 = gam.statistics_.get("pseudo_r2", {})
            expl = (
                float(r2.get("explained_deviance", np.nan))
                if isinstance(r2, dict)
                else np.nan
            )
            gam_rows.append(
                {
                    "arm": arm,
                    "task": task,
                    "task_short": _title(task),
                    "p_gam": p,
                    "explained_deviance": expl,
                    "n": int(m.sum()),
                }
            )
    gam = pd.DataFrame(gam_rows)
    if len(gam):
        ok = np.isfinite(gam["p_gam"].to_numpy())
        gam["q_gam"] = np.nan
        if ok.any():
            gam.loc[ok, "q_gam"] = multipletests(gam.loc[ok, "p_gam"], method="fdr_bh")[
                1
            ]
        gam = gam.sort_values(
            ["arm", "q_gam", "explained_deviance"], ascending=[True, True, False]
        )
    _out(f"{stem}_gam_tasks.csv", gam)

    if len(gam):
        top = (
            gam.dropna(subset=["explained_deviance"])
            .sort_values("explained_deviance", ascending=False)
            .groupby("arm", sort=False)
            .head(1)
        )
        tasks_plot = list(dict.fromkeys(top["task"].tolist()))[:4] or task_cols[:2]
        fig, axs = plt.subplots(
            1, len(tasks_plot), figsize=(3.4 * len(tasks_plot), 3.4), squeeze=False
        )
        for ax, task in zip(axs[0], tasks_plot):
            for arm, lab in zip(spec["arm_order"], spec["arm_labels"]):
                m = bdata.obs["arm"].astype(str).eq(arm).to_numpy()
                if m.sum() < 20:
                    continue
                order = np.argsort(pt[m])
                tt = pt[m][order]
                yy = bdata.obs[task].to_numpy(dtype=float)[m][order]
                bins = np.linspace(tt.min(), tt.max(), 12)
                dig = np.digitize(tt, bins)
                mu = [
                    yy[dig == i].mean() if (dig == i).any() else np.nan
                    for i in range(1, len(bins))
                ]
                xc = 0.5 * (bins[:-1] + bins[1:])
                ax.plot(xc, mu, label=lab, lw=1.5)
            ax.set_title(_title(task), fontsize=10, fontweight="bold")
            ax.set_xlabel("pseudotime")
            ax.set_ylabel("task score")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axs[0][-1].legend(frameon=False, fontsize=7, loc="best")
        fig.suptitle(f"{stem} · GAM trends", fontweight="bold", y=1.02)
        fig.tight_layout()
        _out(f"{stem}_gam_trends.png", fig)
        plt.close(fig)
    print(f"{stem}: EM progression written", flush=True)


def _arm_table(names, obs_stats, genotypes, treatments, pmap=None):
    means = obs_stats["means"].detach().cpu().numpy()
    arms = obs_stats["arms"]
    rows = []
    for j, name in enumerate(names):
        row = {
            "task": name,
            "tet2": float(obs_stats["tet2"][j]),
            "il1": float(obs_stats["il1"][j]),
            "interaction": float(obs_stats["interaction"][j]),
            **{f"mean_{a}": float(means[i, j]) for i, a in enumerate(arms)},
        }
        if pmap is not None and name in pmap.index:
            row["p_perm"] = float(pmap.loc[name])
        rows.append(row)
    return pd.DataFrame(rows)


def _import_de():
    try:
        from sccellfie.stats.differential_analysis import scanpy_differential_analysis

        return scanpy_differential_analysis
    except ImportError:
        return importlib.import_module(
            "sccellfie.stats.differential_analysis"
        ).scanpy_differential_analysis


def _rxn_de(adata, spec, device):
    if not hasattr(adata, "reactions"):
        raise ValueError(f"{spec['stem']}: adata.reactions missing")
    db = spec["db"]
    rx = adata.reactions.copy()
    for c in ("genotype", "treatment", "arm", "sample_name"):
        if c in adata.obs.columns:
            rx.obs[c] = adata.obs[c].astype(str).to_numpy()
    rx.obs["lineage"] = "HSPC"
    genotypes, treatments = GENOTYPES, spec["treatments"]
    g0, g1 = genotypes
    t0, t1 = treatments
    pairs = [
        (f"{g0}_{t0}", f"{g0}_{t1}"),
        (f"{g1}_{t0}", f"{g1}_{t1}"),
        (f"{g0}_{t0}", f"{g1}_{t0}"),
        (f"{g0}_{t1}", f"{g1}_{t1}"),
        (f"{g0}_{t0}", f"{g1}_{t1}"),
    ]
    de = _import_de()(
        rx,
        cell_type="HSPC",
        cell_type_key="lineage",
        condition_key="arm",
        condition_pairs=pairs,
        min_cells=20,
    )
    X = _to_dense(rx.X).astype(np.float64)
    Y_m, g_m, t_m, _ = _mouse_means(X, rx.obs)
    y = torch.tensor(Y_m, device=device, dtype=torch.float32)
    obs = _arm_stats(y, g_m, t_m, genotypes, treatments, device)
    names = list(map(str, rx.var_names))
    tab = _arm_table(names, obs, genotypes, treatments)
    tab = tab.rename(columns={"task": "reaction"})
    panel = _panel_rxns(db)
    tab["subsystem"] = [
        "; ".join(s for s in SUBSYSTEMS if s in panel.get(r, ()))
        for r in tab["reaction"].astype(str)
    ]
    tab["in_panel"] = tab["subsystem"].astype(str).ne("")

    def _pair(g1, g2):
        return de.loc[de["group1"].eq(g1) & de["group2"].eq(g2)].drop_duplicates(
            "feature"
        )

    if de is not None and not de.empty:
        inter_p = _pair(f"{g0}_{t0}", f"{g1}_{t1}")[
            ["feature", "adj_p_value", "cohens_d", "log2FC"]
        ].rename(
            columns={
                "feature": "reaction",
                "adj_p_value": "p_arm",
                "cohens_d": "d_arm",
                "log2FC": "log2FC_arm",
            }
        )
        tab = tab.merge(inter_p, on="reaction", how="left")
        treat_wt = _pair(f"{g0}_{t0}", f"{g0}_{t1}")
        treat_ko = _pair(f"{g1}_{t0}", f"{g1}_{t1}")
        geno_t0 = _pair(f"{g0}_{t0}", f"{g1}_{t0}")
        geno_t1 = _pair(f"{g0}_{t1}", f"{g1}_{t1}")
        if not treat_wt.empty:
            tab["d_treat_wt"] = tab["reaction"].map(
                treat_wt.set_index("feature")["cohens_d"]
            )
            tab["p_treat_wt"] = tab["reaction"].map(
                treat_wt.set_index("feature")["adj_p_value"]
            )
        if not treat_ko.empty:
            tab["d_treat_ko"] = tab["reaction"].map(
                treat_ko.set_index("feature")["cohens_d"]
            )
            tab["p_treat_ko"] = tab["reaction"].map(
                treat_ko.set_index("feature")["adj_p_value"]
            )
        if "d_treat_wt" in tab.columns and "d_treat_ko" in tab.columns:
            tab["d_interaction"] = tab["d_treat_ko"] - tab["d_treat_wt"]
        if not geno_t0.empty:
            tab["d_geno_t0"] = tab["reaction"].map(
                geno_t0.set_index("feature")["cohens_d"]
            )
        if not geno_t1.empty:
            tab["d_geno_t1"] = tab["reaction"].map(
                geno_t1.set_index("feature")["cohens_d"]
            )
        de = de.copy()
        de["subsystem"] = [
            "; ".join(s for s in SUBSYSTEMS if s in panel.get(str(r), ()))
            for r in de["feature"].astype(str)
        ]
    _out(f"{spec['stem']}_rxn_de.csv", de)
    return tab


def _rxn_bar(rxn_tab, spec):
    if rxn_tab is None or not len(rxn_tab):
        return
    score = "d_interaction" if "d_interaction" in rxn_tab.columns else "interaction"
    sub = rxn_tab.copy()
    if "in_panel" in sub.columns:
        sub = sub.loc[sub["in_panel"].astype(bool)]
    sub = sub.loc[np.isfinite(sub[score].to_numpy(dtype=float))].copy()
    if sub.empty:
        return
    sub["_abs"] = sub[score].abs()
    sub = sub.sort_values("_abs", ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.28 * len(sub) + 1.2)))
    y = np.arange(len(sub))[::-1]
    vals = sub[score].to_numpy(dtype=float)
    colors = ["#C44E52" if v >= 0 else "#4C72B0" for v in vals]
    ax.barh(y, vals, color=colors, height=0.7)
    labels = []
    for r, ss in zip(
        sub["reaction"].astype(str),
        sub.get("subsystem", pd.Series([""] * len(sub))).astype(str),
    ):
        lab = r if not ss else f"{r} · {ss.split(';')[0]}"
        labels.append(lab[:48])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel("Cohen's d interaction (KO treat − WT treat)")
    ax.set_title(
        ("Mice" if spec["stem"] == "mice" else "Human")
        + " · reaction differential activity",
        fontweight="bold",
    )
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    _out(f"{spec['stem']}_rxn_de.png", fig)
    plt.close(fig)


def _escher_export(rxn_tab, spec):
    if rxn_tab is None or not len(rxn_tab):
        return
    score = "d_interaction" if "d_interaction" in rxn_tab.columns else "interaction"
    sub = rxn_tab
    if "in_panel" in sub.columns:
        sub = sub.loc[sub["in_panel"].astype(bool)]
    data = {
        str(r): float(v)
        for r, v in zip(sub["reaction"].astype(str), sub[score].to_numpy(dtype=float))
        if np.isfinite(v)
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{spec['stem']}_escher_d_interaction.json").write_text(
        json.dumps(data, indent=2, sort_keys=True)
    )


def _set_mean_perm(
    M, genotype, treatment, genotypes, treatments, gene_sets, genes, device
):
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
    names = list(adata.var_names.astype(str))
    idx = {g: i for i, g in enumerate(names)}
    hyp = {
        g
        for ts in _axis_tasks(db).values()
        for t in ts
        if t in tbg.index
        for g in map(str, tbg.loc[t][tbg.loc[t] > 0].index)
    }
    keep = [g for g in names if g in hyp]
    if not keep:
        raise ValueError(f"{stem}: no hypothesis genes after subsystem filter")
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
    print(
        f"{stem}: VNN on {device}, n={Xt.size(0)} genes={Xt.size(1)} "
        f"tasks={len(graph.tasks)}",
        flush=True,
    )
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
    X_rate, rate_names = _rxn_proxy(adata, db)
    for j, name in enumerate(rate_names):
        cells[name] = X_rate[:, j]
    _out(f"{stem}_cells.csv", cells)

    Y_m, g_m, t_m, _mice = _mouse_means(st, adata.obs)
    obs = _arm_stats(Y_m, g_m, t_m, genotypes, treatments, device)
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
    tab = _arm_table(graph.tasks, obs, genotypes, treatments)
    tab = tab.merge(
        perm[["task", "p_value"]].rename(columns={"p_value": "p_perm"}),
        on="task",
        how="left",
    )
    tab["n_perm"] = N_PERM
    tab["epochs"] = EPOCHS
    tab["task_short"] = [_title(t) for t in tab["task"]]
    tab["subsystem"] = [
        next((ax for ax, ts in _axis_tasks(db).items() if t in ts), "")
        for t in tab["task"]
    ]
    _out(f"{stem}.csv", tab)

    rxn_tab = _rxn_de(adata, spec, device)
    _out(f"{stem}_ions.csv", rxn_tab)
    _rxn_bar(rxn_tab, spec)
    _escher_export(rxn_tab, spec)

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
    for ti_i, task in enumerate(graph.tasks):
        gidx = np.flatnonzero(A_tg[ti_i] > 0)
        if gidx.size == 0:
            continue
        sub = eff.iloc[gidx].copy()
        sub["task"] = task
        rows.append(sub)
    if not rows:
        raise ValueError(f"{stem}: no task gene edges for gene cloud")
    genes = pd.concat(rows, ignore_index=True)
    genes["p_perm"] = genes["task"].map(tab.set_index("task")["p_perm"])
    genes["task_short"] = [_title(t) for t in genes["task"]]
    _out(f"{stem}_genes.csv", genes)
    _plot_gene_cloud(genes, spec)
    _plot_task_2x2(tab, spec)
    _em(adata, cells, tab, spec)


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
        rxn = _read_table(f"{stem}_ions.csv")
        if rxn is not None and "reaction" in rxn.columns:
            _rxn_bar(rxn, spec)
            _escher_export(rxn, spec)
        cells = _read_table(f"{stem}_cells.csv")
        if cells is not None:
            _em(_load_hspc(spec), cells, tab, spec)


def _gpu_free_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    return [int(x.strip()) for x in out.splitlines() if x.strip()]


def _ready():
    for s in SPECIES:
        stem = s["stem"]
        t = _read_table(f"{stem}.csv")
        g = _read_table(f"{stem}_genes.csv")
        c = _read_table(f"{stem}_cells.csv")
        ions = _read_table(f"{stem}_ions.csv")
        if t is None or g is None or c is None:
            return False
        if "p_perm" not in t or "n_perm" not in t or "epochs" not in t:
            return False
        if int(t["n_perm"].iloc[0]) != N_PERM or int(t["epochs"].iloc[0]) != EPOCHS:
            return False
        if ions is None or "reaction" not in ions.columns:
            return False
        if "d_interaction" not in ions.columns and "interaction" not in ions.columns:
            return False
        if not any(sub in c.columns for sub in SUBSYSTEMS):
            return False
    return True


def main():
    if _ready():
        replot_saved()
        return
    gpus = [i for i, m in enumerate(_gpu_free_mib()) if m >= 4000]
    if not gpus:
        raise SystemExit("need a CUDA GPU with ≥4 GiB free")
    if len(gpus) >= 2:
        ctx = mp.get_context("spawn")
        ps = [
            ctx.Process(target=_run_species, args=(s, g)) for s, g in zip(SPECIES, gpus)
        ]
        for p in ps:
            p.start()
        for p in ps:
            p.join()
            if p.exitcode:
                raise SystemExit(f"worker failed: {p.exitcode}")
        return
    for s in SPECIES:
        _run_species(s, gpus[0])


if __name__ == "__main__":
    main()
