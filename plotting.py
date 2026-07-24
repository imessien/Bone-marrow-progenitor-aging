"""Branch / score plots and shallow tree-GNN path readout for BM aging.

Finish target for the age-core atlas product contract
(``docs/plans/2026-07-23-003-feat-shallow-tree-gnn-path-persistence-plan.md``):

  - Branch skeleton anchored at **HSPC → Myeloid_prog** (finer labels later)
  - **age_bin** as vertical time; **pseudotime** for path start / persist / die-off
  - Shallow tree-GNN on that skeleton (hierarchy-inspired; not full T-GNN)
  - Task / EM scores at every step; GSEA on long vs short route genes

Today this module already draws embeddings, branch streams, and marker lanes from
``explore.py`` inputs. Tree-GNN train/infer and path-persistence exports land here.

Inputs from ``explore`` (or elsewhere):
  - ``adata.obsm[basis]`` — usually ``X_umap`` or scGen latent projected to 2D
  - ``adata.obs[branch_key]`` — ``lineage`` / ``cell_type`` (HSPC / Myeloid_prog)
  - ``adata.obs["age_bin"]`` — early / mid / late (vertical axis)
  - ``adata.obs[pt_key]`` — differentiation pseudotime for path dynamics
  - ``adata.obs[color]`` — EM / CHIP / gene scores or categoricals
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

# Display order for HSC→GMP axis (not a learned fate vocabulary).
BM_BRANCHES: tuple[str, ...] = ("HSPC", "Myeloid_prog")

# Axis markers highlighted on branch streams (early HSPC → late GMP).
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

# EM / metabolic task genes (use whichever are present in var_names).
EM_TASK_GENES: tuple[str, ...] = (
    "Myc",
    "Atp5a1",
    "Atp5b",
    "Atp5c1",
    "Cox5a",
    "Cox6a1",
    "Ndufs1",
    "Ndufa1",
    "Glul",
    "Gls",
    "Got1",
    "Got2",
)

# Optional gene highlights for multi-panel expression plots (not CellOracle).
DEFAULT_GENE_COLORS: tuple[str, ...] = AXIS_MARKERS

AGE_BIN_ORDER: tuple[str, ...] = ("early", "mid", "late")

PathLike = Union[str, Path]


def greengrey2red():
    """Diverging cmap for continuous scores."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "greengrey2red", ["#2ca02c", "#bdbdbd", "#d62728", "#8b0000"]
    )


def grey2red():
    """Sequential cmap for expression-like values."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("grey2red", ["#bdbdbd", "#d62728"])


def _as_2d(adata: AnnData, basis: str) -> np.ndarray:
    if basis not in adata.obsm:
        raise KeyError(f"Missing adata.obsm[{basis!r}]; have {list(adata.obsm)}")
    xy = np.asarray(adata.obsm[basis])
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"adata.obsm[{basis!r}] must be (n_cells, ≥2), got {xy.shape}")
    return xy[:, :2]


def _color_values(adata: AnnData, color: str) -> pd.Series:
    if color in adata.obs:
        return adata.obs[color]
    if color in adata.var_names:
        idx = adata.var_names.get_loc(color)
        x = adata[:, idx].X
        if hasattr(x, "toarray"):
            x = x.toarray()
        return pd.Series(np.asarray(x).ravel(), index=adata.obs_names, name=color)
    raise KeyError(f"{color!r} not in obs or var_names")


def plot_embedding(
    adata: AnnData,
    *,
    color: str = "lineage",
    basis: str = "X_umap",
    ax: Optional[Axes] = None,
    s: float = 4.0,
    alpha: float = 0.7,
    cmap: Any = None,
    title: Optional[str] = None,
    show_legend: bool = True,
) -> Axes:
    """Scatter on a 2D embedding colored by obs column or gene."""
    xy = _as_2d(adata, basis)
    vals = _color_values(adata, color)
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(6, 5))
    assert ax is not None

    if pd.api.types.is_numeric_dtype(vals):
        sc = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=vals.to_numpy(),
            s=s,
            alpha=alpha,
            cmap=cmap or greengrey2red(),
            rasterized=True,
        )
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=color)
    else:
        cats = pd.Categorical(vals.astype(str))
        for cat in cats.categories:
            m = cats == cat
            ax.scatter(
                xy[m, 0],
                xy[m, 1],
                s=s,
                alpha=alpha,
                label=str(cat),
                rasterized=True,
            )
        if show_legend:
            ax.legend(markerscale=2, frameon=False, loc="best")

    ax.set_xlabel(f"{basis}1")
    ax.set_ylabel(f"{basis}2")
    ax.set_title(title or color)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_branch_streams(
    adata: AnnData,
    *,
    color: str = "lineage",
    pt_key: str = "dpt_pseudotime",
    branch_key: str = "lineage",
    branch_order: Sequence[str] = BM_BRANCHES,
    ax: Optional[Axes] = None,
    s: float = 6.0,
    alpha: float = 0.6,
    cmap: Any = None,
    jitter: float = 0.35,
    title: Optional[str] = None,
    highlight_genes: Optional[Sequence[str]] = None,
) -> Axes:
    """Lane plot: x = pseudotime, y = branch lane, color = score or category.

    Needs a differentiation pseudotime column (DPT, etc.) already on ``adata.obs``.
    If ``highlight_genes`` is set, draw mean±SEM expression curves (secondary axis)
    so early vs late markers are readable on the same stream.
    """
    if pt_key not in adata.obs:
        raise KeyError(
            f"Missing obs[{pt_key!r}] — compute pseudotime in an analysis script first"
        )
    if branch_key not in adata.obs:
        raise KeyError(f"Missing obs[{branch_key!r}]")

    pt = pd.to_numeric(adata.obs[pt_key], errors="coerce").to_numpy()
    branches = adata.obs[branch_key].astype(str)
    order = [b for b in branch_order if (branches == b).any()]
    extras = sorted(set(branches.unique()) - set(order))
    order = order + extras
    lane = {b: i for i, b in enumerate(order)}
    y = np.array([lane[b] for b in branches], dtype=float)
    rng = np.random.default_rng(0)
    y = y + rng.uniform(-jitter, jitter, size=y.shape)

    vals = _color_values(adata, color)
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(8, 3 + 0.6 * len(order)))
    assert ax is not None

    if pd.api.types.is_numeric_dtype(vals):
        sc = ax.scatter(
            pt,
            y,
            c=vals.to_numpy(),
            s=s,
            alpha=alpha,
            cmap=cmap or greengrey2red(),
            rasterized=True,
        )
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=color)
    else:
        cats = pd.Categorical(vals.astype(str))
        for cat in cats.categories:
            m = cats == cat
            ax.scatter(pt[m], y[m], s=s, alpha=alpha, label=str(cat), rasterized=True)
        ax.legend(markerscale=2, frameon=False, loc="upper left")

    if highlight_genes:
        _overlay_marker_trends(adata, ax, pt_key=pt_key, genes=highlight_genes)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel(pt_key)
    ax.set_title(title or f"{color} along {pt_key}")
    return ax


def _overlay_marker_trends(
    adata: AnnData,
    ax: Axes,
    *,
    pt_key: str,
    genes: Sequence[str],
    n_bins: int = 25,
) -> None:
    """Mean expression vs pseudotime on a twin y-axis (labeled by gene)."""
    present = [g for g in genes if g in adata.var_names]
    if not present:
        return
    pt = pd.to_numeric(adata.obs[pt_key], errors="coerce")
    ok = pt.notna().to_numpy()
    pt_v = pt.to_numpy()[ok]
    edges = np.linspace(np.nanmin(pt_v), np.nanmax(pt_v), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax2 = ax.twinx()
    cmap = plt.get_cmap("tab10")
    for i, gene in enumerate(present):
        expr = _color_values(adata, gene).to_numpy()[ok]
        means = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (pt_v >= lo) & (pt_v <= hi if hi == edges[-1] else pt_v < hi)
            means.append(float(np.nanmean(expr[m])) if m.any() else np.nan)
        ax2.plot(
            centers,
            means,
            color=cmap(i % 10),
            lw=2.0,
            label=gene,
            zorder=5,
        )
    ax2.set_ylabel("marker mean expr")
    ax2.legend(frameon=False, loc="upper right", fontsize=8, title="markers")


def plot_marker_branch_streams(
    adata: AnnData,
    *,
    genes: Sequence[str] = AXIS_MARKERS,
    pt_key: str = "dpt_pseudotime",
    branch_key: str = "lineage",
    context_color: str = "age_bin",
    fig_size: tuple[float, float] = (4.2, 2.8),
) -> Figure:
    """Context stream (age_bin) + one stream panel per axis marker gene.

    Use this to verify HSC→GMP orientation: Procr/Kit should peak early;
    Mpo/Elane/Ms4a3 late.
    """
    present = [g for g in genes if g in adata.var_names]
    if not present:
        raise KeyError(f"none of {list(genes)} found in var_names")

    n = 1 + len(present)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_size[0] * ncols, fig_size[1] * nrows),
        squeeze=False,
    )
    panels: list[tuple[str, Any]] = [(context_color, None)]
    panels.extend((g, grey2red()) for g in present)

    for i, (color, cmap) in enumerate(panels):
        r, c = divmod(i, ncols)
        plot_branch_streams(
            adata,
            color=color,
            pt_key=pt_key,
            branch_key=branch_key,
            ax=axes[r][c],
            cmap=cmap,
            s=4.0,
            alpha=0.55,
            title=color,
            highlight_genes=None,
        )
        # Bold gene title so the QC read is obvious
        if color in present:
            axes[r][c].set_title(f"★ {color}", fontweight="bold")

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    fig.suptitle(
        "Branch streams: age_bin context + axis markers (early→late)",
        y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_score_panels(
    adata: AnnData,
    colors: Sequence[str],
    *,
    basis: str = "X_umap",
    fig_size: tuple[float, float] = (4.0, 3.5),
    s: float = 3.0,
    cmap: Any = None,
    save: Optional[PathLike] = None,
) -> Figure:
    """Grid of embedding plots for EM / CHIP / gene score columns."""
    n = len(colors)
    if n == 0:
        raise ValueError("colors must be non-empty")
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_size[0] * ncols, fig_size[1] * nrows),
        squeeze=False,
    )
    for i, color in enumerate(colors):
        r, c = divmod(i, ncols)
        plot_embedding(
            adata,
            color=color,
            basis=basis,
            ax=axes[r][c],
            s=s,
            cmap=cmap or greengrey2red(),
            show_legend=False,
        )
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")
    fig.tight_layout()
    if save is not None:
        path = Path(save)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_branch_labels(
    adata: AnnData,
    *,
    color: str = "lineage",
    basis: str = "X_umap",
    **kwargs: Any,
) -> Axes:
    """Categorical branch / lineage labels on the embedding."""
    return plot_embedding(adata, color=color, basis=basis, **kwargs)


def plot_gene_panels(
    adata: AnnData,
    genes: Sequence[str] = DEFAULT_GENE_COLORS,
    *,
    basis: str = "X_umap",
    save: Optional[PathLike] = None,
) -> Figure:
    """Expression panels for a gene list (present genes only)."""
    present = [g for g in genes if g in adata.var_names]
    if not present:
        raise KeyError(f"none of {list(genes)} found in var_names")
    return plot_score_panels(
        adata, present, basis=basis, cmap=grey2red(), save=save
    )


# ---------------------------------------------------------------------------
# Shallow tree-GNN path persistence (finish target for plan 003)
# ---------------------------------------------------------------------------


def _resolve_branch_key(adata: AnnData, branch_key: str = "lineage") -> str:
    if branch_key in adata.obs:
        return branch_key
    if "cell_type" in adata.obs:
        return "cell_type"
    raise KeyError("Need obs['lineage'] or obs['cell_type']")


def _resolve_latent(adata: AnnData, use_rep: str = "corrected_latent") -> np.ndarray:
    for key in (use_rep, "X_scgen", "latent"):
        if key in adata.obsm:
            return np.asarray(adata.obsm[key], dtype=np.float32)
    raise KeyError(f"Missing latent in obsm; have {list(adata.obsm)}")


def _pt_bin_codes(
    pt: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return integer bin codes 0..n_bins-1 and bin centers in PT units."""
    ok = np.isfinite(pt)
    edges = np.quantile(pt[ok], np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.linspace(np.nanmin(pt[ok]), np.nanmax(pt[ok]), n_bins + 1)
    # digitize → 1..len-1; clip to 0..n_bins-1
    codes = np.digitize(pt, edges[1:-1], right=False)
    codes = np.clip(codes, 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if len(centers) < n_bins:
        centers = np.pad(centers, (0, n_bins - len(centers)), mode="edge")
    return codes.astype(int), centers[:n_bins]


def build_branch_skeleton(
    adata: AnnData,
    *,
    pt_key: str = "dpt_pseudotime",
    branch_key: str = "lineage",
    age_key: str = "age_bin",
    genotype_key: str = "genotype",
    use_rep: str = "corrected_latent",
    n_bins: int = 20,
    latent_dims: int = 16,
) -> dict[str, Any]:
    """Aggregate cells into a PT×lineage branch skeleton (HSPC → Myeloid_prog)."""
    bk = _resolve_branch_key(adata, branch_key)
    if pt_key not in adata.obs:
        raise KeyError(f"Missing obs[{pt_key!r}]")
    pt = pd.to_numeric(adata.obs[pt_key], errors="coerce").to_numpy()
    lineage = adata.obs[bk].astype(str).to_numpy()
    age = (
        adata.obs[age_key].astype(str).to_numpy()
        if age_key in adata.obs
        else np.array(["NA"] * adata.n_obs)
    )
    geno = (
        adata.obs[genotype_key].astype(str).to_numpy()
        if genotype_key in adata.obs
        else np.array(["NA"] * adata.n_obs)
    )
    Z = _resolve_latent(adata, use_rep)
    if Z.shape[1] > latent_dims:
        # cheap PCA-like: first dims of latent (scGen already ordered)
        Z = Z[:, :latent_dims]

    codes, centers = _pt_bin_codes(pt, n_bins)
    lanes = [b for b in BM_BRANCHES if b in set(lineage)]
    if len(lanes) < 1:
        raise ValueError(f"No BM_BRANCHES in {bk}; saw {sorted(set(lineage))[:8]}")

    node_rows: list[dict[str, Any]] = []
    node_index: dict[tuple[int, str], int] = {}
    for b in range(n_bins):
        for lane in lanes:
            m = (codes == b) & (lineage == lane) & np.isfinite(pt)
            n = int(m.sum())
            if n == 0:
                continue
            ix = len(node_rows)
            node_index[(b, lane)] = ix
            z_mean = Z[m].mean(axis=0)
            age_counts = pd.Series(age[m]).value_counts(normalize=True)
            geno_counts = pd.Series(geno[m]).value_counts(normalize=True)
            node_rows.append(
                {
                    "node_id": ix,
                    "pt_bin": b,
                    "pt_center": float(centers[b]),
                    "lineage": lane,
                    "n_cells": n,
                    "myeloid_frac": 1.0 if lane == "Myeloid_prog" else 0.0,
                    "age_early": float(age_counts.get("early", 0.0)),
                    "age_mid": float(age_counts.get("mid", 0.0)),
                    "age_late": float(age_counts.get("late", 0.0)),
                    "geno_WT": float(geno_counts.get("WT", 0.0)),
                    "geno_IL1R1KO": float(geno_counts.get("IL1R1KO", 0.0)),
                    **{f"z{i}": float(z_mean[i]) for i in range(Z.shape[1])},
                }
            )
    nodes = pd.DataFrame(node_rows)
    if nodes.empty:
        raise ValueError("empty skeleton — check pt_key / lineage coverage")

    # Forward-PT edges + HSPC→Myeloid within same / next bin
    edges: list[tuple[int, int]] = []
    for (b, lane), i in node_index.items():
        for b2 in (b, b + 1):
            if b2 >= n_bins:
                continue
            for lane2 in lanes:
                if b2 == b and lane2 == lane:
                    continue
                # only forward or same-bin HSPC→Myeloid
                if b2 < b:
                    continue
                if lane == "Myeloid_prog" and lane2 == "HSPC":
                    continue
                j = node_index.get((b2, lane2))
                if j is not None:
                    edges.append((i, j))
    edge_index = (
        np.array(edges, dtype=np.int64).T
        if edges
        else np.zeros((2, 0), dtype=np.int64)
    )

    # Occupancy strata: pt_bin × lineage × age_bin / genotype
    occ_rows = []
    for b in range(n_bins):
        for lane in lanes:
            for strat_key, strat_vals in (
                ("age_bin", AGE_BIN_ORDER),
                ("genotype", ("WT", "IL1R1KO")),
            ):
                labels = age if strat_key == "age_bin" else geno
                for lab in strat_vals:
                    m = (
                        (codes == b)
                        & (lineage == lane)
                        & (labels == lab)
                        & np.isfinite(pt)
                    )
                    n = int(m.sum())
                    if n == 0:
                        continue
                    occ_rows.append(
                        {
                            "pt_bin": b,
                            "pt_center": float(centers[b]),
                            "lineage": lane,
                            "stratum": strat_key,
                            "level": lab,
                            "n_cells": n,
                        }
                    )
    occupancy = pd.DataFrame(occ_rows)
    return {
        "nodes": nodes,
        "edge_index": edge_index,
        "occupancy": occupancy,
        "n_bins": n_bins,
        "pt_centers": centers,
        "branch_key": bk,
        "codes": codes,
        "lineage": lineage,
        "age": age,
        "geno": geno,
        "pt": pt,
    }


def path_persistence_table(
    occupancy: pd.DataFrame,
    *,
    lane: str = "Myeloid_prog",
    stratum: str = "age_bin",
    min_cells: int = 30,
) -> pd.DataFrame:
    """Longest contiguous myeloid PT span per stratum level (persist vs die-off)."""
    sub = occupancy[
        (occupancy["lineage"] == lane) & (occupancy["stratum"] == stratum)
    ].copy()
    rows = []
    for level, g in sub.groupby("level", sort=False):
        g = g.sort_values("pt_bin")
        alive = g["n_cells"].to_numpy() >= min_cells
        bins = g["pt_bin"].to_numpy()
        # longest True run
        best_len, best_start, best_end = 0, None, None
        cur_len, cur_start = 0, None
        for i, on in enumerate(alive):
            if on:
                if cur_start is None:
                    cur_start = int(bins[i])
                cur_len += 1
                if cur_len > best_len:
                    best_len = cur_len
                    best_start = cur_start
                    best_end = int(bins[i])
            else:
                cur_len, cur_start = 0, None
        rows.append(
            {
                "stratum": stratum,
                "level": level,
                "lane": lane,
                "persist_len_bins": best_len,
                "persist_start_bin": best_start,
                "persist_end_bin": best_end,
                "total_cells": int(g["n_cells"].sum()),
                "n_active_bins": int(alive.sum()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["rank_persist"] = out["persist_len_bins"].rank(ascending=False, method="min")
        out["is_longest"] = out["persist_len_bins"] == out["persist_len_bins"].max()
        out["is_shortest"] = out["persist_len_bins"] == out["persist_len_bins"].min()
    return out


class ShallowTreeGNN(nn.Module):
    """Two-layer GraphSAGE on the branch skeleton (hierarchy along PT edges)."""

    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return self.head(h).squeeze(-1)


def train_shallow_tree_gnn(
    skeleton: dict[str, Any],
    *,
    epochs: int = 80,
    lr: float = 1e-2,
    seed: int = 0,
) -> tuple[pd.DataFrame, ShallowTreeGNN]:
    """Fit GNN to predict myeloid_frac on skeleton nodes; return node table + model."""
    torch.manual_seed(seed)
    nodes = skeleton["nodes"].copy()
    feat_cols = [
        c
        for c in nodes.columns
        if c.startswith("z")
        or c
        in (
            "myeloid_frac",
            "age_early",
            "age_mid",
            "age_late",
            "geno_WT",
            "geno_IL1R1KO",
            "pt_center",
        )
    ]
    # don't leak exact myeloid label as sole cue — drop myeloid_frac from X
    x_cols = [c for c in feat_cols if c != "myeloid_frac"]
    x = torch.tensor(nodes[x_cols].to_numpy(dtype=np.float32))
    y = torch.tensor(nodes["myeloid_frac"].to_numpy(dtype=np.float32))
    ei = torch.tensor(skeleton["edge_index"], dtype=torch.long)
    if ei.numel() == 0:
        # isolated nodes: still train MLP via empty edges
        ei = torch.zeros((2, 0), dtype=torch.long)
    data = Data(x=x, edge_index=ei, y=y)
    model = ShallowTreeGNN(in_dim=x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(data.x, data.edge_index)
        loss = F.mse_loss(pred, data.y)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).numpy()
    nodes["gnn_myeloid_pred"] = pred
    return nodes, model


def step_task_scores(
    adata: AnnData,
    skeleton: dict[str, Any],
    *,
    genes: Sequence[str] | None = None,
    age_key: str = "age_bin",
) -> pd.DataFrame:
    """Mean gene / task expression at every (pt_bin, lineage, age_bin) step."""
    gene_list = list(genes) if genes is not None else list(AXIS_MARKERS) + list(EM_TASK_GENES)
    present = [g for g in gene_list if g in adata.var_names]
    codes = skeleton["codes"]
    lineage = skeleton["lineage"]
    age = skeleton["age"]
    rows = []
    for b in range(skeleton["n_bins"]):
        for lane in BM_BRANCHES:
            for ab in AGE_BIN_ORDER:
                m = (codes == b) & (lineage == lane) & (age == ab)
                n = int(m.sum())
                if n == 0:
                    continue
                row: dict[str, Any] = {
                    "pt_bin": b,
                    "pt_center": float(skeleton["pt_centers"][b]),
                    "lineage": lane,
                    "age_bin": ab,
                    "n_cells": n,
                }
                if present:
                    # mean over cells for each gene
                    sub = adata[m, present]
                    X = sub.X
                    if hasattr(X, "toarray"):
                        X = X.toarray()
                    means = np.asarray(X, dtype=np.float64).mean(axis=0)
                    for g, val in zip(present, means):
                        row[g] = float(val)
                    row["task_mean"] = float(np.nanmean(means))
                rows.append(row)
    return pd.DataFrame(rows)


def rank_genes_long_vs_short(
    adata: AnnData,
    skeleton: dict[str, Any],
    persistence: pd.DataFrame,
    *,
    n_top: int = 500,
) -> pd.DataFrame:
    """Rank genes by mean expr in longest vs shortest myeloid age_bin strata."""
    if persistence.empty or "is_longest" not in persistence.columns:
        return pd.DataFrame(columns=["gene", "score", "mean_long", "mean_short"])
    long_lv = persistence.loc[persistence["is_longest"], "level"].astype(str).tolist()
    short_lv = persistence.loc[persistence["is_shortest"], "level"].astype(str).tolist()
    # if all equal length, compare late vs early when both present
    if set(long_lv) == set(short_lv):
        long_lv, short_lv = ["late"], ["early"]
    codes = skeleton["codes"]
    lineage = skeleton["lineage"]
    age = skeleton["age"]
    pers = persistence.set_index("level")

    def _mask(levels: list[str]) -> np.ndarray:
        m = np.zeros(adata.n_obs, dtype=bool)
        for lv in levels:
            if lv not in pers.index:
                continue
            start = pers.loc[lv, "persist_start_bin"]
            end = pers.loc[lv, "persist_end_bin"]
            if pd.isna(start) or pd.isna(end):
                continue
            m |= (
                (lineage == "Myeloid_prog")
                & (age == lv)
                & (codes >= int(start))
                & (codes <= int(end))
            )
        return m

    m_long = _mask(long_lv)
    m_short = _mask(short_lv)
    if m_long.sum() < 10 or m_short.sum() < 10:
        return pd.DataFrame(columns=["gene", "score", "mean_long", "mean_short"])

    # subsample genes if huge — use highly variable if flagged else all
    genes = list(adata.var_names)
    if "highly_variable" in adata.var.columns:
        hv = adata.var_names[adata.var["highly_variable"].to_numpy()].tolist()
        if hv:
            genes = hv
    # cap for speed
    if len(genes) > 3000:
        genes = genes[:3000]

    Xl = adata[m_long, genes].X
    Xs = adata[m_short, genes].X
    if hasattr(Xl, "toarray"):
        Xl = Xl.toarray()
        Xs = Xs.toarray()
    ml = np.asarray(Xl, dtype=np.float64).mean(axis=0)
    ms = np.asarray(Xs, dtype=np.float64).mean(axis=0)
    score = ml - ms
    order = np.argsort(-np.abs(score))[:n_top]
    return pd.DataFrame(
        {
            "gene": [genes[i] for i in order],
            "score": score[order],
            "mean_long": ml[order],
            "mean_short": ms[order],
        }
    )


def run_gsea_prerank(ranked: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    """Best-effort gseapy prerank; always leaves ranked TSV from caller."""
    if ranked.empty:
        return None
    try:
        import gseapy as gp
    except ImportError:
        return None
    rnk = ranked[["gene", "score"]].dropna()
    rnk_path = out_dir / "path_long_vs_short_ranked.tsv"
    rnk.to_csv(rnk_path, sep="\t", index=False, header=False)
    try:
        res = gp.prerank(
            rnk=str(rnk_path),
            gene_sets="GO_Biological_Process_2021",
            organism="Mouse",
            outdir=str(out_dir / "gsea_prerank"),
            format="png",
            seed=0,
            no_plot=True,
            verbose=False,
        )
        table = out_dir / "gsea_long_vs_short.tsv"
        res.res2d.to_csv(table, sep="\t", index=False)
        return table
    except Exception as exc:  # enrichment is best-effort
        (out_dir / "gsea_skip_reason.txt").write_text(str(exc) + "\n")
        return None


def plot_path_occupancy_heatmap(
    occupancy: pd.DataFrame,
    *,
    lane: str = "Myeloid_prog",
    stratum: str = "age_bin",
    save: Optional[PathLike] = None,
) -> Figure:
    """Heatmap: vertical age_bin (or genotype) × horizontal PT bin, myeloid n_cells."""
    sub = occupancy[
        (occupancy["lineage"] == lane) & (occupancy["stratum"] == stratum)
    ]
    if sub.empty:
        fig, ax = plt.subplots()
        ax.set_title("no occupancy")
        return fig
    mat = sub.pivot_table(
        index="level", columns="pt_bin", values="n_cells", aggfunc="sum", fill_value=0
    )
    if stratum == "age_bin":
        order = [a for a in AGE_BIN_ORDER if a in mat.index]
        mat = mat.reindex(order)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="YlOrRd", origin="upper")
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(list(mat.index))
    ax.set_xlabel("pt_bin (pseudotime →)")
    ax.set_ylabel(stratum)
    ax.set_title(f"{lane} occupancy (paths persist / die across {stratum})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="n_cells")
    fig.tight_layout()
    if save is not None:
        path = Path(save)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def run_path_gnn_pipeline(
    adata: AnnData,
    out_dir: PathLike,
    *,
    n_bins: int = 20,
    max_cells: Optional[int] = None,
    gnn_epochs: int = 80,
    min_cells_persist: int = 30,
) -> dict[str, Path]:
    """End-to-end: skeleton → persistence → GNN → step scores → GSEA → figures."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ad_use = adata
    if max_cells is not None and adata.n_obs > max_cells:
        rng = np.random.default_rng(0)
        ix = rng.choice(adata.n_obs, size=max_cells, replace=False)
        ad_use = adata[ix].copy()

    sk = build_branch_skeleton(ad_use, n_bins=n_bins)
    nodes_path = out / "path_skeleton_nodes.csv"
    sk["nodes"].to_csv(nodes_path, index=False)
    sk["occupancy"].to_csv(out / "path_occupancy.csv", index=False)

    pers_age = path_persistence_table(
        sk["occupancy"], stratum="age_bin", min_cells=min_cells_persist
    )
    pers_geno = path_persistence_table(
        sk["occupancy"], stratum="genotype", min_cells=min_cells_persist
    )
    pers = pd.concat([pers_age, pers_geno], ignore_index=True)
    pers_path = out / "path_persistence.csv"
    pers.to_csv(pers_path, index=False)

    nodes_pred, _model = train_shallow_tree_gnn(sk, epochs=gnn_epochs)
    gnn_path = out / "path_gnn_nodes.csv"
    nodes_pred.to_csv(gnn_path, index=False)

    scores = step_task_scores(ad_use, sk)
    scores_path = out / "path_step_task_scores.csv"
    scores.to_csv(scores_path, index=False)

    ranked = rank_genes_long_vs_short(ad_use, sk, pers_age)
    ranked_path = out / "path_long_vs_short_ranked.tsv"
    ranked.to_csv(ranked_path, sep="\t", index=False)
    gsea_path = run_gsea_prerank(ranked, out)

    fig_path = out / "path_occupancy_age_bin.png"
    plot_path_occupancy_heatmap(sk["occupancy"], save=fig_path)
    plt.close("all")
    fig_geno = out / "path_occupancy_genotype.png"
    plot_path_occupancy_heatmap(
        sk["occupancy"], stratum="genotype", save=fig_geno
    )
    plt.close("all")

    written = {
        "nodes": nodes_path,
        "occupancy": out / "path_occupancy.csv",
        "persistence": pers_path,
        "gnn": gnn_path,
        "step_scores": scores_path,
        "ranked": ranked_path,
        "heatmap_age": fig_path,
        "heatmap_geno": fig_geno,
    }
    if gsea_path is not None:
        written["gsea"] = gsea_path
    return written


def _self_check() -> None:
    """Minimal synthetic check for skeleton + GNN + persistence (no pytest)."""
    rng = np.random.default_rng(0)
    n = 400
    pt = np.concatenate([rng.uniform(0, 0.4, 200), rng.uniform(0.5, 1.0, 200)])
    lin = np.array(["HSPC"] * 200 + ["Myeloid_prog"] * 200)
    age = np.array(
        ["early"] * 100 + ["late"] * 100 + ["early"] * 50 + ["mid"] * 50 + ["late"] * 100
    )
    geno = np.array(["WT"] * 300 + ["IL1R1KO"] * 100)
    Z = rng.normal(size=(n, 8)).astype(np.float32)
    X = rng.poisson(1.0, size=(n, 12)).astype(np.float32)
    var = [f"g{i}" for i in range(10)] + ["Myc", "Mpo"]
    adata = AnnData(X=X, obs=pd.DataFrame({"dpt_pseudotime": pt, "lineage": lin, "age_bin": age, "genotype": geno}), var=pd.DataFrame(index=var))
    adata.obsm["corrected_latent"] = Z
    sk = build_branch_skeleton(adata, n_bins=8, latent_dims=8)
    assert len(sk["nodes"]) >= 2
    assert sk["edge_index"].shape[0] == 2
    pers = path_persistence_table(sk["occupancy"], min_cells=5)
    assert not pers.empty
    nodes, _ = train_shallow_tree_gnn(sk, epochs=5)
    assert "gnn_myeloid_pred" in nodes.columns
    assert np.isfinite(nodes["gnn_myeloid_pred"]).all()
    scores = step_task_scores(adata, sk)
    assert not scores.empty
    print("plotting._self_check: OK")


# Back-compat aliases (old PHLOWER-era names → custom embedding plots).
plot_stream = plot_branch_streams
plot_stream_sc = plot_embedding
plot_tf_stream = plot_gene_panels
plot_gene_stream = plot_gene_panels


if __name__ == "__main__":
    _self_check()
