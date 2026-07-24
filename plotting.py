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
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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

# Optional gene highlights for multi-panel expression plots (not CellOracle).
DEFAULT_GENE_COLORS: tuple[str, ...] = AXIS_MARKERS

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


# Back-compat aliases (old PHLOWER-era names → custom embedding plots).
plot_stream = plot_branch_streams
plot_stream_sc = plot_embedding
plot_tf_stream = plot_gene_panels
plot_gene_stream = plot_gene_panels
