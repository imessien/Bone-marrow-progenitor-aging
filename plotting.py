"""PHLOWER stream-tree visualization helpers for the BM aging atlas.

Requires a PHLOWER tree already on ``adata`` (``harmonic_stream_tree`` / STREAM
fields). These wrappers only style and plot — they do not replace the age cVAE.

Docs: https://phlower.readthedocs.io/en/latest/api.html#module-phlower.pl
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

import phlower
from anndata import AnnData

from .cvae import BM_BRANCHES

# IL-1R / inflammasome TF layer from README (CellOracle targets).
DEFAULT_STREAM_COLORS: tuple[str, ...] = (
    "Rela",
    "Nfkb1",
    "Cebpb",
    "Irf3",
    "Irf7",
    "Stat1",
)

ColorSpec = Union[str, Sequence[str], None]
Preference = Optional[Mapping[str, Sequence[str]]]


def greengrey2red():
    """Diverging cmap for continuous scores / TF activity on stream trees."""
    return phlower.custom_cmap(["green", "grey", "red", "red"])


def grey2red():
    """Sequential cmap for gene expression on stream trees."""
    return phlower.custom_cmap(["grey", "red"])


def ensure_workdir(adata: AnnData, workdir: str = "") -> AnnData:
    """STREAM helpers write under ``adata.uns['workdir']``; default cwd."""
    adata.uns["workdir"] = workdir
    return adata


def branch_preference(
    order: Sequence[str] = BM_BRANCHES,
    *,
    key: str = "original",
) -> dict[str, list[str]]:
    """Branch display order for ``preference=`` in ``plot_stream*``."""
    return {key: list(order)}


def preference_from_anno(
    anno_dic: Mapping[Any, str],
    *,
    key: str = "original",
) -> dict[str, list[str]]:
    """Kidney-style cluster→label map → STREAM preference order."""
    return {key: list(anno_dic.values())}


def harmonic_stream_tree(
    adata: AnnData,
    *,
    trajs_clusters: str = "annotation",
    retain_clusters: Optional[Sequence[str]] = None,
    min_bin_number: int = 20,
    cut_threshold: float = 1.5,
    verbose: bool = True,
    **kwargs: Any,
) -> AnnData:
    """Thin wrap of ``phlower.tl.harmonic_stream_tree`` (needed before plotting)."""
    ensure_workdir(adata)
    kw: dict[str, Any] = dict(
        trajs_clusters=trajs_clusters,
        min_bin_number=min_bin_number,
        cut_threshold=cut_threshold,
        verbose=verbose,
        **kwargs,
    )
    if retain_clusters is not None:
        kw["retain_clusters"] = list(retain_clusters)
    phlower.tl.harmonic_stream_tree(adata, **kw)
    return adata


def plot_stream(
    adata: AnnData,
    *,
    color: ColorSpec = None,
    preference: Preference = None,
    fig_size: tuple[float, float] = (7, 5),
    dist_scale: float = 1.2,
    factor_min_win: float = 1.2,
    cmap_continous: Any = None,
    return_fig: bool = False,
    **kwargs: Any,
):
    """Density-level STREAM tree (``phlower.ext.plot_stream``)."""
    ensure_workdir(adata)
    if preference is None and color is None:
        preference = branch_preference()
    return phlower.ext.plot_stream(
        adata,
        color=color,
        preference=preference,
        fig_size=fig_size,
        dist_scale=dist_scale,
        factor_min_win=factor_min_win,
        cmap_continous=cmap_continous or "viridis",
        return_fig=return_fig,
        **kwargs,
    )


def plot_stream_sc(
    adata: AnnData,
    *,
    color: ColorSpec = None,
    preference: Preference = None,
    fig_size: tuple[float, float] = (9, 7),
    dist_scale: float = 0.4,
    s: Union[int, float, tuple[float, float]] = (2, 20),
    alpha: float = 0.8,
    show_text: bool = True,
    text_attr: str = "original",
    show_graph: bool = True,
    show_legend: bool = True,
    cmap_continous: Any = None,
    return_fig: bool = False,
    **kwargs: Any,
):
    """Single-cell subway-map STREAM tree (``phlower.ext.plot_stream_sc``).

    Pass ``s=(lo, hi)`` to size points by continuous ``color`` intensity
    (TF activity / gene expression / Δz_age).
    """
    ensure_workdir(adata)
    if preference is None:
        preference = branch_preference()
    return phlower.ext.plot_stream_sc(
        adata,
        color=color,
        preference=preference,
        fig_size=fig_size,
        dist_scale=dist_scale,
        s=s,
        alpha=alpha,
        show_text=show_text,
        text_attr=text_attr,
        show_graph=show_graph,
        show_legend=show_legend,
        cmap_continous=cmap_continous or greengrey2red(),
        return_fig=return_fig,
        **kwargs,
    )


def plot_tf_stream(
    adata: AnnData,
    genes: Sequence[str] = DEFAULT_STREAM_COLORS,
    *,
    cmap: Any = None,
    preference: Preference = None,
    fig_size: tuple[float, float] = (9, 7),
    **kwargs: Any,
):
    """Stream-sc colored by IL-1R / NF-κB axis TFs (or any gene/TF list)."""
    return plot_stream_sc(
        adata,
        color=list(genes),
        preference=preference,
        fig_size=fig_size,
        cmap_continous=cmap or greengrey2red(),
        **kwargs,
    )


def plot_gene_stream(
    adata: AnnData,
    genes: Sequence[str],
    *,
    cmap: Any = None,
    preference: Preference = None,
    fig_size: tuple[float, float] = (9, 7),
    **kwargs: Any,
):
    """Stream-sc for expression (grey→red); MAGIC-impute first if sparse."""
    return plot_stream_sc(
        adata,
        color=list(genes),
        preference=preference,
        fig_size=fig_size,
        cmap_continous=cmap or grey2red(),
        **kwargs,
    )


def plot_branch_labels(
    adata: AnnData,
    color: str = "group_str",
    *,
    fig_size: tuple[float, float] = (8, 5),
    dist_scale: float = 1.0,
    s: float = 10,
    show_legend: bool = False,
    **kwargs: Any,
):
    """Categorical branch / cluster labels on the subway map."""
    return plot_stream_sc(
        adata,
        color=color,
        fig_size=fig_size,
        dist_scale=dist_scale,
        s=s,
        show_legend=show_legend,
        cmap_continous="viridis",
        **kwargs,
    )


plot_stream_tree_embedding = phlower.pl.plot_stream_tree_embedding
plot_fate_tree = phlower.pl.plot_fate_tree
harmonic_backbone = phlower.pl.harmonic_backbone
plot_trajectory_harmonic_lines = phlower.pl.plot_trajectory_harmonic_lines
