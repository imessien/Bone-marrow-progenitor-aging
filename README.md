# Bone-marrow progenitor aging

This project asks whether chronological aging redistributes mouse bone-marrow
`Myeloid_prog` cells along a shared progenitor continuum and whether Il1r1 loss
partly reverses that occupancy shift while changing metabolism at matched
pseudotime. Mitchell et al. showed that Il1r1 loss rebalances aged blood and
niche phenotypes using discrete age cohorts, but did not build a continuous-age
myeloid pseudotime or test reversal of trajectory occupancy [1]:

> "Old *Il1r1*−/− mice also showed reduced features of blood aging … none of the
> characteristic myeloid cell expansion, B cell loss, and anemia were observed
> … we found an attenuation of MPP3 expansion with diminished myeloid cell
> production"
>
> "young mice were 6-12 weeks of age, middle-aged mice were 13 months old, and
> old mice were all ≥ 18 months of age"

Pietras et al. established that chronic IL-1 drives precocious myeloid
differentiation and that these effects reverse upon IL-1 withdrawal [2]:

> "chronic IL-1 exposure restricts HSC lineage output, severely erodes HSC
> self-renewal capacity … Importantly, these damaging effects are transient and
> fully reversible upon IL-1 withdrawal."
>
> "these results demonstrate that the damaging effects of chronic IL-1 exposure
> on HSC regenerative functions are fully reversible upon interruption of IL-1
> exposure."

So partial reversal by Il1r1 loss is biologically expected; the open step is to
quantify age-linked occupancy on a shared `Myeloid_prog` continuum and read
metabolism at matched pseudotime. `explore.py` therefore keeps only the
substantially overlapping `Myeloid_prog` state, runs scGen batch removal
(`dataset` only) → neighbors → diffusion map → DPT, and retains chronological
age as `age_months` / `age_label` (never as a technical batch).

## What is `Myeloid_prog` here?

| Source | Definition |
|---|---|
| TMS (CELLxGENE Census) | Ontology label `granulocyte monocyte progenitor cell` (GMP; CL:0000557) |
| Mitchell (GSE169162) | Marker-scored lineage using `Elane`, `Mpo`, `Ctsg`, `Ms4a3`, `Cebpe` |
| CellTypist encyclopedia | GMP = “hematopoietic granulocyte-monocyte progenitors that are committed to the granulocyte and monocyte lineage”; curated markers ELANE, MPO, PRTN3 [3] |

Local counts after annotation: TMS **148** GMPs; Mitchell **898** `Myeloid_prog`
cells. That is the only state with substantial cross-study UMAP overlap.

**Valid target?** Yes, biologically: Pietras shows chronic IL-1 specifically
amplifies GMPs [2], and Mitchell’s Il1r1-null aging rescue includes attenuated
myeloid-biased progenitor expansion [1]. Marker sets agree with CellTypist GMP
(ELANE/MPO). Caveats: (i) CellTypist’s built-in `Immune_All_Low` GMP label is
**human**; there is no stock mouse-BM GMP model—use Census/TMS labels + shared
markers (or convert a human model) for QC, not as the primary label; (ii) TMS
n=148 is thin for multi-age density, so treat TMS as the age-trend reference
and compute KO reversal inside Mitchell.

--------
REFERENCES
[1] Mitchell, C. A. et al. "Stromal inflammation is a targetable driver of hematopoietic aging." *Nature Cell Biology* (2023). doi:10.1038/s41556-022-01053-0
    https://citations.gxl.ai/papers/PMC7614279#L31,L38
[2] Pietras, E. M. et al. "Chronic interleukin-1 drives haematopoietic stem cells towards precocious myeloid differentiation at the expense of self-renewal." *Nature Cell Biology* (2016). doi:10.1038/ncb3346
    https://citations.gxl.ai/papers/PMC4884136#L9,L20,L27
[3] CellTypist encyclopedia, Immune/v1 GMP (CL:0000557). https://www.celltypist.org/encyclopedia/Immune/v1/?celltype=GMP
