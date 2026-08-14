# Bone-marrow progenitor aging

IL-1 drives HSCs toward myeloid/GMP; Mitchell tests that genetically via Il1r1.
We integrate age-core cohorts with scGen (technical batch only), then ask which
cells are biased toward a **GMP-committed sink** vs **agedHSC (Mk-biased)
persistence**.

## Workflow

```bash
source .venv/bin/activate
python preprocess.py --dataset age_core --annotate
python explore.py --train          # scGen + UMAP + DPT (once)
python explore.py                  # fate UMAP + drivers + GSEA
python -c "import explore; explore._self_check()"
```

| File | Role |
|------|------|
| `preprocess.py` | Per-study QC, lineage, age_bin |
| `explore.py` | scGen joint + GMP vs agedHSC fate package |

Outputs under `results/joint_hsc_aging/`.

**Known hole:** Mitchell IL1R1KO in [GSE169162](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE169162) is **old-only** (no young KO).

## CHIP metabolic VNN (Tet2 × IL-1)

Single entrypoint: **`factor.py`**. Young McClatchy GSE209994 marker-`HSPC`
2×2 only (WT/Tet2 × vehicle/IL-1). No GMP/HSC mix, no age head, no Kovtonyuk,
no GO nodes.

A small **visible / biologically-informed net** (VNN/BiNN): named scCellFie
tasks as nodes, gene→task→subsystem→system edges from the mouse DB, gated
message passing. The net reconstructs **cell** gene scores (no 4-arm classifier).
Each `sample_name` well is treated as a **mouse**. Every HSPC stays in the
2×2: the point estimate is the mean of random 4-cell tuples (one cell per
arm), which equals `(Tet2_IL1 − WT_IL1) − (Tet2_vehicle − WT_vehicle)` on
cell arm means. `combo_frac_pos` is the fraction of those tuples with a
positive interaction. P-values reassign treatment among mice within genotype
(all of a mouse's cells move together). Confirmatory tests are the three
axis rows; per-task rows are exploratory.

Family: DCell, P-NET. Message passing: Ma 2019 FGNN. Knowledge graph: scCellFie.

```bash
source .venv/bin/activate
export NUMBA_CACHE_DIR=/tmp/numba_cache_bm
python factor.py
```

Mouse DB: `/cis/net/r41/data/iessien1/bone/sccellfie/mus_musculus`  
Outputs: `/cis/net/r41/data/iessien1/bone_marrow_results/chip_metabolic_graph/`

- `vnn_axis_2x2_tet2_il1.csv` — confirmatory cell-tuple contrasts
- `vnn_task_state_2x2_tet2_il1.csv` — per-task contrasts (exploratory)
- `vnn_cell_combo_interaction_tet2_il1.csv` — sampled 4-cell interaction tuples (axes)
- `vnn_interaction_attention_tet2_il1.csv` — cell-mean named-task attention
- `vnn_attention_heatmap_tet2_il1.png` — attention by arm
- `vnn_gene_task_weights_tet2_il1.csv` — learned gene→task edge weights

CUDA required. Training uses cells. Permutation unit is the mouse (`sample_name`).
