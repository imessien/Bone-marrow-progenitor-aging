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

`factor.py` is the CHIP entrypoint. Young McClatchy GSE209994 marker-HSPC
only (WT/Tet2 × vehicle/IL-1). No GMP mix, no age head, no extra GEOs in
the four arm labels.

A VNN reconstructs cell gene scores on named scCellFie tasks
(gene→task→subsystem→system), then freezes encodings. Each `sample_name`
is a mouse. `mice.png` is the interaction map: one Normalized 2×2 per
metabolic task (glycolysis, Complex I/II, TCA NADH, pyruvate, HMP, ribose-5-P).
The statistic is arm means of those task encodings:
`(Tet2_IL1 − WT_IL1) − (Tet2_vehicle − WT_vehicle)`. $p = k/36$ under each
panel. `mice_genes.png` is interpretability for that map: the same 2×2
effects on frozen gene embeddings, not log counts.

GSE285379 FACS-HSPC TET2 × LPS is the same VNN map in human (`human.png`,
`human_genes.png`; 4 libraries, 4 combos, CTRL/LPS). Not merged into McClatchy.

Family: DCell, P-NET. Message passing: Ma 2019 FGNN. Knowledge graph: scCellFie.

```bash
source .venv/bin/activate
export NUMBA_CACHE_DIR=/tmp/numba_cache_bm
python factor.py
```

Mouse DB: `/cis/net/r41/data/iessien1/bone/sccellfie/mus_musculus`  
Human DB: `/cis/net/r41/data/iessien1/bone/sccellfie/homo_sapiens`  
Outputs: `/cis/net/r41/data/iessien1/bone_marrow_results/chip_metabolic_graph/`

- `mice.png` / `mice.csv` — mouse VNN per-task 2×2 map
- `mice_genes.png` / `mice_genes.csv` — mouse VNN gene-embedding clouds
- `human.png` / `human.csv` — human VNN per-task 2×2 map
- `human_genes.png` / `human_genes.csv` — human VNN gene-embedding clouds

CUDA required. Training uses cells. Permutation unit is the mouse (`sample_name`).
