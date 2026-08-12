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

## CHIP metabolic Dynamic Deep Factor Graph

Single entrypoint: **`chip_metabolic.py`**.

Hierarchy: **gene → task → subsystem → system** (mouse scCellFie).  
Dynamic edge gates + deep unrolled message passing.  
Biology: Tet2 × IL-1β 2×2 on N-glycosylation / glycolysis / OXPHOS  
(GSE209994 is all young — no age axis in this graph).

```bash
source .venv/bin/activate
export NUMBA_CACHE_DIR=/tmp/numba_cache_bm
unset CUDA_VISIBLE_DEVICES   # use all visible GPUs (DataParallel)
python chip_metabolic.py
```

Mouse DB: `/cis/net/r41/data/iessien1/bone/sccellfie/mus_musculus`  
Outputs: `/cis/net/r41/data/iessien1/bone_marrow_results/chip_metabolic_graph/` (repo `results/` symlink)

Training requires CUDA. Batch size scales as `1024 × n_GPU` via `nn.DataParallel` across every visible device.
