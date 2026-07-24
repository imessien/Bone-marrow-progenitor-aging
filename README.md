# Bone-marrow progenitor aging

IL-1 drives HSCs toward myeloid/GMP; Mitchell tests that genetically via Il1r1.
We integrate age-core cohorts with scGen (technical batch only), then ask how
**condition** (age / genotype) shifts HSC→GMP branch fate and EM metabolic
programs along pseudotime — not “scGen proves aging.”

## Workflow

```bash
source .venv/bin/activate
python preprocess.py --dataset age_core --annotate   # QC + annotate per study
python explore.py --skip-transfer                    # joint scGen + UMAP + DPT
python explore.py --pseudotime-only                  # refresh PT / plots only
python explore.py --path-gnn                         # shallow tree-GNN path persistence
python -c "import plotting; plotting._self_check()"  # synthetic smoke
```

| File | Role |
|------|------|
| `preprocess.py` | Per-study QC, lineage, age_bin |
| `explore.py` | Joint concat, scGen, DPT, path-GNN CLI |
| `plotting.py` | Figures + shallow tree-GNN path persistence |

Outputs live under `results/joint_hsc_aging/` on the bone store
(`/cis/net/r41/data/iessien1/bone/results/joint_hsc_aging`), symlinked as
`results/` in this repo.

**Plot / model convention:** `age_bin` vertical (early→mid→late); pseudotime
drives path start / persist / die-off on an HSPC→Myeloid_prog skeleton.
Task scores at every step; GSEA on long vs short routes. Finer labels later.
No early cells in Myeloid_prog would be surprising — early myeloid is expected.
Mitchell IL1R1KO is old-only — frame genotype contrasts accordingly.
