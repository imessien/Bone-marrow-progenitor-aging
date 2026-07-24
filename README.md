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
```

| File | Role |
|------|------|
| `preprocess.py` | Per-study QC, lineage, age_bin |
| `explore.py` | Joint concat, scGen, DPT, CLI |
| `plotting.py` | Figures; small GNN fate (planned) |

Outputs live under `results/joint_hsc_aging/` on the bone store
(`/cis/net/r41/data/iessien1/bone/results/joint_hsc_aging`), symlinked as
`results/` in this repo.

**Plot convention:** age on x, pseudotime on y; condition modulates myeloid
start / slow / stop. No early cells in `Myeloid_prog` is expected (myeloid
comes later). Mitchell IL1R1KO is old-only — frame genotype contrasts accordingly.
