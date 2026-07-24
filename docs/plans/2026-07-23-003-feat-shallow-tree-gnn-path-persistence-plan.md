---
title: "Shallow tree-GNN path persistence on age-core atlas - Plan"
date: 2026-07-23
type: feat
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-brainstorm
execution: code
supersedes: docs/plans/2026-07-23-001-feat-hsc-gmp-em-stream-occupancy-plan.md
---

## Goal Capsule

**Objective:** Finish `plotting.py` (with `explore.py` inputs) so a shallow tree-GNN on the HSC→Myeloid_prog branch skeleton reports which paths start, persist, or die across `age_bin`, with task scores at every step and GSEA on long vs short route genes. scGen-predict remains optional validation of the same condition axis.

**Authority:** Session-settled decisions below. Coarse axis labels (`HSPC`, `Myeloid_prog`) are the v1 contract; finer labels are deferred.

**Stop when:** `explore.py --path-gnn` writes persistence CSV, step-score CSV, GSEA table, and age_bin×PT path figure under `results/joint_hsc_aging/`; synthetic self-check in `plotting.py` passes; no pytest suite restored.

---

## Product Contract

### Summary

On the existing age-core scGen joint, treat **age_bin as the vertical time axis** and **pseudotime as path dynamics** (start / stop / die-off). Fit a data-driven branch skeleton between fixed **start = HSPC** and **end = Myeloid_prog**. A **shallow tree-GNN** (hierarchy-inspired aggregation only — not full T-GNN) learns how routes flow under condition. Emit task/EM scores along every step and GSEA on genes that mark long-lived vs dying routes. Deliver this by finishing `plotting.py` against CLI outputs from `preprocess.py` → `explore.py`.

### Problem Frame

Occupancy-only or PHLOWER-stream-only plans miss the intended story: condition changes which differentiation routes survive over calendar age. The joint object already has two axis labels and dense cells per age_bin; the missing piece is path persistence plus condition-aware flow on a tree, not deeper cell-type ontology or a deep kNN GNN.

### Key Decisions

- **Three-file CLI spine** — `preprocess.py` → `explore.py` → `plotting.py`; no marimo, no pytest suite for this work. `(session-settled: user-directed — chosen over marimo+tests workflow: get to the biology point)`
- **Outputs under `results/joint_hsc_aging/`** on the bone store (repo `results/` symlink). `(session-settled: user-directed — chosen over data/joint_hsc_aging: separate results from raw study data)`
- **age_bin vertical; pseudotime = path start/stop** — calendar progression up→down; PT governs which routes live or die. `(session-settled: user-directed — chosen over age-as-x / PT-as-y panel framing and over genotype-as-primary vertical)`
- **Fixed start/end labels** — v1 uses only `HSPC` → `Myeloid_prog`; data-driven splits inside that axis are allowed. `(session-settled: user-directed — chosen over requiring finer GMP/MDP labels first: start and end are what matter)`
- **Finer labels later** — optional enrichment from literature / Paperclip + `citations.bib`, not a v1 blocker. `(session-settled: user-approved — chosen over blocking on re-annotation)`
- **Shallow tree-GNN on branch skeleton** — few message-passing layers; hierarchical aggregation along the tree (T-GNN-inspired), not the full heterogeneous T-GNN stack and not a deep cell-kNN GNN. `(session-settled: user-approved — chosen over adopting CARTA/SLICE/DDRTree as the core model and over skipping GNN entirely)`
- **scGen-predict = optional validation** — young↔old or WT↔KO transfer validates the condition axis; it is not the main deliverable and not demoted QC. `(session-settled: user-directed — chosen over scGen-predict as primary story or as throwaway QC)`
- **No multi-panel genotype overlays in v1** — branch trees and path readouts first. `(session-settled: user-directed — chosen over genotype overlay panels)`
- **Reject CARTA / SLICE / DDRTree as core** — CARTA needs lineage trees; SLICE/DDRTree only compete with trajectory geometry already covered by DPT + branch skeleton. `(session-settled: user-approved — chosen over switching the core plan to those tools)`

### Requirements

**Workflow and artifacts**

- R1. The runnable spine remains `preprocess.py` → `explore.py` → `plotting.py` with joint artifacts under `results/joint_hsc_aging/`.
- R2. `plotting.py` is the home for branch-skeleton construction helpers, shallow tree-GNN train/infer, path-persistence summaries, step-wise task scores, and GSEA handoff tables/figures.

**Geometry and labels**

- R3. Plots and model use **age_bin** as the vertical time axis (early → mid → late).
- R4. **Pseudotime** defines path progression used to decide route start, persistence, and die-off.
- R5. Branch skeleton is anchored at **start = HSPC** and **end = Myeloid_prog**; additional splits are data-driven within that axis.
- R6. v1 must not require finer cell-type labels than `HSPC` / `Myeloid_prog`.

**Path persistence and condition**

- R7. The system reports routes that persist vs die off across age_bin (longest vs shortest / surviving vs collapsing occupancy along PT).
- R8. Condition comparison (at minimum genotype WT vs IL1R1KO where both exist; age_bin as the vertical) uses path-level summaries suitable for side-by-side comparison without forcing a single two-root DAG.
- R9. Early Myeloid_prog is treated as expected biology, not a data bug.

**Scores and enrichment**

- R10. Task / EM (and related) scores are computed and plottable **at every step** along retained routes (PT bins or tree nodes), not only at terminals.
- R11. Genes associated with long-lived vs dying routes support a **GSEA** (or equivalent enrichment) table under results.

**Model**

- R12. The predictive model is a **shallow tree-GNN** on the branch skeleton with hierarchical aggregation along tree paths; depth tracks tree depth (few layers).
- R13. scGen batch-corrected latent remains the embedding input from `explore.py`; scGen condition-predict is optional validation only.

**Deferred annotation**

- R14. Finer labels, when added later, may be informed by Paperclip lookups against entries in `citations.bib` without rewriting the v1 start/end contract.

### Key Flows

1. Load `results/joint_hsc_aging/age_core_scgen.h5ad` (scGen latent, `age_bin`, `lineage`/`cell_type`, `genotype`, `dpt_pseudotime`).
2. Build / refine branch skeleton HSPC → Myeloid_prog; allow data-driven internal splits.
3. Train or fit shallow tree-GNN; score path persistence across age_bin (and genotype where available).
4. At each step, attach task scores; export long vs short route gene sets → GSEA.
5. Optionally run scGen-predict as validation that the same condition axis moves.

```text
age_bin (vertical: early → mid → late)
        │
HSPC ───┼──► data-driven splits ──► Myeloid_prog
        │         │
        │    paths persist or die (PT dynamics)
        │         │
        └──── task scores @ every step → GSEA (long vs short)
```

### Acceptance Examples

- AE1. When only `HSPC` and `Myeloid_prog` labels exist, path persistence and step-wise scores still run end-to-end.
- AE2. When early Myeloid_prog cells are present, the pipeline does not treat them as an error; persistence is about route survival across age_bin.
- AE3. When IL1R1KO lacks young/mid coverage, genotype path comparison is framed on overlapping age support (old-heavy KO) without inventing missing bins.
- AE4. When scGen-predict is skipped, the tree-GNN path + score + GSEA path still completes.

### Success Criteria

- A reader can see which HSPC→myeloid routes strengthen or die across age_bin.
- Task scores are visible along the route, not only at the end.
- Long vs short route genes yield an enrichment table.
- `plotting.py` is the finished analysis surface; `explore.py` stays CLI integration + PT/embedding producer.

### Scope Boundaries

**In scope:** Shallow tree-GNN on branch skeleton; path persistence; step-wise task scores; GSEA; vertical age_bin; start/end HSPC→Myeloid_prog; optional scGen-predict validation; finish `plotting.py`.

**Deferred:** Finer cell-type re-annotation (Paperclip + `citations.bib`); multi-panel genotype overlays; CARTA/SLICE/DDRTree as primary; restoring marimo or pytest.

**Out of scope:** TMS occupancy; CellOracle stromal IL-1; full T-GNN stack; scGen-predict as main claim.

### Dependencies / Assumptions

- Joint object already provides `age_bin`, `lineage`/`cell_type` ∈ {HSPC, Myeloid_prog}, `genotype`, `dpt_pseudotime`, and scGen latent.
- Mitchell IL1R1KO is old-skewed; asymmetric condition support is expected.
- `torch-geometric` and `gseapy` are already project dependencies.

### Outstanding Questions

**Resolve Before Planning:** none.

**Deferred to Planning:** resolved in Planning Contract KTDs below.

### Sources / Research

- Supersedes: `docs/plans/2026-07-23-001-feat-hsc-gmp-em-stream-occupancy-plan.md`
- Joint snapshot: `results/joint_hsc_aging/age_core_scgen.h5ad`
- T-GNN inspiration (hierarchy only): https://arxiv.org/abs/2008.10003
- Code surface: `plotting.py`, `explore.py`, `preprocess.py`

---

## Planning Contract

### Key Technical Decisions

- KTD1. **Skeleton = PT quantile bins × lineage lanes** — aggregate cells into nodes `(pt_bin, lineage)` with forward-PT edges and HSPC→Myeloid_prog cross edges; no STREAM/DDRTree dependency. `(session-settled: user-approved — chosen over DDRTree principal graph as required geometry)`
- KTD2. **Shared skeleton + stratified occupancy** — one skeleton from all cells; persistence tables split by `age_bin` and by `genotype` where both labels exist.
- KTD3. **ShallowTreeGNN = 2× SAGEConv on skeleton graph** — node features include latent summary, myeloid_frac, age_bin mix, genotype mix; target = myeloid_frac; CPU-ok few epochs; message pass only along forward-PT edges.
- KTD4. **Task scores every step** — mean expression for `AXIS_MARKERS` plus EM panel genes present in `var_names`; written per `(pt_bin, lineage, age_bin)`.
- KTD5. **Long vs short routes** — Myeloid_prog lane persistence length = contiguous PT span with occupancy ≥ threshold per age_bin; longest vs shortest strata drive ranked genes → `gseapy.prerank` best-effort.
- KTD6. **Verification without pytest** — `plotting._self_check()` synthetic AnnData assert + `--path-gnn` smoke. `(session-settled: user-directed — chosen over restoring tests/)`
- KTD7. **CLI** — `explore.py --path-gnn` loads joint h5ad and calls `plotting.run_path_gnn_pipeline`.

### Technical Design

Aggregate first (O(n_cells) → O(n_bins × 2) nodes), then GNN on the tiny graph. Plots: age_bin on vertical in a path-occupancy heatmap (pt_bin × age_bin for Myeloid_prog fraction).

### Assumptions

- `dpt_pseudotime` and `corrected_latent` already on joint h5ad.
- If GSEA gene sets fail offline, still write ranked gene TSV.

### Sequencing

U1 skeleton+persistence → U2 tree-GNN → U3 step scores+GSEA → U4 CLI+self-check → U5 README.

---

## Implementation Units

### U1. Branch skeleton and path persistence

**Goal:** Build PT×lineage skeleton and persistence tables by age_bin / genotype.

**Requirements:** R2–R9

**Files:** modify `plotting.py`

**Approach:** Quantile-bin `dpt_pseudotime`; nodes per (bin, HSPC|Myeloid_prog); occupancy + contiguous persistence spans; CSV under results.

**Test scenarios:**
- Synthetic AnnData with early HSPC / late Myeloid yields forward edges only.
- Early Myeloid_prog present does not raise.
- Empty genotype×bin strata are omitted, not fabricated.

### U2. Shallow tree-GNN on skeleton

**Goal:** Train 2-layer SAGEConv; write predicted myeloid flow per node.

**Requirements:** R12–R13

**Files:** modify `plotting.py`

**Approach:** `torch_geometric.data.Data` from skeleton; short Adam train; CSV of predictions.

**Test scenarios:**
- Runs on CPU with ≤100 nodes.
- Predictions finite and aligned to nodes.

### U3. Step-wise task scores and GSEA

**Goal:** Scores at every PT step; long vs short gene rank + enrichment table.

**Requirements:** R10–R11

**Files:** modify `plotting.py`

**Approach:** Per-bin mean marker/EM expression; contrast longest vs shortest myeloid persistence; `gseapy.prerank` best-effort; always write ranked TSV.

**Test scenarios:**
- Step score table has rows per (pt_bin, lineage, age_bin).
- Ranked gene file non-empty when expression present.

### U4. explore CLI + package exports

**Goal:** `--path-gnn` entrypoint; export new symbols from `__init__.py`.

**Requirements:** R1–R2

**Files:** modify `explore.py`, `__init__.py`

**Approach:** Flag loads `JOINT_H5AD`, calls `run_path_gnn_pipeline`.

**Test scenarios:**
- `--path-gnn` appears in help.
- Missing h5ad raises FileNotFoundError.

### U5. README path-gnn one-liner

**Goal:** Document `--path-gnn` in README workflow.

**Requirements:** R1

**Files:** modify `README.md`

**Test scenarios:** README mentions `--path-gnn` and `results/joint_hsc_aging`.

---

## Verification Contract

- `source .venv/bin/activate && python -c "import plotting; plotting._self_check()"`
- `python explore.py --path-gnn` with optional `max_cells` subsample for smoke
- No pytest suite (session-settled)

**Execution direction:** smoke-first self-check, then optional real h5ad smoke.

---

## Definition of Done

- [ ] U1–U5 complete
- [ ] Self-check passes
- [ ] Artifacts path documented; PR opened from LFG
- [ ] Product Contract R1–R14 addressed or explicitly deferred in PR body
