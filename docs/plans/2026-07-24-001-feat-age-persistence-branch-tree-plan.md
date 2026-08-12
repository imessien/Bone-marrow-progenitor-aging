---
title: "Age-persistence branch tree - Plan"
date: 2026-07-24
type: feat
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-brainstorm
execution: code
supersedes_hero: results/joint_hsc_aging/tree_paga_leiden.png
---

# Age-persistence branch tree - Plan

## Goal Capsule

**Objective:** Ship one primary figure: a **WT-only age-persistence branch tree** (Mitchell young→old) where **branch length = `age_bin` span**, **dots = cells by pseudotime**, type-only labels, plus EM/GSEA on long vs short WT routes. Support = **side-by-side late WT | late KO type maps** on the same skeleton (no age-length). Document young IL1R1KO as a **missing hole**.

**Authority:** Session-settled Product Contract below. Demotes `tree_paga_leiden.png` (PT-as-x PAGA) and path-occupancy heatmaps as heroes. GNN / scGen-predict remain validation-only.

**Stop when:** `explore.py --age-tree` (or renamed `--tree`) writes WT hero PNG, late genotype support PNG, node/edge CSVs, long-vs-short ranked genes + best-effort GSEA under `results/joint_hsc_aging/`; synthetic self-check passes; README notes the young-KO hole; no Ferchen download required.

**Open blockers:** None.

---

## Product Contract

### Problem Frame

Calendar age and differentiation pseudotime are different biology. Putting both on one shared geometric axis lies. The atlas product is to **show WT route kinetics across age**: which routes persist across `age_bin`, how cells sit along PT on those routes, and EM scores on dying vs surviving routes — then an honest late-age IL-1 genetic contrast.

**Known missing hole (locked — do not paper over):** Mitchell **IL1R1KO in GSE169162 is old-only** (`Old_IL1R1KO_*`; **0 young KO**). Twin age-persistence trees for WT|KO invent an age axis KO does not have. **Age extension is not the fix.**

### Users

- Primary: investigator building the BM aging / IL-1 story from the joint age-core atlas.
- Secondary: readers who need WT age map + late genotype contrast without a fake KO age continuum.

### Key Decisions

1. **Hero geometry** — length = occupied `age_bin` count; dots = PT; bins ≠ shared axis with PT. `(session-settled: user-directed)`
2. **Hero = WT only** — Mitchell young→old; no KO age-length tree. `(session-settled: user-directed)`
3. **Support = side-by-side late type maps** — same skeleton/labels as hero; no age-length. `(session-settled: user-directed)`
4. **Labels = type only** (`HSC` / `agedHSC` / `MPP` / `GMP`); Leiden backend, Ferchen later. `(session-settled: user-directed)`
5. **Length score = bin span**; occupancy threshold CLI-tunable. `(session-settled: user-directed)`
6. **Long vs short** from WT bin-span; GSEA + EM on those routes.
7. **GNN / scGen-predict = validation only**, not the claim.

### Visual Contract

```
Hero — WT (Mitchell young → old)

age persistence ──────────────►

HSC ──●●●── MPP ──●●── GMP ●●●●
      ↑ PT along branch
```

```
Support — late age only

late WT                         late IL1R1KO
HSC ──●●●── MPP ──●●── GMP      HSC ──●●── MPP ──●── GMP
(same skeleton + type labels; length ≠ age survival)
```

Caption / methods: young IL1R1KO absent; KO age-persistence not estimated.

### Requirements

**Geometry and panels**

- R1. Hero figure is WT-only; branch drawn length encodes number of occupied `age_bin`s for that route (early/mid/late as present in data).
- R2. Cell dots on hero branches are positioned by `dpt_pseudotime` (not by age).
- R3. Visible node labels are type-only (`HSC`, `agedHSC`, `MPP`, `GMP`); no `Leiden*` strings on the figure.
- R4. Support figure is two panels: late WT | late KO, sharing hero skeleton topology and type labels, without age-persistence length encoding.
- R5. Methods/README state the young-KO missing hole; no code path invents young KO for length.

**Persistence and scores**

- R6. A bin is occupied under an implementer default threshold exposed as a CLI flag; tune from occupancy CSVs.
- R7. Long vs short routes are defined by WT age_bin span; write ranked genes and best-effort GSEA (reuse existing prerank path if present).
- R8. EM/task gene-set proxy scores attach to route steps or nodes (scCellFie optional/deferred if import broken).

**Workflow**

- R9. Runnable spine remains `preprocess.py` → `explore.py` → `plotting.py`; artifacts under `results/joint_hsc_aging/`.
- R10. Naming backend for v1 is Leiden (±markers) → type collapse; Ferchen GSE266609 deferred.
- R11. Existing `tree_paga_leiden.png` may remain as QC; new hero filename must not claim KO age survival.

### Acceptance Examples

- AE1. WT hero renders with type-only labels and length varying by occupied bin count.
- AE2. Support late KO panel runs with old-only KO cells; does not error for missing young KO; does not draw age-length.
- AE3. CLI occupancy threshold change alters which bins count toward length and is reflected in CSV.
- AE4. Ferchen data absent: pipeline still completes end-to-end.

### Success Criteria

- Reader can state: length ≠ differentiation; dots = PT; hero = WT age survival.
- No figure claims KO age-persistence length.
- Long vs short WT CSVs + best-effort GSEA exist; EM tags or scores on nodes/routes.
- Visible labels never include `Leiden*`.

### Scope Boundaries

**In scope:** WT age-persistence hero; late WT|KO support maps; Leiden→type; bin-span length; CLI threshold; long/short GSEA; EM proxies; missing-hole docs.

**Out of scope:** Twin KO age-length trees; age-extension / deconvolution / Phung-Villatoro as substitute young KO; Ferchen for v1; scANVI-on-scGen; mid-bin as powered stratum; GNN/scGen-predict as claim.

### Assumptions

- `results/joint_hsc_aging/age_core_scgen.h5ad` has `corrected_latent`, `dpt_pseudotime`, `age_bin`, `genotype`, and lineage/type columns.
- WT has early and late coverage; KO is late-skewed.
- Multiple Leiden clusters may collapse to one display type.

### Outstanding Questions

**Blocking:** none.

**Deferred to implementer:** occupancy threshold default + CLI; max-branch / merge policy for readability.

### Sources / Research

- Prior hero code: `plotting.annotate_leiden_named`, `plot_paga_tree`, `run_tree_pipeline` (PT-as-x — to be replaced as hero).
- Persistence helpers to reuse conceptually: `build_branch_skeleton`, `path_persistence_table` (today = contiguous **PT** span — hero needs **age_bin** span instead).
- CLI: `explore.py --tree` / `run_tree_on_saved`.
- Hole evidence: Mitchell GSE169162 `Old_IL1R1KO_*` only.
- Citations: `citations.bib` Mitchell / Villatoro / Phung — support narrative only.

---

## Planning Contract

### Key Technical Decisions

- KTD1. **Reuse Leiden naming, change display** — keep `leiden_fine` / marker scoring; add `obs['display_type']` (or strip `Leiden{k}_` for plot labels). CSV may keep fine ids. `(session-settled: Leiden now, Ferchen later)`
- KTD2. **Route = edge or path on type-collapsed skeleton** — build connectivity on fine Leiden (or type nodes) via PAGA/neighbors; collapse to display-type graph for drawing; length from age_bin occupancy on that route among **WT** cells.
- KTD3. **Length = count of occupied age_bins** — not contiguous PT span from `path_persistence_table`; new helper (e.g. `age_bin_span_table`) with `min_cells` CLI (default ~10–30, match existing persistence defaults).
- KTD4. **Shared skeleton for support** — fit topology once (prefer all WT or all cells for edges); subset dots to late WT vs late KO for the two panels; fix layout coordinates across panels.
- KTD5. **Hero plot API** — new `plot_age_persistence_tree` (do not overload `plot_paga_tree` semantics); x or radial length = bin span; jitter/offset for multiple branches; PT maps to position *along* each branch.
- KTD6. **CLI** — extend `explore.py --tree` (or `--age-tree`) with `--min-cells-persist` (or sibling flag); call `run_age_persistence_pipeline`; keep old PAGA PNG optional via flag only if cheap.
- KTD7. **GSEA** — reuse long-vs-short ranked-gene + `gseapy.prerank` path from path-GNN pipeline; fix any stale `PosixPath.endswith` bug if still present when touching that code.
- KTD8. **Verification** — synthetic AnnData self-check in `plotting.py` (no pytest suite); smoke via CLI on joint h5ad.

### Technical Design

```text
age_core_scgen.h5ad
    → annotate_leiden_named → display_type collapse
    → WT subset → age_bin occupancy per route → length
    → plot_age_persistence_tree (hero)
    → late WT | late KO on shared layout (support)
    → EM tags + long/short genes → GSEA
    → CSVs + PNGs under results/joint_hsc_aging/
```

Demote: `tree_paga_leiden.png` as primary; path occupancy heatmaps stay QC.

### Assumptions

- `age_bin` ∈ {early, mid, late, …}; mid may be thin — length still counts occupied bins only.
- Genotype column distinguishes WT vs IL1R1KO (or project equivalents already in obs).

### Risks

| Risk | Mitigation |
|------|------------|
| Too many Leiden→type edges → spaghetti | Cap branches / merge low-n edges; CLI later |
| Mid bin empty → lengths mostly 1 or 2 | Document; early+late is the powered contrast |
| Support panels look like twin age heroes | Caption + no length axis; equal branch geometry |

### Sequencing

U1 display types → U2 age_bin span tables → U3 hero plot → U4 support panels → U5 CLI + GSEA + self-check + README.

---

## Implementation Units

### U1. Type-only display labels from Leiden

**Goal:** Produce plot-safe type labels without Ferchen.

**Requirements:** R3, R10

**Files:** modify `plotting.py` (`annotate_leiden_named` or thin wrapper)

**Approach:** After marker naming, write `display_type` = type suffix; keep `leiden_named` / `leiden_fine` for CSV.

**Test scenarios:**
- Every cell has `display_type` in `{HSC, agedHSC, MPP, GMP}` (or NA only if markers missing — then fail loud).
- Figure annotation strings never contain `Leiden`.

### U2. WT age_bin span / occupancy tables

**Goal:** Score routes by occupied age_bin count under CLI `min_cells`.

**Requirements:** R1, R6, R7

**Files:** modify `plotting.py`

**Approach:** New helper distinct from `path_persistence_table` (PT contiguous). For each route/edge/node set among WT: count bins with n≥threshold; write `age_persistence_routes.csv` (length, bins_hit, n_cells).

**Test scenarios:**
- Synthetic: route in early+late only → length 2.
- Raising min_cells drops a sparse bin and shortens length.
- KO-only cells never enter WT length table.

### U3. Hero age-persistence tree plot

**Goal:** Draw WT tree: length = bin span; dots = PT; type labels.

**Requirements:** R1–R3, R11

**Files:** modify `plotting.py`

**Approach:** `plot_age_persistence_tree` + `run_age_persistence_pipeline` writing e.g. `tree_age_persistence_wt.png`, node/edge CSVs. Optional EM tags via existing `_node_em_tags` on display types.

**Test scenarios:**
- PNG exists; axes/legend communicate age persistence ≠ PT.
- Dot positions correlate with PT within a branch (monotonic check on synthetic).

### U4. Late WT | late KO support maps

**Goal:** Side-by-side type maps; shared skeleton; no age-length.

**Requirements:** R4, R5

**Files:** modify `plotting.py`

**Approach:** Fix node positions from shared topology; subset late cells by genotype; equal branch lengths (topology only); write e.g. `tree_late_wt_vs_ko.png`.

**Test scenarios:**
- Runs with zero young KO.
- Two panels share node coordinates (same skeleton).
- No colorbar/axis labeled as age persistence length.

### U5. CLI, GSEA handoff, self-check, README hole note

**Goal:** Runnable entrypoint + verification + honest docs.

**Requirements:** R5, R7–R9, R11

**Files:** modify `explore.py`, `plotting.py`, `README.md`

**Approach:** Wire `--tree` (or `--age-tree`) + `--min-cells-persist`; long vs short genes → ranked TSV + best-effort GSEA; `_self_check` asserts AE1–AE3 synthetically; README bullet on young-KO hole.

**Test scenarios:**
- CLI help documents threshold flag.
- Self-check passes without joint h5ad.
- Smoke on real h5ad writes hero + support + CSVs (manual/implementer).

---

## Verification Contract

- **Synthetic:** `python -c 'from plotting import _self_check; _self_check()'` (extend existing or age-tree-specific check).
- **Smoke:** `source .venv/bin/activate && python explore.py --tree` (with threshold flag) against `results/joint_hsc_aging/age_core_scgen.h5ad`.
- **Artifacts:** `tree_age_persistence_wt.png`, `tree_late_wt_vs_ko.png`, route/node CSVs, long-vs-short ranked TSV; GSEA table or skip reason with string-safe paths.
- **Visual QA:** hero legend length=age bins; support has no age-length; labels type-only.

---

## Definition of Done

- All R1–R11 satisfied; AE1–AE4 hold.
- U1–U5 complete; self-check green; smoke produces hero + support.
- Young-KO hole stated in README (and figure caption text if written).
- Ferchen not required; GNN/scGen-predict not sold as the result.
- Plan readiness remains implementation-ready; execution tracked in git, not this file.

---

## Appendix

### Rejected alternatives (record)

- Twin WT|KO age-persistence trees.
- Continuous `age_months` as drawn length.
- Occupancy bars instead of late type maps for support.
- Ferchen-required v1.
- Filling hole via Phung bulk / Villatoro / deconvolution.
