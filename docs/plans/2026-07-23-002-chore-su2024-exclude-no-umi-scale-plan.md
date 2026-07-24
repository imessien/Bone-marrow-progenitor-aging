---
title: "Su stays excluded — no UMI-scale path - Plan"
date: 2026-07-23
type: chore
artifact_contract: ce-unified-plan/v1
artifact_readiness: requirements-only
product_contract_source: ce-brainstorm
execution: code
---

## Goal Capsule

**Objective:** Lock the age-core scGen policy that Su2024 Smart-seq2 stays out of the primary joint; treat UMI-style depth scaling as a non-path into that joint.

**Authority:** Session-settled decisions below. Current `explore.py` default (`exclude_datasets=("su2024",)`) already encodes the joint rule; this plan does not reopen it.

**Open blockers:** None. No code or README changes are required in this round.

---

## Product Contract

### Summary

Record that Su2024 is excluded from the primary age-core scGen experiment and that rescaling Smart-seq2 counts to “match” 10x UMI depth is not a fix. Leave the existing exclude-default and holdout path unchanged for now.

### Problem Frame

Su2024 is plate-based Smart-seq2; the age-core joint is droplet 10x UMI. Force-include already failed under scGen (`technical_batch`) with pinpoint streaks after symbol/QC fixes. Depth scaling does not convert assays, cannot invent shared markers missing from Su vs the joint, and does not make Su the same experiment as the 10x atlas.

### Key Decisions

- **Su stays out of primary scGen.** (session-settled: user-directed — chosen over force re-integrate: Smart-seq2 vs 10x is a different assay; scaling cannot make it fit.)
- **No UMI-style scale path.** (session-settled: user-directed — chosen over depth-match / downsample recipes: depth is not the binding failure; gene space and chemistry are.)
- **No second-model transfer as “the same experiment.”** (session-settled: user-directed — chosen over projecting Su onto the 10x embedding with another model: that is a different analysis.)
- **No implementation this round.** (session-settled: user-directed — chosen over documenting now or removing holdout: leave code and docs as-is; this plan is the durable record.)

### Requirements

**Joint policy**

- R1. Primary age-core scGen joint continues to exclude `su2024` by default.
- R2. Do not treat library-size scaling, UMI-target downsampling, or similar depth transforms as a path to include Su in that joint.
- R3. Do not claim Su and the 10x core are the same scGen experiment via a separate transfer / projection model.

**Stability**

- R4. Existing `--su-holdout` adult/juvenile summary behavior may remain; this plan does not require enabling, disabling, or redesigning it.
- R5. Ship no code or README edits solely to satisfy this plan; the artifact itself locks the decision until a later change is explicitly requested.

### Scope Boundaries

**In scope**

- Decision record: exclude Su from primary joint; reject UMI-scale as a remedy.

**Deferred for later**

- Optional prose in `data/Su2024_CD49b_HSC/README.md` or joint docs spelling out “why not rescale.”
- Any future re-integration attempt with a different integrator or cohort strategy (not approved here).

**Outside this decision**

- Replacing scGen.
- Stacking scANVI/scArches on scGen latents for Su.
- Changing 10x age-core membership (Mitchell, White, Yang, Hérault, Elias).

### Acceptance Examples

- AE1. A later attempt to “fix Su for the joint by scaling UMIs” is rejected by pointing at R2 and the Key Decisions, without reopening assay-conversion debate.
- AE2. Running the age-core joint with default flags still excludes Su; force-include remains an explicit opt-in, not the documented path.

### Assumptions and Dependencies

- Assumption: Current default exclude in `explore.py` remains the runtime source of truth until code is deliberately changed.
- Assumption: Holdout summary (shared genes; no UMAP retrain) is compatible with “no changes for now” and needs no action under this plan.
- Dependency: Aligns with occupancy plan R1 (Su holdout excluded from primary joint by default) and that plan’s out-of-scope “rebuilding Su into the primary scGen UMAP.”
