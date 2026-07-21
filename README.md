# Bone-marrow progenitor aging

This project tests whether aging redistributes mouse bone-marrow
`Myeloid_prog` cells along a shared progenitor continuum and whether Il1r1 loss
partly reverses that shift while changing metabolism at matched pseudotime.

Mitchell itself did not do this. The paper that generated your KO data uses discrete cohorts (~2 mo, 13 mo, ~24 mo) [1], is niche/stroma-focused, and shows Il1r1 loss rebalances progenitor abundance and lineage commitment — but it never builds a continuous-age myeloid pseudotime or tests reversal of pseudotime/branch occupancy. That's the exact gap you'd fill.
The reversibility premise is already established mechanistically — Pietras 2016 showed chronic IL-1 drives precocious myeloid differentiation and that this is reversible on IL-1 withdrawal [5]. So "Il1r1 loss partially reverses" is biologically expected; your novelty is quantifying it as trajectory-occupancy over age at matched branch position, with a metabolic readout, not the bare claim that IL-1 is reversible.

`explore.py` keeps only the substantially overlapping `Myeloid_prog` state,
runs scGen batch removal (`dataset` only) → neighbors → diffusion map → DPT,
and retains chronological age as `age_months`/`age_label`. Age is never treated
as a technical batch.
