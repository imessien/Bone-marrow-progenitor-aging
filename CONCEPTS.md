# Concepts

Shared domain vocabulary for this project — entities, named processes, and status concepts with project-specific meaning. Seeded with core domain vocabulary, then accretes as ce-compound and ce-compound-refresh process learnings; direct edits are fine. Glossary only, not a spec or catch-all.

## Hematopoietic aging atlas

### Age-core
The primary joint mouse bone-marrow single-cell cohort used for aging and IL-1 analyses, built from a fixed set of GEO studies and integrated with technical-batch correction only (age and genotype kept out of the batch key).

### Display type
A coarse lineage label used for atlas visualization and some analyses (for example HSC, agedHSC, MPP, GMP). It is an annotation layer, not an independent biological measurement of fate.

### Type-circular fate
An absorption or trajectory readout whose absorbing terminals are defined from the same cell-type labels the plot claims to discover. High probability mass then largely recolors the type map rather than revealing aging, cytokine, or genotype biology.

## Gate and cargo

### Export gate
The marrow-side conditions that shape which myeloid cells are produced and available to leave the bone marrow — here chiefly chronological aging and IL-1 / inflammaging context.

### CHIP cargo
The genotype of a clonal hematopoiesis clone (for example Tet2, Dnmt3a, or Asxl1 loss) carried by marrow-derived myeloid cells; treated as a phenotype of the cells that may infiltrate tissues, not as a gene-set expression score alone.

### Paired CHIP GEO pilot
The settled discovery package that pairs a bone-marrow IL-1 × Tet2 study with a Tet2-versus-Dnmt3a brain-engraftment study so export phenotype and CNS infiltration can be read together.

### CHIP metabolic-graph transfer
The discovery hero: a small VNN/BiNN on **young McClatchy** GSE209994 marker-`HSPC` only (Tet2 × IL-1β 2×2). Named nodes are glycolysis, OXPHOS/TCA, and PPP scCellFie tasks. The net reconstructs cell gene scores through the graph. Each `sample_name` is a mouse. The 2×2 uses every HSPC: random 4-cell tuples (one per arm) give the interaction distribution; their mean is the cell-arm contrast. P-values permute treatment among mice within genotype (cells in a mouse move together). Confirmatory p-values are the three axis interaction terms.
*Avoid:* pooling GMP/Mono/Gran with HSPC so a 2×2 can be a lineage-mix shift; treating this as an HSC↔GMP fate/transition analysis
*Avoid:* age/Kovtonyuk/Caiado as training heads; GO as co-equal nodes; training or scoring a 4-arm classifier at 2 mice/arm; shuffling **cells** as if they were independent 2×2 units; treating Burns/Niño/Kim as extra training GEOs; novel TET2-selective cytokine discovery (requires wet lab)

### Graph factors
Named scCellFie tasks on the hypothesis axes (glycolysis ATP-from-glucose, Krebs + Complex I/II, PPP HMP/ribose-5-P) → subsystem → system. Family: visible / biologically-informed nets (DCell, P-NET). Unrolled gated message passing: Ma2019FGNN, implemented as PyG `MessagePassing` on a `HeteroData` gene/task/subsystem/system graph. Genotype, treatment, and the four arms are **labels**, not graph nodes.
*Avoid:* calling genotype, age, IL-1β, or Il1r1 “factors” in this graph sense; wiring GO or N-glycosylation in as co-equal nodes; calling this a gene–gene GAT/GCN

### Age context (not in this 2×2)
Inflammaging enters as **IL-1 treatment** on young McClatchy marrow. Chronological age is motivation, not a factor. Caiado bulk HSC Tet2×IL-1α is not an Old training cohort. Kovtonyuk age×Il1r1 is a separate IL-1 necessity experiment.
*Avoid:* claiming Caiado RNA is chronological aged marrow; pretrain→finetune or soft age priors on Mitchell/Kovtonyuk; LOO age accuracy as biology

### Cell-pooled graph training
The VNN trains on cells. Each McClatchy `sample_name` is a mouse. The 2×2 permutation reassigns treatment among mice within genotype. Edges are scCellFie gene→hypothesis-task priors with input-dependent gates. Study identity is not a graph node.
*Avoid:* treating cell count as the factorial n; shuffling cell labels independently of mouse; calling gated prior edges new biology connections

### Bulk metabolic prior
Public bulk Tet2 RNA-seq used only to check that metabolic gene programs move with genotype outside McClatchy (currently GSE132090). Not a graph-transfer cohort and not “Niño validation” while GSE314014 / MW ST004480–6532 remain private or empty.

## Clonal hematopoiesis and brain

### PACT
Passenger Assisted Clone Tracking: using aggregated passenger somatic mutations from blood clones to quantify the same clones in matched brain tissue, without requiring prior knowledge of the driver mutation.

### Microglia replacement
The contribution of marrow-derived myeloid clones to the aged human microglial pool, measured as the fraction of microglia carrying blood-clone markers.
*Avoid:* treating all brain myeloid cells as blood contamination or as perivascular macrophages by default

## Flagged ambiguities

- "'CHIP score' had been used for curated driver gene-set expression on age-core streams — that is not the same as CHIP cargo from mutant GEO genotypes."
- "'Fate' on the joint UMAP had been read as competing aging biology — when terminals are display types, prefer calling it type-circular absorption."
- "Caiado PRJEB56666 is adult Tet2×IL-1α bulk HSC — not a chronological Old training cohort for the CHIP 2×2."
- "Coarse-aligning chronological old WT into McClatchy `WT_vehicle` / `WT_IL1` collapses age into cargo labels — keep the 2×2 on young McClatchy only."
