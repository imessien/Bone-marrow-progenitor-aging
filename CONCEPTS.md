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
The discovery hero: a graph model with scCellFie metabolic module features that learns age_cohort × genotype × IL-1 classes on a shared gene→task→subsystem→system graph (McClatchy Young + Caiado Old), then transfers to GSE298597 Tet2 versus WT myeloid/Mono_Mac–like cells, with Dnmt3a as a hard negative.
*Avoid:* calling plain scCellFie heatmaps on McClatchy the result; treating Burns/Niño/Kim as extra training GEOs; novel TET2-selective cytokine discovery (requires wet lab)

### Graph factors
Metabolic task (and higher) nodes in the deep factor graph — e.g. scCellFie tasks such as glycolysis or N-linked glycosylation — not experimental conditions.
*Avoid:* calling genotype, age, IL-1β, or Il1r1 “factors” in this graph sense

### Age context (observed class on shared graph)
Young = GSE209994 McClatchy (scRNA Tet2×IL-1β). Old = PRJEB56666 Caiado (bulk HSC Tet2×IL-1α, ~6–9 mo adult proxy — not chronological 18–24 mo aging RNA). Both scored onto the same genes/tasks and trained as 8-way `age_cohort × genotype × treatment` classes in `chip_metabolic.py`. Gene→task edge weights and per-gene input scales are learnable (`Task_by_Gene` only defines which edges exist). Optional `--age-prior` initializes/scales those edges by Young vs Old task Cohen's d; `--young-only` ablates Caiado.
*Avoid:* claiming Caiado RNA is chronological aged marrow; mapping old WT into young `WT_vehicle` / `WT_IL1` labels; treating Il1r1KO/GF as the age bin itself

### Cell-pooled graph training
The deep factor graph is per-row (gene→task→subsystem→system with gated prior edges). Mixing McClatchy cells and Caiado bulk samples is allowed when features share the same Task_by_Gene graph; study identity is not a graph node. Age is an observed class on that shared graph — pooling alone does not invent aged-Tet2 biology.
*Avoid:* “samples don’t matter so age is learned automatically”; calling gated prior edges new “age connections” in the structural sense

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
- "Caiado PRJEB56666 is adult Tet2×IL-1α bulk HSC used as proxy Old vs McClatchy Young — not 18–24 mo chronological aging RNA."
- "Coarse-aligning chronological old WT into McClatchy `WT_vehicle` / `WT_IL1` collapses age into cargo labels — prefer explicit `age_cohort` classes on the shared graph."
