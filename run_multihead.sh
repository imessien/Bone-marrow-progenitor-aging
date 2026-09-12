#!/usr/bin/env bash
# Multi-head VNN (metabolic + EM rate) — must see /dev/nvidia*.
# Cursor agent sandboxes often hide GPU devices; run this in a host terminal.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
nvidia-smi --query-gpu=index,name,memory.free --format=csv
# force retrain: drop multihead stamp if present
python - <<'PY'
from pathlib import Path
import pandas as pd
for stem in ("mice", "human"):
    for base in (Path("results/chip_metabolic_graph"), Path("/cis/net/r41/data/iessien1/bone_marrow_results/chip_metabolic_graph")):
        p = base / f"{stem}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "multihead" in df.columns:
            df = df.drop(columns=["multihead"])
            df.to_csv(p, index=False)
            print("cleared multihead flag", p)
PY
python factor.py
echo "done — check results/chip_metabolic_graph/{mice,human}_{dual,rates,ions}.*"
