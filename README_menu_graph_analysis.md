
# Menu-graph & Planning-Uncertainty Analyses

This folder contains a **reproducible pipeline** for the Johnson-graph (menu reconfiguration), recency/caching,
planning-uncertainty, directed exploration, MB-consistency, policy complexity, and experience-weighted centrality analyses.

## Files
- `menu_graph_analysis.py` — standalone script (Python 3.9+) that runs everything and writes outputs.
- `menu_graph_analysis.ipynb` — notebook wrapper that calls the script and displays the source.
- `requirements_menu_graph_analysis.txt` — minimal package list.

## Default inputs
The script looks for these files (any subset that exists will be processed):
- `/mnt/data/hddm2_fixed_final_3states.csv`
- `/mnt/data/hddm2_fixed_final_4states.csv`
- `/mnt/data/hddm2_fixed_final_5states.csv`
- `/mnt/data/hddm2_fixed_final_2states.csv` (optional)

## How to run
From a shell:
```
python /mnt/data/menu_graph_analysis.py --outdir /mnt/data
```
Or open the notebook and run the first cell.

## Outputs
See the script header or the assistant’s previous message for the full list (CSV, TXT, PNG). By default everything is placed in `/mnt/data`.

## Notes
- Plots use matplotlib only; no style/colors are forced.
- Regressions use statsmodels with **cluster-robust SEs by subject** where appropriate.
- The script reconstructs the first-stage menu pairs from `state1` using the lexicographic Johnson-graph mapping,
  computes trial-wise **planning uncertainty** from Beta posteriors, **Johnson distance**, **recency lags**,
  and the **MB-consistency** signal.
