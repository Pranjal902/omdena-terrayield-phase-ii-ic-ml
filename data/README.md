# Data

The scripts expect two CSV files in this folder. They aren't included because the data belongs to the Omdena TerraYield Analytics Phase II project and isn't mine to redistribute. If you have access to the project, copy them here; the file names have to match.

| File | Shape | What it is | SHA-256 |
|---|---|---|---|
| `master_dataset.csv` | 580 x 154 | interim joined table from Sprint 1 (region x crop x 16-day window, 2020 to April 2026) | `5b3e8a1c956e834c2670777fccff9372773299b2105973bc5117f46a2ffc03f7` |
| `Gold_Dataset_Candidate_v1.csv` | 580 x 81 | Gold dataset candidate handed to the ML team in Sprint 3 (2020 to 2025) | `c11a7ee352cd1f035d2df5d7bd647f2e526476320c92dde9a9398dee73c2be8f` |

Both cover US-Iowa and US-Illinois, corn and soybean. Another copy of the Gold candidate in the project repository writes the window dates in a single format; the loader reads either version and the results are the same. The underlying sources are public: Sentinel-2 L2A imagery aggregated per 16-day window, Open-Meteo weather, USDA NASS Quick Stats yields, and grain prices.

You can keep the files somewhere else and point the scripts at it:

```bash
python src/audit_yield_label.py --data-dir /path/to/folder
```

For the notebook, set `TERRAYIELD_DATA_DIR=/path/to/folder` before starting Jupyter.

Nothing in this repo needs an API key. If you ever rebuild these tables from the raw sources, keys such as a NASS Quick Stats key or an Earth Engine service account belong in a local `.env` file. That file is already in `.gitignore`, so it never reaches git.
