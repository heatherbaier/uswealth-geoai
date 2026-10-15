# Code — GeoAI Wealth Index

## Files

| File                  | Purpose                                                                                                                               |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `download_acs.py`     | Downloads all 63 ACS 5-year variables at census tract level for all 50 states + DC via the Census API and saves the two national CSVs |
| `run_wealth_index.py` | Computes all six wealth indices using national income-correlation weights and saves output CSVs and diagnostic plots                  |

---

## Requirements

```
pandas, numpy, scikit-learn, matplotlib, requests
```

---

## Running the Analysis

### Step 1 — Download ACS data

```bash
python download_acs.py
```

This downloads all 63 ACS variables in a single run (two API calls per state to stay under the Census 50-variable limit) and writes both period CSVs to `data/`. Set your Census API key in the `CENSUS_API_KEY` constant at the top of the script.

### Step 2 — Compute wealth indices

```bash
python run_wealth_index.py
```

By default the script runs for **Ohio**. To change the target:

```python
# Edit these three lines at the top of run_wealth_index.py:
STATE_FIPS = '39'       # Ohio=39, Pennsylvania=42, Michigan=26, California=06, Arizona=04
STATE_NAME = 'Ohio'
STATE_ABBR = 'OH'

# Set STATE_FIPS = None to run for the full United States:
STATE_FIPS = None
STATE_NAME = 'United States'
STATE_ABBR = 'USA'
```

Output is saved to `output/<STATE_ABBR>/`:

- `wealth_indices_<year>_<STATE_ABBR>.csv` — all indices + raw variables
- Diagnostic plots (score distributions, income/poverty scatter, feature weights)

---
