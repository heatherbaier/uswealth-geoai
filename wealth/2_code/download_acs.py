"""
download_acs.py
===============
Downloads all ACS 5-year tract-level variables needed for the wealth
index pipeline and saves two national CSVs.

  data/acs_wealth_variables_2015_2019_US.csv   (ACS 2015–2019, year=2019)
  data/acs_wealth_variables_2010_2014_US.csv   (ACS 2010–2014, year=2014)

The Census API caps at 50 variables per request.  The full variable set
(63 columns) is split into two groups that are each fetched separately
and then joined on geoid.

Usage
-----
  python download_acs.py

Requires a Census API key (free): https://api.census.gov/data/key_signup.html
Set it in the CENSUS_API_KEY constant below.
"""

import os
import time
import requests
import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────

CENSUS_API_KEY = "YOUR_CENSUS_API_KEY"   # replace with your key
DATA_DIR = "data"

STATE_FIPS = [
    '01','02','04','05','06','08','09','10','11','12','13','15','16','17','18',
    '19','20','21','22','23','24','25','26','27','28','29','30','31','32','33',
    '34','35','36','37','38','39','40','41','42','44','45','46','47','48','49',
    '50','51','53','54','55','56',
]

SENTINEL_VALUES = [-666666666, -999999999, -555555555, -1]

# ── Variable definitions ───────────────────────────────────────────────────────
# Split into two groups to stay under the Census API 50-variable limit.

ACS_VARS_CORE = {
    # Property
    "median_home_value":                        "B25077_001E",
    "median_year_built":                        "B25035_001E",
    "median_gross_rent":                        "B25064_001E",
    "median_rooms":                             "B25018_001E",
    "total_housing_units":                      "B25003_001E",
    "owner_occupied_units":                     "B25003_002E",
    "total_units_in_structure":                 "B25024_001E",
    "units_1_detached":                         "B25024_002E",
    "units_1_attached":                         "B25024_003E",
    "total_bedrooms":                           "B25041_001E",
    "crowding_total_households":                "B25014_001E",
    "owner_crowded_1_01_to_1_50":               "B25014_005E",
    "owner_severely_crowded_1_51_plus":         "B25014_006E",
    "renter_crowded_1_01_to_1_50":              "B25014_011E",
    "renter_severely_crowded_1_51_plus":        "B25014_012E",
    # Transport assets
    "vehicles_total_households":                "B25044_001E",
    "owner_1_vehicle":                          "B25044_003E",
    "owner_2_vehicles":                         "B25044_004E",
    "owner_3_plus_vehicles":                    "B25044_005E",
    "renter_1_vehicle":                         "B25044_008E",
    "renter_2_vehicles":                        "B25044_009E",
    "renter_3_plus_vehicles":                   "B25044_010E",
    # Utilities & digital (internet/computer NaN in 2014 — set to NaN after download)
    "internet_total_households":                "B28002_001E",
    "broadband_internet":                       "B28002_002E",
    "computer_total_households":                "B28001_001E",
    "has_computer":                             "B28001_002E",
    "heating_total_households":                 "B25040_001E",
    "utility_gas_heat":                         "B25040_002E",
    "electricity_heat":                         "B25040_004E",
    "fuel_oil_kerosene_heat":                   "B25040_005E",
    "wood_heat":                                "B25040_007E",
    # Socioeconomic & financial
    "median_household_income":                  "B19013_001E",
    "per_capita_income":                        "B19301_001E",
    "poverty_total_population":                 "B17001_001E",
    "population_below_poverty_line":            "B17001_002E",
    "labor_force_total":                        "B23025_002E",
    "unemployed":                               "B23025_005E",
    "rent_burden_universe":                     "B25070_001E",
    "rent_burdened_30_34":                      "B25070_007E",
    "rent_burdened_35_39":                      "B25070_008E",
    "rent_burdened_40_49":                      "B25070_009E",
    "rent_burdened_50_plus":                    "B25070_010E",
    "owner_mortgage_universe":                  "B25095_001E",
    "owner_mortgaged_units":                    "B25091_002E",
    "owner_mortgaged_30_34":                    "B25091_008E",
    "owner_mortgaged_35_39":                    "B25091_009E",
    "owner_mortgaged_40_49":                    "B25091_010E",
    "owner_mortgaged_50_plus":                  "B25091_011E",
}

# Variables not released at tract level for the 2010–2014 ACS vintage
NOT_IN_2014 = {"internet_total_households", "broadband_internet",
               "computer_total_households", "has_computer"}

ACS_VARS_SUPPLEMENTAL = {
    # Property — units-in-structure breakdown
    "units_2":              "B25024_004E",
    "units_3_4":            "B25024_005E",
    "units_5_9":            "B25024_006E",
    "units_10_19":          "B25024_007E",
    "units_20_49":          "B25024_008E",
    "units_50plus":         "B25024_009E",
    "mobile_home":          "B25024_010E",
    # Utilities — additional heating fuels
    "propane_heat":         "B25040_003E",
    "solar_heat":           "B25040_008E",
    "no_fuel_heat":         "B25040_010E",
    # Transport — commute mode
    "commute_universe":     "B08301_001E",
    "commute_car_alone":    "B08301_003E",
    "commute_transit":      "B08301_010E",
    "commute_walk":         "B08301_019E",
    "commute_work_home":    "B08301_021E",
}

# ── Download helpers ───────────────────────────────────────────────────────────

def download_state(variables, state_fips, year, max_retries=5):
    base_url = f"https://api.census.gov/data/{year}/acs/acs5"
    var_list = ",".join(variables.values())
    params = {
        "key":  CENSUS_API_KEY,
        "get":  f"NAME,{var_list}",
        "for":  "tract:*",
        "in":   f"state:{state_fips}",
    }
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(base_url, params=params, timeout=120)
            if r.status_code == 200:
                data = r.json()
                df = pd.DataFrame(data[1:], columns=data[0])
                df = df.rename(columns={v: k for k, v in variables.items()})
                df["geoid"] = df["state"] + df["county"] + df["tract"]
                for col in variables.keys():
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                        for s in SENTINEL_VALUES:
                            df[col] = df[col].replace(s, np.nan)
                keep = ["geoid"] + [c for c in variables.keys() if c in df.columns]
                return df[keep]
            else:
                print(f"  HTTP {r.status_code} for state {state_fips}")
                return None
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  Attempt {attempt} failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  All {max_retries} attempts failed for state {state_fips}: {e}")
                return None


def download_all_states(variables, year):
    parts = []
    failed = []
    for i, fips in enumerate(STATE_FIPS, 1):
        df = download_state(variables, fips, year)
        if df is not None:
            parts.append(df)
            print(f"    [{i:>2}/{len(STATE_FIPS)}] state {fips}: {len(df):,} tracts")
        else:
            failed.append(fips)
    if failed:
        print(f"  WARNING: failed states: {failed}")
    return pd.concat(parts, ignore_index=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def download_year(year, core_vars, supp_vars):
    print(f"\n  Downloading core variables ({len(core_vars)})...")
    core_df = download_all_states(core_vars, year)

    print(f"\n  Downloading supplemental variables ({len(supp_vars)})...")
    supp_df = download_all_states(supp_vars, year)

    core_df["geoid"]  = core_df["geoid"].astype(str).str.zfill(11)
    supp_df["geoid"]  = supp_df["geoid"].astype(str).str.zfill(11)

    combined = core_df.merge(supp_df, on="geoid", how="left")
    print(f"\n  Combined: {len(combined):,} tracts × {combined.shape[1]} columns")
    return combined


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # ── 2019 ──────────────────────────────────────────────────────────────────
    print("=" * 65)
    print("  ACS 2015–2019 (year=2019)")
    print("=" * 65)
    df19 = download_year(2019, ACS_VARS_CORE, ACS_VARS_SUPPLEMENTAL)
    out19 = os.path.join(DATA_DIR, "acs_wealth_variables_2015_2019_US.csv")
    df19.to_csv(out19, index=False)
    print(f"  Saved: {out19}  ({len(df19):,} tracts, {df19.shape[1]} columns)")

    # ── 2014 ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  ACS 2010–2014 (year=2014)")
    print("=" * 65)
    core_2014 = {k: v for k, v in ACS_VARS_CORE.items() if k not in NOT_IN_2014}
    df14 = download_year(2014, core_2014, ACS_VARS_SUPPLEMENTAL)
    for col in NOT_IN_2014:
        df14[col] = float("nan")
    out14 = os.path.join(DATA_DIR, "acs_wealth_variables_2010_2014_US.csv")
    df14.to_csv(out14, index=False)
    print(f"  Saved: {out14}  ({len(df14):,} tracts, {df14.shape[1]} columns)")
    print(f"  Note: {NOT_IN_2014} set to NaN (not available at tract level before 2015–2019 ACS)")

    print("\n" + "=" * 65)
    print("  Done.")
    print("=" * 65)


if __name__ == "__main__":
    main()
