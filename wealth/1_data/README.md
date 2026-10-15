# Data — GeoAI Wealth Index

## Files

| File                          | Period        | Tracts | Columns |
| ----------------------------- | ------------- | ------ | ------- |
| `wealth_indices_2019_USA.csv` | 2015–2019 ACS | 73,056 | 109     |
| `wealth_indices_2014_USA.csv` | 2010–2014 ACS | 73,056 | 109     |

Each file contains all raw ACS variables, all derived composite variables, all six wealth index scores, and their corresponding quintile labels.

---

## Wealth Indices

### `wealth_index` — Primary Index (8 composite variables, 14 raw ACS variables)

An income-weighted composite of housing and neighborhood characteristics — home value, gross rent, room count, building type, housing age, homeownership rate, and crowding

| Composite variable     | Derived from                                                                                             |
| ---------------------- | -------------------------------------------------------------------------------------------------------- |
| `log_home_value`       | log(`median_home_value`)                                                                                 |
| `log_gross_rent`       | log(`median_gross_rent`)                                                                                 |
| `median_rooms`         | direct ACS variable                                                                                      |
| `median_year_built`    | direct ACS variable                                                                                      |
| `single_family_pct`    | (`units_1_detached` + `units_1_attached`) / `total_units_in_structure`                                   |
| `owner_pct`            | `owner_occupied_units` / `total_housing_units`                                                           |
| `crowded_pct`          | (`owner_crowded_1_01_to_1_50` + `renter_crowded_1_01_to_1_50`) / `crowding_total_households`             |
| `severely_crowded_pct` | (`owner_severely_crowded_1_51_plus` + `renter_severely_crowded_1_51_plus`) / `crowding_total_households` |

---

### `wealth_index_sat` — Satellite Index (6 composite variables, 9 raw ACS variables)

A restricted composite built from housing characteristics that are physically observable from overhead imagery — home value, gross rent, room count, building type, housing age, and mobile home prevalence

| Composite variable  | Derived from                                                           |
| ------------------- | ---------------------------------------------------------------------- |
| `log_home_value`    | log(`median_home_value`)                                               |
| `log_gross_rent`    | log(`median_gross_rent`)                                               |
| `median_rooms`      | direct ACS variable                                                    |
| `median_year_built` | direct ACS variable                                                    |
| `single_family_pct` | (`units_1_detached` + `units_1_attached`) / `total_units_in_structure` |
| `mobile_home_pct`   | `mobile_home` / `total_units_in_structure`                             |

---

### `wealth_index_housing_core` — Housing Core Index (4 composite variables, 8 raw ACS variables)

A compact composite focused purely on ownership and occupancy conditions — home value, homeownership rate, mild crowding, and severe crowding

| Composite variable     | Derived from                                                                                             |
| ---------------------- | -------------------------------------------------------------------------------------------------------- |
| `log_home_value`       | log(`median_home_value`)                                                                                 |
| `owner_pct`            | `owner_occupied_units` / `total_housing_units`                                                           |
| `crowded_pct`          | (`owner_crowded_1_01_to_1_50` + `renter_crowded_1_01_to_1_50`) / `crowding_total_households`             |
| `severely_crowded_pct` | (`owner_severely_crowded_1_51_plus` + `renter_severely_crowded_1_51_plus`) / `crowding_total_households` |

---

### `wealth_index_transport` — Transport Index (2 composite variables, 7 raw ACS variables)

A composite of household vehicle access — the share of households with no vehicle and the share with two or more vehicles — intended to capture the transportation asset dimension of neighborhood wealth.

| Composite variable   | Derived from                                                                                                                  |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `pct_no_vehicle`     | (`vehicles_total_households` − households with ≥1 vehicle) / `vehicles_total_households`                                      |
| `pct_2plus_vehicles` | (`owner_2_vehicles` + `owner_3_plus_vehicles` + `renter_2_vehicles` + `renter_3_plus_vehicles`) / `vehicles_total_households` |

---

### `wealth_index_financial` — Financial Index (4 composite variables, 14 raw ACS variables)

A composite of economic burden and deprivation indicators — poverty rate, unemployment rate, rent burden, and owner housing cost burden

| Composite variable          | Derived from                                                                                                                        |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `poverty_rate`              | `population_below_poverty_line` / `poverty_total_population`                                                                        |
| `unemployment_rate`         | `unemployed` / `labor_force_total`                                                                                                  |
| `rent_burdened_30plus_pct`  | (`rent_burdened_30_34` + `rent_burdened_35_39` + `rent_burdened_40_49` + `rent_burdened_50_plus`) / `rent_burden_universe`          |
| `owner_burdened_30plus_pct` | (`owner_mortgaged_30_34` + `owner_mortgaged_35_39` + `owner_mortgaged_40_49` + `owner_mortgaged_50_plus`) / `owner_mortgaged_units` |

---

### `wealth_index_utilities` — Utilities Index (3 composite variables, 6 raw ACS variables)

A composite of digital and infrastructure access — broadband internet subscription, gas heating prevalence, and poor heating fuel prevalence — capturing the utilities and connectivity dimension of neighborhood wealth, with broadband available for the 2015–2019 period only.

| Composite variable | Derived from                                                          |
| ------------------ | --------------------------------------------------------------------- |
| `broadband_pct`    | `broadband_internet` / `internet_total_households`                    |
| `gas_heat_pct`     | `utility_gas_heat` / `heating_total_households`                       |
| `poor_heat_pct`    | (`fuel_oil_kerosene_heat` + `wood_heat`) / `heating_total_households` |

> **Note:** `broadband_pct`, `gas_heat_pct`, and `poor_heat_pct` are present in the dataset for both periods but are **not used in the primary `wealth_index` calculation** — they are used only for this sub-index. Additionally, `broadband_pct` is NaN in the 2010–2014 file (the Census Bureau did not release broadband variables at census tract level until the 2015–2019 ACS vintage), so the utilities sub-index for 2014 reflects heating variables only.

---

## Output Columns

Each data file contains the following column groups:

| Column group                                                 | Description                                                                                                                           |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| Raw ACS variables (68 cols)                                  | All downloaded ACS variables including building-type breakdown (B25024), commute mode (B08301), and additional heating fuels (B25040) |
| Derived composite variables                                  | All intermediate computed variables (e.g. `log_home_value`, `single_family_pct`, `poverty_rate`)                                      |
| `wealth_index` + `wealth_quintile`                           | Primary 8-variable index score and quintile label                                                                                     |
| `wealth_index_sat` + `wealth_quintile_sat`                   | Satellite 6-variable index score and quintile label                                                                                   |
| `wealth_index_housing_core` + `wealth_quintile_housing_core` | Housing core 4-variable index score and quintile label                                                                                |
| `wealth_index_transport` + `wealth_quintile_transport`       | Transport 2-variable index score and quintile label                                                                                   |
| `wealth_index_financial` + `wealth_quintile_financial`       | Financial 4-variable index score and quintile label                                                                                   |
| `wealth_index_utilities` + `wealth_quintile_utilities`       | Utilities 3-variable index score and quintile label                                                                                   |

Quintile labels: `Poorest`, `Poor`, `Middle`, `Rich`, `Richest` — assigned within each file using national score distributions.

---

## All Derived Variables in the Dataset

The table below covers every derived variable computed from the raw ACS columns. Variables marked **not used** are retained in the dataset for reference but excluded from all indices; the reason is given.

### Property & Housing

| Derived variable        | Derived from                                                                                             | Used in index                                                   | Notes                                                                                                                            |
| ----------------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `log_home_value`        | log(`median_home_value`)                                                                                 | `wealth_index`, `wealth_index_sat`, `wealth_index_housing_core` |                                                                                                                                  |
| `log_gross_rent`        | log(`median_gross_rent`)                                                                                 | `wealth_index`, `wealth_index_sat`                              |                                                                                                                                  |
| `median_rooms`          | direct ACS variable                                                                                      | `wealth_index`, `wealth_index_sat`                              |                                                                                                                                  |
| `median_year_built`     | direct ACS variable                                                                                      | `wealth_index`, `wealth_index_sat`                              |                                                                                                                                  |
| `single_family_pct`     | (`units_1_detached` + `units_1_attached`) / `total_units_in_structure`                                   | `wealth_index`, `wealth_index_sat`                              |                                                                                                                                  |
| `owner_pct`             | `owner_occupied_units` / `total_housing_units`                                                           | `wealth_index`, `wealth_index_housing_core`                     |                                                                                                                                  |
| `crowded_pct`           | (`owner_crowded_1_01_to_1_50` + `renter_crowded_1_01_to_1_50`) / `crowding_total_households`             | `wealth_index`, `wealth_index_housing_core`                     |                                                                                                                                  |
| `severely_crowded_pct`  | (`owner_severely_crowded_1_51_plus` + `renter_severely_crowded_1_51_plus`) / `crowding_total_households` | `wealth_index`, `wealth_index_housing_core`                     |                                                                                                                                  |
| `mobile_home_pct`       | `mobile_home` / `total_units_in_structure`                                                               | `wealth_index_sat`                                              |                                                                                                                                  |
| `units_2_pct`           | `units_2` / `total_units_in_structure`                                                                   | **not used**                                                    | Weak national correlation (r=−0.175); urban/rural confound — `single_family_pct` already captures the primary split              |
| `units_3_4_pct`         | `units_3_4` / `total_units_in_structure`                                                                 | **not used**                                                    | Weak national correlation (r=−0.207); same confound as above                                                                     |
| `units_5_9_pct`         | `units_5_9` / `total_units_in_structure`                                                                 | **not used**                                                    | Weak national correlation (r=−0.190); same confound as above                                                                     |
| `units_10_19_pct`       | `units_10_19` / `total_units_in_structure`                                                               | **not used**                                                    | Weak national correlation (r=−0.137); luxury towers and low-income housing projects are indistinguishable in national aggregates |
| `units_20_49_pct`       | `units_20_49` / `total_units_in_structure`                                                               | **not used**                                                    | Weak national correlation (r=−0.104); same confound as above                                                                     |
| `units_50plus_pct`      | `units_50plus` / `total_units_in_structure`                                                              | **not used**                                                    | Near-zero national correlation (r=−0.028); same confound as above                                                                |
| `avg_bedrooms_per_unit` | `total_bedrooms` / `total_housing_units`                                                                 | **not used**                                                    | Weak negative correlation (r=−0.139); urban apartments have fewer bedrooms than rural homes regardless of wealth                 |

### Transport & Commute

| Derived variable     | Derived from                                                                                                                  | Used in index            | Notes                                                                                                                                                                      |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pct_no_vehicle`     | (`vehicles_total_households` − households with ≥1 vehicle) / `vehicles_total_households`                                      | `wealth_index_transport` |                                                                                                                                                                            |
| `pct_2plus_vehicles` | (`owner_2_vehicles` + `owner_3_plus_vehicles` + `renter_2_vehicles` + `renter_3_plus_vehicles`) / `vehicles_total_households` | `wealth_index_transport` |                                                                                                                                                                            |
| `car_alone_pct`      | `commute_car_alone` / `commute_universe`                                                                                      | **not used**             | Near-zero national correlation (r=+0.002); driving alone is near-universal across all wealth levels outside dense cities                                                   |
| `transit_pct`        | `commute_transit` / `commute_universe`                                                                                        | **not used**             | Urban/rural confound — wealthy urban tracts have high transit use; national correlation does not reflect a wealth signal (r=+0.044)                                        |
| `walk_pct`           | `commute_walk` / `commute_universe`                                                                                           | **not used**             | Weak correlation (r=−0.111); walking to work is associated with dense urban areas regardless of wealth                                                                     |
| `work_from_home_pct` | `commute_work_home` / `commute_universe`                                                                                      | **not used**             | Moderate correlation (r=+0.412) but reflects professional occupation type, not a physically observable housing characteristic; not suitable as a satellite training target |

### Utilities & Digital

| Derived variable | Derived from                                                          | Used in index            | Notes                                                                                                                               |
| ---------------- | --------------------------------------------------------------------- | ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| `broadband_pct`  | `broadband_internet` / `internet_total_households`                    | `wealth_index_utilities` | NaN in 2010–2014 (not released at tract level until 2015–2019 ACS)                                                                  |
| `computer_pct`   | `has_computer` / `computer_total_households`                          | **not used**             | Strong correlation (r=+0.609) but not physically observable from satellite imagery                                                  |
| `gas_heat_pct`   | `utility_gas_heat` / `heating_total_households`                       | `wealth_index_utilities` |                                                                                                                                     |
| `poor_heat_pct`  | (`fuel_oil_kerosene_heat` + `wood_heat`) / `heating_total_households` | `wealth_index_utilities` |                                                                                                                                     |
| `solar_heat_pct` | `solar_heat` / `heating_total_households`                             | **not used**             | Nationally sparse (mean ≈ 0.15% of households); concentrated in California, Arizona, and Hawaii — not a reliable national predictor |
| `propane_pct`    | `propane_heat` / `heating_total_households`                           | **not used**             | Near-zero correlation (r=−0.054); primarily a rural geography indicator rather than a wealth signal                                 |
| `no_fuel_pct`    | `no_fuel_heat` / `heating_total_households`                           | **not used**             | Near-zero correlation (r=−0.016); no meaningful wealth signal nationally                                                            |

### Financial & Socioeconomic

| Derived variable            | Derived from                                                 | Used in index            | Notes |
| --------------------------- | ------------------------------------------------------------ | ------------------------ | ----- |
| `poverty_rate`              | `population_below_poverty_line` / `poverty_total_population` | `wealth_index_financial` |       |
| `unemployment_rate`         | `unemployed` / `labor_force_total`                           | `wealth_index_financial` |       |
| `rent_burdened_30plus_pct`  | sum of rent burden bands / `rent_burden_universe`            | `wealth_index_financial` |       |
| `owner_burdened_30plus_pct` | sum of owner mortgage burden bands / `owner_mortgaged_units` | `wealth_index_financial` |       |

---

## Data Source

American Community Survey (ACS) 5-year estimates, U.S. Census Bureau.
Downloaded via Census API: `api.census.gov/data/{year}/acs/acs5`

ACS tables used: B25077, B25035, B25064, B25018, B25003, B25024, B25041, B25014, B25044, B28002, B28001, B25040, B08301, B19013, B19301, B17001, B23025, B25070, B25091, B25095
