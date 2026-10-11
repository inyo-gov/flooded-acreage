# Sentinel-based flooding model for BWMA — design note (SCREEN)

Date: 2026-10-11 · Status: **design only, no model run** · Owner: Zac Nelson

> Every output of this model is a **SCREEN**. A cell the model fills in is labeled `inferred` with a probability, kept apart from `observed` cells, and is never presented as a determination of flooded acreage. Reported acreage stays the satellite-observed number. Screen acreage is shown next to it.

## 1. Questions
1. If we know the state of a few **indicator points** (pond center, pond end, conveyance head), can we predict flooding across the rest of a unit?
2. Can we **fill in** flooded ground under emergent and woody vegetation that Sentinel misses, where hydraulics say it must be wet?
3. What is the **flooding history** of each parcel and unit, and how many **days inundated** per parcel per year?

## 2. Conveyance trigger rule (Zac's rule, as a deterministic layer)
For each pond or subbasin *P* fed by conveyance *C*:
- IF the indicator at P's **center** AND the indicator at P's **end** (downstream or terminal end) are observed wet in the same composite,
- THEN set C to `on`, add P's **prespecified acreage** to the screen acreage, and add P's **outline** to the flooded-acreage spatial output, tagged `trigger_rule` (not `observed`).
- If only one of the two is wet, the rule does not fire. The probabilistic layer (Section 5.4) handles that case.

Three primary chains (from Zac; **routing below is to be confirmed**):

| Chain | Conveyance | Feeds | Candidate polygons (from `subunits.shp`) |
|---|---|---|---|
| 1 | Blackrock Ditch | West Winterton | W1–W8, W10 (Flood_Unit "West Winterton", 508.7 ac in total) |
| 2 | Blackrock Ditch (eastern branch) | Waggoner | WAG1–WAG6 (470.3 ac) |
| 3 | LA Aqueduct → Thibaut | Thibaut, then South Winterton | TH1–TH11, SW2 (SW2 is attributed to Flood_Unit "Thibaut", 389.5 ac), SW1 (South Winterton, 94.7 ac) |

## 3. Inputs (verified paths)

| Input | Path | Notes |
|---|---|---|
| Sentinel-2 flooded composites | box `/workspace/repo-hygiene/clones/flooded-acreage/` (`flood_report.py`, `flood_report_s1.py`, `flood_report_fusion.py`, `sar_utils.py`, `dem_utils.py`, `validate_flood_model.py`; HEAD 4dfc223, 2026-09-30); live copy on mac `~/workspace/flooded-acreage` | GEE. NIR (B8) reflectance < **0.14** on a **15-day cloud-masked median**, with a single-scene mode. Outputs are in `docs/flood_reports/csv_output/` and `reports/` (clipped GeoJSON, false-color TIFF). **The 2026-09-26_0.14 composite totals 114.15 ac, not ~116**: Thibaut 38.38, Waggoner 45.47, West Winterton 21.03, Thibaut Ponds 9.27, Drew 1.56, East and South Winterton 0. Nearby runs: 09-22 composite 85.67 ac, 09-29 scene 104.3 ac |
| Sentinel-1 | `flood_report_s1.py` / `sar_utils.py` | dVV drop (−1.5 dB) against a dry baseline, or absolute VV < −18 / VH < −22 dB. Can see through cloud and partly under sparse canopy |
| BWMA ditch network v2c | box `/workspace/bwma-ditch-v2/outputs/gpkg/bwma_ditch_v2c_m5_vote_SCREEN.gpkg`, layer `features_ditch` | 4,996 lines, ~317 km, EPSG:6340. Fields include berm flags, width_est_m, type. This is itself a SCREEN. Method: `docs/2026-10-09_bwma-ditch-algorithm-bakeoff.md` |
| DTM | box `/workspace/bwma-ditch-v2/work/bwma_hybrid_dtm_1m_biascorr_v2.tif` | 1 m, EPSG:6340, 77 MB |
| REM | mac `~/workspace/lidar-data/data/processed/bwma_2024_topo/exports/bwma_hybrid_rem_v1_cog.tif` | 87 MB COG. Not copied |
| BWMA units | box `/workspace/bwma-ditch-v2/inputs/bwma_units.geojson` | 7 units (EPSG:4326): Thibaut 2,106 · Drew 548 · Waggoner 1,246 · West Winterton 806 · East Winterton 335 · South Winterton 513 · Thibaut Ponds 738 ac |
| BWMA subbasins (pond polygons) | mac `~/workspace/bwma-avian/data/subunits.shp` (EPSG:3857); AGOL `BWMA_Subunits_oct2021` | 33 subbasins with `Acres`, edited 2021-10 to 2023-05. **No Drew subbasins.** Best existing candidate for "prespecified acreage" |
| LADWP flows and reported flooded area | box `/workspace/ladwp-feeds/db/stations.csv`, `observations_daily.{parquet,csv}` | Daily, 2009 to 2026-10-07. `LORP:BWMA_{DREW,THIBAUT,WAGGONER,WINTERTON}` (flooded_area, flow_rate_set), `BWMA_TOTAL_FLOODED_AREA`, `THIBAUT_POND_FLOODED_AREA`, `BLACKROCK_DITCH_RETURN`, `NDR:THIBAUT_SPILLGATES` and `NDR:LORP_INFLOW_BLACKROCK` (2018–2026), `LORPLOG:BLACKROCK_RETURN_DITCH_GAGE` (gage height, 2018–2026). Parquet needs pyarrow on the box venv; the CSV works now |
| Field validation points | mac `~/workspace/flooded-acreage/data/validation/` (README only) | `validate_flood_model.py` expects `validation_points_YYYY-MM-DD.geojson`. **No point files are present** |
| Parcels | mac `~/workspace/parcels_canonical/data/v2/ladwp_vegetation_parcels_v2.geojson` (EPSG:32611) | Intersect in EPSG:6340 |

## 4. Validation sources
- **Planet 2023 (Jun–Aug high flows): not in hand.** The archive can be bought. Nothing has been purchased, and Zac decides.
- **LORP aerials:** Feb/Mar 2024 (~7 cm RGB, plus a 2 ft DEM, per `2024 LORP Mapping.docx`) and June 2025 (~5 cm, `LORP_2025_Tiling_Scheme`). The rasters have **not been located**. Hand-digitize water and wet-under-canopy polygons on sample tiles.
- **LADWP reported flooded acres** per unit, daily. This is an independent operational check, not ground truth (methods unknown).
- **Field GPS yes/no points** using the existing `validate_flood_model.py` workflow.

## 5. Approach

### 5.1 Unit and conveyance graph
- Nodes: sources (Blackrock Ditch intake or return, LA Aqueduct turnout, Thibaut spillgates), conveyance segments (from `features_ditch`, snapped and attributed by Zac), subbasins, and units.
- Edges are directed flow paths with attributes: `conveyance_id`, `feeds_subbasin`, gauge (LADWP station code), and lag.
- Build by snapping ditch lines to subbasin boundaries, then hand-confirm. **The model will not guess routing. Unconfirmed edges are excluded.**

### 5.2 Fill-and-spill on the DTM
- Run depression filling (priority-flood) on the 1 m DTM inside each subbasin to get a spill elevation, spill point and stage–area curve.
- Given an indicator that is observed wet at elevation *z*, the hydraulically consistent wet area is the connected region below *z* (bathtub with connectivity). The REM is used where gradients run along a channel.
- Watch for berm/ditch artifacts in the DTM. `features_ditch` berm flags mark barriers.

### 5.3 Indicator pixels
- For each subbasin, choose 2–4 indicator locations: center, terminal end, and inlet. Choose open-water spots with good historic Sentinel-2 visibility (low vegetation, a reliably detected NIR drop).
- Track each indicator's wet/dry state over time from S2 composites, with S1 as a fallback when cloudy.

### 5.4 Probabilistic propagation
- Per-pixel logistic model, fit on dates and areas where S2 sees clearly (open water):
  `logit P(wet) ~ indicator states + (z − indicator stage) + REM + flow lag (LADWP) + S1 dVV + veg class/NDVI + distance along graph`.
- Bayesian version: hierarchical by subbasin, with a prior from fill-and-spill (P near 1 below indicator stage and connected; prior decays above it). The posterior blends that prior with S1/S2 evidence.
- Cells under canopy get **P(wet)** and a flag `inferred_under_veg`. They are reported only as a screen band (e.g. P ≥ 0.8), never merged into observed acreage.
- Trigger-rule polygons (Section 2) are a separate deterministic layer that overrides only for display and tally within the `trigger_rule` tag.

### 5.5 Time series
- Run every composite date from 2017 (S2 L2A) to the present. LADWP daily flows interpolate state between composites, and only within the stated lag window. Gaps stay `NA`.

## 6. Outputs (all labeled SCREEN)
- `unit_subbasin_timeseries.csv`: date, unit, subbasin, observed_ac, trigger_rule_ac, inferred_ac (P≥0.8), LADWP_reported_ac, source scene/composite.
- `parcel_inundation_timeseries.csv`: date, PCL, observed_ac, screen_ac, frac.
- `parcel_days_inundated.csv`: PCL, water_year, days_observed_wet, days_screen_wet, days_no_data.
- GPKG layers: `observed_flood_<date>`, `trigger_rule_<date>`, `inferred_prob_<date>` (raster, COG).
- Each record carries method, version and threshold (0.14) provenance.

## 7. Needed from Zac
1. **Routing confirmation**: which ponds or subbasins each conveyance feeds for the three chains, including whether SW2 belongs to Thibaut or South Winterton, and the Blackrock Ditch branch point between West Winterton and Waggoner.
2. **Prespecified acreages** per pond: confirm `subunits.shp` Acres or supply official numbers. Drew needs polygons.
3. **Indicator locations**: center and end for each pond. A GPS or hand-picked point is fine.
4. Lag and turn-on conventions: which LADWP station signals "conveyance on" (e.g. `NDR:THIBAUT_SPILLGATES`, `LORP:BLACKROCK_DITCH_RETURN`).
5. Imagery: paths to the 2024 and 2025 LORP orthos, and a decision on Planet 2023.
6. Any field yes/no validation points (the `data/validation/` folder is empty).

## 8. Build order
1. Graph and trigger rule on the three chains, using existing S2 outputs plus `subunits.shp`. Compare with the LADWP reported acres.
2. Fill-and-spill plus indicators per subbasin. Back-cast 2017–2026 and produce per-parcel observed and trigger-rule days inundated.
3. Probabilistic under-vegetation layer. Validate against the 2024/2025 aerials and Planet 2023 if acquired, and report accuracy before any screen acreage is shown.
