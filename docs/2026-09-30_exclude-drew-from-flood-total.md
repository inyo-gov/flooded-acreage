# ICM — Exclude Drew from flooded acreage Total

**Date:** 2026-09-30 (PT)  
**Repo:** `flooded-acreage` (`inyo-gov/flooded-acreage`)  
**Site:** https://inyo-gov.github.io/flooded-acreage/

---

## Goal

Exclude **Drew** from the reported flooded-acreage **Total** everywhere totals are computed, while keeping the Drew unit row visible for QA. Desk note: Drew is not in the current wet-up rotation; detections are often Blackrock ditch / reddish false positives.

Screenshot context: latest usable scene (2026-09-29, NIR 0.16) showed **137 ac** with Drew **19** → **~118 ac** without Drew.

## Decision

| Item | Behavior |
|------|----------|
| Drew row | Kept in unit tables / stacked chart |
| **Total** (CSV, report HTML, KPI cards, `latest_scene.json`) | Sum of units **excluding Drew** |
| UI note | “Drew excluded from Total (not in current rotation; detections often Blackrock ditch / reddish false positives).” |

## Code / data changes

1. **`flood_report.py`** — `UNITS_EXCLUDED_FROM_TOTAL = {Drew}`; Total row uses `units_for_total()`; report HTML marks Drew `(not in Total)` + footnote; latest-scene sidecar carries `drew_note` / `acres_include_drew: false`.
2. **`flood_report_s1.py`**, **`flood_report_fusion.py`** — same Total exclusion for consistency.
3. **`index.qmd`** — KPI / season-peak totals recomputed from unit rows excluding Drew (works even if an old CSV Total still included Drew); units bar + chart caption note; scene card shows `drew_note`.
4. **`styles.css`** — `.scene-card-drew-note`.
5. **Existing products** — recomputed Total on flood-report CSVs under `flood_reports/csv_output/` and `docs/flood_reports/csv_output/`; rebuilt matching non-archive / non-S1 report HTML tables; updated `includes/latest_scene.json`.

## Latest scene (displayed)

| | With Drew | **Without Drew (new)** |
|--|-----------|-------------------------|
| Sep 29 · 0.16 | 137.49 ac | **118.09 ac** |
| Drew alone | 19.4 ac | (row kept) |

Report: `flood_reports/reports/bwma_flood_report_scene_2026-09-29_0.16.html`  
Pages: https://inyo-gov.github.io/flooded-acreage/flood_reports/reports/bwma_flood_report_scene_2026-09-29_0.16.html  
Dashboard: https://inyo-gov.github.io/flooded-acreage/

## Out of scope

- Did not re-run Earth Engine; unit-level flooded acres unchanged — only Total aggregation + UI.
- Archive HTML under `reports/archive/` and S1 report HTML not rebuilt (S1 CSVs Total were recomputed where present).
- Stacked monthly chart still shows Drew as a segment for QA; KPI cards exclude it.

## Follow-up

- If Drew returns to rotation, remove it from `UNITS_EXCLUDED_FROM_TOTAL` and re-render.
