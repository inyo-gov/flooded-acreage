# ICM — Season peak = current flood season only

**Date:** 2026-09-30 (PT)  
**Repo:** `flooded-acreage` (`inyo-gov/flooded-acreage`)  
**Site:** https://inyo-gov.github.io/flooded-acreage/

---

## Problem

SEASON PEAK card showed **2334.5 ac in Mar 2025**.

Two bugs:

1. **Wrong season** — KPI took the max across *all* Nov–Mar months since 2024-09-01 (all-time), not the current flood season.
2. **Overestimate** — `index.qmd` loaded every threshold-experiment CSV for a date and summed unit rows, so Mar 2025 (five thresholds on 2025-03-26) inflated to ~2334 ac. True single-threshold (0.14) that day was ~370 ac excl. Drew.

Screenshot context: card paired with stacked chart where Mar ’25 looked like an all-time spike.

## Decision

| Item | Behavior |
|------|----------|
| Flood season window | **Nov 1 – Mar 31** |
| “Current” season (as of date *D*) | If month ≥ 11 → season starting that Nov; else → season that started the prior Nov. **Apr–Oct** shows the just-completed season. |
| As of 2026-09-30 | **2025–26** = 2025-11-01 → 2026-03-31 |
| Peak value | Max Drew-excluded total among **deduped** reports in that window only |
| Multi-threshold CSVs | One CSV per `report_date` (threshold closest to **0.14**) before unit sums |
| Drew | Still excluded from KPI totals (same as Total / Latest cards) |

## New season peak (production)

| | Value |
|--|-------|
| Season | **2025–26** |
| Peak | **788.5 ac** |
| Month | **Mar 2026** (report date 2026-03-01, NIR 0.14) |
| Drew | Excluded (~8.7 ac that day kept in unit row / chart only) |

Dashboard: https://inyo-gov.github.io/flooded-acreage/

## Code

- **`index.qmd`** — CSV dedupe by date; season window from `Sys.Date()`; season-peak `slice_max` inside window only; card subtitle shows season label (`2025–26 · Nov–Mar · 500 ac target`).

## Out of scope

- Did not delete historical multi-threshold CSVs (still useful for QA / dives).
- Stacked monthly chart still one bar per month (latest label date); not changed beyond benefiting from CSV dedupe.
- Upcoming **2026–27** season (Nov 2026–Mar 2027) will become current automatically once the calendar rolls into Nov 2026 (or stay on 2025–26 through Oct 2026).

## Follow-up

- Optional: pin canonical threshold per archived HTML report instead of “closest to 0.14” for pre-Apr-2026 dates (current rule already matches published peaks for 2025–26).
