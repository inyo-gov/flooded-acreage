# Work Log

This document tracks development work, fixes, and improvements to the flooded acreage monitoring system.

**When to add an entry:** After generating monthly reports, changing the dashboard (`index.qmd`, Quarto render), or any operational/documentation update (README, `_quarto.yml`, `.gitignore`, scripts, staff workflow). Keep entries brief; link filenames and thresholds so future staff can reproduce the month.

---

## 2026-06-12: June 2026 report and staff documentation

### Report generated
- **Start date**: 2026-06-01 (15-day window through ~16th)
- **Threshold**: 0.14
- **Scenes in composite**: 7 (cloud-masked median; not a single scene)
- **Imagery label in filename**: 2026-06-01
- **Total flooded acreage**: ~12.8 acres (composite; lower than an earlier run with only 1 scene in the window, ~44 ac)

### Actions taken
1. Ran `python flood_report.py 2026-06-01 0.14`
2. `quarto render`; updated **Most Recent Report** iframe to `flood_reports/reports/bwma_flood_report_2026-06-01_0.14.html`
3. **README**: monthly staff workflow (run mid-month, use `YYYY-MM-01`, composite behavior); SAR/fusion marked **upcoming**, not in dashboard; optical is production
4. **worklog.md**: this entry; note to update log on doc/report changes
5. **`scripts/run_apr_may_2026_bwma.sh`**: post-run messages aligned with Quarto + `docs/flood_reports/`

### Notes for staff
- Acreage uses a **median composite** over all qualifying Sentinel-2 scenes in the window—not one acquisition. Run around **mid-month** so several passes are usually available.
- SAR scripts exist but are **not** incorporated into monthly reporting yet; planned for emergent veg masking optical NIR.

---

## 2026-05-18: April–May 2026 reports and published site layout

### Reports generated
| Month (start date) | Scene date | Threshold | Total acres (approx.) |
|--------------------|------------|-----------|------------------------|
| 2026-04-01 | 2026-04-02 | 0.14 | 334 |
| 2026-05-01 | 2026-05-07 | 0.14 | 167 |

### Infrastructure / handoff
1. **`flood_report.py`**: removed broken `geemap.foliumap` import (use top-level `geemap` only)
2. **`_quarto.yml`**: render `index.qmd` + `about.qmd` only; removed Avian from navbar
3. **`.gitignore`**: `/flood_reports/` at repo root (local); `avian.qmd`, `avian_data/`, `code/`, `avian_todo_notes.md` staged locally; **`scripts/`** tracked
4. **Dashboard links**: `index.qmd` uses `flood_reports/reports/`; Quarto copies resources to **`docs/flood_reports/`** (replacing old `docs/reports/` layout)
5. Committed **Apr and May 2026 Acreage**; avian products not on `main`

---

## 2026-01-02: January 2026 Report Generation

### Report Generated
- **Date**: January 2, 2026 (first available image in date range)
- **Threshold**: 0.14 (reduced from initial 0.16 due to false positives)
- **Total Flooded Acreage**: 679.15 acres
- **Key Results**:
  - West Winterton: 242.11 acres (23.98% flooded)
  - Waggoner: 208.88 acres (13.37% flooded)
  - Thibaut: 103.98 acres (4.21% flooded)

### Actions Taken
1. Generated initial report at 0.16 threshold (877.40 acres total)
2. Reduced threshold to 0.14 to minimize false positives
3. Removed 0.16 threshold files before commit
4. Updated iframe in `index.qmd` to new report
5. Rendered Quarto website
6. Committed and pushed to GitHub

---

## 2025-12-01: December 2025 Report Generation

### Report Generated
- **Date**: December 1, 2025
- **Threshold**: 0.16 (after testing 0.21 and 0.15)
- **Process**: Tested multiple thresholds (0.21 → 0.15 → 0.16) to find optimal balance

### Actions Taken
1. Generated reports at multiple thresholds for comparison
2. Selected 0.16 as optimal threshold
3. Removed test reports (0.15 and 0.21)
4. Updated website with final report

---

## 2025-11-10: November 2025 Report Generation

### Report Generated
- **Date**: November 10, 2025
- **Threshold**: 0.21 (reduced from 0.22 due to false positives)

### Context
- Previous November reports used 0.22 threshold
- Reduced to 0.21 to minimize false positive detections

---

## 2025-11-01: November 2025 Report Generation

### Report Generated
- **Date**: November 1, 2025
- **Threshold**: 0.22 (initial), then adjusted to 0.21, then 0.16
- **Process**: Iterative threshold refinement based on visual inspection

---

## 2025-10-15: October 2025 Report Generation

### Report Generated
- **Date**: October 15, 2025
- **Threshold**: 0.2

---

## 2025-09-29: Data Recovery and System Restoration

### Problem Identified
- `copy_reports.sh` script was overwriting and deleting files in `docs/` directory
- Website would not render
- Missing geotiffs and supporting files

### Actions Taken
1. **Data Recovery**: Used `git restore .` to recover deleted files from GitHub
2. **Script Removal**: Deleted `copy_reports.sh` to prevent future overwrites
3. **Cloud Threshold Update**: Changed `CLOUDY_PIXEL_PERCENTAGE` filter from 30% to 50% in `flood_report.py`
4. **Report Generation**: Generated September 29, 2025 report at 0.15 threshold
5. **Path Consolidation**: Moved report storage to `flood_reports/` directory structure

### Key Changes
- Updated `flood_report.py` to output to `flood_reports/reports/` and `flood_reports/csv_output/`
- Added `resources` configuration to `_quarto.yml` to copy files to `docs/` during render
- Fixed all relative paths in generated HTML reports

---

## Website Structure Improvements

### Layout Changes
1. **Plot Positioning**: Moved monthly flooded acreage plot to top of page
2. **Plot Title**: Changed from "Monthly Flooded Acres by Unit" to "Monthly Flooded Acreage by Unit"
3. **Table Header**: Changed table caption to "Monthly Reports"
4. **Main Header**: Kept "Blackrock Waterfowl Management Area" as primary header
5. **Section Header**: Added "Most Recent Report" header before iframe
6. **Iframe Size**: Increased height from 800px to 1000px for better visibility

### Data Filtering
- Limited plot X-axis to start from September 2024 to exclude older historical data
- Updated target line annotation dates in plot code

---

## Path and Deployment Fixes

### Problem: Quarto Cleaning Output Directory
- Newer Quarto versions (>=1.5) clean the `output-dir` (`docs/`) before rendering
- Files manually copied to `docs/` were being deleted

### Solution: Resources Configuration
- Added `resources` section to `_quarto.yml`:
  ```yaml
  resources:
    - flood_reports/csv_output/**
    - flood_reports/reports/**
  ```
- This ensures Quarto copies files from source to output directory on each render
- Files are now stored in `flood_reports/` and automatically copied to `docs/` during render

### Problem: GitHub Pages 404 Errors
- Links were pointing to incorrect paths for GitHub Pages deployment
- Iframe and table links were returning 404 errors

### Solution: Path Standardization
- Updated `index.qmd` to scan `docs/reports/` for files but generate links using `reports/` prefix
- Fixed all relative paths in generated HTML reports for proper GitHub Pages compatibility
- Ensured download links (CSV, GeoJSON, GeoTIFF) use correct relative paths

---

## Download Links and Map Rendering Fixes

### Problem: Download Links Not Working
- Links within HTML reports were using incorrect relative paths
- CSV, GeoJSON, and GeoTIFF downloads failed from both iframe and direct report access

### Solution: Relative Path Corrections
- Updated `flood_report.py` to generate correct relative paths:
  - GeoJSON: `clipped_flooded_areas_*.geojson` (same directory)
  - CSV: `../csv_output/flood_report_data_*.csv` (parent directory)
  - GeoTIFF: `false_color_composite_*.tif` (same directory)

### Problem: Map Not Displaying
- Interactive maps in reports were not showing layers
- GeoJSON and PNG overlays were not loading

### Solution: Map Overlay Path Fixes
- Updated `folium.GeoJson` paths to be relative to map HTML location
- Fixed `ImageOverlay` paths for false color composite PNG
- Ensured `subunits.geojson` path is correct relative to map file

### Legacy Report Fixes
- Created temporary script (`fix_report_paths.py`) to update hardcoded paths in older HTML reports
- Script updated paths in existing reports to match new structure
- Script was deleted after use

---

## Earth Engine Configuration

### Project ID Update
- Updated `ee.Initialize()` to use registered project: `ee-zjn-2022`
- Resolved 403 authentication errors

### Cloud Filter Adjustment
- Changed cloud pixel percentage filter from 30% to 50%
- Allows more images to be available for analysis, especially in cloudy seasons

---

## Workflow Streamlining

### Removed Obsolete Scripts
- Deleted `copy_reports.sh` (was causing data overwrites)
- Deleted `run_flood_report.sh` (replaced by direct Python execution)
- Deleted `copy_csvs.sh` (temporary utility, no longer needed)
- Deleted `update_website.sh` (replaced by Quarto render)

### Current Workflow (Sentinel-2 monthly — production)

1. **Mid-month:** `python flood_report.py 'YYYY-MM-01' 0.14` (15-day window, median composite)
2. **Render site:** `quarto render` (copies `flood_reports/` → `docs/flood_reports/`)
3. **Optional:** set **Most Recent Report** iframe in `index.qmd` to the new HTML
4. Review `docs/index.html` locally
5. Commit and push **`docs/`** (and source changes) for GitHub Pages

SAR / fusion scripts are experimental; see README **upcoming** section—not part of this workflow yet.

### File Organization
- **Source (local, gitignored):** `/flood_reports/reports/` and `/flood_reports/csv_output/`
- **Published (after render):** `docs/flood_reports/reports/` and `docs/flood_reports/csv_output/`
- **Dashboard:** `index.qmd` → `docs/index.html`

---

## Technical Improvements

### Code Quality
- Standardized all file paths to use relative paths
- Improved error handling for Earth Engine API calls
- Better organization of output directories

### Documentation
- Updated README with current workflow
- Added validation protocol documentation
- Created this work log for tracking changes

---

## Notes

- Production monthly reports use **Sentinel-2** surface reflectance only (`flood_report.py`).
- **Composite:** cloud-masked **median** of all scenes in the 15-day window (start date = 1st of month); not a single-scene threshold.
- **When to run:** mid-month so multiple passes are usually in the archive (~5–7 scenes).
- Threshold **0.14** for current monthly series; adjust using field validation.
- Report **filename date** comes from the first image in the collection; acreage still reflects the full-window composite.
- **SAR / fusion:** scripts in repo; **upcoming** for emergent vegetation where optical NIR is masked—not in dashboard workflow yet.
- Update **worklog.md** when reports or operational docs change (see top of this file).
