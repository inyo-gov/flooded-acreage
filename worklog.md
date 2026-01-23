# Work Log

This document tracks development work, fixes, and improvements to the flooded acreage monitoring system.

---

## 2025-01-02: January 2026 Report Generation

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

### Current Workflow
1. Generate report: `python flood_report.py 'YYYY-MM-DD' <threshold>`
2. Copy supporting files to `docs/` directory
3. Update iframe in `index.qmd` to new report
4. Render website: `quarto render index.qmd`
5. Manual review of `docs/index.html`
6. Commit and push to GitHub

### File Organization
- **Source files**: `flood_reports/reports/` and `flood_reports/csv_output/`
- **Deployed files**: `docs/reports/` and `docs/csv_output/` (copied by Quarto)
- **Single source of truth**: All reports generated in `flood_reports/`, then copied to `docs/` for deployment

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

- All reports use Sentinel-2 Surface Reflectance imagery
- Threshold values are adjusted based on visual inspection and field validation
- Reports are generated using a 15-day window from the requested date
- The system automatically selects the first available cloud-free image in the date range
