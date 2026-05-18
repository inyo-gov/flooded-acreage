#!/usr/bin/env bash
# Generate BWMA flood reports for April and May 2026 using the same NIR threshold
# as the March 2026 report (0.14). Filenames use the *actual* cloud-free image date
# from Earth Engine (see "Image date:" in script output), not necessarily the start date.
#
# Requires: Earth Engine project auth (ee.Initialize) and a working geemap install.
# From repo root:
#   bash scripts/run_apr_may_2026_bwma.sh

set -euo pipefail
cd "$(dirname "$0")/.."

python3 flood_report.py 2026-04-01 0.14
python3 flood_report.py 2026-05-01 0.14

echo ""
echo "Next: copy flood_reports/reports/* and flood_reports/csv_output/* into the site if needed,"
echo "      run 'quarto render', and set the iframe src in index.qmd to the latest"
echo "      bwma_flood_report_<ImageDate>_0.14.html under docs/reports/."
