#!/usr/bin/env bash
# Example: generate monthly BWMA reports (adjust dates/threshold as needed).
# Workflow: see README → "Monthly BWMA acreage workflow (for staff)".
#
# From repo root:
#   bash scripts/run_apr_may_2026_bwma.sh

set -euo pipefail
cd "$(dirname "$0")/.."

python3 flood_report.py 2026-04-01 0.14
python3 flood_report.py 2026-05-01 0.14

echo ""
echo "Next: quarto render"
echo "      Update the iframe src in index.qmd if needed (flood_reports/reports/bwma_flood_report_*_0.14.html)"
echo "      Commit docs/ for GitHub Pages (reports copied to docs/flood_reports/)."
