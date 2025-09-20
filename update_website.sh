#!/bin/bash

echo "🌊 Updating Flooded Acreage Website..."

# Step 1: Run the Python script to generate new reports and CSVs
echo "📊 Running flood report script..."
echo "Please run: python flood_report.py 'YYYY-MM-DD' 0.XX"
read -p "Then press Enter to continue..."

# Step 2: Copy reports to all necessary directories
echo "🔄 Copying reports to all directories..."
./copy_reports.sh

# Step 3: Render the Quarto website
echo "🌐 Rendering Quarto website..."
quarto render

# Step 4: Update the main iframe to show the latest report
echo "🔄 Updating main iframe to latest report..."
LATEST_REPORT=$(ls -t docs/reports/bwma_flood_report_*.html | head -1 | xargs basename)
if [ ! -z "$LATEST_REPORT" ]; then
    echo "Latest report: $LATEST_REPORT"
    # Update the iframe in index.html to point to the latest report
    sed -i.bak "s|src=\"docs/reports/bwma_flood_report_.*\.html\"|src=\"docs/reports/$LATEST_REPORT\"|g" docs/index.html
    echo "✅ Updated iframe to show: $LATEST_REPORT"
else
    echo "⚠️ No reports found to update iframe"
fi

echo "✅ Website update complete!"
echo "🌐 Open docs/index.html to view the website"
