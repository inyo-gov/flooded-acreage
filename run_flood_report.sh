#!/bin/bash

echo "🌊 Running Flood Report and Updating Website..."

# Check if date and threshold arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <date> <threshold>"
    echo "Example: $0 '2025-09-15' 0.15"
    exit 1
fi

DATE=$1
THRESHOLD=$2

echo "📊 Running flood report for date: $DATE, threshold: $THRESHOLD"

# Activate virtual environment and run the Python script
source venv/bin/activate
python flood_report.py "$DATE" "$THRESHOLD"

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "✅ Python script completed successfully"
    
    # Copy reports to all necessary directories
    echo "🔄 Copying reports to all directories..."
    ./copy_reports.sh
    
    # Render the Quarto website
    echo "🌐 Rendering Quarto website..."
    quarto render
    
    # Update the main iframe to show the latest report
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
else
    echo "❌ Python script failed. Please check the error messages above."
    exit 1
fi
