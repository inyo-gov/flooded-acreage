#!/bin/bash

echo "🔄 Copying flood reports to all necessary directories..."

# Ensure all target directories exist
mkdir -p docs/reports
mkdir -p docs/docs/reports
mkdir -p docs/docs/docs/reports

# Copy HTML reports to all three locations
echo "📄 Copying HTML reports..."
cp docs/reports/*.html docs/docs/reports/ 2>/dev/null || echo "No HTML files to copy to docs/docs/reports"
cp docs/reports/*.html docs/docs/docs/reports/ 2>/dev/null || echo "No HTML files to copy to docs/docs/docs/reports"

# Copy CSV data to the nested directory
echo "📊 Copying CSV data..."
mkdir -p docs/docs/docs/reports/csv_output
cp flood_reports/csv_output/*.csv docs/docs/docs/reports/csv_output/ 2>/dev/null || echo "No CSV files to copy from flood_reports"
cp docs/docs/reports/csv_output/*.csv docs/docs/docs/reports/csv_output/ 2>/dev/null || echo "No CSV files to copy from docs/docs/reports"

# Also copy CSV files to the intermediate nested directory
mkdir -p docs/docs/reports/csv_output
cp flood_reports/csv_output/*.csv docs/docs/reports/csv_output/ 2>/dev/null || echo "No CSV files to copy to docs/docs/reports"

# Copy other report assets (maps, images, etc.)
echo "🗺️ Copying report assets..."
cp docs/reports/*.html docs/docs/reports/ 2>/dev/null || echo "No HTML files to copy"
cp docs/reports/*.geojson docs/docs/reports/ 2>/dev/null || echo "No GeoJSON files to copy"
cp docs/reports/*.tif docs/docs/reports/ 2>/dev/null || echo "No TIF files to copy"
cp docs/reports/*.png docs/docs/reports/ 2>/dev/null || echo "No PNG files to copy"

# Copy to the deepest nested directory
cp docs/reports/*.html docs/docs/docs/reports/ 2>/dev/null || echo "No HTML files to copy to deepest directory"
cp docs/reports/*.geojson docs/docs/docs/reports/ 2>/dev/null || echo "No GeoJSON files to copy to deepest directory"
cp docs/reports/*.tif docs/docs/docs/reports/ 2>/dev/null || echo "No TIF files to copy to deepest directory"
cp docs/reports/*.png docs/docs/docs/reports/ 2>/dev/null || echo "No PNG files to copy to deepest directory"

echo "✅ Report copying complete!"
echo "📁 Reports now available in:"
echo "   - docs/reports/"
echo "   - docs/docs/reports/"
echo "   - docs/docs/docs/reports/"
echo "   - docs/docs/docs/reports/csv_output/"
