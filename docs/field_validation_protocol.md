# Field Validation Protocol for Flood Mapping

## Overview

This protocol guides biologists collecting GPS validation points during bird surveys to assess the accuracy of the automated flood mapping model. The validation data helps calibrate thresholds and improve model performance.

## Data Collection

### What to Collect

**GPS Points** with the following information:
- **Location**: GPS coordinates (automatically captured)
- **Date/Time**: When the observation was made
- **Flooded Status**: Yes or No (binary)
- **Unit Name**: Which BWMA unit you're in (optional but helpful)
- **Observer Name**: Your name or ID
- **Notes**: Optional - vegetation type, water depth, etc.

### Flooded Status Criteria

**YES (Flooded)** if:
- Standing water is visible
- Water covers the ground surface
- Area is clearly inundated

**NO (Dry)** if:
- No standing water
- Ground is dry or only damp
- No visible inundation

**Note**: Be consistent - if you're unsure, note it in the comments field.

### Collection Strategy

**Frequency**: 
- Biweekly or whenever bird surveys occur
- Opportunistic collection during regular field work

**Number of Points**:
- Target: 20-50 points per survey
- Adjust based on time available and survey route

**Sampling Approach**:

1. **Stratified Sampling**: 
   - Collect points in areas predicted as flooded AND dry
   - Don't just sample obvious water - also sample areas the model says are dry

2. **Boundary Sampling**:
   - Extra points along visible water edges
   - These are critical for assessing edge detection accuracy

3. **Representative Sampling**:
   - Points throughout each unit to capture heterogeneity
   - Include different vegetation types and terrain

4. **Coverage**:
   - Try to cover most of the flooded extent line during your survey
   - Focus on areas you're already visiting for bird surveys

## ArcGIS Online Workflow

### Setup (One-time)

1. Feature layer created in ArcGIS Online with fields:
   - Date
   - Latitude/Longitude (auto-captured)
   - Flooded (Yes/No dropdown)
   - Unit (optional dropdown)
   - Observer (text)
   - Notes (optional text)

2. Install ArcGIS Field Maps or Survey123 on your mobile device

3. Download the flood validation survey/form

### During Field Work

1. Open ArcGIS Field Maps or Survey123 app
2. Navigate to your survey location
3. For each validation point:
   - Tap to create new point (GPS location captured automatically)
   - Select "Yes" or "No" for flooded status
   - Optionally select unit name
   - Add your name as observer
   - Add any notes if needed
   - Submit/save the point
4. Points automatically sync to ArcGIS Online when you have connectivity

### After Field Work

1. Points sync automatically when device connects to internet
2. Verify points in ArcGIS Online web map
3. Export will be done by project coordinator weekly/biweekly

## Data Export and Analysis

### Export Process

1. Project coordinator exports from ArcGIS Online to GeoJSON or CSV
2. Exports saved to `data/validation/` directory
3. Naming convention: `validation_points_YYYY-MM-DD.geojson`

### Analysis

Validation analysis is run using:
```bash
python validate_flood_model.py data/validation/validation_points_2025-12-08.geojson 2025-12-08 --threshold 0.18
```

This generates:
- Accuracy metrics (overall accuracy, precision, recall)
- Confusion matrix
- Spatial visualization map
- Recommendations for threshold adjustment

## Tips for Biologists

1. **Be Consistent**: Use the same criteria for "Yes" vs "No" throughout the season
2. **Match Dates**: Try to collect validation points on dates close to satellite imagery dates (ideally same day)
3. **Cover Boundaries**: Focus extra attention on water edges - these are where the model often struggles
4. **Don't Overthink It**: Quick yes/no is fine - you don't need detailed measurements
5. **Use Existing Routes**: No need for special trips - collect during your regular bird survey routes
6. **Quality over Quantity**: 20 well-distributed points is better than 50 clustered points

## Questions?

Contact the project coordinator for:
- ArcGIS Online access issues
- App installation help
- Questions about flooded/dry criteria
- Data export requests

