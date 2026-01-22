"""
Validate flood mapping model using field-collected GPS points.

This script compares field observations (yes/no flooded) to model predictions
and generates accuracy reports.
"""

import argparse
import ee
import geopandas as gpd
import geemap
import pandas as pd
import os
from datetime import datetime
from validation_utils import (
    load_validation_points,
    extract_model_predictions,
    calculate_accuracy_metrics,
    spatial_accuracy_analysis,
    export_validation_report
)

# Example terminal command
# > python validate_flood_model.py data/validation/validation_points_2025-12-08.geojson 2025-12-08 --threshold 0.18


def load_flood_mask_from_report(report_date, threshold, bounding_box_geometry):
    """
    Load or regenerate flood mask for a given report date.
    
    Parameters:
    -----------
    report_date : str
        Date of the flood report (YYYY-MM-DD)
    threshold : float
        NIR threshold used
    bounding_box_geometry : ee.Geometry
        Area of interest
    
    Returns:
    --------
    tuple
        (flood_mask_geojson_path, flood_mask_ee_image)
    """
    # Check if flood report GeoJSON exists
    geojson_path = f"flood_reports/reports/clipped_flooded_areas_{report_date}_{threshold}.geojson"
    
    if os.path.exists(geojson_path):
        print(f"Loading existing flood mask from {geojson_path}")
        return geojson_path, None
    else:
        print(f"Flood mask not found at {geojson_path}")
        print("Regenerating flood mask...")
        
        # Regenerate flood mask (reuse logic from flood_report.py)
        from datetime import timedelta
        end_date = (datetime.strptime(report_date, '%Y-%m-%d') + timedelta(days=15)).strftime('%Y-%m-%d')
        
        sentinel_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(bounding_box_geometry) \
            .filterDate(report_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50)) \
            .select(['B4','B8', 'SCL','B11'])
        
        cloud_free_composite = sentinel_collection.map(
            lambda img: img.updateMask(img.select('SCL').neq(9))
        ).median()
        
        binary_image = cloud_free_composite.select('B8').divide(10000).lt(threshold).selfMask()
        
        # Export to GeoJSON for later use
        os.makedirs(os.path.dirname(geojson_path), exist_ok=True)
        flooded_vectors = binary_image.reduceToVectors(
            geometryType='polygon',
            reducer=ee.Reducer.countEvery(),
            scale=10,
            geometry=bounding_box_geometry,
            maxPixels=1e8
        )
        geemap.ee_export_vector(flooded_vectors, filename=geojson_path)
        print(f"Flood mask exported to {geojson_path}")
        
        return geojson_path, binary_image


def main(validation_data_path, report_date, threshold=None, output_dir=None):
    """
    Main validation workflow.
    
    Parameters:
    -----------
    validation_data_path : str
        Path to validation points (GeoJSON or CSV)
    report_date : str
        Date of the flood report to validate (YYYY-MM-DD)
    threshold : float, optional
        Threshold used in the model (will try to infer from filename if not provided)
    output_dir : str, optional
        Directory for output files (default: flood_reports/validation)
    """
    # Initialize Earth Engine
    print("Initializing Earth Engine...")
    ee.Initialize(project='ee-zjn-2022')
    
    # Set output directory
    if output_dir is None:
        output_dir = "flood_reports/validation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Infer threshold from filename if not provided
    if threshold is None:
        # Try to find existing report with this date
        import glob
        pattern = f"flood_reports/reports/clipped_flooded_areas_{report_date}_*.geojson"
        matches = glob.glob(pattern)
        if matches:
            # Extract threshold from filename
            import re
            match = re.search(r'_(\d+\.\d+)\.geojson$', matches[0])
            if match:
                threshold = float(match.group(1))
                print(f"Inferred threshold from existing report: {threshold}")
        else:
            # Default threshold
            threshold = 0.18
            print(f"No existing report found, using default threshold: {threshold}")
    
    # Define bounding box
    bbox = [[-118.23240736400778, 36.84651455123723],
            [-118.17232588207419, 36.84651455123723],
            [-118.17232588207419, 36.924364295139625],
            [-118.23240736400778, 36.924364295139625]]
    bounding_box_geometry = ee.Geometry.Polygon(bbox)
    
    # Load validation points
    print(f"Loading validation points from {validation_data_path}...")
    validation_points = load_validation_points(validation_data_path)
    print(f"Loaded {len(validation_points)} validation points")
    
    # Filter to points matching the report date (if date column exists)
    if 'date' in validation_points.columns:
        validation_points['date'] = pd.to_datetime(validation_points['date']).dt.date
        report_date_obj = datetime.strptime(report_date, '%Y-%m-%d').date()
        validation_points = validation_points[validation_points['date'] == report_date_obj]
        print(f"Filtered to {len(validation_points)} points matching report date {report_date}")
    
    if len(validation_points) == 0:
        print("Warning: No validation points found for this date!")
        return
    
    # Load flood mask
    print("Loading flood mask...")
    flood_mask_path, flood_mask_image = load_flood_mask_from_report(
        report_date, threshold, bounding_box_geometry
    )
    
    # Extract model predictions
    print("Extracting model predictions at validation point locations...")
    if flood_mask_image is None:
        # Use GeoJSON approach (spatial join)
        validation_with_predictions = extract_model_predictions(
            validation_points, flood_mask_path, report_date, threshold
        )
    else:
        # Use Earth Engine image approach
        validation_with_predictions = extract_model_predictions(
            validation_points, flood_mask_image, report_date, threshold
        )
    
    # Ensure we have the required columns
    if 'flooded_binary' not in validation_with_predictions.columns:
        # Try to recreate from flooded column
        if 'flooded' in validation_with_predictions.columns:
            validation_with_predictions['flooded'] = validation_with_predictions['flooded'].astype(str).str.upper()
            validation_with_predictions['flooded_binary'] = validation_with_predictions['flooded'].isin(
                ['YES', 'TRUE', '1', 'Y', 'FLOODED']
            ).astype(int)
        else:
            print("Error: Validation data must contain 'flooded' column")
            return
    
    if 'predicted_flooded' not in validation_with_predictions.columns:
        print("Error: Could not extract model predictions")
        return
    
    # Convert to DataFrame for metrics calculation (keep geometry if GeoDataFrame)
    if isinstance(validation_with_predictions, gpd.GeoDataFrame):
        validation_df = pd.DataFrame(validation_with_predictions.drop(columns=['geometry']))
        validation_gdf = validation_with_predictions  # Keep for map
    else:
        validation_df = validation_with_predictions
        validation_gdf = None
    
    # Calculate accuracy metrics
    print("Calculating accuracy metrics...")
    metrics = calculate_accuracy_metrics(validation_df)
    
    # Load units for spatial analysis
    units_gdf = None
    try:
        units_gdf = gpd.read_file("data/unitsBwma2800.geojson")
        print("Loaded unit boundaries for spatial analysis")
    except Exception as e:
        print(f"Could not load unit boundaries - skipping spatial analysis: {e}")
    
    # Spatial accuracy analysis
    print("Performing spatial accuracy analysis...")
    # Use GeoDataFrame if available, otherwise regular DataFrame
    spatial_metrics = spatial_accuracy_analysis(
        validation_gdf if validation_gdf is not None else validation_df, 
        units_gdf
    )
    
    # Print summary
    print("\n" + "="*50)
    print("VALIDATION RESULTS SUMMARY")
    print("="*50)
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall: {metrics['recall']:.2%}")
    print(f"Specificity: {metrics['specificity']:.2%}")
    print(f"F1 Score: {metrics['f1_score']:.2%}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives: {metrics['true_positives']}")
    print(f"  True Negatives: {metrics['true_negatives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print("="*50)
    
    # Generate HTML report
    report_filename = os.path.join(output_dir, f"validation_report_{report_date}_{threshold}.html")
    print(f"\nGenerating validation report...")
    export_validation_report(
        validation_df,
        metrics,
        spatial_metrics,
        report_filename,
        report_date,
        threshold
    )
    
    # Export CSV with point-by-point comparison
    csv_filename = os.path.join(output_dir, f"validation_points_{report_date}_{threshold}.csv")
    validation_export = validation_df.copy()
    
    # Add correctness column
    validation_export['correct'] = (
        validation_export['flooded_binary'] == validation_export['predicted_flooded']
    )
    validation_export['error_type'] = 'Correct'
    validation_export.loc[
        (validation_export['flooded_binary'] == 0) & (validation_export['predicted_flooded'] == 1),
        'error_type'
    ] = 'False Positive'
    validation_export.loc[
        (validation_export['flooded_binary'] == 1) & (validation_export['predicted_flooded'] == 0),
        'error_type'
    ] = 'False Negative'
    
    # Select relevant columns for export
    export_cols = ['date', 'flooded', 'flooded_binary', 'predicted_flooded', 'correct', 'error_type']
    if 'unit' in validation_export.columns:
        export_cols.insert(3, 'unit')
    if 'observer' in validation_export.columns:
        export_cols.append('observer')
    
    validation_export[export_cols].to_csv(csv_filename, index=False)
    print(f"Validation points CSV saved to {csv_filename}")
    
    # Create map visualization
    print("Creating validation map...")
    try:
        import folium
        
        # Use GeoDataFrame if available, otherwise try to get coordinates from DataFrame
        if validation_gdf is not None:
            map_data = validation_gdf
            map_center = [
                validation_gdf.geometry.y.mean(),
                validation_gdf.geometry.x.mean()
            ]
        elif 'latitude' in validation_df.columns and 'longitude' in validation_df.columns:
            map_data = validation_df
            map_center = [
                validation_df['latitude'].mean(),
                validation_df['longitude'].mean()
            ]
        else:
            print("Cannot create map - no geometry or lat/lon columns")
            map_data = None
        
        if map_data is not None:
            val_map = folium.Map(location=map_center, zoom_start=12)
            
            # Add validation points colored by correctness
            for idx, row in map_data.iterrows():
                if validation_gdf is not None:
                    lat, lon = row['geometry'].y, row['geometry'].x
                else:
                    lat, lon = row['latitude'], row['longitude']
                
                observed = row.get('flooded_binary', 0)
                predicted = row.get('predicted_flooded', 0)
                
                if observed == predicted:
                    color = 'green'
                    icon = 'check'
                    popup = f"Correct: Observed={'Flooded' if observed else 'Dry'}, Predicted={'Flooded' if predicted else 'Dry'}"
                elif predicted == 1 and observed == 0:
                    color = 'red'
                    icon = 'times'
                    popup = "False Positive: Predicted Flooded but Actually Dry"
                else:
                    color = 'orange'
                    icon = 'exclamation'
                    popup = "False Negative: Predicted Dry but Actually Flooded"
                
                folium.Marker(
                    location=[lat, lon],
                    popup=popup,
                    icon=folium.Icon(color=color, icon=icon, prefix='fa')
                ).add_to(val_map)
            
            # Add legend
            legend_html = """
            <div style="position: fixed; bottom: 50px; left: 50px; width: 200px; height: 120px; 
                        background-color: white; z-index:9999; border:2px solid grey; padding: 10px; font-size:14px">
            <h4>Validation Points</h4>
            <p><span style="color:green">●</span> Correct</p>
            <p><span style="color:red">●</span> False Positive</p>
            <p><span style="color:orange">●</span> False Negative</p>
            </div>
            """
            val_map.get_root().html.add_child(folium.Element(legend_html))
            
            map_filename = os.path.join(output_dir, f"validation_map_{report_date}_{threshold}.html")
            val_map.save(map_filename)
            print(f"Validation map saved to {map_filename}")
    except Exception as e:
        print(f"Could not create map visualization: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nValidation complete! Report: {report_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate flood mapping model using field-collected GPS points"
    )
    parser.add_argument('validation_data', type=str,
                       help='Path to validation points file (GeoJSON or CSV)')
    parser.add_argument('report_date', type=str,
                       help='Date of flood report to validate (YYYY-MM-DD)')
    parser.add_argument('--threshold', type=float,
                       help='NIR threshold used (will infer from existing report if not provided)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for validation reports (default: flood_reports/validation)')
    args = parser.parse_args()
    
    main(args.validation_data, args.report_date, 
         threshold=args.threshold, output_dir=args.output_dir)

