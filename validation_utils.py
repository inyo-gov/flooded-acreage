"""
Validation utilities for flood mapping model accuracy assessment.

This module provides functions for loading validation data, comparing field
observations to model predictions, and calculating accuracy metrics.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
try:
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
except ImportError:
    raise ImportError("scikit-learn is required for validation. Install with: pip install scikit-learn")
import ee
import geemap


def load_validation_points(file_path):
    """
    Load validation points from GeoJSON or CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to validation data file (GeoJSON or CSV)
    
    Returns:
    --------
    gpd.GeoDataFrame
        Validation points with geometry and attributes
    """
    if file_path.endswith('.geojson') or file_path.endswith('.json'):
        gdf = gpd.read_file(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        # Create GeoDataFrame from CSV with lat/lon columns
        if 'latitude' in df.columns and 'longitude' in df.columns:
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df.longitude, df.latitude),
                crs='EPSG:4326'
            )
        else:
            raise ValueError("CSV must contain 'latitude' and 'longitude' columns")
    else:
        raise ValueError("File must be GeoJSON (.geojson, .json) or CSV (.csv)")
    
    # Standardize flooded column (handle Yes/No, True/False, 1/0, etc.)
    if 'flooded' in gdf.columns:
        gdf['flooded'] = gdf['flooded'].astype(str).str.upper()
        gdf['flooded_binary'] = gdf['flooded'].isin(['YES', 'TRUE', '1', 'Y', 'FLOODED', '1.0']).astype(int)
    elif 'flooded_binary' in gdf.columns:
        # Already binary, ensure it's 0/1
        gdf['flooded_binary'] = gdf['flooded_binary'].astype(int)
        if 'flooded' not in gdf.columns:
            gdf['flooded'] = gdf['flooded_binary'].map({1: 'Yes', 0: 'No'})
    else:
        raise ValueError("Validation data must contain 'flooded' or 'flooded_binary' column")
    
    return gdf


def extract_model_predictions(validation_points, flood_mask, date, threshold=None):
    """
    Extract model predictions (flooded/dry) at validation point locations.
    
    Parameters:
    -----------
    validation_points : gpd.GeoDataFrame
        Validation points with geometry
    flood_mask : ee.Image or str
        Flood mask image (ee.Image) or path to flood mask GeoJSON
    date : str
        Date of the flood report (for reference)
    threshold : float, optional
        Threshold used (for reference in output)
    
    Returns:
    --------
    gpd.GeoDataFrame
        GeoDataFrame with validation points and model predictions
    """
    # If flood_mask is a string (path), use spatial join with GeoPandas
    if isinstance(flood_mask, str):
        # Load flood polygons from GeoJSON
        flood_polygons = gpd.read_file(flood_mask)
        
        # Ensure same CRS
        if validation_points.crs != flood_polygons.crs:
            flood_polygons = flood_polygons.to_crs(validation_points.crs)
        
        # Spatial join to check if points are within flood polygons
        validation_with_predictions = validation_points.copy()
        joined = gpd.sjoin(validation_points, flood_polygons, how='left', predicate='within')
        
        # If point intersects any flood polygon, it's predicted as flooded
        validation_with_predictions['predicted_flooded'] = (
            joined.index_right.notna().astype(int)
        )
        
    else:
        # flood_mask is an ee.Image - sample at point locations using Earth Engine
        validation_ee = geemap.geopandas_to_ee(validation_points)
        
        def sample_at_point(feature):
            point = feature.geometry()
            # Sample the flood mask at the point location
            sample = flood_mask.sample(
                region=point,
                scale=10,
                numPixels=1
            )
            # Get the first (and only) pixel value
            predicted = sample.first().get(flood_mask.bandNames().get(0))
            is_flooded = predicted.eq(1)  # Assuming binary mask (1 = flooded, 0 = dry)
            return feature.set({'predicted_flooded': is_flooded})
        
        validation_with_predictions_ee = validation_ee.map(sample_at_point)
        
        # Convert back to GeoDataFrame
        validation_info = validation_with_predictions_ee.getInfo()
        predictions = []
        
        for feature in validation_info['features']:
            props = feature['properties']
            predicted_value = props.get('predicted_flooded', False)
            predictions.append(1 if predicted_value else 0)
        
        validation_with_predictions = validation_points.copy()
        validation_with_predictions['predicted_flooded'] = predictions
    
    # Add metadata
    validation_with_predictions['validation_date'] = date
    if threshold is not None:
        validation_with_predictions['threshold'] = threshold
    
    return validation_with_predictions


def calculate_accuracy_metrics(validation_df):
    """
    Calculate accuracy metrics from validation data.
    
    Parameters:
    -----------
    validation_df : pd.DataFrame
        DataFrame with 'flooded_binary' (observed) and 'predicted_flooded' columns
    
    Returns:
    --------
    dict
        Dictionary with accuracy metrics and confusion matrix
    """
    y_true = validation_df['flooded_binary'].values
    y_pred = validation_df['predicted_flooded'].values
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = recall_score(1 - y_true, 1 - y_pred, zero_division=0)  # Recall of negative class
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    metrics = {
        'overall_accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'confusion_matrix': cm,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'total_points': len(validation_df)
    }
    
    return metrics


def spatial_accuracy_analysis(validation_df, units_gdf=None):
    """
    Calculate accuracy metrics by spatial unit and other spatial factors.
    
    Parameters:
    -----------
    validation_df : pd.DataFrame or gpd.GeoDataFrame
        Validation data with predictions
    units_gdf : gpd.GeoDataFrame, optional
        Unit boundaries for spatial aggregation
    
    Returns:
    --------
    dict
        Dictionary with spatial accuracy metrics
    """
    results = {}
    
    # Overall metrics
    overall_metrics = calculate_accuracy_metrics(validation_df)
    results['overall'] = overall_metrics
    
    # Accuracy by unit if units provided
    if units_gdf is not None and 'unit' in validation_df.columns:
        validation_gdf = gpd.GeoDataFrame(validation_df) if not isinstance(validation_df, gpd.GeoDataFrame) else validation_df
        
        # Spatial join to assign units
        if 'geometry' in validation_gdf.columns:
            validation_with_units = gpd.sjoin(validation_gdf, units_gdf, how='left', predicate='within')
        else:
            validation_with_units = validation_gdf.copy()
        
        # Calculate metrics by unit
        unit_metrics = {}
        for unit in validation_with_units['unit'].dropna().unique():
            unit_data = validation_with_units[validation_with_units['unit'] == unit]
            if len(unit_data) > 0:
                unit_metrics[unit] = calculate_accuracy_metrics(unit_data)
        
        results['by_unit'] = unit_metrics
    
    return results


def export_validation_report(validation_df, metrics, spatial_metrics, output_path, 
                            report_date, threshold=None, map_image_path=None):
    """
    Generate HTML validation report with metrics and visualizations.
    
    Parameters:
    -----------
    validation_df : pd.DataFrame
        Validation data with predictions
    metrics : dict
        Overall accuracy metrics
    spatial_metrics : dict
        Spatial accuracy analysis results
    output_path : str
        Path to save HTML report
    report_date : str
        Date of the flood report
    threshold : float, optional
        Threshold used in model
    map_image_path : str, optional
        Path to map image for visualization
    """
    # Create summary table
    summary_html = f"""
    <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
        <tr style="background-color: #f0f0f0;">
            <th style="padding: 10px; border: 1px solid #ddd;">Metric</th>
            <th style="padding: 10px; border: 1px solid #ddd;">Value</th>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd;">Overall Accuracy</td>
            <td style="padding: 10px; border: 1px solid #ddd;">{metrics['overall_accuracy']:.2%}</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd;">Precision</td>
            <td style="padding: 10px; border: 1px solid #ddd;">{metrics['precision']:.2%}</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd;">Recall (Sensitivity)</td>
            <td style="padding: 10px; border: 1px solid #ddd;">{metrics['recall']:.2%}</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd;">Specificity</td>
            <td style="padding: 10px; border: 1px solid #ddd;">{metrics['specificity']:.2%}</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd;">F1 Score</td>
            <td style="padding: 10px; border: 1px solid #ddd;">{metrics['f1_score']:.2%}</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd;">Total Validation Points</td>
            <td style="padding: 10px; border: 1px solid #ddd;">{metrics['total_points']}</td>
        </tr>
    </table>
    """
    
    # Confusion matrix table
    cm = metrics['confusion_matrix']
    cm_html = f"""
    <table style="width: 300px; border-collapse: collapse; margin: 20px 0;">
        <tr>
            <th style="padding: 10px; border: 1px solid #ddd;"></th>
            <th style="padding: 10px; border: 1px solid #ddd;">Predicted: Dry</th>
            <th style="padding: 10px; border: 1px solid #ddd;">Predicted: Flooded</th>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Observed: Dry</td>
            <td style="padding: 10px; border: 1px solid #ddd;">{metrics['true_negatives']}</td>
            <td style="padding: 10px; border: 1px solid #ddd; background-color: #ffcccc;">{metrics['false_positives']}</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Observed: Flooded</td>
            <td style="padding: 10px; border: 1px solid #ddd; background-color: #ffe6cc;">{metrics['false_negatives']}</td>
            <td style="padding: 10px; border: 1px solid #ddd;">{metrics['true_positives']}</td>
        </tr>
    </table>
    """
    
    # Unit-level metrics if available
    unit_metrics_html = ""
    if 'by_unit' in spatial_metrics and spatial_metrics['by_unit']:
        unit_rows = ""
        for unit, unit_met in spatial_metrics['by_unit'].items():
            unit_rows += f"""
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;">{unit}</td>
                <td style="padding: 10px; border: 1px solid #ddd;">{unit_met['overall_accuracy']:.2%}</td>
                <td style="padding: 10px; border: 1px solid #ddd;">{unit_met['precision']:.2%}</td>
                <td style="padding: 10px; border: 1px solid #ddd;">{unit_met['recall']:.2%}</td>
                <td style="padding: 10px; border: 1px solid #ddd;">{unit_met['total_points']}</td>
            </tr>
            """
        
        unit_metrics_html = f"""
        <h3>Accuracy by Unit</h3>
        <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
            <tr style="background-color: #f0f0f0;">
                <th style="padding: 10px; border: 1px solid #ddd;">Unit</th>
                <th style="padding: 10px; border: 1px solid #ddd;">Accuracy</th>
                <th style="padding: 10px; border: 1px solid #ddd;">Precision</th>
                <th style="padding: 10px; border: 1px solid #ddd;">Recall</th>
                <th style="padding: 10px; border: 1px solid #ddd;">Points</th>
            </tr>
            {unit_rows}
        </table>
        """
    
    # Recommendations
    recommendations = []
    if metrics['precision'] < 0.8:
        recommendations.append("Consider increasing threshold to reduce false positives")
    if metrics['recall'] < 0.8:
        recommendations.append("Consider decreasing threshold to reduce false negatives")
    if metrics['overall_accuracy'] > 0.9:
        recommendations.append("Model performance is excellent - current threshold appears optimal")
    
    recommendations_html = ""
    if recommendations:
        recommendations_html = f"""
        <h3>Recommendations</h3>
        <ul>
            {''.join([f'<li>{rec}</li>' for rec in recommendations])}
        </ul>
        """
    
    # Full HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Flood Mapping Validation Report: {report_date}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                line-height: 1.6;
            }}
            h1, h2, h3 {{
                color: #2E86AB;
            }}
            .summary {{
                background-color: #f9f9f9;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <h1>Flood Mapping Model Validation Report</h1>
        <p><strong>Report Date:</strong> {report_date}</p>
        {f'<p><strong>Threshold:</strong> {threshold}</p>' if threshold else ''}
        
        <div class="summary">
            <h2>Summary Metrics</h2>
            {summary_html}
        </div>
        
        <h2>Confusion Matrix</h2>
        {cm_html}
        <p style="color: #666; font-size: 0.9em;">
            <span style="background-color: #ffcccc; padding: 2px 5px;">Red</span> = False Positives (predicted flooded but actually dry)<br>
            <span style="background-color: #ffe6cc; padding: 2px 5px;">Orange</span> = False Negatives (predicted dry but actually flooded)
        </p>
        
        {unit_metrics_html}
        
        {recommendations_html}
        
        <h2>Validation Points Summary</h2>
        <p>Total validation points: {len(validation_df)}</p>
        <p>Correct predictions: {metrics['true_positives'] + metrics['true_negatives']}</p>
        <p>Incorrect predictions: {metrics['false_positives'] + metrics['false_negatives']}</p>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Validation report saved to {output_path}")

