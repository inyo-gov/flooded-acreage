import argparse
import ee
import geopandas as gpd
import geemap
import pandas as pd
import folium
from folium.raster_layers import ImageOverlay
import geemap.foliumap as geemap
from datetime import datetime, timedelta
import os
import imageio.v2 as imageio
from sar_utils import s1_composites, s1_flood_indicators, s1_flood_mask

# Example terminal command
# > python flood_report_fusion.py '2024-10-22' 0.22 '2024-08-01' '2024-08-31' --fusion-mode union
# > python flood_report_fusion.py '2024-11-01' 0.22 '2024-08-01' '2024-08-31' --fusion-mode confidence --s1-thresh -1.5

def process_s2(start_date, threshold, bounding_box_geometry):
    """
    Process Sentinel-2 data to create flood mask (reused from flood_report.py logic).
    
    Returns:
    --------
    tuple: (binary_image, cloud_free_composite, image_date_str, pixel_size)
    """
    # Filter Sentinel-2 surface reflectance imagery
    sentinel_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(bounding_box_geometry) \
        .filterDate(start_date, (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=15)).strftime('%Y-%m-%d')) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50)) \
        .select(['B4','B8', 'SCL','B11'])

    size = sentinel_collection.size().getInfo()
    print(f"Number of S2 images in collection: {size}")

    pixel_size = sentinel_collection.first().select('B8').projection().nominalScale().getInfo()

    # Extract the date information
    image_info = sentinel_collection.first().getInfo()
    image_date = image_info['properties']['system:time_start']
    image_date_str = pd.to_datetime(image_date, unit='ms').strftime('%Y-%m-%d')

    # Mask clouds and compute composite
    cloud_free_composite = sentinel_collection.map(lambda img: img.updateMask(img.select('SCL').neq(9))).median()
    binary_image = cloud_free_composite.select('B8').divide(10000).lt(threshold).selfMask()
    
    return binary_image.rename('flood_s2'), cloud_free_composite, image_date_str, pixel_size


def process_s1(start_date, dry_start, dry_end, bounding_box_geometry, dvv_thresh=-1.5, vv_vh_ratio_max=3, orbit_pass=None):
    """
    Process Sentinel-1 data to create flood mask.
    
    Returns:
    --------
    tuple: (binary_image, wet_composite, pixel_size)
    """
    wet_end = (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=15)).strftime('%Y-%m-%d')
    
    # Create composites
    dry_composite, wet_composite = s1_composites(
        bounding_box_geometry,
        dry_start, dry_end,
        start_date, wet_end,
        orbit_pass=orbit_pass
    )
    
    # Compute flood indicators and mask
    s1_indicators = s1_flood_indicators(dry_composite, wet_composite)
    binary_image = s1_flood_mask(s1_indicators, dvv_thresh=dvv_thresh, vv_vh_ratio_max=vv_vh_ratio_max).selfMask()
    
    pixel_size = 10.0  # S1 IW mode is 10m
    
    return binary_image, wet_composite, pixel_size


def fuse_masks(flood_s2, flood_s1, fusion_mode='union'):
    """
    Fuse S2 and S1 flood masks using specified mode.
    
    Parameters:
    -----------
    flood_s2 : ee.Image
        Binary S2 flood mask
    flood_s1 : ee.Image
        Binary S1 flood mask
    fusion_mode : str
        'union', 'intersection', or 'confidence'
    
    Returns:
    --------
    ee.Image
        Fused flood mask or confidence class map
    """
    if fusion_mode == 'union':
        # Simple union: flood if either sensor flags it
        fused = flood_s2.Or(flood_s1).rename('flood_fused')
        return fused
    
    elif fusion_mode == 'intersection':
        # Conservative: only if both sensors flag it
        fused = flood_s2.And(flood_s1).rename('flood_fused')
        return fused
    
    elif fusion_mode == 'confidence':
        # Confidence classes:
        # 0 = dry (no sensors flag)
        # 1 = flood by S2 only (open water, optically obvious)
        # 2 = flood by S1 only (likely under vegetation or cloudy)
        # 3 = flood by both S1 and S2 (highest confidence)
        both = flood_s2.And(flood_s1).multiply(3)
        s1_only = flood_s1.And(flood_s2.Not()).multiply(2)
        s2_only = flood_s2.And(flood_s1.Not()).multiply(1)
        
        confidence_map = both.add(s1_only).add(s2_only).rename('confidence_class')
        return confidence_map
    
    else:
        raise ValueError(f"Unknown fusion mode: {fusion_mode}. Use 'union', 'intersection', or 'confidence'.")


def main(start_date, s2_threshold, dry_start, dry_end, dvv_thresh=-1.5, vv_vh_ratio_max=3, 
         fusion_mode='union', orbit_pass=None):
    """
    Generate fused flood report combining Sentinel-1 and Sentinel-2 data.
    
    Parameters:
    -----------
    start_date : str
        Start date for wet/flood period in YYYY-MM-DD format
    s2_threshold : float
        S2 NIR threshold for flood detection
    dry_start : str
        Start date for dry baseline period in YYYY-MM-DD format
    dry_end : str
        End date for dry baseline period in YYYY-MM-DD format
    dvv_thresh : float, default=-1.5
        S1 dVV threshold in dB
    vv_vh_ratio_max : float, default=3
        S1 VV-VH ratio threshold in dB
    fusion_mode : str, default='union'
        Fusion mode: 'union', 'intersection', or 'confidence'
    orbit_pass : str, optional
        S1 orbit pass direction: 'ASCENDING' or 'DESCENDING'
    """
    # Initialize Earth Engine
    print("Initializing Earth Engine...")
    ee.Initialize(project='ee-zjn-2022')

    print(f"Processing fused flood report for date: {start_date}")
    print(f"Fusion mode: {fusion_mode}")

    # Define the bounding box coordinates
    print("Defining bounding box and geometry...")
    bbox = [[-118.23240736400778, 36.84651455123723],
            [-118.17232588207419, 36.84651455123723],
            [-118.17232588207419, 36.924364295139625],
            [-118.23240736400778, 36.924364295139625]]

    bounding_box_geometry = ee.Geometry.Polygon(bbox)

    # Process S2
    print("Processing Sentinel-2 data...")
    flood_s2, s2_composite, image_date_str, s2_pixel_size = process_s2(start_date, s2_threshold, bounding_box_geometry)
    print(f"S2 image date: {image_date_str}")

    # Process S1
    print("Processing Sentinel-1 data...")
    flood_s1, s1_composite, s1_pixel_size = process_s1(start_date, dry_start, dry_end, bounding_box_geometry, 
                                                       dvv_thresh=dvv_thresh, vv_vh_ratio_max=vv_vh_ratio_max, 
                                                       orbit_pass=orbit_pass)

    # Fuse masks
    print(f"Fusing masks using {fusion_mode} mode...")
    fused_mask = fuse_masks(flood_s2, flood_s1, fusion_mode=fusion_mode)

    # Use S2 pixel size for calculations (both are 10m, but S2 is the reference)
    pixel_size = s2_pixel_size

    # Define output directories and filenames
    html_subdirectory = "flood_reports/reports"
    os.makedirs(html_subdirectory, exist_ok=True)

    s2_thresh_str = str(s2_threshold).replace('.', '_')
    dvv_str = str(dvv_thresh).replace('-', 'neg').replace('.', '_')
    report_filename = os.path.join(html_subdirectory, f"bwma_flood_report_fusion_{image_date_str}_{fusion_mode}.html")
    map_filename = os.path.join(html_subdirectory, f"flooded_area_map_fusion_{image_date_str}_{fusion_mode}.html")

    # Export composites for visualization
    print("Exporting composite images...")
    false_color_vis = {'min': 0, 'max': 3000, 'bands': ['B11', 'B8', 'B4'], 'gamma': 1.4}
    s2_composite_filename_tif = f"flood_reports/reports/false_color_composite_{image_date_str}_{s2_thresh_str}.tif"
    s2_composite_image = s2_composite.select(['B11', 'B8', 'B4']).visualize(**false_color_vis)
    geemap.ee_export_image(s2_composite_image, filename=s2_composite_filename_tif, scale=10, region=bbox)
    
    s2_composite_filename_png = f"flood_reports/reports/false_color_composite_{image_date_str}_{s2_thresh_str}.png"
    s2_image = imageio.imread(s2_composite_filename_tif)
    imageio.imwrite(s2_composite_filename_png, s2_image)

    # Load units
    print("Loading units from geojson...")
    gdf = gpd.read_file("data/unitsBwma2800.geojson")
    units = geemap.geopandas_to_ee(gdf)
    units = units.filterBounds(bounding_box_geometry)
    units_clipped = units.map(lambda feature: feature.intersection(bounding_box_geometry))

    # Calculate areas for each sensor and fused result
    def compute_refined_flood_area(feature):
        geom = feature.geometry()
        
        # Reference for total pixels (use S2 composite)
        total_pixels = s2_composite.select('B8').reduceRegion(
            reducer=ee.Reducer.count(), geometry=geom, scale=10).get('B8')
        
        # Count flooded pixels for each sensor and fused
        s2_pixels = flood_s2.reduceRegion(
            reducer=ee.Reducer.count(), geometry=geom, scale=10).get('flood_s2')
        s1_pixels = flood_s1.reduceRegion(
            reducer=ee.Reducer.count(), geometry=geom, scale=10).get('flood_s1')
        
        if fusion_mode == 'confidence':
            # For confidence mode, count pixels in each class
            confidence_pixels = fused_mask.reduceRegion(
                reducer=ee.Reducer.frequencyHistogram(), geometry=geom, scale=10).get('confidence_class')
            # Sum all non-zero classes as flooded
            fused_pixels = ee.Dictionary(confidence_pixels).values().reduce(ee.Reducer.sum())
        else:
            fused_pixels = fused_mask.reduceRegion(
                reducer=ee.Reducer.count(), geometry=geom, scale=10).get(fused_mask.bandNames().get(0))
        
        pixel_area_m2 = ee.Number(pixel_size).multiply(pixel_size)
        pixel_area_acres = pixel_area_m2.multiply(0.000247105)
        
        s2_acres = ee.Number(s2_pixels).multiply(pixel_area_acres)
        s1_acres = ee.Number(s1_pixels).multiply(pixel_area_acres)
        fused_acres = ee.Number(fused_pixels).multiply(pixel_area_acres)
        
        unit_name = feature.get('Flood_Unit')
        centroid = feature.geometry().centroid().coordinates()
        
        result = {
            'total_pixels': total_pixels,
            's2_flooded_pixels': s2_pixels,
            's1_flooded_pixels': s1_pixels,
            'fused_flooded_pixels': fused_pixels,
            's2_acres_flooded': s2_acres,
            's1_acres_flooded': s1_acres,
            'fused_acres_flooded': fused_acres,
            'centroid': centroid
        }
        
        if fusion_mode == 'confidence':
            result['confidence_pixels'] = confidence_pixels
        
        return feature.set(result)
    
    print("Calculating areas for S2, S1, and fused results...")
    units_with_area = units_clipped.map(lambda f: f.set({'unit_acres': f.geometry().area().divide(4046.86)}))
    units_with_calculations = units_with_area.map(compute_refined_flood_area)
    
    # Export GeoJSON
    geemap.ee_export_vector(units_with_calculations, filename="flood_reports/reports/subunits_fusion.geojson")
    
    # Convert to DataFrame
    units_df = pd.DataFrame(units_with_calculations.getInfo()['features'])
    units_df = pd.json_normalize(units_df['properties'])
    
    # Select and round columns
    cols = ['Flood_Unit', 'unit_acres', 's2_acres_flooded', 's1_acres_flooded', 'fused_acres_flooded']
    units_df = units_df[cols].round(2)
    
    # Add totals
    totals = pd.DataFrame([{
        'Flood_Unit': 'Total',
        'unit_acres': units_df['unit_acres'].sum(),
        's2_acres_flooded': units_df['s2_acres_flooded'].sum(),
        's1_acres_flooded': units_df['s1_acres_flooded'].sum(),
        'fused_acres_flooded': units_df['fused_acres_flooded'].sum()
    }])
    units_df = pd.concat([units_df.dropna(), totals], ignore_index=True).round(2)
    
    print(units_df.to_string(index=False))
    
    # Create HTML map
    print("Creating HTML map...")
    Map = folium.Map(location=[36.8795, -118.202], zoom_start=12)
    
    # Add S2 composite
    Map.add_child(ImageOverlay(
        name="S2 False Color",
        image=s2_composite_filename_png,
        bounds=[[36.84651455123723, -118.23240736400778], [36.924364295139625, -118.17232588207419]],
        opacity=1
    ))
    
    # Add flood layers
    if fusion_mode == 'confidence':
        # For confidence mode, we'd need to export and visualize the confidence map
        # For now, just show the fused binary
        fused_binary = fused_mask.gt(0).selfMask()
    else:
        fused_binary = fused_mask.selfMask()
    
    # Export and add flooded polygons
    flooded_vectors = fused_binary.reduceToVectors(
        geometryType='polygon',
        reducer=ee.Reducer.countEvery(),
        scale=10,
        geometry=bounding_box_geometry,
        maxPixels=1e8
    )
    flooded_geojson = f"flood_reports/reports/clipped_flooded_areas_fusion_{image_date_str}_{fusion_mode}.geojson"
    geemap.ee_export_vector(flooded_vectors, filename=flooded_geojson)
    
    Map.add_child(folium.GeoJson(
        flooded_geojson,
        name="Fused Flood Areas",
        style_function=lambda x: {
            "color": "blue",
            "weight": 1,
            "fillColor": "blue",
            "fillOpacity": 0.5
        }
    ))
    
    # Add unit boundaries
    Map.add_child(folium.GeoJson(
        "flood_reports/reports/subunits_fusion.geojson",
        name="Unit Boundaries",
        style_function=lambda x: {
            "color": "red",
            "weight": 2,
            "fillColor": "#00000000",
            "fillOpacity": 0
        }
    ))
    
    # Add labels
    for feature in units_with_calculations.getInfo()['features']:
        unit_name = feature['properties']['Flood_Unit']
        centroid = feature['properties']['centroid']
        if isinstance(centroid, list) and len(centroid) == 2:
            folium.map.Marker(
                location=[centroid[1], centroid[0]],
                icon=folium.DivIcon(html=f"""<div style="font-size: 12px; color: black;">{unit_name}</div>""")
            ).add_to(Map)
    
    Map.add_child(folium.LayerControl())
    Map.save(map_filename)
    print(f"Map saved to {map_filename}")
    
    # Save CSV
    csv_subdirectory = "flood_reports/csv_output"
    os.makedirs(csv_subdirectory, exist_ok=True)
    csv_filename = os.path.join(csv_subdirectory, f'flood_report_data_fusion_{image_date_str}_{fusion_mode}.csv')
    units_df.to_csv(csv_filename, index=False)
    print(f"CSV file saved to {csv_filename}")
    
    # Create HTML report
    units_df_display = units_df[['Flood_Unit', 's2_acres_flooded', 's1_acres_flooded', 'fused_acres_flooded']].copy()
    units_df_display.columns = ['BWMA Unit', 'S2 Acres', 'S1 Acres', 'Fused Acres']
    units_df_display = units_df_display.round(0).astype(int)
    
    html_table = (
        units_df_display.style
        .set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center'), ('font-size', '14px')]},
            {'selector': 'td', 'props': [('font-size', '12px')]},
        ])
        .set_properties(subset=pd.IndexSlice[units_df_display.index[-1], :], **{'font-weight': 'bold'})
        .hide(axis='index')
        .to_html()
    )
    html_table = html_table.replace('<th></th>', '')
    
    html_report = f"""
    <html>
    <head>
        <title>BWMA Flooded Extent (Fused S1+S2): {image_date_str}</title>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .container {{ display: flex; justify-content: space-between; }}
            .left {{ width: 30%; }}
            .right {{ width: 70%; text-align: center; }}
            h1, h2 {{ text-align: center; }}
            .notes {{ margin-top: 20px; padding: 10px; border-top: 1px solid #000; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ text-align: center; }}
            td:nth-child(n+2) {{ text-align: right; }}
            tr:last-child {{ font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>BWMA Flooded Acres and Extent (Fused S1+S2): {image_date_str}</h1>
        <div class="container">
            <div class="left">
                <h2>Flooded Acres Comparison</h2>
                {html_table}
                <p><strong>Fusion Mode:</strong> {fusion_mode}</p>
            </div>
            <div class="right">
                <h2>Spatial Extent</h2>
                <p>Imagery Date: {image_date_str}</p>
                <p>Dry Baseline: {dry_start} to {dry_end}</p>
                <iframe src="./flooded_area_map_fusion_{image_date_str}_{fusion_mode}.html" width="90%" height="500"></iframe>
            </div>
        </div>
        <div class="notes">
            <h3>Technical Notes</h3>
            <p>This report combines Sentinel-2 optical and Sentinel-1 SAR flood detection results.</p>
            <p><strong>Sentinel-2:</strong> NIR threshold = {s2_threshold} (excellent for open water)</p>
            <p><strong>Sentinel-1:</strong> dVV threshold = {dvv_thresh} dB, VV-VH ratio = {vv_vh_ratio_max} dB (sensitive to water under vegetation)</p>
            <p><strong>Fusion Mode ({fusion_mode}):</strong> {'Union combines both sensors' if fusion_mode == 'union' else 'Intersection requires both sensors' if fusion_mode == 'intersection' else 'Confidence classes: 1=S2-only, 2=S1-only, 3=both'}</p>
            <p>CSV data: <a href="../csv_output/flood_report_data_fusion_{image_date_str}_{fusion_mode}.csv" download>Download</a></p>
        </div>
    </body>
    </html>
    """
    
    with open(report_filename, "w") as file:
        file.write(html_report)
    print(f"Report saved to {report_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fused flood report combining Sentinel-1 and Sentinel-2 data.")
    parser.add_argument('start_date', type=str, help='Start date for wet/flood period in YYYY-MM-DD format')
    parser.add_argument('s2_threshold', type=float, help='S2 NIR threshold for flood detection')
    parser.add_argument('dry_start', type=str, help='Start date for dry baseline period in YYYY-MM-DD format')
    parser.add_argument('dry_end', type=str, help='End date for dry baseline period in YYYY-MM-DD format')
    parser.add_argument('--s1-thresh', type=float, default=-1.5, dest='dvv_thresh', help='S1 dVV threshold in dB (default: -1.5)')
    parser.add_argument('--vv-vh-ratio', type=float, default=3, dest='vv_vh_ratio_max', help='S1 VV-VH ratio threshold in dB (default: 3)')
    parser.add_argument('--fusion-mode', type=str, default='union', choices=['union', 'intersection', 'confidence'],
                       help='Fusion mode: union, intersection, or confidence (default: union)')
    parser.add_argument('--orbit-pass', type=str, choices=['ASCENDING', 'DESCENDING'],
                       help='S1 orbit pass direction (optional)')
    args = parser.parse_args()
    main(args.start_date, args.s2_threshold, args.dry_start, args.dry_end,
         dvv_thresh=args.dvv_thresh, vv_vh_ratio_max=args.vv_vh_ratio_max,
         fusion_mode=args.fusion_mode, orbit_pass=args.orbit_pass)
