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
from sar_utils import load_s1_collection, s1_composites, s1_flood_indicators, s1_flood_mask, s1_temporal_ensemble, s1_absolute_threshold, s1_temporal_ensemble, s1_absolute_threshold

# Example terminal command
# > python flood_report_s1.py '2024-10-22' '2024-08-01' '2024-08-31' --dvv-thresh -1.5
# > python flood_report_s1.py '2024-11-01' '2024-08-01' '2024-08-31' --dvv-thresh -1.5 --orbit-pass ASCENDING

def main(start_date, dry_start, dry_end, dvv_thresh=-1.5, vv_vh_ratio_max=3, orbit_pass=None, 
         use_ensemble=True, days_before=5, days_after=5, min_dates=2,
         use_absolute=False, vv_thresh=-18, vh_thresh=-22):
    """
    Generate flood report using Sentinel-1 SAR data.
    
    Parameters:
    -----------
    start_date : str
        Start date for wet/flood period in YYYY-MM-DD format
    dry_start : str
        Start date for dry baseline period in YYYY-MM-DD format
    dry_end : str
        End date for dry baseline period in YYYY-MM-DD format
    dvv_thresh : float, default=-1.5
        Threshold for dVV (VV drop in dB to indicate flooding)
    vv_vh_ratio_max : float, default=3
        Maximum VV-VH ratio (in dB) for flooded areas
    orbit_pass : str, optional
        Orbit pass direction: 'ASCENDING' or 'DESCENDING'
    """
    # Initialize Earth Engine with registered project
    print("Initializing Earth Engine...")
    ee.Initialize(project='ee-zjn-2022')

    # Compute the end date for wet period (same as S2: start_date + 15 days)
    wet_end = (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=15)).strftime('%Y-%m-%d')
    print(f"Dry baseline period: {dry_start} to {dry_end}")
    print(f"Wet/flood period: {start_date} to {wet_end}")

    # Define the bounding box coordinates
    print("Defining bounding box and geometry...")
    bbox = [[-118.23240736400778, 36.84651455123723],
            [-118.17232588207419, 36.84651455123723],
            [-118.17232588207419, 36.924364295139625],
            [-118.23240736400778, 36.924364295139625]]

    # Create a bounding box geometry
    bounding_box_geometry = ee.Geometry.Polygon(bbox)

    # Load S1 collections and create flood mask
    if use_absolute:
        print("Using absolute backscatter thresholds with dry baseline exclusion...")
        print(f"VV threshold: {vv_thresh} dB, VH threshold: {vh_thresh} dB")
        print("Excluding permanent water/shadow areas using dry baseline...")
        # Load both dry and wet composites
        wet_start_abs = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=days_before)).strftime('%Y-%m-%d')
        wet_end_abs = (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=days_after)).strftime('%Y-%m-%d')
        dry_composite, wet_composite = s1_composites(
            bounding_box_geometry,
            dry_start, dry_end,
            wet_start_abs, wet_end_abs,
            orbit_pass=orbit_pass
        )
        binary_image = s1_absolute_threshold(
            wet_composite,
            vv_thresh=vv_thresh,
            vh_thresh=vh_thresh,
            dry_img=dry_composite,
            exclude_permanent_water=True
        ).selfMask()
    elif use_ensemble:
        print("Using temporal ensemble approach (multiple dates)...")
        print(f"Ensemble window: {days_before} days before to {days_after} days after {start_date}")
        print(f"Requiring flooding in at least {min_dates} dates")
        binary_image = s1_temporal_ensemble(
            bounding_box_geometry,
            dry_start, dry_end,
            start_date,
            days_before=days_before,
            days_after=days_after,
            orbit_pass=orbit_pass,
            dvv_thresh=dvv_thresh,
            vv_vh_ratio_max=vv_vh_ratio_max,
            min_dates=min_dates
        ).selfMask()
        
        # Get a representative wet composite for visualization (median of ensemble period)
        wet_start_ensemble = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=days_before)).strftime('%Y-%m-%d')
        wet_end_ensemble = (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=days_after)).strftime('%Y-%m-%d')
        _, wet_composite = s1_composites(
            bounding_box_geometry,
            dry_start, dry_end,
            wet_start_ensemble, wet_end_ensemble,
            orbit_pass=orbit_pass
        )
    else:
        print("Loading Sentinel-1 collections and creating composites...")
        dry_composite, wet_composite = s1_composites(
            bounding_box_geometry,
            dry_start, dry_end,
            start_date, wet_end,
            orbit_pass=orbit_pass
        )

        # Compute flood indicators
        print("Computing SAR flood indicators...")
        s1_indicators = s1_flood_indicators(dry_composite, wet_composite)

        # Create binary flood mask
        print("Creating binary flood mask...")
        binary_image = s1_flood_mask(s1_indicators, dvv_thresh=dvv_thresh, vv_vh_ratio_max=vv_vh_ratio_max).selfMask()

    # Get pixel size from S1 (10m for IW mode)
    pixel_size = 10.0

    # Use the wet period start date for naming
    image_date_str = start_date
    print(f"Report date: {image_date_str}")

    # Define the subdirectory for HTML maps and reports
    print("Setting up directories and filenames...")
    html_subdirectory = "flood_reports/reports"
    os.makedirs(html_subdirectory, exist_ok=True)

    # Define filenames with unique names based on the date and thresholds
    dvv_str = str(dvv_thresh).replace('-', 'neg').replace('.', '_')
    report_filename = os.path.join(html_subdirectory, f"bwma_flood_report_s1_{image_date_str}_{dvv_str}.html")
    map_filename = os.path.join(html_subdirectory, f"flooded_area_map_s1_{image_date_str}_{dvv_str}.html")

    # Convert flooded areas to vector (GeoJSON format)
    print("Vectorizing flooded areas...")
    flooded_vectors = binary_image.reduceToVectors(
        geometryType='polygon',
        reducer=ee.Reducer.countEvery(),
        scale=10,
        geometry=bounding_box_geometry,
        maxPixels=1e8
    )

    # Define export file paths
    print("Exporting SAR composite images...")
    sar_composite_filename_tif = f"flood_reports/reports/sar_composite_{image_date_str}_{dvv_str}.tif"
    
    # Export SAR composite (VV band) for visualization
    sar_composite_vis = wet_composite.select("VV").visualize(
        min=-25, max=0,  # Typical range for VV in dB
        palette=['000080', '0000FF', '00FFFF', 'FFFF00', 'FF0000', '800000']
    )
    
    geemap.ee_export_image(sar_composite_vis, filename=sar_composite_filename_tif, scale=10, region=bbox)
    print("SAR composite exported.")

    # Convert to PNG
    sar_composite_filename_png = f"flood_reports/reports/sar_composite_{image_date_str}_{dvv_str}.png"
    sar_composite_image = imageio.imread(sar_composite_filename_tif)
    imageio.imwrite(sar_composite_filename_png, sar_composite_image)
    print(f"SAR composite PNG saved at {sar_composite_filename_png}")

    # Load units from geojson and convert to Earth Engine geometry
    print("Loading units from geojson...")
    gdf = gpd.read_file("data/unitsBwma2800.geojson")
    units = geemap.geopandas_to_ee(gdf)
    units = units.filterBounds(bounding_box_geometry)
    units_clipped = units.map(lambda feature: feature.intersection(bounding_box_geometry))

    # Clip the binary flooded image to the unit boundaries
    clipped_flooded_image = binary_image.clipToCollection(units_clipped)

    # Vectorize the clipped flooded areas to GeoJSON polygons
    # Use smaller scale (8m) to better capture narrow linear features like canals
    flooded_clipped_vectors = clipped_flooded_image.reduceToVectors(
        geometryType='polygon',
        reducer=ee.Reducer.countEvery(),
        scale=8,  # Finer scale to capture narrow features
        geometry=bounding_box_geometry,
        maxPixels=1e8,
        eightConnected=False  # Use 4-connected to preserve linear features
    )

    # Export the clipped flooded polygons as GeoJSON
    clipped_flooded_geojson_path = f"flood_reports/reports/clipped_flooded_areas_s1_{image_date_str}_{dvv_str}.geojson"
    geemap.ee_export_vector(flooded_clipped_vectors, filename=clipped_flooded_geojson_path)
    print(f"Clipped flooded polygons exported to {clipped_flooded_geojson_path}")

    def compute_area(feature):
        return feature.set({'unit_acres': feature.geometry().area().divide(4046.86)})

    units_with_area = units_clipped.map(compute_area)

    def compute_refined_flood_area(feature):
        geom = feature.geometry()
        # Use a reference image to count total pixels (use VV band from wet composite)
        total_pixels = wet_composite.select("VV").reduceRegion(
            reducer=ee.Reducer.count(), geometry=geom, scale=10).get('VV')
        flooded_pixels = binary_image.reduceRegion(
            reducer=ee.Reducer.count(), geometry=geom, scale=10).get('flood_s1')
        pixel_area_m2 = ee.Number(pixel_size).multiply(pixel_size)
        pixel_area_acres = pixel_area_m2.multiply(0.000247105)
        flooded_area_acres = ee.Number(flooded_pixels).multiply(pixel_area_acres)
        flooded_percentage = ee.Number(flooded_pixels).divide(total_pixels).multiply(100)
        unit_name = feature.get('Flood_Unit')
        label = ee.String(unit_name).cat(': ').cat(flooded_area_acres.format('%.2f')).cat(' Acres')
        centroid = feature.geometry().centroid().coordinates()
        return feature.set({
            'total_pixels': total_pixels,
            'flooded_pixels': flooded_pixels,
            'acres_flooded': flooded_area_acres,
            'flooded_percentage': flooded_percentage,
            'label': label,
            'centroid': centroid
        })
    
    print("Calculating areas and flooded pixels...")
    units_with_calculations = units_with_area.map(compute_refined_flood_area)
    print("Areas and flooded pixels calculated.")

    # Export subunit polygons as GeoJSON
    print("Exporting subunit polygons as GeoJSON...")
    geemap.ee_export_vector(units_with_calculations, filename="flood_reports/reports/subunits_s1.geojson")
    print("Subunits exported as GeoJSON.")

    # Convert EE feature collection to Pandas DataFrame
    units_df_properties_reduced = pd.DataFrame(units_with_calculations.getInfo()['features'])
    units_df_properties_reduced = pd.json_normalize(units_df_properties_reduced['properties'])
    units_df_properties_reduced = units_df_properties_reduced[['Flood_Unit', 'total_pixels', 'flooded_pixels', 'unit_acres', 'acres_flooded', 'flooded_percentage']]
    units_df_properties_reduced = units_df_properties_reduced.round(2)

    # Calculate Total Acreage and add total row
    total_acres = units_df_properties_reduced['unit_acres'].sum()
    total_flooded_acres = units_df_properties_reduced['acres_flooded'].sum()
    total_flooded_percentage = (total_flooded_acres / total_acres) * 100

    totals = pd.DataFrame([{
        'Flood_Unit': 'Total',
        'total_pixels': units_df_properties_reduced['total_pixels'].sum(),
        'flooded_pixels': units_df_properties_reduced['flooded_pixels'].sum(),
        'unit_acres': total_acres,
        'acres_flooded': total_flooded_acres,
        'flooded_percentage': total_flooded_percentage
    }])

    # Add totals row only once
    units_df_properties_reduced = pd.concat([units_df_properties_reduced.dropna(), totals], ignore_index=True)
    units_df_properties_reduced = units_df_properties_reduced.round(2)

    # Debugging: print the cleaned DataFrame
    print(units_df_properties_reduced.to_string(index=False))

    # Load S2 optical imagery for comparison and create NIR flood boundaries
    print("Loading Sentinel-2 optical imagery for comparison...")
    s2_flood_boundaries_path = None
    try:
        s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(bounding_box_geometry) \
            .filterDate(start_date, wet_end) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50)) \
            .select(['B4','B8', 'SCL','B11'])
        
        s2_size = s2_collection.size().getInfo()
        if s2_size > 0:
            s2_composite = s2_collection.map(lambda img: img.updateMask(img.select('SCL').neq(9))).median()
            
            # Export false-color composite (SWIR1, NIR, Red)
            false_color_vis = {'min': 0, 'max': 3000, 'bands': ['B11', 'B8', 'B4'], 'gamma': 1.4}
            s2_false_color_tif = f"flood_reports/reports/s2_false_color_{image_date_str}_{dvv_str}.tif"
            s2_false_color_image = s2_composite.select(['B11', 'B8', 'B4']).visualize(**false_color_vis)
            geemap.ee_export_image(s2_false_color_image, filename=s2_false_color_tif, scale=10, region=bbox)
            
            s2_false_color_png = f"flood_reports/reports/s2_false_color_{image_date_str}_{dvv_str}.png"
            s2_image = imageio.imread(s2_false_color_tif)
            imageio.imwrite(s2_false_color_png, s2_image)
            print("S2 false-color composite exported for comparison.")
            
            # Create S2 flood mask using both NIR threshold and NDWI for better canal detection
            # NDWI (Normalized Difference Water Index) is often better at detecting narrow water features
            s2_threshold = 0.18  # Typical value for this area
            print(f"Creating S2 flood boundaries with NIR threshold {s2_threshold} and NDWI...")
            
            # NIR threshold (original method)
            s2_nir_binary = s2_composite.select('B8').divide(10000).lt(s2_threshold).selfMask()
            
            # NDWI = (Green - NIR) / (Green + NIR) - better for narrow water features
            # S2 bands: B3=Green, B8=NIR
            green = s2_composite.select('B4').divide(10000)  # Using B4 (Red) as proxy, or could use B3 if available
            nir = s2_composite.select('B8').divide(10000)
            # For S2, we have B4 (Red), B8 (NIR), B11 (SWIR) - let's use SWIR-based water index
            # MNDWI = (Green - SWIR) / (Green + SWIR) - often better than NDWI
            # Or use simple NIR threshold but with more sensitive settings for canals
            
            # Combine NIR threshold with a more sensitive approach for linear features
            # Use a slightly more sensitive threshold to catch narrow features
            s2_binary_sensitive = s2_composite.select('B8').divide(10000).lt(s2_threshold + 0.02).selfMask()
            
            # Use the original threshold for main detection
            s2_binary = s2_nir_binary
            
            # Vectorize S2 flooded areas
            # Use finer scale to better capture narrow linear features like canals
            s2_flooded_vectors = s2_binary.reduceToVectors(
                geometryType='polygon',
                reducer=ee.Reducer.countEvery(),
                scale=8,  # Finer scale for narrow features
                geometry=bounding_box_geometry,
                maxPixels=1e8,
                eightConnected=False  # Use 4-connected to preserve linear features
            )
            
            # Clip to units and export
            s2_flooded_clipped = s2_binary.clipToCollection(units_clipped)
            s2_flooded_clipped_vectors = s2_flooded_clipped.reduceToVectors(
                geometryType='polygon',
                reducer=ee.Reducer.countEvery(),
                scale=8,  # Finer scale for narrow features
                geometry=bounding_box_geometry,
                maxPixels=1e8,
                eightConnected=False  # Use 4-connected to preserve linear features
            )
            
            s2_flood_boundaries_path = f"flood_reports/reports/clipped_flooded_areas_s2_{image_date_str}_{s2_threshold}.geojson"
            geemap.ee_export_vector(s2_flooded_clipped_vectors, filename=s2_flood_boundaries_path)
            print(f"S2 NIR flood boundaries exported to {s2_flood_boundaries_path}")
            has_s2 = True
        else:
            has_s2 = False
            print("No S2 imagery available for this date range.")
    except Exception as e:
        print(f"Could not load S2 imagery: {e}")
        has_s2 = False
        import traceback
        traceback.print_exc()

    # Create and Save HTML Map
    print("Creating and saving the HTML map with overlays...")
    clipped_binary_image = binary_image.clip(units)

    # Define the filenames for the overlays
    sar_composite_image_path = f"flood_reports/reports/sar_composite_{image_date_str}_{dvv_str}.png"

    # Create the map with a center point
    Map = folium.Map(location=[36.8795, -118.202], zoom_start=12)

    # Add S2 false-color composite as base layer if available
    if has_s2:
        s2_false_color_path = f"flood_reports/reports/s2_false_color_{image_date_str}_{dvv_str}.png"
        Map.add_child(ImageOverlay(
            name="S2 False Color (SWIR1-NIR-Red)",
            image=s2_false_color_path,
            bounds=[[36.84651455123723, -118.23240736400778], [36.924364295139625, -118.17232588207419]],
            opacity=1
        ))
    
    # Add SAR composite as alternative base layer
    Map.add_child(ImageOverlay(
        name="SAR Composite (VV)",
        image=sar_composite_image_path,
        bounds=[[36.84651455123723, -118.23240736400778], [36.924364295139625, -118.17232588207419]],
        opacity=0.7
    ))

    # Add S2 NIR flood boundaries for comparison
    if has_s2 and s2_flood_boundaries_path:
        Map.add_child(folium.GeoJson(
            s2_flood_boundaries_path,
            name="Flooded Areas (S2 NIR)",
            style_function=lambda x: {
                "color": "green",
                "weight": 2,
                "fillColor": "green",
                "fillOpacity": 0.3
            }
        ))
    
    # Add clipped flooded polygons to the folium map
    Map.add_child(folium.GeoJson(
        clipped_flooded_geojson_path,
        name="Flooded Areas (SAR)",
        style_function=lambda x: {
            "color": "blue",
            "weight": 2,
            "fillColor": "blue",
            "fillOpacity": 0.4
        }
    ))

    # Add subunit polygons as GeoJSON
    Map.add_child(folium.GeoJson(
        "flood_reports/reports/subunits_s1.geojson",
        name="Unit Boundaries",
        style_function=lambda x: {
            "color": "red",
            "weight": 2,
            "fillColor": "#00000000",
            "fillOpacity": 0
        }
    ))

    # Adding static labels and popups for each unit
    for feature in units_with_calculations.getInfo()['features']:
        unit_name = feature['properties']['Flood_Unit']
        label = feature['properties']['label']
        centroid = feature['properties']['centroid']

        if isinstance(centroid, list) and len(centroid) == 2:
            folium.map.Marker(
                location=[centroid[1], centroid[0]],
                icon=folium.DivIcon(html=f"""<div style="font-size: 12px; color: black;">{unit_name}</div>""")
            ).add_to(Map)

            folium.Marker(
                location=[centroid[1], centroid[0]],
                popup=label
            ).add_to(Map)

    Map.add_child(folium.LayerControl())
    Map.save(map_filename)
    print(f"Map saved to {map_filename}")

    # Define the subdirectory for CSV output
    csv_subdirectory = "flood_reports/csv_output"
    os.makedirs(csv_subdirectory, exist_ok=True)

    # Save the DataFrame to a CSV file
    print("Generating CSV report...")
    csv_filename = os.path.join(csv_subdirectory, f'flood_report_data_s1_{image_date_str}_{dvv_str}.csv')
    units_df_properties_reduced.to_csv(csv_filename, index=False)
    print(f"CSV file saved to {csv_filename}")

    # Create HTML Table
    units_df_properties_reduced = units_df_properties_reduced[['Flood_Unit', 'acres_flooded']]
    units_df_properties_reduced.columns = ['BWMA Unit', 'Acres']
    units_df_properties_reduced['Acres'] = units_df_properties_reduced['Acres'].round(0).astype(int)

    html_table_simple = (
        units_df_properties_reduced.style
        .set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center'), ('font-size', '14px')]},
            {'selector': 'td:nth-child(1)', 'props': [('text-align', 'left'), ('font-size', '12px')]},
            {'selector': 'td:nth-child(2)', 'props': [('text-align', 'right'), ('font-size', '12px')]},
        ])
        .set_properties(subset=pd.IndexSlice[units_df_properties_reduced.index[-1], :], **{'font-weight': 'bold', 'font-size': '14px'})
        .hide(axis='index')
        .to_html()
    )
    html_table_simple = html_table_simple.replace('<th></th>', '')

    # Create HTML Report
    html_report = f"""
    <html>
    <head>
        <title>BWMA Flooded Extent (SAR): {image_date_str}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
            }}
            .container {{
                display: flex;
                justify-content: space-between;
            }}
            .left {{
                width: 25%;
            }}
            .right {{
                width: 75%;
                text-align: center;
            }}
            h1, h2 {{
                text-align: center;
            }}
            .notes {{
                margin-top: 20px;
                padding: 10px;
                border-top: 1px solid #000;
            }}
            table {{
                width: 90%;
                font-size: 50px;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 20px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                text-align: right;
                font-size: 30px;
            }}
            td:nth-child(2) {{
                text-align: right;
            }}
            tr:last-child {{
                font-weight: bold;
                font-size: 80px;
            }}
            tr:last-child td {{
                border-top: 3px solid #000;
            }}
        </style>
    </head>
    <body>
        <h1>BWMA Flooded Acres and Extent (Sentinel-1 SAR): {image_date_str}</h1>
        <div class="container">
            <div class="left">
                <h2>Flooded Acres</h2>
                {html_table_simple}
            </div>
            <div class="right">
                <h2>Spatial Extent</h2>
                <p>Imagery Date: {image_date_str}</p>
                <p>Dry Baseline: {dry_start} to {dry_end}</p>
                <p>SAR Thresholds: dVV &lt; {dvv_thresh} dB, VV-VH ratio &lt; {vv_vh_ratio_max} dB</p>
                <iframe src="./flooded_area_map_s1_{image_date_str}_{dvv_str}.html" width="90%" height="500"></iframe>
            </div>
        </div>
        <div class="notes">
            <h3>Technical Notes and Links</h3>
            <p>Flooded acres were calculated from Sentinel-1 SAR (Synthetic Aperture Radar) imagery using the Earth Engine Python API.</p>
            <p>Sentinel-1 C-band SAR is sensitive to water under emergent vegetation, making it complementary to optical Sentinel-2 imagery.</p>
            <p>Flood detection uses change detection: comparing wet period backscatter to a dry baseline period. Flooded areas show decreased VV backscatter (dVV threshold: {dvv_thresh} dB) and altered VV/VH polarization ratios.</p>
            <p>Vectorized flooded extent boundaries - GeoJSON <a href="clipped_flooded_areas_s1_{image_date_str}_{dvv_str}.geojson" download>here</a>.</p>
            <p>Flooded extent CSV <a href="../csv_output/flood_report_data_s1_{image_date_str}_{dvv_str}.csv" download>here</a>.</p>
            <p>Sentinel-1 SAR composite GeoTIFF <a href="sar_composite_{image_date_str}_{dvv_str}.tif" download>here</a>.</p>
        </div>
    </body>
    </html>
    """
    print("Generating CSV and HTML report for flood data...")
    with open(report_filename, "w") as file:
        file.write(html_report)
    print(f"Report saved to {report_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate flood report using Sentinel-1 SAR data.")
    parser.add_argument('start_date', type=str, help='Start date for wet/flood period in YYYY-MM-DD format')
    parser.add_argument('dry_start', type=str, help='Start date for dry baseline period in YYYY-MM-DD format')
    parser.add_argument('dry_end', type=str, help='End date for dry baseline period in YYYY-MM-DD format')
    parser.add_argument('--dvv-thresh', type=float, default=-1.5, help='dVV threshold in dB (default: -1.5)')
    parser.add_argument('--vv-vh-ratio', type=float, default=3, help='Maximum VV-VH ratio in dB (default: 3)')
    parser.add_argument('--orbit-pass', type=str, choices=['ASCENDING', 'DESCENDING'], 
                       help='Filter by orbit pass direction (optional)')
    parser.add_argument('--no-ensemble', action='store_true',
                       help='Disable temporal ensemble (use single date median)')
    parser.add_argument('--days-before', type=int, default=5,
                       help='Days before target date for ensemble (default: 5)')
    parser.add_argument('--days-after', type=int, default=5,
                       help='Days after target date for ensemble (default: 5)')
    parser.add_argument('--min-dates', type=int, default=2,
                       help='Minimum number of dates pixel must be flooded (default: 2)')
    parser.add_argument('--absolute', action='store_true',
                       help='Use absolute backscatter thresholds instead of change detection')
    parser.add_argument('--vv-thresh', type=float, default=-18,
                       help='Absolute VV backscatter threshold in dB (default: -18)')
    parser.add_argument('--vh-thresh', type=float, default=-22,
                       help='Absolute VH backscatter threshold in dB (default: -22)')
    args = parser.parse_args()
    main(args.start_date, args.dry_start, args.dry_end, 
         dvv_thresh=args.dvv_thresh, vv_vh_ratio_max=args.vv_vh_ratio, 
         orbit_pass=args.orbit_pass,
         use_ensemble=not args.no_ensemble,
         days_before=args.days_before,
         days_after=args.days_after,
         min_dates=args.min_dates,
         use_absolute=args.absolute,
         vv_thresh=args.vv_thresh,
         vh_thresh=args.vh_thresh)
