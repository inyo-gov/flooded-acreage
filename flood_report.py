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

# Example terminal command
# > python flood_report.py '2024-10-22' .22
# > python flood_report.py '2024-11-01' .22

def main(start_date, threshold):
    # Initialize Earth Engine with registered project
    print("Initializing Earth Engine...")
    ee.Initialize(project='ee-zjn-2022')

    # Compute the end date by adding days to the start date
    end_date = (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=15)).strftime('%Y-%m-%d')
    print(f"Date range: {start_date} to {end_date}")

    # Define the bounding box coordinates
    print("Defining bounding box and geometry...")
    bbox = [[-118.23240736400778, 36.84651455123723],
            [-118.17232588207419, 36.84651455123723],
            [-118.17232588207419, 36.924364295139625],
            [-118.23240736400778, 36.924364295139625]]

    # Create a bounding box geometry
    bounding_box_geometry = ee.Geometry.Polygon(bbox)

    # Define visualization parameters for a better false-color composite
    print("Setting false-color visualization parameters...")
    false_color_vis = {
        'min': 0,
        'max': 3000,
        'bands': ['B11', 'B8', 'B4'],  # SWIR1, NIR, Red
        'gamma': 1.4
    }

    # Filter Sentinel-2 surface reflectance imagery and extract dates, pixel size
    print("Filtering Sentinel-2 collection...")
    sentinel_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(bounding_box_geometry) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50)) \
        .select(['B4','B8', 'SCL','B11'])

    size = sentinel_collection.size().getInfo()
    print(f"Number of images in collection: {size}")

    # Select a single band before retrieving the projection information
    pixel_size = sentinel_collection.first().select('B8').projection().nominalScale().getInfo()

    # Extract the date information
    print("Extracting image date information...")
    image_info = sentinel_collection.first().getInfo()
    image_date = image_info['properties']['system:time_start']
    image_date_str = pd.to_datetime(image_date, unit='ms').strftime('%Y-%m-%d')
    print(f"Image date: {image_date_str}")

    # Define the subdirectory for HTML maps and reports
    print("Setting up directories and filenames...")
    html_subdirectory = "flood_reports/reports"
    os.makedirs(html_subdirectory, exist_ok=True)

    # Define filenames with unique names based on the image date
    report_filename = os.path.join(html_subdirectory, f"bwma_flood_report_{image_date_str}_{threshold}.html")
    map_filename = os.path.join(html_subdirectory, f"flooded_area_map_{image_date_str}_{threshold}.html")

    def mask_clouds(image):
        cloud_prob = image.select('SCL')
        is_cloud = cloud_prob.eq(9)
        return image.updateMask(is_cloud.Not())

    # Mask clouds and compute composite
    print("Creating cloud-free composite and binary image...")
    cloud_free_composite = sentinel_collection.map(lambda img: img.updateMask(img.select('SCL').neq(9))).median()
    binary_image = cloud_free_composite.select('B8').divide(10000).lt(threshold).selfMask()
    
    # Convert flooded areas to vector (GeoJSON format)
    print("Vectorizing flooded areas...")
    flooded_vectors = binary_image.reduceToVectors(
        geometryType='polygon',
        reducer=ee.Reducer.countEvery(),
        scale=10,
        geometry=bounding_box_geometry,
        maxPixels=1e8
    )

    # Define export file paths with date and threshold
    print("Exporting false-color and flooded pixels images...")
    false_color_filename_tif = f"flood_reports/reports/false_color_composite_{image_date_str}_{threshold}.tif"
    # flooded_pixels_filename_tif = f"docs/reports/flooded_pixels_{image_date_str}_{threshold}.tif"
    
    
    # Export false-color composite as TIF with the correct bands
    false_color_image = cloud_free_composite.select(['B11', 'B8', 'B4']).visualize(**false_color_vis)
    
    geemap.ee_export_image(false_color_image, filename=false_color_filename_tif, scale=10, region=bbox)
    print("False-color composite exported.")

    # Create the visualized image for flooded pixels only with a transparent background
    # Apply selfMask() and visualize the binary image for flooded areas only
    flooded_pixels_visualized = binary_image.selfMask().visualize(
        palette=['blue'],  # Set only flooded areas to blue
        min=0, max=1  # This limits values to binary (0 or 1) for transparency
    )

    false_color_filename_png = f"flood_reports/reports/false_color_composite_{image_date_str}_{threshold}.png"
    # flooded_pixels_filename_png = f"docs/reports/flooded_pixels_{image_date_str}_{threshold}.png"
    
    # Read the TIFF file and save it as PNG
    false_color_image = imageio.imread(false_color_filename_tif)
    imageio.imwrite(false_color_filename_png, false_color_image)
    print(f"False-color PNG saved at {false_color_filename_png}")

    # Load units from geojson and convert to Earth Engine geometry
    print("Loading units from geojson...")
    gdf = gpd.read_file("data/unitsBwma2800.geojson")
    units = geemap.geopandas_to_ee(gdf)
    units = units.filterBounds(bounding_box_geometry)
    units_clipped = units.map(lambda feature: feature.intersection(bounding_box_geometry))

    # units_flattened = units_clipped.flatten()
        
    # Clip the binary flooded image to the flattened unit boundaries
    clipped_flooded_image = binary_image.clipToCollection(units_clipped)

    # Vectorize the clipped flooded areas to GeoJSON polygons
    flooded_clipped_vectors = clipped_flooded_image.reduceToVectors(
        geometryType='polygon',
        reducer=ee.Reducer.countEvery(),
        scale=10,
        geometry=bounding_box_geometry,
        maxPixels=1e8
    )

    # Export the clipped flooded polygons as GeoJSON
    clipped_flooded_geojson_path = f"flood_reports/reports/clipped_flooded_areas_{image_date_str}_{threshold}.geojson"
    geemap.ee_export_vector(flooded_clipped_vectors, filename=clipped_flooded_geojson_path)
    print(f"Clipped flooded polygons exported to {clipped_flooded_geojson_path}")


    def compute_area(feature):
        return feature.set({'unit_acres': feature.geometry().area().divide(4046.86)})

    units_with_area = units_clipped.map(compute_area)

    def compute_refined_flood_area(feature):
        geom = feature.geometry()
        total_pixels = cloud_free_composite.select('B8').reduceRegion(
            reducer=ee.Reducer.count(), geometry=geom, scale=10).get('B8')
        flooded_pixels = binary_image.reduceRegion(
            reducer=ee.Reducer.count(), geometry=geom, scale=10).get('B8')
        pixel_area_m2 = ee.Number(pixel_size).multiply(pixel_size)
        pixel_area_acres = pixel_area_m2.multiply(0.000247105)
        flooded_area_acres = ee.Number(flooded_pixels).multiply(pixel_area_acres)
        flooded_percentage = ee.Number(flooded_pixels).divide(total_pixels).multiply(100)
        unit_name = feature.get('Flood_Unit')
        label = ee.String(unit_name).cat(': ').cat(flooded_area_acres.format('%.2f')).cat(' Acres')
        centroid = feature.geometry().centroid().coordinates()  # Compute centroid here
        return feature.set({
            'total_pixels': total_pixels,
            'flooded_pixels': flooded_pixels,
            'acres_flooded': flooded_area_acres,
            'flooded_percentage': flooded_percentage,
            'label': label,
            'centroid': centroid  # Store the centroid
        })
    print("Calculating areas and flooded pixels...")
    units_with_calculations = units_with_area.map(compute_refined_flood_area)
    print("Areas and flooded pixels calculated.")

    # Export subunit polygons as GeoJSON
    print("Exporting subunit polygons as GeoJSON...")
    geemap.ee_export_vector(units_with_calculations, filename="flood_reports/reports/subunits.geojson")
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

    # Debugging: print the cleaned DataFrame to ensure no duplicates
    print(units_df_properties_reduced.to_string(index=False))

    # Create and Save HTML Map
    print("Creating and saving the HTML map with overlays...")
    clipped_binary_image = binary_image.clip(units)

    # Define the filenames for the overlays, matching the export paths
    false_color_image_path = f"flood_reports/reports/false_color_composite_{image_date_str}_{threshold}.png"
    # flooded_pixels_image_path = f"docs/reports/flooded_pixels_{image_date_str}_{threshold}.png"

    # Create the map with a center point
    Map = folium.Map(location=[36.8795, -118.202], zoom_start=12)

    # Add static image overlay for False Color Composite with dynamic path
    # Use full path for creation, but the map will work with relative paths when served
    Map.add_child(ImageOverlay(
        name="False Color Composite",
        image=false_color_image_path,
        bounds=[[36.84651455123723, -118.23240736400778], [36.924364295139625, -118.17232588207419]],
        opacity=1
    ))

    
    # Add clipped flooded polygons to the folium map
    Map.add_child(folium.GeoJson(
        clipped_flooded_geojson_path,
        name="Flooded Areas (Clipped)",
        style_function=lambda x: {
            "color": "blue",
            "weight": 1,
            "fillColor": "blue",
            "fillOpacity": 0.5
        }
    ))

    # Add subunit polygons as GeoJSON (exported to a local file, if necessary)
    Map.add_child(folium.GeoJson(
        "flood_reports/reports/subunits.geojson",
    name="Unit Boundaries", 
    style_function=lambda x: {
        "color": "red",  # Boundary color
        "weight": 2,
        "fillColor": "#00000000",  # Transparent fill
        "fillOpacity": 0  # Ensures fill is transparent
    }
    ))

    # Adding static labels and popups for each unit at the pre-computed centroid
    for feature in units_with_calculations.getInfo()['features']:
        unit_name = feature['properties']['Flood_Unit']
        label = feature['properties']['label']
        centroid = feature['properties']['centroid']

        # Ensure centroid coordinates are valid before adding
        if isinstance(centroid, list) and len(centroid) == 2:
            # Static label using DivIcon for always visible text
            folium.map.Marker(
                location=[centroid[1], centroid[0]],  # Latitude, Longitude
                icon=folium.DivIcon(html=f"""<div style="font-size: 12px; color: black;">{unit_name}</div>""")
            ).add_to(Map)

            # Popup with detailed information (e.g., acreage)
            folium.Marker(
                location=[centroid[1], centroid[0]],  # Latitude, Longitude
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
    csv_filename = os.path.join(csv_subdirectory, f'flood_report_data_{image_date_str}_{threshold}.csv')
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
        <title>BWMA Flooded Extent: {image_date_str}</title>
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
        <h1>BWMA Flooded Acres and Extent: {image_date_str}</h1>
        <div class="container">
            <div class="left">
                <h2>Flooded Acres</h2>
                {html_table_simple}
            </div>
            <div class="right">
                <h2>Spatial Extent</h2>
                <p>Imagery Date: {image_date_str}</p>
                <iframe src="./flooded_area_map_{image_date_str}_{threshold}.html" width="90%" height="500"></iframe>
            </div>
        </div>
        <div class="notes">
            <h3>Technical Notes and Links</h3>
            <p>Flooded acres were calculated from Sentinel-2 Surface Reflectance imagery using the Earth Engine Python API.</p>
            <p>NIR band was used to identify flooded areas by applying a threshold to isolate water.</p>
            <p>Vectorized flooded extent boundaries - GeoJSON <a href="clipped_flooded_areas_{image_date_str}_{threshold}.geojson" download>here</a>.</p>
            <p>Flooded extent CSV <a href="../csv_output/flood_report_data_{image_date_str}_{threshold}.csv" download>here</a>.</p>
            <p>Sentinel 2 false color composite GeoTIFF <a href="false_color_composite_{image_date_str}_{threshold}.tif" download>here</a>.</p>
        </div>
        </div>
    </body>
    </html>
    """
    print("Generating CSV and HTML report for flood data...")
    with open(report_filename, "w") as file:
        file.write(html_report)
    print(f"Report saved to {report_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate flood report based on start date and threshold.")
    parser.add_argument('start_date', type=str, help='Start date in YYYY-MM-DD format')
    parser.add_argument('threshold', type=float, help='Surface reflectance threshold value')
    args = parser.parse_args()
    main(args.start_date, args.threshold)
