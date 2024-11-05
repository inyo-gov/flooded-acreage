import argparse
import ee
import geopandas as gpd
import geemap
import pandas as pd
import folium
import geemap.foliumap as geemap
from datetime import datetime, timedelta
import os

# Example terminal command
# > python flood_report.py '2024-10-22' .22
# > python flood_report.py '2024-11-01' .22

def main(start_date, threshold):
    # Initialize Earth Engine
    ee.Initialize()

    # Compute the end date by adding days to the start date
    end_date = (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=15)).strftime('%Y-%m-%d')

    # Define the bounding box coordinates
    bbox = [[-118.23240736400778, 36.84651455123723],
            [-118.17232588207419, 36.84651455123723],
            [-118.17232588207419, 36.924364295139625],
            [-118.23240736400778, 36.924364295139625]]

    # Create a bounding box geometry
    bounding_box_geometry = ee.Geometry.Polygon(bbox)


    # Define visualization parameters for a better false-color composite
    false_color_vis = {
        'min': 0,
        'max': 3000,
        'bands': ['B11', 'B8', 'B4'],  # SWIR1, NIR, Red
        'gamma': 1.4
    }

    # Filter Sentinel-2 surface reflectance imagery and extract dates, pixel size
    sentinel_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(bounding_box_geometry) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
        .select(['B4','B8', 'SCL','B11'])

    size = sentinel_collection.size().getInfo()
    print(f"Number of images in collection: {size}")

    # Select a single band before retrieving the projection information
    pixel_size = sentinel_collection.first().select('B8').projection().nominalScale().getInfo()

    # Extract the date information
    image_info = sentinel_collection.first().getInfo()
    image_date = image_info['properties']['system:time_start']
    image_date_str = pd.to_datetime(image_date, unit='ms').strftime('%Y-%m-%d')

    # Define the subdirectory for HTML maps and reports
    html_subdirectory = "docs/reports"
    os.makedirs(html_subdirectory, exist_ok=True)

    # Define filenames with unique names based on the image date
    report_filename = os.path.join(html_subdirectory, f"bwma_flood_report_{image_date_str}_{threshold}.html")
    map_filename = os.path.join(html_subdirectory, f"flooded_area_map_{image_date_str}_{threshold}.html")

    def mask_clouds(image):
        cloud_prob = image.select('SCL')
        is_cloud = cloud_prob.eq(9)
        return image.updateMask(is_cloud.Not())

    cloud_free_composite = sentinel_collection.map(mask_clouds).median()
    binary_image = cloud_free_composite.select('B8').divide(10000).lt(threshold).selfMask()

    # Load units from geojson and convert to Earth Engine geometry
    gdf = gpd.read_file("data/unitsBwma2800.geojson")
    units = geemap.geopandas_to_ee(gdf)
    units = units.filterBounds(bounding_box_geometry)
    units_clipped = units.map(lambda feature: feature.intersection(bounding_box_geometry))

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
        return feature.set({
            'total_pixels': total_pixels,
            'flooded_pixels': flooded_pixels,
            'acres_flooded': flooded_area_acres,
            'flooded_percentage': flooded_percentage,
            'label': label
        })

    units_with_calculations = units_with_area.map(compute_refined_flood_area)

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

    # units_df_properties_reduced = pd.concat([units_df_properties_reduced, totals], ignore_index=True)
    # units_df_properties_reduced = units_df_properties_reduced.round(2)
  

    # Create and Save HTML Map
    clipped_binary_image = binary_image.clip(units)
    Map = geemap.Map(center=[36.8795, -118.202], zoom=12)
    # Add the false-color composite to the map
    Map.addLayer(sentinel_collection.median(), false_color_vis, 'False-Color Composite')

    Map.addLayer(clipped_binary_image, {'palette': ['blue'], 'opacity': 0.5}, 'Flooded Pixels')
    units_style = {'color': 'red', 'fillColor': '00000000'}
    Map.addLayer(units_with_calculations.style(**units_style), {}, 'Unit Boundaries')

    labels = units_with_calculations.aggregate_array('label').getInfo()

    def get_centroid(feature):
        return feature.geometry().centroid().coordinates()

    centroids = units_with_calculations.map(lambda f: f.set('centroid', get_centroid(f)))
    centroid_info = centroids.aggregate_array('centroid').getInfo()

    for label, centroid in zip(labels, centroid_info):
        folium.Marker(
            location=[centroid[1], centroid[0]],
            icon=None,
            popup=label
        ).add_to(Map)

    Map.add_child(folium.LayerControl())

    Map.save(map_filename)
    print(f"Map saved to {map_filename}")

    # Define the subdirectory for CSV output
    csv_subdirectory = "flood_reports/csv_output"
    os.makedirs(csv_subdirectory, exist_ok=True)

    # Save the DataFrame to a CSV file
    csv_filename = os.path.join(csv_subdirectory, f'flood_report_data_{image_date_str}_{threshold}.csv')
    units_df_properties_reduced.to_csv(csv_filename, index=False)
    print(f"CSV file saved to {csv_filename}")

    # Create HTML Table
    units_df_properties_reduced = units_df_properties_reduced[['Flood_Unit', 'acres_flooded']]
    units_df_properties_reduced.columns = ['BWMA Unit', 'Acres']
    units_df_properties_reduced['Acres'] = units_df_properties_reduced['Acres'].round(0).astype(int)
    # total_acres = units_df_properties_reduced['Acres'].sum()
    # totals_row = pd.DataFrame([{'BWMA Unit': 'Total', 'Acres': total_acres}])
    # units_df_properties_reduced = pd.concat([units_df_properties_reduced, totals_row], ignore_index=True)

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
                <iframe src="flooded_area_map_{image_date_str}_{threshold}.html" width="90%" height="500"></iframe>
            </div>
        </div>
        <div class="notes">
            <h3>Technical Notes</h3>
            <p>Flooded acres were calculated from Sentinel-2 Surface Reflectance imagery using the Earth Engine Python API in a Jupyter notebook.  Sentinel-2 (S2) is a wide-swath, high-resolution, multispectral imaging mission with a global 5-day revisit frequency.</p>
            <p>The S2 Multispectral Instrument (MSI) samples 13 spectral bands: Visible and NIR at 10 meters, red edge and SWIR at 20 meters, and atmospheric bands at 60 meters spatial resolution. The Near Infrared (NIR) band was used to identify flooded areas by applying a threshold to isolate water.</p>
            <p>The flooded extent estimates are validated during routine field checks throughout the seasonal flooding cycle September through April.</p>
        </div>
    </body>
    </html>
    """
    with open(report_filename, "w") as file:
        file.write(html_report)
    print(f"Report saved to {report_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate flood report based on start date and threshold.")
    parser.add_argument('start_date', type=str, help='Start date in YYYY-MM-DD format')
    parser.add_argument('threshold', type=float, help='Surface reflectance threshold value')
    args = parser.parse_args()
    main(args.start_date, args.threshold)
