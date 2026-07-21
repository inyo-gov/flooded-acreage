import argparse
import ee
import geopandas as gpd
import geemap
import pandas as pd
import folium
from datetime import datetime, timedelta
import html
import os
import imageio.v2 as imageio

# Example terminal command
# > python flood_report.py '2024-10-22' .22
# > python flood_report.py '2024-11-01' .22

UNIT_ORDER = [
    "Drew", "Waggoner", "West Winterton", "East Winterton",
    "South Winterton", "Thibaut Ponds", "Thibaut",
]


def order_units_df(df):
    """Sort flood units to match dashboard legend order; Total row last."""
    units_only = df[df["Flood_Unit"] != "Total"].copy()
    units_only["Flood_Unit"] = pd.Categorical(
        units_only["Flood_Unit"], categories=UNIT_ORDER, ordered=True
    )
    units_only = units_only.sort_values("Flood_Unit")
    total_row = df[df["Flood_Unit"] == "Total"]
    return pd.concat([units_only, total_row], ignore_index=True)


def build_report_html(
    table_df,
    image_date_str,
    start_date,
    end_date,
    threshold,
    scene_count,
    map_basename,
    clipped_geojson_basename,
    csv_basename,
    tif_basename,
):
    month_label = pd.to_datetime(start_date).strftime("%b %Y")
    total_acres = int(table_df.loc[table_df["BWMA Unit"] == "Total", "Acres"].iloc[0])

    rows_html = []
    for _, row in table_df.iterrows():
        unit = html.escape(str(row["BWMA Unit"]))
        acres = int(row["Acres"])
        row_class = "total-row" if unit == "Total" else ""
        rows_html.append(
            f'<tr class="{row_class}">'
            f'<td class="unit-name">{unit}</td>'
            f'<td class="unit-acres"><span class="acres-num">{acres}</span> ac</td>'
            f"</tr>"
        )
    rows_html = "\n".join(rows_html)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BWMA Flood Report · {html.escape(month_label)}</title>
  <style>
    :root {{
      --rs-nasa-blue: #0b3d91;
      --rs-nasa-red: #fc3d21;
      --rs-dark: #0d1117;
      --rs-border: #c5cdd8;
      --rs-muted: #5a6578;
      --rs-mono: ui-monospace, "SF Mono", Menlo, monospace;
      --rs-sans: "Segoe UI", system-ui, sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: var(--rs-sans);
      font-size: 14px;
      line-height: 1.45;
      color: #1a2332;
      background-color: #eef1f5;
      background-image:
        linear-gradient(rgba(11, 61, 145, 0.06) 1px, transparent 1px),
        linear-gradient(90deg, rgba(11, 61, 145, 0.06) 1px, transparent 1px);
      background-size: 18px 18px;
    }}
    .report-wrap {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 0.75rem 1rem 1.5rem;
    }}
    .report-header {{
      padding: 0.55rem 0.75rem;
      margin-bottom: 0.65rem;
      border: 1px solid #8a96a8;
      border-left: 4px solid var(--rs-nasa-blue);
      border-radius: 3px;
      background: var(--rs-dark);
      color: #e8ecf2;
    }}
    .report-tag {{
      font-family: var(--rs-mono);
      font-size: 0.62rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: #7eb8ff;
      margin-bottom: 0.15rem;
    }}
    .report-header h1 {{
      margin: 0;
      font-size: 1.15rem;
      font-weight: 700;
      color: #fff;
    }}
    .report-meta {{
      margin: 0.25rem 0 0;
      font-family: var(--rs-mono);
      font-size: 0.68rem;
      color: #9aa8bc;
    }}
    .report-meta strong {{
      color: #dce8f8;
      font-weight: 600;
    }}
    .report-grid {{
      display: grid;
      grid-template-columns: minmax(220px, 28%) 1fr;
      gap: 0.55rem;
      align-items: stretch;
    }}
    .report-panel {{
      background: #fff;
      border: 1px solid var(--rs-border);
      border-radius: 3px;
      overflow: hidden;
      box-shadow: 1px 1px 0 rgba(0, 0, 0, 0.03);
    }}
    .report-panel h2 {{
      margin: 0;
      padding: 0.4rem 0.65rem;
      font-family: var(--rs-mono);
      font-size: 0.68rem;
      font-weight: 700;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: #fff;
      background: var(--rs-dark);
      border-bottom: 2px solid var(--rs-nasa-blue);
    }}
    .report-panel h2::before {{
      content: "▸ ";
      color: var(--rs-nasa-red);
    }}
    .report-table-wrap {{
      padding: 0.35rem 0.5rem 0.5rem;
    }}
    .report-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.78rem;
    }}
    .report-table th {{
      font-family: var(--rs-mono);
      font-size: 0.6rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      text-align: left;
      color: var(--rs-muted);
      padding: 0.35rem 0.45rem;
      border-bottom: 2px solid var(--rs-border);
    }}
    .report-table th:last-child {{
      text-align: right;
    }}
    .report-table td {{
      padding: 0.32rem 0.45rem;
      border-bottom: 1px solid #e4e9ef;
    }}
    .report-table tr:nth-child(even) td {{
      background: #f7f9fc;
    }}
    .report-table .unit-name {{
      font-family: var(--rs-mono);
      font-size: 0.72rem;
      color: #1a2332;
    }}
    .report-table .unit-acres {{
      text-align: right;
      font-family: var(--rs-mono);
      font-size: 0.68rem;
      color: var(--rs-muted);
    }}
    .report-table .acres-num {{
      font-family: Georgia, Cambria, serif;
      font-variant-numeric: tabular-nums;
      font-size: 0.95rem;
      font-weight: 700;
      color: var(--rs-nasa-blue);
    }}
    .report-table tr.total-row td {{
      border-top: 2px solid var(--rs-nasa-blue);
      background: #eef3fa !important;
      font-weight: 700;
    }}
    .report-table tr.total-row .acres-num {{
      font-size: 1.05rem;
      color: var(--rs-nasa-red);
    }}
    .report-map-panel iframe {{
      display: block;
      width: 100%;
      height: min(560px, 72vh);
      border: none;
      background: #fff;
    }}
    .report-summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.35rem 0.75rem;
      padding: 0.45rem 0.65rem;
      border-top: 1px solid var(--rs-border);
      background: #f7f9fc;
      font-family: var(--rs-mono);
      font-size: 0.65rem;
      color: var(--rs-muted);
    }}
    .report-summary span {{
      white-space: nowrap;
    }}
    .report-footer {{
      margin-top: 0.65rem;
      padding: 0.55rem 0.65rem;
      background: #fff;
      border: 1px solid var(--rs-border);
      border-left: 3px solid var(--rs-nasa-blue);
      border-radius: 3px;
    }}
    .report-footer h3 {{
      margin: 0 0 0.35rem;
      font-family: var(--rs-mono);
      font-size: 0.65rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--rs-nasa-blue);
    }}
    .report-footer p {{
      margin: 0.2rem 0;
      font-size: 0.78rem;
      color: var(--rs-muted);
    }}
    .report-footer a {{
      color: var(--rs-nasa-blue);
      font-family: var(--rs-mono);
      font-size: 0.72rem;
      text-decoration: underline;
      text-underline-offset: 2px;
    }}
    .report-footer a:hover {{
      color: var(--rs-nasa-red);
    }}
    .download-links {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.35rem 0.75rem;
      margin-top: 0.35rem;
    }}
    @media (max-width: 820px) {{
      .report-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="report-wrap">
    <header class="report-header">
      <div class="report-tag">Sentinel-2 SR · Cloud-masked median composite</div>
      <h1>BWMA Flood Report · {html.escape(month_label)}</h1>
      <p class="report-meta">
        <strong>{html.escape(start_date)}</strong> to <strong>{html.escape(end_date)}</strong>
        · NIR threshold <strong>{threshold:g}</strong>
        · <strong>{scene_count}</strong> scene{"s" if scene_count != 1 else ""} in composite
        · label date {html.escape(image_date_str)}
      </p>
    </header>

    <div class="report-grid">
      <section class="report-panel">
        <h2>Flooded acreage</h2>
        <div class="report-table-wrap">
          <table class="report-table">
            <thead>
              <tr>
                <th>BWMA unit</th>
                <th>Acres flooded</th>
              </tr>
            </thead>
            <tbody>
              {rows_html}
            </tbody>
          </table>
        </div>
        <div class="report-summary">
          <span>Total: <strong>{total_acres} ac</strong></span>
          <span>Composite window: 15 days</span>
        </div>
      </section>

      <section class="report-panel report-map-panel">
        <h2>Spatial extent</h2>
        <iframe src="./{html.escape(map_basename)}" title="BWMA flooded extent map"></iframe>
      </section>
    </div>

    <footer class="report-footer">
      <h3>Data products</h3>
      <p>Sentinel-2 surface reflectance (harmonized). NIR band threshold applied to a cloud-masked median composite—not a single acquisition.</p>
      <div class="download-links">
        <a href="{html.escape(clipped_geojson_basename)}" download>Flooded extent GeoJSON</a>
        <a href="../csv_output/{html.escape(csv_basename)}" download>Unit acreage CSV</a>
        <a href="{html.escape(tif_basename)}" download>False-color GeoTIFF</a>
      </div>
    </footer>
  </div>
</body>
</html>
"""

def build_flood_map(
    map_basename,
    false_color_basename,
    clipped_geojson_basename,
    subunits_basename,
    unit_features,
    start_date,
    end_date,
    threshold,
    scene_count,
):
    """Build a telemetry-themed folium map for the monthly report iframe."""
    bbox_bounds = [
        [36.84651455123723, -118.23240736400778],
        [36.924364295139625, -118.17232588207419],
    ]
    month_label = pd.to_datetime(start_date).strftime("%b %Y")

    m = folium.Map(location=[36.8795, -118.202], tiles=None, control_scale=True)

    folium.TileLayer(
        tiles="CartoDB positron",
        name="Light basemap",
        overlay=False,
        control=True,
        show=True,
    ).add_to(m)

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri World Imagery",
        name="Satellite basemap",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    folium.raster_layers.ImageOverlay(
        name="S2 false-color composite",
        image=false_color_basename,
        bounds=bbox_bounds,
        opacity=1,
        interactive=True,
        cross_origin=False,
        zindex=2,
    ).add_to(m)

    folium.GeoJson(
        clipped_geojson_basename,
        name="Flooded extent",
        style_function=lambda x: {
            "color": "#0b3d91",
            "weight": 1.5,
            "fillColor": "#2e86ab",
            "fillOpacity": 0.55,
        },
        highlight_function=lambda x: {
            "weight": 2.5,
            "fillOpacity": 0.72,
            "color": "#fc3d21",
        },
    ).add_to(m)

    folium.GeoJson(
        subunits_basename,
        name="Unit boundaries",
        style_function=lambda x: {
            "color": "#fc3d21",
            "weight": 2,
            "fillColor": "#00000000",
            "fillOpacity": 0,
            "dashArray": "5 4",
        },
    ).add_to(m)

    for feature in unit_features:
        unit_name = feature["properties"]["Flood_Unit"]
        label = feature["properties"]["label"]
        centroid = feature["properties"]["centroid"]
        if isinstance(centroid, list) and len(centroid) == 2:
            folium.Marker(
                location=[centroid[1], centroid[0]],
                icon=folium.DivIcon(
                    icon_size=(None, None),
                    class_name="unit-label-wrap",
                    html=(
                        f'<div class="unit-map-label">{html.escape(unit_name)}</div>'
                    ),
                ),
                popup=folium.Popup(
                    f'<div class="unit-popup">{html.escape(label)}</div>',
                    max_width=240,
                ),
                tooltip=html.escape(label),
            ).add_to(m)

    folium.LayerControl(collapsed=True, position="topright").add_to(m)
    m.fit_bounds(bbox_bounds, padding=(12, 12))

    map_css = """
    <style>
      html, body, .folium-map { width: 100%; height: 100%; margin: 0; padding: 0; }
      .leaflet-container {
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
        background: #eef1f5;
      }
      .leaflet-control-layers {
        font-family: ui-monospace, "SF Mono", Menlo, monospace !important;
        font-size: 10px !important;
        border: 1px solid #c5cdd8 !important;
        border-radius: 3px !important;
        box-shadow: 1px 1px 0 rgba(0,0,0,0.05) !important;
      }
      .leaflet-control-layers-expanded {
        padding: 4px 8px !important;
        background: rgba(255,255,255,0.95) !important;
      }
      .leaflet-control-scale-line {
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
        font-size: 9px;
        border-color: #0b3d91 !important;
        background: rgba(255,255,255,0.85) !important;
      }
      .unit-map-label {
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
        font-size: 9px;
        font-weight: 600;
        letter-spacing: 0.03em;
        color: #fff;
        background: rgba(13, 17, 23, 0.88);
        border: 1px solid #0b3d91;
        border-left: 2px solid #fc3d21;
        padding: 1px 5px;
        border-radius: 2px;
        white-space: nowrap;
        box-shadow: 0 1px 3px rgba(0,0,0,0.25);
      }
      .unit-label-wrap {
        background: transparent !important;
        border: none !important;
      }
      .unit-popup {
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
        font-size: 11px;
        color: #1a2332;
      }
      .map-legend {
        position: fixed;
        bottom: 24px;
        left: 10px;
        z-index: 9999;
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
        font-size: 9px;
        line-height: 1.45;
        color: #c8d4e4;
        background: rgba(13, 17, 23, 0.92);
        padding: 6px 8px;
        border: 1px solid #8a96a8;
        border-left: 3px solid #fc3d21;
        border-radius: 3px;
        pointer-events: none;
      }
      .map-legend-title {
        color: #7eb8ff;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        font-size: 8px;
        margin-bottom: 4px;
      }
      .map-legend-meta {
        margin-top: 4px;
        color: #9aa8bc;
        font-size: 8px;
      }
      .swatch-flood { color: #2e86ab; }
      .swatch-unit { color: #fc3d21; }
    </style>
    """
    m.get_root().html.add_child(folium.Element(map_css))

    scene_label = "scene" if scene_count == 1 else "scenes"
    legend_html = f"""
    <div class="map-legend">
      <div class="map-legend-title">BWMA · {html.escape(month_label)} composite</div>
      <div><span class="swatch-flood">■</span> Flooded extent (NIR &lt; {threshold:g})</div>
      <div><span class="swatch-unit">—</span> Unit boundary</div>
      <div class="map-legend-meta">
        {html.escape(start_date)} – {html.escape(end_date)} ·
        {scene_count} {scene_label}
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(map_basename)


def main(start_date, threshold):
    # Project ID can be set via EARTH_ENGINE_PROJECT_ID environment variable
    # Falls back to default if not set
    project_id = os.getenv('EARTH_ENGINE_PROJECT_ID', 'ee-zjn-2022')
    print("Initializing Earth Engine...")
    ee.Initialize(project=project_id)

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

    # Create and Save HTML Map (use paths relative to map file so it renders from docs/reports/)
    print("Creating and saving the HTML map with overlays...")
    clipped_binary_image = binary_image.clip(units)

    false_color_basename = f"false_color_composite_{image_date_str}_{threshold}.png"
    clipped_geojson_basename = f"clipped_flooded_areas_{image_date_str}_{threshold}.geojson"
    subunits_basename = "subunits.geojson"
    map_basename = f"flooded_area_map_{image_date_str}_{threshold}.html"
    unit_features = units_with_calculations.getInfo()["features"]

    orig_cwd = os.getcwd()
    try:
        os.chdir(html_subdirectory)
        build_flood_map(
            map_basename=map_basename,
            false_color_basename=false_color_basename,
            clipped_geojson_basename=clipped_geojson_basename,
            subunits_basename=subunits_basename,
            unit_features=unit_features,
            start_date=start_date,
            end_date=end_date,
            threshold=threshold,
            scene_count=size,
        )
        print(f"Map saved to {map_filename}")
    finally:
        os.chdir(orig_cwd)

    # Define the subdirectory for CSV output
    csv_subdirectory = "flood_reports/csv_output"
    os.makedirs(csv_subdirectory, exist_ok=True)

    # Save the DataFrame to a CSV file
    print("Generating CSV report...")
    csv_filename = os.path.join(csv_subdirectory, f'flood_report_data_{image_date_str}_{threshold}.csv')
    units_df_properties_reduced.to_csv(csv_filename, index=False)
    print(f"CSV file saved to {csv_filename}")

    # Build styled HTML report table (legend unit order)
    report_table_df = units_df_properties_reduced[["Flood_Unit", "acres_flooded"]].copy()
    report_table_df = order_units_df(report_table_df)
    report_table_df.columns = ["BWMA Unit", "Acres"]
    report_table_df["Acres"] = report_table_df["Acres"].round(0).astype(int)

    csv_basename = f"flood_report_data_{image_date_str}_{threshold}.csv"
    tif_basename = f"false_color_composite_{image_date_str}_{threshold}.tif"

    html_report = build_report_html(
        table_df=report_table_df,
        image_date_str=image_date_str,
        start_date=start_date,
        end_date=end_date,
        threshold=threshold,
        scene_count=size,
        map_basename=map_basename,
        clipped_geojson_basename=clipped_geojson_basename,
        csv_basename=csv_basename,
        tif_basename=tif_basename,
    )
    print("Generating HTML report for flood data...")
    with open(report_filename, "w") as file:
        file.write(html_report)
    print(f"Report saved to {report_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate flood report based on start date and threshold.")
    parser.add_argument('start_date', type=str, help='Start date in YYYY-MM-DD format')
    parser.add_argument('threshold', type=float, help='Surface reflectance threshold value')
    args = parser.parse_args()
    main(args.start_date, args.threshold)
