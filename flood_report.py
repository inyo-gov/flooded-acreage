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
import shutil
import subprocess
import json
import tempfile

# Example terminal command
# > python flood_report.py '2024-10-22' .22
# > python flood_report.py '2024-11-01' .22
# Single clear scene after wet-up (does not replace the 15-day composite):
# > python flood_report.py 2026-09-20 0.14 --mode latest-scene

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
    product="composite",
    cloud_note="",
):
    month_label = pd.to_datetime(start_date).strftime("%b %Y")
    if product == "scene":
        report_tag = "Sentinel-2 SR · Single scene (not a median composite)"
        report_title = f"BWMA single-scene flood report · {image_date_str}"
        page_title = report_title
        report_meta = (
            f"Scene date <strong>{html.escape(image_date_str)}</strong>"
            f" · NIR threshold <strong>{threshold:g}</strong>"
            f" · one acquisition, not a 15-day composite"
        )
        if cloud_note:
            report_meta += f"<br>{html.escape(cloud_note)}"
        summary_note = "Single scene · release calibration"
        footer_note = (
            "Sentinel-2 surface reflectance (harmonized). "
            "NIR threshold applied to this one cloud-masked scene — not a median composite."
        )
    else:
        scene_word = "scene" if scene_count == 1 else "scenes"
        report_tag = "Sentinel-2 SR · Cloud-masked median composite"
        report_title = f"BWMA Flood Report · {month_label}"
        page_title = report_title
        report_meta = (
            f"<strong>{html.escape(start_date)}</strong> to <strong>{html.escape(end_date)}</strong>"
            f" · NIR threshold <strong>{threshold:g}</strong>"
            f" · <strong>{scene_count}</strong> {scene_word} in composite"
            f" · label date {html.escape(image_date_str)}"
        )
        summary_note = "Composite window: 15 days"
        footer_note = (
            "Sentinel-2 surface reflectance (harmonized). "
            "NIR band threshold applied to a cloud-masked median composite—not a single acquisition."
        )
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
  <title>{html.escape(page_title)}</title>
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
      <div class="report-tag">{html.escape(report_tag)}</div>
      <h1>{html.escape(report_title)}</h1>
      <p class="report-meta">
        {report_meta}
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
          <span>{html.escape(summary_note)}</span>
        </div>
      </section>

      <section class="report-panel report-map-panel">
        <h2>Spatial extent</h2>
        <iframe src="./{html.escape(map_basename)}" title="BWMA flooded extent map"></iframe>
      </section>
    </div>

    <footer class="report-footer">
      <h3>Data products</h3>
      <p>{html.escape(footer_note)}</p>
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


AOI_BBOX_BOUNDS = [
    [36.84651455123723, -118.23240736400778],
    [36.924364295139625, -118.17232588207419],
]


def _gdal_exe(name):
    """Resolve gdalwarp/gdalinfo from PATH or known conda envs on this Mac."""
    found = shutil.which(name)
    if found:
        return found
    for base in (
        "/opt/homebrew/Caskroom/miniconda/base/envs/lidar/bin",
        "/opt/homebrew/Caskroom/miniconda/base/envs/ee-tools/bin",
        "/opt/homebrew/Caskroom/miniconda/base/bin",
    ):
        cand = os.path.join(base, name)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def prepare_folium_overlay_png(tif_path, png_path):
    """Reproject a GeoTIFF to EPSG:4326 for Folium ImageOverlay.

    geemap/EE exports are typically UTM-aligned. Folium ImageOverlay only
    accepts an axis-aligned lon/lat rectangle, so stretching a UTM PNG onto the
    AOI lon/lat box misplaces the imagery relative to GeoJSON (tens of meters
    inside BWMA; >100 m at corners). Warping to WGS84 before PNG export fixes
    that real overlay bug.

    Returns Folium bounds [[south, west], [north, east]].
    """
    gdalwarp = _gdal_exe("gdalwarp")
    gdalinfo = _gdal_exe("gdalinfo")
    if not gdalwarp or not gdalinfo:
        raise RuntimeError(
            "gdalwarp/gdalinfo required to build a WGS84 Folium overlay PNG "
            "(install GDAL or use the lidar/ee-tools conda env)."
        )

    os.makedirs(os.path.dirname(os.path.abspath(png_path)) or ".", exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bwma_overlay_") as tmp:
        warped = os.path.join(tmp, "wgs84.tif")
        cmd = [
            gdalwarp,
            "-t_srs", "EPSG:4326",
            "-r", "bilinear",
            "-dstalpha",
            "-overwrite",
            tif_path,
            warped,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        info = subprocess.run(
            [gdalinfo, "-json", warped],
            check=True, capture_output=True, text=True,
        )
        meta = json.loads(info.stdout)
        # cornerCoordinates: upperLeft/lowerRight etc as [lon, lat]
        corners = meta["cornerCoordinates"]
        ul = corners["upperLeft"]
        lr = corners["lowerRight"]
        west = min(ul[0], lr[0], corners["upperRight"][0], corners["lowerLeft"][0])
        east = max(ul[0], lr[0], corners["upperRight"][0], corners["lowerLeft"][0])
        south = min(ul[1], lr[1], corners["upperRight"][1], corners["lowerLeft"][1])
        north = max(ul[1], lr[1], corners["upperRight"][1], corners["lowerLeft"][1])
        arr = imageio.imread(warped)
        # Drop alpha if present for a simple RGB PNG the map already expects.
        if arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[:, :, :3]
        imageio.imwrite(png_path, arr)

    bounds = [[south, west], [north, east]]
    print(
        f"Folium overlay PNG (EPSG:4326) saved at {png_path} "
        f"bounds S/W–N/E {south:.6f}/{west:.6f}–{north:.6f}/{east:.6f}"
    )
    return bounds


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
    product="composite",
    overlay_bounds=None,
):
    """Build a telemetry-themed folium map for the monthly report iframe."""
    # Image overlay must use the PNG's true WGS84 bounds (see prepare_folium_overlay_png).
    # fit_bounds can stay on the AOI box so the view framing stays consistent.
    bbox_bounds = AOI_BBOX_BOUNDS
    image_bounds = overlay_bounds or AOI_BBOX_BOUNDS
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

    overlay_name = "S2 false-color scene" if product == "scene" else "S2 false-color composite"
    folium.raster_layers.ImageOverlay(
        name=overlay_name,
        image=false_color_basename,
        bounds=image_bounds,
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
    if product == "scene":
        legend_title = f"BWMA · {start_date} single scene"
        legend_meta = f"{html.escape(start_date)} · 1 scene · not a composite"
    else:
        legend_title = f"BWMA · {month_label} composite"
        legend_meta = (
            f"{html.escape(start_date)} – {html.escape(end_date)} · "
            f"{scene_count} {scene_label}"
        )
    legend_html = f"""
    <div class="map-legend">
      <div class="map-legend-title">{html.escape(legend_title)}</div>
      <div><span class="swatch-flood">■</span> Flooded extent (NIR &lt; {threshold:g})</div>
      <div><span class="swatch-unit">—</span> Unit boundary</div>
      <div class="map-legend-meta">
        {legend_meta}
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(map_basename)


# SCL values counted as cloud when scoring a scene over the BWMA box.
# 8 medium cloud, 9 high cloud, 10 thin cirrus.
SCENE_CLOUD_SCL = (8, 9, 10)
# Removed before the single-scene NIR test (composites still mask only SCL 9).
# 0 no data, 3 cloud shadow, 8/9/10 cloud and cirrus.
SCENE_MASK_SCL = (0, 3, 8, 9, 10)


def select_latest_clear_scene(collection, geometry, max_cloud_frac, scale=20):
    """Newest Sentinel-2 scene whose BWMA-box cloud fraction is within the limit.

    Cloud rule (acceptance, not the granule metadata):
      aoi_cloud_frac = count(SCL in {8, 9, 10}) / count(SCL != 0)
      computed inside ``geometry`` at ``scale`` meters.
    Usable when valid pixels > 0 and aoi_cloud_frac <= max_cloud_frac.
    Among usable scenes, the greatest system:time_start wins.
    Returns (chosen_props_or_None, all_rows_newest_first).
    """

    def annotate(img):
        scl = img.select("SCL")
        valid = scl.neq(0).rename("valid")
        cloudy = scl.eq(8).Or(scl.eq(9)).Or(scl.eq(10)).rename("cloudy")
        stats = valid.addBands(cloudy).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=scale,
            maxPixels=1e8,
        )
        valid_n = ee.Number(ee.Algorithms.If(stats.contains("valid"), stats.get("valid"), 0))
        cloudy_n = ee.Number(ee.Algorithms.If(stats.contains("cloudy"), stats.get("cloudy"), 0))
        frac = cloudy_n.divide(valid_n.max(1))
        return img.set({
            "scene_index": img.get("system:index"),
            "scene_time": img.get("system:time_start"),
            "scene_date": img.date().format("YYYY-MM-dd"),
            "aoi_cloud_frac": frac,
            "valid_pixels": valid_n,
            "cloudy_pixels": cloudy_n,
            "granule_cloud_pct": img.get("CLOUDY_PIXEL_PERCENTAGE"),
        })

    annotated = collection.map(annotate)

    def props_feature(img):
        img = ee.Image(img)
        return ee.Feature(None, img.toDictionary([
            "scene_index",
            "scene_time",
            "scene_date",
            "aoi_cloud_frac",
            "valid_pixels",
            "cloudy_pixels",
            "granule_cloud_pct",
        ]))

    info = ee.FeatureCollection(annotated.toList(200).map(props_feature)).getInfo()
    rows = [f["properties"] for f in info.get("features", [])]
    rows.sort(key=lambda r: r.get("scene_time") or 0, reverse=True)
    usable = [
        r for r in rows
        if (r.get("valid_pixels") or 0) > 0
        and r.get("aoi_cloud_frac") is not None
        and float(r["aoi_cloud_frac"]) <= max_cloud_frac
    ]
    return (usable[0] if usable else None), rows


def _print_scene_candidates(rows, max_cloud_frac):
    print(f"Scene candidates (usable if AOI cloud fraction <= {max_cloud_frac:.0%}):")
    print(f"{'date':<12} {'aoi_cloud':>10} {'granule_%':>10} {'valid_px':>10} index")
    for r in rows:
        frac = r.get("aoi_cloud_frac")
        frac_s = f"{float(frac):.1%}" if frac is not None else "n/a"
        gran = r.get("granule_cloud_pct")
        gran_s = f"{float(gran):.1f}" if gran is not None else "n/a"
        print(
            f"{r.get('scene_date',''):<12} {frac_s:>10} {gran_s:>10} "
            f"{int(r.get('valid_pixels') or 0):>10} {r.get('scene_index')}"
        )


def main(start_date, threshold, mode="composite", end_date=None, max_cloud_frac=0.20):
    if mode not in ("composite", "latest-scene"):
        raise SystemExit(f"Unknown mode: {mode}")

    # Project ID can be set via EARTH_ENGINE_PROJECT_ID environment variable
    # Falls back to default if not set
    project_id = os.getenv('EARTH_ENGINE_PROJECT_ID', 'ee-zjn-2022')
    print("Initializing Earth Engine...")
    ee.Initialize(project=project_id)

    search_start = start_date
    search_end = end_date
    scene_cloud_frac = None
    scene_id = None
    granule_cloud_pct = None
    cloud_note = ""
    product = "composite"
    name_prefix = ""

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
    if mode == "latest-scene":
        # Granule CLOUDY_PIXEL_PERCENTAGE is only a cheap prefilter (whole tile).
        # Acceptance is the BWMA-box SCL fraction inside select_latest_clear_scene.
        search_end = search_end or datetime.now().strftime("%Y-%m-%d")
        filter_end = (datetime.strptime(search_end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"Latest-scene search: {search_start} through {search_end} (EE filter end {filter_end})")
        sentinel_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(bounding_box_geometry) \
            .filterDate(search_start, filter_end) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 90)) \
            .select(['B4','B8', 'SCL','B11'])
        size = sentinel_collection.size().getInfo()
        print(f"Number of images after granule prefilter: {size}")
        if size == 0:
            raise SystemExit("No Sentinel-2 scenes in the latest-scene window.")
        chosen, rows = select_latest_clear_scene(
            sentinel_collection, bounding_box_geometry, max_cloud_frac
        )
        _print_scene_candidates(rows, max_cloud_frac)
        if chosen is None:
            raise SystemExit(
                "No usable scene: none had AOI cloud fraction "
                f"<= {max_cloud_frac:.0%} over the BWMA box."
            )
        scene_id = chosen["scene_index"]
        image_date_str = chosen["scene_date"]
        scene_cloud_frac = float(chosen["aoi_cloud_frac"])
        granule_cloud_pct = chosen.get("granule_cloud_pct")
        print(
            f"Selected scene {scene_id} date {image_date_str} "
            f"AOI cloud {scene_cloud_frac:.1%}"
        )
        image = sentinel_collection.filter(ee.Filter.eq("system:index", scene_id)).first()
        scl = image.select("SCL")
        bad = (
            scl.eq(0).Or(scl.eq(3)).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
        )
        cloud_free_composite = image.updateMask(bad.Not())
        pixel_size = image.select("B8").projection().nominalScale().getInfo()
        size = 1
        product = "scene"
        name_prefix = "scene_"
        # Report labels use the true acquisition date, not the search window.
        start_date = image_date_str
        end_date = image_date_str
        gran_txt = (
            f"{float(granule_cloud_pct):.1f}%" if granule_cloud_pct is not None else "n/a"
        )
        cloud_note = (
            f"AOI cloud {scene_cloud_frac:.1%} "
            f"(SCL 8/9/10 over SCL≠0, 20 m, BWMA bbox; usable if ≤ {max_cloud_frac:.0%}). "
            f"Granule CLOUDY_PIXEL_PERCENTAGE {gran_txt} (prefilter only). "
            f"Search {search_start} to {search_end}. "
            "Flood mask drops SCL 0/3/8/9/10 before the NIR test."
        )
    else:
        end_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=15)).strftime("%Y-%m-%d")
        print(f"Date range: {start_date} to {end_date}")
        search_end = end_date
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
        cloud_free_composite = sentinel_collection.map(lambda img: img.updateMask(img.select('SCL').neq(9))).median()

    # Define the subdirectory for HTML maps and reports
    print("Setting up directories and filenames...")
    html_subdirectory = "flood_reports/reports"
    os.makedirs(html_subdirectory, exist_ok=True)

    # Define filenames with unique names based on the image date
    report_filename = os.path.join(html_subdirectory, f"bwma_flood_report_{name_prefix}{image_date_str}_{threshold}.html")
    map_filename = os.path.join(html_subdirectory, f"flooded_area_map_{name_prefix}{image_date_str}_{threshold}.html")

    def mask_clouds(image):
        cloud_prob = image.select('SCL')
        is_cloud = cloud_prob.eq(9)
        return image.updateMask(is_cloud.Not())

    # Mask clouds and compute composite (latest-scene already set cloud_free_composite)
    print("Creating flood mask...")
    if mode != "latest-scene":
        # composite branch assigns cloud_free_composite above; keep this guard so a
        # refactor cannot median-blend a single-scene run.
        pass
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
    false_color_filename_tif = f"flood_reports/reports/false_color_composite_{name_prefix}{image_date_str}_{threshold}.tif"
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

    false_color_filename_png = f"flood_reports/reports/false_color_composite_{name_prefix}{image_date_str}_{threshold}.png"
    # flooded_pixels_filename_png = f"docs/reports/flooded_pixels_{image_date_str}_{threshold}.png"

    # UTM (or other projected) GeoTIFF stays the downloadable product; Folium needs
    # a WGS84-aligned PNG + matching bounds or the unit outlines look offset.
    overlay_bounds = prepare_folium_overlay_png(
        false_color_filename_tif, false_color_filename_png
    )

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
    clipped_flooded_geojson_path = f"flood_reports/reports/clipped_flooded_areas_{name_prefix}{image_date_str}_{threshold}.geojson"
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

    # Export subunit polygons as GeoJSON.
    # latest-scene reuses the shared subunits.geojson already published with composites
    # so this run does not rewrite that file.
    if mode != "latest-scene":
        print("Exporting subunit polygons as GeoJSON...")
        geemap.ee_export_vector(units_with_calculations, filename="flood_reports/reports/subunits.geojson")
        print("Subunits exported as GeoJSON.")
    else:
        print("Skipping shared subunits.geojson export (latest-scene mode).")

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

    false_color_basename = f"false_color_composite_{name_prefix}{image_date_str}_{threshold}.png"
    clipped_geojson_basename = f"clipped_flooded_areas_{name_prefix}{image_date_str}_{threshold}.geojson"
    subunits_basename = "subunits.geojson"
    map_basename = f"flooded_area_map_{name_prefix}{image_date_str}_{threshold}.html"
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
            product=product,
            overlay_bounds=overlay_bounds,
        )
        print(f"Map saved to {map_filename}")
    finally:
        os.chdir(orig_cwd)

    # Define the subdirectory for CSV output
    csv_subdirectory = "flood_reports/csv_output"
    os.makedirs(csv_subdirectory, exist_ok=True)

    # Save the DataFrame to a CSV file
    print("Generating CSV report...")
    csv_filename = os.path.join(csv_subdirectory, f'flood_report_data_{name_prefix}{image_date_str}_{threshold}.csv')
    units_df_properties_reduced.to_csv(csv_filename, index=False)
    print(f"CSV file saved to {csv_filename}")

    # Build styled HTML report table (legend unit order)
    report_table_df = units_df_properties_reduced[["Flood_Unit", "acres_flooded"]].copy()
    report_table_df = order_units_df(report_table_df)
    report_table_df.columns = ["BWMA Unit", "Acres"]
    report_table_df["Acres"] = report_table_df["Acres"].round(0).astype(int)

    csv_basename = f"flood_report_data_{name_prefix}{image_date_str}_{threshold}.csv"
    tif_basename = f"false_color_composite_{name_prefix}{image_date_str}_{threshold}.tif"

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
        product=product,
        cloud_note=cloud_note,
    )
    print("Generating HTML report for flood data...")
    with open(report_filename, "w") as file:
        file.write(html_report)
    print(f"Report saved to {report_filename}")

    if mode == "latest-scene":
        import json
        import shutil
        payload = {
            "product": "latest-usable-scene",
            "wetup_start": search_start,
            "search_end": search_end,
            "scene_date": image_date_str,
            "scene_id": scene_id,
            "threshold": threshold,
            "acres": float(total_flooded_acres),
            "aoi_cloud_frac": scene_cloud_frac,
            "granule_cloud_pct": None if granule_cloud_pct is None else float(granule_cloud_pct),
            "max_cloud_frac": max_cloud_frac,
            "lead": (
                "Newest clear Sentinel-2 pass over BWMA since wet-up began — "
                "for release calibration, not the monthly composite."
            ),
            "method_details": (
                "Prefilter: granule CLOUDY_PIXEL_PERCENTAGE < 90 (whole Sentinel-2 tile, not the AOI). "
                "Acceptance: over the BWMA bounding box at 20 m, "
                "aoi_cloud_frac = count(SCL in {8,9,10}) / count(SCL != 0) must be <= 20%. "
                "The usable scene with the latest system:time_start is kept. "
                "Acreage is NIR reflectance < threshold on that single scene after masking "
                "SCL in {0,3,8,9,10} (no data, cloud shadow, medium cloud, high cloud, cirrus). "
                "This is not a 15-day median composite."
            ),
            # Kept for older dashboard renders that still read cloud_rule.
            "cloud_rule": (
                "Newest clear Sentinel-2 pass over BWMA since wet-up began — "
                "for release calibration, not the monthly composite."
            ),
            "report_href": report_filename.replace(chr(92), "/"),
            "map_note": (
                "False-color PNG for the Folium map is warped to EPSG:4326 so the "
                "image overlay matches unit GeoJSON; downloadable GeoTIFF stays in "
                "the EE export CRS (typically UTM 11N)."
            ),
        }

        os.makedirs("includes", exist_ok=True)
        sidecar = "includes/latest_scene.json"
        with open(sidecar, "w") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        print(f"Wrote {sidecar}")

        docs_reports = "docs/flood_reports/reports"
        docs_csv = "docs/flood_reports/csv_output"
        os.makedirs(docs_reports, exist_ok=True)
        os.makedirs(docs_csv, exist_ok=True)
        copies = [
            (report_filename, os.path.join(docs_reports, os.path.basename(report_filename))),
            (map_filename, os.path.join(docs_reports, os.path.basename(map_filename))),
            (clipped_flooded_geojson_path, os.path.join(docs_reports, os.path.basename(clipped_flooded_geojson_path))),
            (false_color_filename_png, os.path.join(docs_reports, os.path.basename(false_color_filename_png))),
            (false_color_filename_tif, os.path.join(docs_reports, os.path.basename(false_color_filename_tif))),
            (csv_filename, os.path.join(docs_csv, os.path.basename(csv_filename))),
        ]
        for src, dst in copies:
            shutil.copy2(src, dst)
            print(f"Copied {src} -> {dst}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate flood report based on start date and threshold.")
    parser.add_argument('start_date', type=str, help='Start date in YYYY-MM-DD format. For --mode latest-scene this is the search start (wet-up), not the scene date.')
    parser.add_argument('threshold', type=float, help='Surface reflectance threshold value')
    parser.add_argument(
        '--mode',
        choices=('composite', 'latest-scene'),
        default='composite',
        help='composite = 15-day cloud-masked median (default). latest-scene = newest low-cloud single scene from start_date through today.',
    )
    parser.add_argument(
        '--end-date',
        default=None,
        help='Inclusive search end for --mode latest-scene (YYYY-MM-DD). Default: today.',
    )
    parser.add_argument(
        '--max-cloud-frac',
        type=float,
        default=0.20,
        help='Max AOI cloud fraction (SCL 8/9/10) for a usable scene. Default 0.20.',
    )
    args = parser.parse_args()
    main(
        args.start_date,
        args.threshold,
        mode=args.mode,
        end_date=args.end_date,
        max_cloud_frac=args.max_cloud_frac,
    )
