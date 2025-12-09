"""
DEM (Digital Elevation Model) utilities for flood mapping.

This module provides functions for loading DEMs, creating slope masks,
and performing relative elevation analysis. These utilities are designed
to be integrated after field data collection informs the specific needs.
"""

import ee


def load_dem(dem_source='SRTM', aoi=None):
    """
    Load DEM from specified source.
    
    Parameters:
    -----------
    dem_source : str, default='SRTM'
        DEM source: 'SRTM', 'ALOS', or '3DEP'
    aoi : ee.Geometry, optional
        Area of interest to clip DEM
    
    Returns:
    --------
    ee.Image
        DEM image in meters
    """
    if dem_source == 'SRTM':
        dem = ee.Image("USGS/SRTMGL1_003")
    elif dem_source == 'ALOS':
        dem = ee.Image("JAXA/ALOS/AW3D30/V3_2").select('AVE')
    elif dem_source == '3DEP':
        # USGS 3DEP - check availability for specific region
        # This may need to be adjusted based on actual GEE catalog
        dem = ee.ImageCollection("USGS/3DEP/10m").mosaic()
    else:
        raise ValueError(f"Unknown DEM source: {dem_source}. Use 'SRTM', 'ALOS', or '3DEP'.")
    
    if aoi:
        dem = dem.clip(aoi)
    
    return dem


def slope_mask(dem, max_slope_deg=3):
    """
    Create binary mask for acceptable slopes (removes steep terrain).
    
    Parameters:
    -----------
    dem : ee.Image
        DEM image
    max_slope_deg : float, default=3
        Maximum slope in degrees for acceptable areas
    
    Returns:
    --------
    ee.Image
        Binary mask (1 = acceptable slope, 0 = too steep)
    """
    slope = ee.Terrain.slope(dem)
    return slope.lt(max_slope_deg).rename("slope_ok")


def apply_dem_mask(flood_mask, dem_source=None, max_slope_deg=3, aoi=None):
    """
    Apply DEM-based slope mask to flood mask.
    
    Parameters:
    -----------
    flood_mask : ee.Image
        Binary flood mask
    dem_source : str, optional
        DEM source ('SRTM', 'ALOS', '3DEP'). If None, returns original mask.
    max_slope_deg : float, default=3
        Maximum acceptable slope in degrees
    aoi : ee.Geometry, optional
        Area of interest
    
    Returns:
    --------
    ee.Image
        Flood mask with DEM slope mask applied
    """
    if dem_source is None:
        return flood_mask
    
    dem = load_dem(dem_source, aoi=aoi)
    slope_mask_img = slope_mask(dem, max_slope_deg=max_slope_deg)
    
    return flood_mask.updateMask(slope_mask_img)


def relative_elevation_to_drainage(dem, drainage_network=None):
    """
    Calculate relative elevation to nearest drainage.
    
    This is a placeholder for future implementation after field data collection
    informs the specific drainage network and analysis needs.
    
    Parameters:
    -----------
    dem : ee.Image
        DEM image
    drainage_network : ee.FeatureCollection, optional
        Drainage network features
    
    Returns:
    --------
    ee.Image
        Relative elevation to nearest drainage (placeholder)
    """
    # TODO: Implement after field data collection and drainage network definition
    # This would involve:
    # 1. Identifying drainage network (from DEM flow accumulation or provided features)
    # 2. Calculating distance to nearest drainage
    # 3. Computing relative elevation (elevation - drainage elevation)
    raise NotImplementedError("Relative elevation analysis to be implemented after field data collection")


def local_relief_map(dem, kernel_size=3):
    """
    Create local relief visualization.
    
    This is a placeholder for future implementation to visualize local
    topographic variation relative to drainage.
    
    Parameters:
    -----------
    dem : ee.Image
        DEM image
    kernel_size : int, default=3
        Kernel size for local relief calculation
    
    Returns:
    --------
    ee.Image
        Local relief map (placeholder)
    """
    # TODO: Implement after field data collection informs specific needs
    # This could use focal statistics to compute local relief
    raise NotImplementedError("Local relief mapping to be implemented after field data collection")
