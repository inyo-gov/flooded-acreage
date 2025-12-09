"""
SAR (Sentinel-1) processing utilities for flood mapping.

This module provides functions for loading, preprocessing, and analyzing
Sentinel-1 SAR data to detect flooded areas, particularly under emergent vegetation.
"""

import ee


def load_s1_collection(aoi, start_date, end_date, orbit_pass=None):
    """
    Load and preprocess Sentinel-1 GRD collection.
    
    Parameters:
    -----------
    aoi : ee.Geometry
        Area of interest geometry
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    orbit_pass : str, optional
        Orbit pass direction: 'ASCENDING' or 'DESCENDING'. If None, includes both.
    
    Returns:
    --------
    ee.ImageCollection
        Preprocessed Sentinel-1 GRD collection
    """
    col = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(aoi)
           .filterDate(start_date, end_date)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.eq("resolution_meters", 10))
           .filter(ee.Filter.eq("productType", "GRD")))
    
    # Optional: filter by orbit pass direction for consistency
    if orbit_pass:
        col = col.filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
    
    # Basic preprocessing: thermal noise removal, border noise mask
    def prep(img):
        # Crude border noise mask: remove pixels with very low backscatter
        # VV values below -50 dB are typically noise/border artifacts
        img = img.updateMask(img.select("VV").gt(-50))
        
        # GEE S1 GRD is already gamma0 backscatter in dB
        # Select VV and VH bands and preserve properties
        return (img.select(["VV", "VH"])
                    .rename(["VV", "VH"])
                    .copyProperties(img, img.propertyNames()))
    
    return col.map(prep)


def s1_composites(aoi, dry_start, dry_end, wet_start, wet_end, orbit_pass=None):
    """
    Create dry baseline and wet event composites from Sentinel-1 collection.
    
    Parameters:
    -----------
    aoi : ee.Geometry
        Area of interest geometry
    dry_start : str
        Dry baseline period start date in 'YYYY-MM-DD' format
    dry_end : str
        Dry baseline period end date in 'YYYY-MM-DD' format
    wet_start : str
        Wet/flood event period start date in 'YYYY-MM-DD' format
    wet_end : str
        Wet/flood event period end date in 'YYYY-MM-DD' format
    orbit_pass : str, optional
        Orbit pass direction: 'ASCENDING' or 'DESCENDING'
    
    Returns:
    --------
    tuple
        (dry_composite, wet_composite) as ee.Image objects
    """
    dry = (load_s1_collection(aoi, dry_start, dry_end, orbit_pass)
           .median()
           .clip(aoi))
    
    wet = (load_s1_collection(aoi, wet_start, wet_end, orbit_pass)
           .median()
           .clip(aoi))
    
    return dry, wet


def s1_temporal_ensemble(aoi, dry_start, dry_end, target_date, days_before=5, days_after=5, 
                          orbit_pass=None, dvv_thresh=-1.5, vv_vh_ratio_max=3, 
                          min_dates=2, morph_kernel_size=2):
    """
    Create flood mask using temporal ensemble from multiple adjacent dates.
    A pixel must be flagged as flooded in at least min_dates out of the available dates.
    This reduces noise and creates more stable, contiguous flood areas.
    
    Parameters:
    -----------
    aoi : ee.Geometry
        Area of interest geometry
    dry_start : str
        Dry baseline period start date in 'YYYY-MM-DD' format
    dry_end : str
        Dry baseline period end date in 'YYYY-MM-DD' format
    target_date : str
        Central date for the ensemble in 'YYYY-MM-DD' format
    days_before : int, default=5
        Number of days before target_date to include
    days_after : int, default=5
        Number of days after target_date to include
    orbit_pass : str, optional
        Orbit pass direction: 'ASCENDING' or 'DESCENDING'
    dvv_thresh : float, default=-1.5
        dVV threshold for flood detection
    vv_vh_ratio_max : float, default=3
        VV-VH ratio threshold
    min_dates : int, default=2
        Minimum number of dates (out of available) where pixel must be flooded
    morph_kernel_size : int, default=2
        Kernel size for morphological smoothing
    
    Returns:
    --------
    ee.Image
        Binary flood mask from temporal ensemble
    """
    from datetime import datetime, timedelta
    
    # Calculate date range
    target_dt = datetime.strptime(target_date, '%Y-%m-%d')
    start_dt = target_dt - timedelta(days=days_before)
    end_dt = target_dt + timedelta(days=days_after)
    
    start_date = start_dt.strftime('%Y-%m-%d')
    end_date = end_dt.strftime('%Y-%m-%d')
    
    # Load dry baseline composite
    dry_composite = (load_s1_collection(aoi, dry_start, dry_end, orbit_pass)
                     .median()
                     .clip(aoi))
    
    # Load all wet period images (not just median - we want individual dates)
    wet_collection = load_s1_collection(aoi, start_date, end_date, orbit_pass)
    
    # Create flood mask for each image in the collection
    def create_flood_mask(wet_img):
        """Create flood mask for a single wet image."""
        wet_clipped = wet_img.clip(aoi)
        indicators = s1_flood_indicators(dry_composite, wet_clipped)
        mask = s1_flood_mask(indicators, dvv_thresh=dvv_thresh, 
                            vv_vh_ratio_max=vv_vh_ratio_max,
                            morph_kernel_size=morph_kernel_size)
        return mask.select('flood_s1')
    
    # Map over collection to create masks for each date
    flood_masks = wet_collection.map(create_flood_mask)
    
    # Count how many dates each pixel is flooded
    flood_count = flood_masks.sum()
    
    # Require flooding in at least min_dates
    # But also need to handle case where collection might be empty or have fewer images
    collection_size = flood_masks.size()
    ensemble_mask = flood_count.gte(ee.Number(min_dates).min(collection_size))
    
    return ensemble_mask.rename("flood_s1")


def s1_flood_indicators(dry_img, wet_img):
    """
    Derive SAR flood indicators from dry and wet composites.
    
    Flooded emergent vegetation typically shows:
    - VV drops (darker, specular reflection over water)
    - VH often increases due to double-bounce, or behaves differently
    
    Parameters:
    -----------
    dry_img : ee.Image
        Dry baseline composite with VV and VH bands
    wet_img : ee.Image
        Wet/flood event composite with VV and VH bands
    
    Returns:
    --------
    ee.Image
        Image with additional bands: dVV, dVH, VV_VH_ratio
    """
    vv_dry = dry_img.select("VV")
    vv_wet = wet_img.select("VV")
    vh_dry = dry_img.select("VH")
    vh_wet = wet_img.select("VH")
    
    # Difference images (in dB)
    dvv = vv_wet.subtract(vv_dry).rename("dVV")  # Expect negative where flooded
    dvh = vh_wet.subtract(vh_dry).rename("dVH")
    
    # Ratio: VV - VH (in dB space, subtraction is equivalent to ratio)
    vv_vh_ratio = vv_wet.subtract(vh_wet).rename("VV_VH_ratio")
    
    # Add indicators as bands to the wet image
    return wet_img.addBands([dvv, dvh, vv_vh_ratio])


def s1_absolute_threshold(wet_img, vv_thresh=-18, vh_thresh=-22, 
                          morph_kernel_size=2, dry_img=None, 
                          exclude_permanent_water=True):
    """
    Create flood mask using absolute backscatter thresholds.
    Optionally uses dry baseline to exclude permanent water and areas that are always dark.
    
    Parameters:
    -----------
    wet_img : ee.Image
        Wet/flood period SAR image with VV and VH bands
    vv_thresh : float, default=-18
        VV backscatter threshold in dB (pixels below this are water)
    vh_thresh : float, default=-22
        VH backscatter threshold in dB (pixels below this are water)
    morph_kernel_size : int, default=2
        Kernel radius for morphological smoothing
    dry_img : ee.Image, optional
        Dry baseline composite. If provided, used to exclude permanent water
    exclude_permanent_water : bool, default=True
        If True and dry_img provided, exclude areas that are also dark in dry period
        (permanent water, shadows, etc.)
    
    Returns:
    --------
    ee.Image
        Binary flood mask (1 = flooded, 0 = not flooded)
    """
    vv_wet = wet_img.select("VV")
    vh_wet = wet_img.select("VH")
    
    # Water has low backscatter - require both VV and VH below thresholds
    flooded = vv_wet.lt(vv_thresh).And(vh_wet.lt(vh_thresh))
    
    # If dry baseline provided, exclude permanent water/shadow areas
    if dry_img is not None and exclude_permanent_water:
        vv_dry = dry_img.select("VV")
        vh_dry = dry_img.select("VH")
        
        # Areas that are also dark in dry period are likely permanent water/shadows
        # Only keep areas that are dark in wet but NOT dark in dry (new flooding)
        permanent_water = vv_dry.lt(vv_thresh).And(vh_dry.lt(vh_thresh))
        flooded = flooded.And(permanent_water.Not())
    
    # Apply morphological operations to create contiguous, pooled areas
    # Use a smaller kernel first to preserve narrow linear features like canals
    kernel_small = ee.Kernel.circle(radius=1, units='pixels')
    kernel_large = ee.Kernel.circle(radius=morph_kernel_size, units='pixels')
    
    # First pass: light dilation to connect nearby pixels (helps with narrow canals)
    flooded_dilated_light = flooded.focalMax(kernel=kernel_small, iterations=1)
    
    # Second pass: stronger dilation for larger areas
    flooded_dilated = flooded_dilated_light.focalMax(kernel=kernel_large, iterations=1)
    
    # Erode slightly to remove very small isolated pixels, but preserve linear features
    flooded_smoothed = flooded_dilated.focalMin(
        kernel=ee.Kernel.circle(radius=1, units='pixels'), 
        iterations=1
    )
    
    return flooded_smoothed.rename("flood_s1")


def s1_flood_mask(s1_indicators, dvv_thresh=-1.5, vv_vh_ratio_max=3, 
                  morph_kernel_size=2):
    """
    Create binary flood mask from SAR flood indicators using thresholds.
    Applies morphological operations to create contiguous, pooled areas.
    
    Parameters:
    -----------
    s1_indicators : ee.Image
        Image with dVV, dVH, and VV_VH_ratio bands from s1_flood_indicators()
    dvv_thresh : float, default=-1.5
        Threshold for dVV (VV must drop at least this many dB to be considered flooded)
    vv_vh_ratio_max : float, default=3
        Maximum VV-VH difference (in dB) for flooded areas
        Lower values indicate more cross-pol relative to VV (typical of flooded vegetation)
    morph_kernel_size : int, default=2
        Kernel radius (in pixels) for morphological operations to connect nearby flooded pixels.
        Larger values create more contiguous, pooled areas but may over-connect.
    
    Returns:
    --------
    ee.Image
        Binary flood mask (1 = flooded, 0 = not flooded) with morphological smoothing
    """
    dvv = s1_indicators.select("dVV")
    ratio = s1_indicators.select("VV_VH_ratio")
    
    # Flooded areas: VV drops significantly AND cross-pol is relatively higher
    flooded = (dvv.lt(dvv_thresh)
               .And(ratio.lt(vv_vh_ratio_max)))
    
    # Apply morphological operations to create contiguous, pooled areas
    # Use a circular kernel to connect nearby flooded pixels
    kernel = ee.Kernel.circle(radius=morph_kernel_size, units='pixels')
    
    # Dilate to connect nearby flooded pixels (creates pooled appearance)
    flooded_dilated = flooded.focalMax(kernel=kernel, iterations=1)
    
    # Erode slightly to remove very small isolated pixels while keeping connected areas
    flooded_smoothed = flooded_dilated.focalMin(
        kernel=ee.Kernel.circle(radius=1, units='pixels'), 
        iterations=1
    )
    
    return flooded_smoothed.rename("flood_s1")
