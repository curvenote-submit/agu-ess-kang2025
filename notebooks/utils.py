# -----------------
# Helper functions: 
# -----------------
from PIL import Image
from io import BytesIO
import base64
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import geopandas as gpd


# Convert bounds convention from rioxarray to ipyleaflet
def leaflet_bounds(da):
    """
    Convert rioxarray bounds to ipyleaflet bounds format.
    
    Parameters:
    - da: rioxarray DataArray
        The input raster data.
    
    Returns:
    - bounds: list
        Bounds in ipyleaflet format [[min_lat, min_lon], [max_lat, max_lon]].
    """
    rio_bounds = da.rio.bounds()  # (left, bottom, right, top)
    return [[rio_bounds[1], rio_bounds[0]], [rio_bounds[3], rio_bounds[2]]]

# Convert RGBA DataArray to base64 image

def transpose_da(da):
    """Transpose DataArray from any order to (..., y, x)"""
    dims = da.dims
    other_dims = [d for d in dims if d not in ('y', 'x')]
    new_order = other_dims + ['y', 'x']
    return da.transpose(*new_order)

# -------------------------------
# Helper: convert scalar DataArray to colormap overlay
# -------------------------------

def scalar_to_base64_image(da, cmap='viridis', vmin=None, vmax=None):
    """Convert scalar DataArray to colored base64 PNG with transparency for masked values"""
    da = transpose_da(da)
    arr = da.values
    
    if vmin is None:
        vmin = np.nanmin(arr)
    if vmax is None:
        vmax = np.nanmax(arr)
            
    norm = Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    rgba = mapper.to_rgba(arr, bytes=False)  # shape: (y, x, 4)
    
    mask = np.isnan(arr)
    rgba[mask, 3] = 0
    
    rgba_uint8 = (rgba * 255).astype(np.uint8)
    
    img = Image.fromarray(rgba_uint8, mode='RGBA')
    
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    return img_str


def find_intersections(river_gdf, basin_gdf):
    """
    Find easternmost intersection points where rivers cross basin boundaries.
    
    Parameters:
    -----------
    river_gdf : GeoDataFrame with 'name' and 'geometry' (LineString)
    basin_gdf : GeoDataFrame with polygon geometries
    
    Returns:
    --------
    GeoDataFrame with columns: 'river_name', 'geometry' (Point)
    """
    # Get outer boundary of all basins
    outer_boundary = basin_gdf.geometry.unary_union.boundary
    
    # Find intersections
    intersections = []
    for idx, river in river_gdf.iterrows():
        intersection = river.geometry.intersection(outer_boundary)
        if not intersection.is_empty:
            # Collect points
            points = []
            if intersection.geom_type == 'MultiPoint':
                points = list(intersection.geoms)
            elif intersection.geom_type == 'Point':
                points = [intersection]
            
            # Keep easternmost (highest x)
            if points:
                easternmost = max(points, key=lambda p: p.x)
                intersections.append({
                    'river_name': river['GNIS_Name'],
                    'geometry': easternmost
                })
    
    return gpd.GeoDataFrame(intersections, crs=river_gdf.crs)