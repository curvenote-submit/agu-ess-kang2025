"""
Utility functions for interactive geospatial visualization.

This module provides helper functions for converting geospatial data
into formats compatible with ipyleaflet interactive maps.
"""

# -------------------------------
# Imports
# -------------------------------
import base64
from io import BytesIO

import geopandas as gpd
import matplotlib.cm as cm
import numpy as np
from matplotlib.colors import Normalize
from PIL import Image
import ipywidgets as widgets


# -------------------------------
# Coordinate System Conversions
# -------------------------------


def leaflet_bounds(da):
    """
    Convert rioxarray bounds to ipyleaflet bounds format.

    Transforms raster bounds from (left, bottom, right, top) format
    to ipyleaflet's [[south, west], [north, east]] format.

    Parameters
    ----------
    da : rioxarray.DataArray
        Raster data with spatial coordinates and CRS information.

    Returns
    -------
    list of list of float
        Bounds in ipyleaflet format: [[min_lat, min_lon], [max_lat, max_lon]]

    Example
    -------
    >>> bounds = leaflet_bounds(raster_data)
    >>> image_overlay = ImageOverlay(url=image_url, bounds=bounds)
    """
    rio_bounds = da.rio.bounds()  # (left, bottom, right, top)
    return [[rio_bounds[1], rio_bounds[0]], [rio_bounds[3], rio_bounds[2]]]


def transpose_da(da):
    """
    Transpose DataArray to ensure spatial dimensions are last.

    Reorders dimensions to (..., y, x) format, which is required
    for proper image rendering in map overlays.

    Parameters
    ----------
    da : xarray.DataArray
        Input array with arbitrary dimension order.

    Returns
    -------
    xarray.DataArray
        Array with spatial dimensions (y, x) at the end.
    """
    dims = da.dims
    other_dims = [d for d in dims if d not in ("y", "x")]
    new_order = other_dims + ["y", "x"]
    return da.transpose(*new_order)


# -------------------------------
# Image Conversion
# -------------------------------


def scalar_to_base64_image(da, cmap, vmin, vmax):
    """
    Convert scalar raster data to base64-encoded PNG image with colormap.

    Creates a colored image overlay suitable for ipyleaflet ImageOverlay,
    with transparency for NaN/masked values.

    Parameters
    ----------
    da : xarray.DataArray
        2D or 3D raster data to visualize.
    cmap : str
        Matplotlib colormap name (e.g., 'viridis', 'RdBu_r', 'Greens').
    vmin : float
        Minimum value for color scaling. Defaults to data minimum.
    vmax : float
        Maximum value for color scaling. Defaults to data maximum.

    Returns
    -------
    str
        Base64-encoded PNG image as data URI string.

    Example
    -------
    >>> img_url = scalar_to_base64_image(temperature_data, cmap='RdBu_r')
    >>> overlay = ImageOverlay(url=img_url, bounds=bounds, opacity=0.7)
    """
    # Ensure spatial dimensions are last
    da = transpose_da(da)
    arr = da.values

    # Apply colormap to data
    norm = Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba = mapper.to_rgba(arr, bytes=False)  # shape: (y, x, 4)

    # Make NaN values transparent
    mask = np.isnan(arr)
    rgba[mask, 3] = 0

    # Convert to 8-bit image
    rgba_uint8 = (rgba * 255).astype(np.uint8)
    img = Image.fromarray(rgba_uint8, mode="RGBA")

    # Encode as base64 data URI
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    return img_str


# -------------------------------
# Geospatial Analysis
# -------------------------------


def find_intersections(river_gdf, basin_gdf):
    """
    Find easternmost points where rivers exit basin boundaries.

    Parameters
    ----------
    river_gdf : geopandas.GeoDataFrame
        River LineStrings with 'GNIS_Name' column for river names.
    basin_gdf : geopandas.GeoDataFrame
        Basin Polygon geometries defining the study area boundary.

    Returns
    -------
    geopandas.GeoDataFrame
        Point geometries with columns:
        - 'river_name': Name of the river
        - 'geometry': Point at easternmost basin boundary crossing

    Notes
    -----
    - Assumes both inputs share the same CRS
    - Returns only rivers that intersect the basin boundary
    - For rivers with multiple boundary crossings, keeps only the easternmost
    """
    # Get outer boundary of all basins combined
    outer_boundary = basin_gdf.geometry.unary_union.boundary

    # Find intersection points for each river
    intersections = []
    for _, river in river_gdf.iterrows():
        intersection = river.geometry.intersection(outer_boundary)

        if not intersection.is_empty:
            # Handle both Point and MultiPoint geometries
            points = []
            if intersection.geom_type == "MultiPoint":
                points = list(intersection.geoms)
            elif intersection.geom_type == "Point":
                points = [intersection]

            # Keep easternmost point (highest x/longitude)
            if points:
                easternmost = max(points, key=lambda p: p.x)
                intersections.append(
                    {"river_name": river["GNIS_Name"], "geometry": easternmost}
                )

    return gpd.GeoDataFrame(intersections, crs=river_gdf.crs)


def create_colorbar_widget(vmin, vmax, cmap, label, width=80, height=300, bar_width=0.1):
    """
    Create a vertical colorbar widget using matplotlib.

    Parameters
    ----------
    vmin : float
        Minimum value for color scale
    vmax : float
        Maximum value for color scale
    cmap : str
        Matplotlib colormap name
    label : str
        Label for the colorbar
    width : int, default=80
        Width of colorbar in pixels
    height : int, default=300
        Height of colorbar in pixels
    bar_width : float, default=0.15
        Width of the colorbar in inches. Smaller values make it skinnier.

    Returns
    -------
    ipywidgets.HTML
        HTML widget containing the colorbar image
    """
    import matplotlib.pyplot as plt

    # Calculate figure size based on desired pixel dimensions
    dpi = 100
    fig_width = width / dpi
    fig_height = height / dpi

    # Create figure for vertical colorbar
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

    # Fixed margins in inches (independent of dimensions)
    left_margin_in = 0.05
    right_margin_in = 0.5  # For label and ticks
    bottom_margin_in = 0.1
    top_margin_in = 0.1

    # Convert to figure coordinates
    left_margin = left_margin_in / fig_width
    right_margin = right_margin_in / fig_width
    bottom_margin = bottom_margin_in / fig_height
    top_margin = top_margin_in / fig_height

    # Calculate colorbar width in figure coordinates
    cbar_width = bar_width / fig_width

    ax = fig.add_axes([
        left_margin,
        bottom_margin,
        cbar_width,  # Fixed width for skinny colorbar
        1 - bottom_margin - top_margin    # Height fills vertically
    ])

    norm = Normalize(vmin=vmin, vmax=vmax)

    # Create vertical colorbar
    cb = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='vertical',
    )
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=7)

    # Save to bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, transparent=True, facecolor='none')
    plt.close(fig)
    buf.seek(0)

    # Create HTML img tag
    img_base64 = base64.b64encode(buf.read()).decode()
    html = f'<img src="data:image/png;base64,{img_base64}" style="width:{width}px; height:{height}px;"/>'

    return widgets.HTML(value=html)