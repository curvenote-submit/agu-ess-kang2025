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
from matplotlib.colors import Normalize, LogNorm
from PIL import Image
import ipywidgets as widgets

from ipyleaflet import Map, GeoJSON, WidgetControl, LayersControl, basemaps, Popup
from IPython.display import display
from ipywidgets import jslink

# -------------------------------
# Geopandas helpers
# -------------------------------
def format_valley_name(name):
    """
    Convert valley names from ALL CAPS format to title case with parentheses.

    Transforms "SAN JOAQUIN VALLEY - TRACY" to "San Joaquin Valley (Tracy)".

    Parameters
    ----------
    name : str
        Valley name in ALL CAPS format with " - " separator.

    Returns
    -------
    str
        Formatted name in title case with parentheses.

    Examples
    --------
    >>> format_valley_name("SAN JOAQUIN VALLEY - TRACY")
    "San Joaquin Valley (Tracy)"
    >>> format_valley_name("SACRAMENTO VALLEY - RED BLUFF")
    "Sacramento Valley (Red Bluff)"
    """
    if " - " in name:
        parts = name.split(" - ")
        valley_part = parts[0].title()
        location_part = parts[1].title()
        return f"{valley_part} ({location_part})"
    else:
        return name.title()
    
def combine_rivers_gdf(river_gdf, name_column="GNIS_Name"):
    """
    Combine river fragments with the same name into single features.

    Dissolves LineString geometries that share the same river name,
    creating continuous features for visualization and analysis.

    Parameters
    ----------
    river_gdf : GeoDataFrame
        Input rivers with LineString geometries.
    name_column : str, default='GNIS_Name'
        Column containing river names for grouping.

    Returns
    -------
    GeoDataFrame
        Rivers dissolved by name with only name and geometry columns.

    Notes
    -----
    Some rivers may have gaps or duplicate names requiring QC.
    """
    rivers_combined = river_gdf.dissolve(by=name_column).reset_index()
    return rivers_combined[[name_column, "geometry"]]


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

def _norm_wrapper(vmin, vmax, log_scale=False):
    if log_scale:
        if vmin <= 0:
            raise ValueError("Cannot define vmin <= 0 for log colorbar")
        # Use logarithmic normalization
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        # Use linear normalization
        norm = Normalize(vmin=vmin, vmax=vmax)
    return norm



def scalar_to_base64_image(da, cmap, vmin, vmax, log_scale=False):
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
    log_scale : bool, default=False
        If True, use logarithmic color scaling.

    Returns
    -------
    str
        Base64-encoded PNG image as data URI string.

    Example
    -------
    >>> img_url = scalar_to_base64_image(temperature_data, cmap='RdBu_r')
    >>> overlay = ImageOverlay(url=img_url, bounds=bounds, opacity=0.7)

    Notes
    -----
    When using log_scale=True, values <= 0 are clipped to a small positive value.
    """
    # Ensure spatial dimensions are last
    da = transpose_da(da)
    arr = da.values.copy()  # Make a copy to avoid modifying original

    # Apply colormap to data with appropriate normalization
    if log_scale:
        # Clip data values <= 0 to avoid log errors
        # Keep track of where data was invalid (NaN or <= 0)
        invalid_mask = np.isnan(arr) | (arr <= 0)
        arr[arr <= 0] = vmin
    else:
        # Use linear normalization
        invalid_mask = np.isnan(arr)

    norm = _norm_wrapper(vmin, vmax, log_scale)

    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba = mapper.to_rgba(arr, bytes=False)  # shape: (y, x, 4)

    # Make invalid values transparent
    rgba[invalid_mask, 3] = 0

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


def create_colorbar_widget(vmin, vmax, cmap, label, width=80, height=300, bar_width=0.1, log_scale=False):
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
    bar_width : float, default=0.1
        Width of the colorbar in inches. Smaller values make it skinnier.
    log_scale : bool, default=False
        If True, use logarithmic color scaling.

    Returns
    -------
    ipywidgets.HTML
        HTML widget containing the colorbar image

    Notes
    -----
    When using log_scale=True, vmin must be > 0 to avoid log(0) errors.
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

    norm = _norm_wrapper(vmin, vmax, log_scale)

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


#---------------------
# Map Controller class
#---------------------

class DualMapController:
    """
    Controller for synchronized dual maps with interactive controls.

    Manages two side-by-side ipyleaflet maps with:
    - Synchronized zoom and pan
    - Independent data layers
    - Interactive feature highlighting
    - Dynamic dataset selection
    - Dynamic colorbars
    """

    def __init__(
        self,
        width,
        height,
        center,
        zoom,
        subbasins,
        subbasin_column,
        rivers_gdf,
        river_intersections_gdf,
        param_dict,
    ):
        """Initialize dual map controller with configuration and data."""
        self.subbasins = subbasins
        self.subbasin_column = subbasin_column
        self.rivers_gdf = rivers_gdf
        self.river_intersections_gdf = river_intersections_gdf
        self.param_dict = param_dict

        # Create maps with explicit center and zoom
        layout = widgets.Layout(width=width, height=height)
        self.m1 = Map(
            center=center, zoom=zoom, basemap=basemaps.CartoDB.Positron, layout=layout
        )
        self.m2 = Map(
            center=center, zoom=zoom, basemap=basemaps.CartoDB.Positron, layout=layout
        )

        # Synchronize views using JavaScript linking
        jslink((self.m1, "center"), (self.m2, "center"))
        jslink((self.m1, "zoom"), (self.m2, "zoom"))

        # -------------------------------
        # Define Highlight Styles
        # -------------------------------
        # Subbasin highlight style (orange dashed border)
        self.highlight_style = {
            "color": "orange",
            "weight": 3,
            "opacity": 0.8,
            "fillColor": "orange",
            "fillOpacity": 0.1,
            "dashArray": "5, 5",
        }

        # River highlight style (bright blue glow effect)
        self.river_highlight_style = {
            "color": "#0099ff",
            "weight": 5,
            "opacity": 0.8,
            "lineCap": "round",
            "lineJoin": "round",
        }

        # -------------------------------
        # State Tracking
        # -------------------------------
        # Track highlight layers (created on demand)
        self.river_highlight_m1 = None
        self.river_highlight_m2 = None
        self.current_highlight = None  # Subbasin highlight

        # Track popup references for cleanup
        self.current_popup_m1 = None
        self.current_popup_m2 = None

        # Widget for displaying selected river name
        self.river_name_widget = widgets.HTML(value="")

        # Debug output widget
        self.debug_output = widgets.Output()
        
        # Colorbar widgets and controls (initialized later)
        self.colorbar_control_m1 = None
        self.colorbar_control_m2 = None
        
        # Subbasin dropdown widget (initialized later)
        self.subbasin_dropdown = None

    def add_base_layers(self, rivers, subbasins, brackish, fcd_layer, scalar_overlay):
        """Add base layers to both maps."""
        # Add to both maps
        self.m1.add_layer(rivers)
        self.m2.add_layer(rivers)
        self.m1.add_layer(subbasins)
        self.m2.add_layer(subbasins)
        self.m1.add_layer(brackish)
        self.m2.add_layer(brackish)

        # Map-specific overlays
        self.m1.add_layer(fcd_layer)
        self.m2.add_layer(scalar_overlay)

        # Store references for dynamic updates
        self.fcd_layer = fcd_layer
        self.scalar_overlay = scalar_overlay

    def _create_colorbar_m1(self, dataset_name):
        """Create/update colorbar for left map (FCD)."""
        params = self.param_dict[dataset_name]
        
        # Remove existing colorbar if present
        if self.colorbar_control_m1 is not None:
            self.m1.remove_control(self.colorbar_control_m1)
        
        # Create new colorbar
        colorbar_widget = create_colorbar_widget(
            vmin=params['vmin'],
            vmax=params['vmax'],
            cmap=params['cmap'],
            label=params['label'],
            width=80,
            height=300,
            bar_width=0.1,
            log_scale=params['log_scale'],
        )
        
        # Add colorbar at topright (on top of dropdown)
        self.colorbar_control_m1 = WidgetControl(
            widget=colorbar_widget, 
            position='topright'
        )
        self.m1.add_control(self.colorbar_control_m1)
    
    def _create_colorbar_m2(self, dataset_name):
        """Create/update colorbar for right map."""
        params = self.param_dict[dataset_name]
        
        # Remove existing colorbar if present
        if self.colorbar_control_m2 is not None:
            self.m2.remove_control(self.colorbar_control_m2)
        
        # Create new colorbar
        colorbar_widget = create_colorbar_widget(
            vmin=params['vmin'],
            vmax=params['vmax'],
            cmap=params['cmap'],
            label=params['label'],
            width=80,
            height=300,
            bar_width=0.1,
            log_scale=params['log_scale'],
        )
        
        # Add colorbar at topright (on top of dropdown)
        self.colorbar_control_m2 = WidgetControl(
            widget=colorbar_widget, 
            position='topright'
        )
        self.m2.add_control(self.colorbar_control_m2)

    def add_river_intersections(self, intersections_layer):
        """Add clickable river intersection markers to both maps."""
        self.intersections_layer = intersections_layer

        # Register click event handler
        self.intersections_layer.on_click(self._on_intersection_click)

        # Add to both maps
        self.m1.add_layer(self.intersections_layer)
        self.m2.add_layer(self.intersections_layer)

    def _on_intersection_click(self, feature, **kwargs):
        """
        Handle click on river intersection point.

        Highlights the full river geometry on both maps and displays
        a popup with the river name.
        """
        river_name = feature["properties"].get("river_name", "Unknown")
        coords = feature["geometry"]["coordinates"]

        # Update river name display
        self.river_name_widget.value = f"<b>Selected River:</b> {river_name}"

        # -------------------------------
        # Clean Up Previous Popups
        # -------------------------------
        if self.current_popup_m1 is not None:
            self.m1.remove_layer(self.current_popup_m1)
        if self.current_popup_m2 is not None:
            self.m2.remove_layer(self.current_popup_m2)

        # -------------------------------
        # Create New Popups
        # -------------------------------
        popup_html = widgets.HTML(f"<b>{river_name}</b>")
        self.current_popup_m1 = Popup(
            location=(coords[1], coords[0]),
            child=popup_html,
            close_button=True,
            auto_close=False,
            close_on_escape_key=True,
            name="River Info Popup",
        )

        popup_html2 = widgets.HTML(f"<b>{river_name}</b>")
        self.current_popup_m2 = Popup(
            location=(coords[1], coords[0]),
            child=popup_html2,
            close_button=True,
            auto_close=False,
            close_on_escape_key=True,
            name="River Info Popup",
        )

        self.m1.add_layer(self.current_popup_m1)
        self.m2.add_layer(self.current_popup_m2)

        # -------------------------------
        # Highlight River Geometry
        # -------------------------------
        matching_river = self.rivers_gdf[self.rivers_gdf["GNIS_Name"] == river_name]

        if not matching_river.empty:
            # Remove previous river highlights
            if self.river_highlight_m1 is not None:
                self.m1.remove_layer(self.river_highlight_m1)
            if self.river_highlight_m2 is not None:
                self.m2.remove_layer(self.river_highlight_m2)

            # Create new highlight layers
            self.river_highlight_m1 = GeoJSON(
                data=matching_river.__geo_interface__,
                style=self.river_highlight_style,
                name="River Highlight",
            )
            self.river_highlight_m2 = GeoJSON(
                data=matching_river.__geo_interface__,
                style=self.river_highlight_style,
                name="River Highlight",
            )

            self.m1.add_layer(self.river_highlight_m1)
            self.m2.add_layer(self.river_highlight_m2)

    def add_controls(self):
        """Add LayersControl to both maps (call AFTER adding base layers)."""
        self.m1.add_control(LayersControl(position="bottomleft", collapsed=False))
        self.m2.add_control(LayersControl(position="bottomleft", collapsed=False))

    def create_dataset_selector(self, ds, initial_dataset):
        """Create dataset selection controls for right map."""
        self.ds = ds
        self.current_dataset_name = initial_dataset
        self.da_scalar = ds[initial_dataset]

        # -------------------------------
        # Dataset Dropdown (use short_label)
        # -------------------------------
        # Create options mapping from short_label to dataset name
        # Filter out fraction_coarse and spatial_ref (CRS metadata)
        dataset_options = {
            self.param_dict[v]['short_label']: v
            for v in ds.data_vars
            if v not in ["fraction_coarse", "spatial_ref"] and v in self.param_dict
        }
        
        dataset_dropdown = widgets.Dropdown(
            options=list(dataset_options.keys()),
            value=self.param_dict[initial_dataset]['short_label'],
            description="Dataset:",
            style={"description_width": "initial"},
        )
        
        # Store mapping for callback
        self.dataset_options = dataset_options
        dataset_dropdown.observe(self._on_dataset_change, names="value")

        # -------------------------------
        # Threshold Slider
        # -------------------------------
        self.slider = widgets.SelectionSlider(
            options=ds.threshold.values,
            value=20,
            description="FCD Threshold (%)",
            style={"description_width": "initial"},
        )
        self.slider.observe(self._on_threshold_change, names="value")

        # Add dropdown FIRST (will be at back)
        controls = widgets.VBox([dataset_dropdown, self.slider])
        widget_control = WidgetControl(widget=controls, position="topright")
        self.m2.add_control(widget_control)
        
        # Add colorbar AFTER (will be on top)
        self._create_colorbar_m2(initial_dataset)

    def _on_dataset_change(self, change):
        """Handle dataset selection change."""
        # Map from short_label back to dataset name
        short_label = change["new"]
        self.current_dataset_name = self.dataset_options[short_label]
        self.da_scalar = self.ds[self.current_dataset_name]
        
        # Update colorbar
        self._create_colorbar_m2(self.current_dataset_name)
        
        # Update overlay
        self._update_scalar_overlay()

    def _on_threshold_change(self, change):
        """Handle threshold slider change."""
        self._update_scalar_overlay()

    def _update_scalar_overlay(self):
        """Update right map overlay with new dataset/threshold."""
        threshold = self.slider.value
        params = self.param_dict[self.current_dataset_name]

        # Select data by threshold if dimension exists
        if "threshold" in self.da_scalar.dims:
            da_overlay = self.da_scalar.sel(threshold=threshold)
        else:
            da_overlay = self.da_scalar

        # Update image overlay with new data using param_dict values
        self.scalar_overlay.url = scalar_to_base64_image(
            da_overlay,
            cmap=params['cmap'],
            vmin=params['vmin'],
            vmax=params['vmax'],
            log_scale=params['log_scale'],
        )

    def create_subbasin_selector(self, center, zoom):
        """Create subbasin selection dropdown for left map."""
        self.default_center = center
        self.default_zoom = zoom

        # Create dropdown options (no "All Regions")
        subbasin_names = self.subbasins[self.subbasin_column].tolist()
        self.subbasin_dropdown = widgets.Dropdown(
            options=subbasin_names,
            value=None,
            description="Subbasin:",
            style={"description_width": "initial"},
        )
        self.subbasin_dropdown.observe(self._on_subbasin_change, names="value")

        # Add dropdown FIRST (will be at back)
        controls_vbox = widgets.VBox([self.subbasin_dropdown, self.river_name_widget])
        control = WidgetControl(widget=controls_vbox, position="topright")
        self.m1.add_control(control)
        
        # Add colorbar AFTER (will be on top)
        self._create_colorbar_m1('fraction_coarse')

    def _on_subbasin_change(self, change):
        """Handle subbasin selection change - zooms and highlights region."""
        selected_name = change["new"]
        
        # If no selection, do nothing
        if selected_name is None:
            return

        # Remove previous highlight
        if self.current_highlight is not None:
            self.m1.remove_layer(self.current_highlight)
            self.m2.remove_layer(self.current_highlight)
            self.current_highlight = None

        # Find selected subbasin
        selected_subbasin = self.subbasins[
            self.subbasins[self.subbasin_column] == selected_name
        ]

        if selected_subbasin.empty:
            with self.debug_output:
                print("No matching subbasin found!")
            return

        # Zoom to subbasin bounds
        bounds = selected_subbasin.total_bounds
        self.m1.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        # Create highlight overlay
        self.current_highlight = GeoJSON(
            data=selected_subbasin.__geo_interface__,
            style=self.highlight_style,
            name="Region Highlight",
        )
        self.m1.add_layer(self.current_highlight)
        self.m2.add_layer(self.current_highlight)

    def _on_home_click(self, button):
        """Handle home button click - reset to default view."""
        # Remove any subbasin highlight
        if self.current_highlight is not None:
            self.m1.remove_layer(self.current_highlight)
            self.m2.remove_layer(self.current_highlight)
            self.current_highlight = None
        
        # Reset dropdown to no selection
        if self.subbasin_dropdown is not None:
            self.subbasin_dropdown.value = None
        
        # Reset to default view
        self.m1.center = self.default_center
        self.m1.zoom = self.default_zoom

    def create_home_button(self):
        """Create home button above the maps."""
        home_button = widgets.Button(
            description="Home",
            button_style="info",
            tooltip="Reset to default view",
            icon="home",
            layout=widgets.Layout(width="100px")
        )
        home_button.on_click(self._on_home_click)
        
        return home_button

    def display(self):
        """Display the dual map setup with home button above."""
        home_button = self.create_home_button()
        
        display(self.debug_output)
        display(home_button)
        display(widgets.HBox([self.m1, self.m2]))