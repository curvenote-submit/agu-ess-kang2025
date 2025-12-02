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

from ipyleaflet import Map, GeoJSON, WidgetControl, LayersControl, basemaps, Popup, ImageOverlay
from IPython.display import display
from ipywidgets import jslink


# -------------------------------
# Renaming Helpers
# -------------------------------
def format_subbasin_name(name):
    """
    Convert subbasin names from ALL CAPS format to title case with parentheses.

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
    >>> format_subbasin_name("SAN JOAQUIN VALLEY - TRACY")
    "San Joaquin Valley (Tracy)"
    >>> format_subbasin_name("SACRAMENTO VALLEY - RED BLUFF")
    "Sacramento Valley (Red Bluff)"
    """
    if " - " in name:
        parts = name.split(" - ")
        subbasin_part = parts[0].title()
        location_part = parts[1].title()
        return f"{subbasin_part} ({location_part})"
    else:
        return name.title()


# -------------------------------
# Colormap Helper Functions
# -------------------------------
def _is_binary_colormap(cmap):
    """Check if colormap is a ListedColormap with exactly 2 colors."""
    from matplotlib.colors import ListedColormap

    return isinstance(cmap, ListedColormap) and len(cmap.colors) == 2


def _get_legend_color(cmap):
    """Extract the visible color from a binary ListedColormap."""
    colors = cmap.colors
    # Skip first color if it's transparent (None, "none", or RGBA with alpha=0)
    for color in colors:
        if color is None or color == "none":
            continue
        # Check if RGBA tuple with alpha=0
        if isinstance(color, (tuple, list)) and len(color) == 4 and color[3] == 0:
            continue
        return color  # Return first non-transparent color
    return colors[-1]  # Fallback to last color


# -------------------------------
# Geopandas helpers
# -------------------------------


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
                intersections.append({"river_name": river["GNIS_Name"], "geometry": easternmost})

    return gpd.GeoDataFrame(intersections, crs=river_gdf.crs)


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


# ---------------------
# Map Controller
# ---------------------


def create_colorbar_widget(
    vmin, vmax, cmap, label, width=80, height=300, bar_width=0.1, log_scale=False
):
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
    bottom_margin_in = 0.1
    top_margin_in = 0.1

    # Convert to figure coordinates
    left_margin = left_margin_in / fig_width
    bottom_margin = bottom_margin_in / fig_height
    top_margin = top_margin_in / fig_height

    # Calculate colorbar width in figure coordinates
    cbar_width = bar_width / fig_width

    ax = fig.add_axes(
        [
            left_margin,
            bottom_margin,
            cbar_width,  # Fixed width for skinny colorbar
            1 - bottom_margin - top_margin,  # Height fills vertically
        ]
    )

    norm = _norm_wrapper(vmin, vmax, log_scale)

    # Create vertical colorbar
    cb = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation="vertical",
    )
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=7)

    # Save to bytes
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, transparent=True, facecolor="none")
    plt.close(fig)
    buf.seek(0)

    # Create HTML img tag
    img_base64 = base64.b64encode(buf.read()).decode()
    html = f'<img src="data:image/png;base64,{img_base64}" style="width:{width}px; height:{height}px;"/>'

    return widgets.HTML(value=html)


def create_legend_widget(color, label, width=100, height=100):
    """
    Create a simple legend widget with colored box + label centered below.

    Parameters
    ----------
    color : color spec
        Matplotlib color for the legend box
    label : str
        Text label to display (centered below box)
    width : int, default=100
        Width in pixels
    height : int, default=100
        Height in pixels

    Returns
    -------
    ipywidgets.HTML
        HTML widget containing the legend image
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    dpi = 100
    fig_width_in = width / dpi
    fig_height_in = height / dpi

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)
    ax.axis("off")

    # Add colored square centered horizontally, in upper portion
    box_size = 0.4
    box_x = (1 - box_size) / 2  # Center horizontally
    box_y = 0.55  # Upper portion

    rect = mpatches.Rectangle(
        (box_x, box_y), box_size, box_size, facecolor=color, edgecolor="none", linewidth=0
    )
    ax.add_patch(rect)

    # Add label text centered below the box with more spacing
    ax.text(0.5, 0.2, label, fontsize=8, va="center", ha="center", wrap=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Convert to base64
    buf = BytesIO()
    plt.savefig(
        buf,
        format="png",
        dpi=dpi,
        transparent=True,
        facecolor="none",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(fig)
    buf.seek(0)

    img_base64 = base64.b64encode(buf.read()).decode()
    # Force the aspect ratio to match the figure
    html = f'<img src="data:image/png;base64,{img_base64}" style="width:{width}px; height:{height}px;"/>'

    return widgets.HTML(value=html)


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
        stats_data=None,
        stats_map_side="right",
    ):
        """Initialize dual map controller with configuration and data."""
        self.subbasins = subbasins
        self.subbasin_column = subbasin_column
        self.rivers_gdf = rivers_gdf
        self.river_intersections_gdf = river_intersections_gdf
        self.param_dict = param_dict
        self.stats_data = stats_data
        self.stats_map_side = stats_map_side

        # Create maps with explicit center and zoom
        layout = widgets.Layout(width=width, height=height)
        self.m1 = Map(center=center, zoom=zoom, basemap=basemaps.CartoDB.Positron, layout=layout)
        self.m2 = Map(center=center, zoom=zoom, basemap=basemaps.CartoDB.Positron, layout=layout)

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

        # Track scalar overlay layers for each map
        self.scalar_layer_m1 = None
        self.scalar_layer_m2 = None

        # Track widget controls for scalar layers (sliders, dropdowns)
        self.scalar_control_m1 = None
        self.scalar_control_m2 = None

        # Track current state for dynamic scalar layers
        self.current_dataset_m1 = None
        self.current_dataset_m2 = None
        self.dataset_options_m1 = None
        self.dataset_options_m2 = None

        # Track dimension sliders for syncing
        self.dim_slider_m1 = None
        self.dim_slider_m2 = None
        self.dim_name_m1 = None
        self.dim_name_m2 = None

        # Info widget for displaying current selection
        self.info_widget = None

        # Stats widget for displaying statistics
        self.stats_widget = None

        # Flag to prevent infinite loops when syncing sliders
        self._syncing = False

    def add_layer(self, layer):
        """
        Add a single layer to both maps.

        Use this for base vector layers (rivers, subbasins, etc.) that should
        appear on both maps. Call this method multiple times to add multiple layers.

        Parameters
        ----------
        layer : ipyleaflet layer object
            Any ipyleaflet layer (GeoJSON, TileLayer, etc.) to add to both maps
        """
        self.m1.add_layer(layer)
        self.m2.add_layer(layer)

    def add_scalar_layer(
        self,
        ds,
        variable_names,
        map_side="left",
        opacity=1.0,
        name="Raster Dataset",
        initial_variable=None,
        initial_value=None,
    ):
        """
        Add scalar raster layer(s) to one map with automatic controls.

        This unified method handles both single and multiple variables:
        - Single variable (str): Creates a static layer, adds slider if variable has extra dimension
        - Multiple variables (list): Creates dropdown selector + slider for extra dimensions

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset containing the variable(s) to visualize
        variable_names : str or list of str
            Single variable name or list of variable names
        map_side : str, default='left'
            Which map to add the layer to ('left' or 'right')
        opacity : float, default=1.0
            Layer opacity (0.0 to 1.0)
        name : str, default="Raster Dataset"
            Name for the layer (used in layer control)
        initial_variable : str, optional
            For multiple variables, which to display initially. If None, uses first.
        initial_value : optional
            Initial value for slider if variable has extra dimension. If None, uses first value.

        Examples
        --------
        >>> # Single 2D variable - no controls
        >>> controller.add_scalar_layer(ds, "fraction_coarse", map_side="left")

        >>> # Single 3D variable - slider only
        >>> controller.add_scalar_layer(ds, "path_to_no_flow", map_side="left", initial_value=20)

        >>> # Multiple variables - dropdown + slider
        >>> controller.add_scalar_layer(
        ...     ds, ["path_length_norm", "path_to_no_flow"],
        ...     map_side="right", initial_variable="path_length_norm", initial_value=20
        ... )

        Raises
        ------
        ValueError
            If variable has more than 3 dimensions (more than 1 non-spatial dimension)
        """
        # Normalize input: convert string to single-item list
        if isinstance(variable_names, str):
            variable_names = [variable_names]
            has_dropdown = False
        else:
            has_dropdown = True

        # Determine initial variable
        if initial_variable is None:
            initial_variable = variable_names[0]
        elif initial_variable not in variable_names:
            raise ValueError(
                f"initial_variable '{initial_variable}' not in variable_names: {variable_names}"
            )

        # Get initial data array and parameters
        da = ds[initial_variable]
        params = self.param_dict[initial_variable]

        # Check dimensionality
        spatial_dims = {"x", "y"}
        non_spatial_dims = [d for d in da.dims if d not in spatial_dims]

        if len(non_spatial_dims) > 1:
            raise ValueError(
                f"Variable '{initial_variable}' has {len(non_spatial_dims)} non-spatial dimensions. "
                f"Only variables with 0 or 1 non-spatial dimension are supported. "
                f"Non-spatial dimensions found: {non_spatial_dims}"
            )

        # Determine if we need a dimension slider
        # If dropdown exists, check if ANY variable has extra dimensions
        has_dim_slider = False
        dim_name = None
        dim_values = None

        if has_dropdown:
            # Check all variables in the dropdown to find if any have extra dimensions
            for var_name in variable_names:
                var_da = ds[var_name]
                var_non_spatial_dims = [d for d in var_da.dims if d not in spatial_dims]
                if len(var_non_spatial_dims) == 1:
                    has_dim_slider = True
                    dim_name = var_non_spatial_dims[0]
                    dim_values = var_da[dim_name].values
                    break
        else:
            # Single variable case - only create slider if this variable has extra dimension
            if len(non_spatial_dims) == 1:
                has_dim_slider = True
                dim_name = non_spatial_dims[0]
                dim_values = da[dim_name].values

        # Select initial data slice
        if len(non_spatial_dims) == 1:
            # Initial variable has extra dimension
            if initial_value is None:
                initial_value = da[non_spatial_dims[0]].values[0]
            elif initial_value not in da[non_spatial_dims[0]].values:
                raise ValueError(
                    f"initial_value {initial_value} not in {non_spatial_dims[0]} values: {da[non_spatial_dims[0]].values}"
                )
            da_selected = da.sel({non_spatial_dims[0]: initial_value})
        else:
            # Use data as-is for 2D initial variable
            da_selected = da
            # Set initial_value for slider even if initial variable is 2D
            if has_dim_slider and initial_value is None:
                initial_value = dim_values[0]

        # Create ImageOverlay
        layer = ImageOverlay(
            url=scalar_to_base64_image(
                da_selected,
                cmap=params["cmap"],
                vmin=params["vmin"],
                vmax=params["vmax"],
                log_scale=params["log_scale"],
            ),
            bounds=leaflet_bounds(da_selected),
            opacity=opacity,
            name=name,
        )

        # Determine target map
        target_map = self.m1 if map_side == "left" else self.m2

        # Add layer to map
        target_map.add_layer(layer)

        # Store layer reference
        if map_side == "left":
            self.scalar_layer_m1 = layer
            self.current_dataset_m1 = initial_variable
        else:
            self.scalar_layer_m2 = layer
            self.current_dataset_m2 = initial_variable

        # Build controls (dropdown and/or slider)
        controls = []

        # Add dataset dropdown if multiple variables
        if has_dropdown:
            # Create options mapping from short_label to dataset name
            dataset_options = {
                self.param_dict[v]["short_label"]: v for v in variable_names if v in self.param_dict
            }

            dataset_dropdown = widgets.Dropdown(
                options=list(dataset_options.keys()),
                value=self.param_dict[initial_variable]["short_label"],
                description="Dataset:",
                style={"description_width": "initial"},
            )

            # Store mapping and create callback
            if map_side == "left":
                self.dataset_options_m1 = dataset_options
            else:
                self.dataset_options_m2 = dataset_options

            # Create dataset change callback
            def on_dataset_change(change):
                short_label = change["new"]
                options = self.dataset_options_m1 if map_side == "left" else self.dataset_options_m2
                new_variable = options[short_label]

                # Update current dataset
                if map_side == "left":
                    self.current_dataset_m1 = new_variable
                else:
                    self.current_dataset_m2 = new_variable

                # Check if new variable has extra dimensions
                new_da = ds[new_variable]
                spatial_dims = {"x", "y"}
                non_spatial_dims = [d for d in new_da.dims if d not in spatial_dims]
                new_var_has_dim = len(non_spatial_dims) == 1

                # Get the slider for this side
                current_slider = self.dim_slider_m1 if map_side == "left" else self.dim_slider_m2

                # Show/hide slider based on new variable's dimensionality
                if current_slider is not None:
                    if new_var_has_dim:
                        current_slider.layout.display = "flex"  # Show slider
                    else:
                        current_slider.layout.display = "none"  # Hide slider

                # Update colorbar
                if map_side == "left":
                    self._create_colorbar_m1(new_variable)
                else:
                    self._create_colorbar_m2(new_variable)

                # Update the layer
                self._update_scalar_layer(ds, new_variable, map_side, current_slider)

                # Update stats widget
                self._update_info_widget()

            dataset_dropdown.observe(on_dataset_change, names="value")
            controls.append(dataset_dropdown)

        # Add dimension slider if needed
        dim_slider = None
        if has_dim_slider:
            dim_slider = widgets.SelectionSlider(
                options=dim_values.tolist(),
                value=initial_value,
                description=f"{dim_name.title()}:",
                style={"description_width": "initial"},
            )

            # Hide slider initially if the initial variable doesn't need it
            if len(non_spatial_dims) == 0:
                dim_slider.layout.display = "none"

            # Store slider and dimension name reference
            if map_side == "left":
                self.dim_slider_m1 = dim_slider
                self.dim_name_m1 = dim_name
            else:
                self.dim_slider_m2 = dim_slider
                self.dim_name_m2 = dim_name

            # Create dimension change callback
            def on_dimension_change(change):
                current_var = (
                    self.current_dataset_m1 if map_side == "left" else self.current_dataset_m2
                )
                self._update_scalar_layer(ds, current_var, map_side, dim_slider)
                self._update_info_widget()

            dim_slider.observe(on_dimension_change, names="value")
            controls.append(dim_slider)

            # Sync sliders if both sides have same dimension
            self._sync_sliders()

        # Add controls to map if any exist
        if controls:
            controls_widget = widgets.VBox(controls)
            # Always use topright to stack properly with colorbar/legend
            control = WidgetControl(widget=controls_widget, position="topright")
            target_map.add_control(control)

            # Store control reference
            if map_side == "left":
                self.scalar_control_m1 = control
            else:
                self.scalar_control_m2 = control

        # Add colorbar
        if map_side == "left":
            self._create_colorbar_m1(initial_variable)
        else:
            self._create_colorbar_m2(initial_variable)

    def _sync_sliders(self):
        """Sync sliders if they represent the same dimension."""
        if (
            self.dim_slider_m1 is not None
            and self.dim_slider_m2 is not None
            and self.dim_name_m1 == self.dim_name_m2
        ):
            # Create bidirectional sync callbacks using observe
            # Note: We can't use jslink because SelectionSlider.value is not syncable

            def sync_m1_to_m2(change):
                """Sync left slider changes to right slider."""
                if not self._syncing:
                    self._syncing = True
                    try:
                        self.dim_slider_m2.value = change["new"]
                    finally:
                        self._syncing = False

            def sync_m2_to_m1(change):
                """Sync right slider changes to left slider."""
                if not self._syncing:
                    self._syncing = True
                    try:
                        self.dim_slider_m1.value = change["new"]
                    finally:
                        self._syncing = False

            # Attach observers
            self.dim_slider_m1.observe(sync_m1_to_m2, names="value")
            self.dim_slider_m2.observe(sync_m2_to_m1, names="value")

    def _update_info_widget(self):
        """Update the info widget with current selection state and statistics."""
        if self.info_widget is None:
            return

        # Get current threshold/dimension value (for stats only, not display)
        current_threshold = None
        if self.dim_slider_m1 is not None or self.dim_slider_m2 is not None:
            # Use whichever slider exists (they're synced if both exist)
            slider = self.dim_slider_m1 if self.dim_slider_m1 is not None else self.dim_slider_m2
            current_threshold = slider.value

        # Get current subbasin (for stats lookup - use original name)
        current_basin = "Full Valley"  # Default
        if self.subbasin_dropdown is not None and self.subbasin_dropdown.value is not None:
            display_name = self.subbasin_dropdown.value
            # Convert display name to original name for stats lookup
            current_basin = self.subbasin_display_to_original.get(display_name, display_name)

        # Hide the info widget since we're not displaying anything in it
        self.info_widget.value = ""

        # Update statistics widget
        self._update_stats_widget(current_threshold, current_basin)

    def _update_stats_widget(self, threshold, basin_name):
        """Update statistics widget based on current selections."""
        if self.stats_widget is None or self.stats_data is None:
            return

        # make sure the threshold is a string for the stats
        threshold = str(int(threshold))

        # Get current variable from the selected map side
        var_name = (
            self.current_dataset_m1 if self.stats_map_side == "left" else self.current_dataset_m2
        )

        # Check if we have valid data to display
        if threshold is None or threshold not in self.stats_data:
            self.stats_widget.value = "<i>No statistics available</i>"
            return

        if basin_name not in self.stats_data[threshold]:
            self.stats_widget.value = f"<i>No data for {basin_name}</i>"
            return

        # Extract land use name from variable name (e.g., "land_use_Forest" -> "Forest")
        if var_name and var_name.startswith("land_use_"):
            land_use_name = var_name.replace("land_use_", "")
        else:
            # For non-land use variables, use the variable name directly
            land_use_name = var_name

        if not land_use_name or land_use_name not in self.stats_data[threshold][basin_name]:
            # self.stats_widget.value = "<i>No statistics for current variable</i>"
            # TODO remove DEBUG:

            return

        # Get statistics and variable label
        stats = self.stats_data[threshold][basin_name][land_use_name]
        var_label = self.param_dict.get(var_name, {}).get("short_label", land_use_name)

        # Conversion factors
        to_acre = 247.105
        to_macreft = 0.810713

        # Calculate values in different units
        # total_area_km2 = stats['total_area']
        # total_area_Macre = total_area_km2 * to_acre / 1e6
        suitable_area_km2 = stats["suitable_area"]
        suitable_area_Macre = suitable_area_km2 * to_acre / 1e6

        # total_vol_km3 = stats['total_vol']
        # total_vol_Macreft = total_vol_km3 * to_macreft
        suitable_vol_km3 = stats["suitable_vol"]
        suitable_vol_Macreft = suitable_vol_km3 * to_macreft

        # # Calculate percentages
        # area_percent = (suitable_area_km2 / total_area_km2 * 100) if total_area_km2 > 0 else 0
        # vol_percent = (suitable_vol_km3 / total_vol_km3 * 100) if total_vol_km3 > 0 else 0

        # Build HTML with statistics (no header, just the stats)
        html_parts = ["<div style='font-size: 11px; padding: 5px; line-height: 1.4;'>"]
        html_parts.append(f"<b>{var_label} suitable:</b><br/>")
        # html_parts.append(f"&nbsp;&nbsp;{suitable_area_km2/1e3:.0f} K km² ({suitable_area_Macre:.0f} M acre) - {area_percent:.0f}% of total<br/>")
        # html_parts.append(f"&nbsp;&nbsp;{suitable_vol_km3:.0f} km³ ({suitable_vol_Macreft:.0f} M acre-ft) - {vol_percent:.0f}% of total")
        # html_parts.append(f"&nbsp;&nbsp;{suitable_area_km2/1e3:.0f} K km² ({suitable_area_Macre:.0f} M acre)<br/>")
        # html_parts.append(f"&nbsp;&nbsp;{suitable_vol_km3:.0f} km³ ({suitable_vol_Macreft:.0f} M acre-ft)")
        html_parts.append(
            f"&nbsp;&nbsp;{suitable_area_km2:.1f} km² ({suitable_area_Macre:.3f} M acre)<br/>"
        )
        html_parts.append(
            f"&nbsp;&nbsp;{suitable_vol_km3:.1f} km³ ({suitable_vol_Macreft:.2f} M acre-ft)"
        )
        html_parts.append("</div>")

        self.stats_widget.value = "".join(html_parts)

    def _update_scalar_layer(self, ds, variable_name, map_side, dim_slider=None):
        """Update scalar layer with new variable and/or dimension value."""
        params = self.param_dict[variable_name]
        da = ds[variable_name]

        # Check if this variable has non-spatial dimensions
        spatial_dims = {"x", "y"}
        non_spatial_dims = [d for d in da.dims if d not in spatial_dims]

        # Select data by dimension if needed
        if len(non_spatial_dims) == 1:
            dim_name = non_spatial_dims[0]
            # Get dimension value from slider if available, otherwise use first value
            if dim_slider is not None:
                dim_value = dim_slider.value
            else:
                # Fallback: use first value if no slider exists
                dim_value = da[dim_name].values[0]
            da_selected = da.sel({dim_name: dim_value})
        elif len(non_spatial_dims) == 0:
            # 2D variable, use as-is
            da_selected = da
        else:
            raise ValueError(
                f"Variable '{variable_name}' has {len(non_spatial_dims)} non-spatial dimensions. "
                f"Only variables with 0 or 1 non-spatial dimension are supported."
            )

        # Get the layer to update
        layer = self.scalar_layer_m1 if map_side == "left" else self.scalar_layer_m2

        # Update image overlay
        layer.url = scalar_to_base64_image(
            da_selected,
            cmap=params["cmap"],
            vmin=params["vmin"],
            vmax=params["vmax"],
            log_scale=params["log_scale"],
        )

    def _create_colorbar_m1(self, dataset_name):
        """Create/update colorbar or legend for left map."""
        params = self.param_dict[dataset_name]

        # Remove existing colorbar/legend if present
        if self.colorbar_control_m1 is not None:
            self.m1.remove_control(self.colorbar_control_m1)

        cmap = params["cmap"]

        # Auto-detect: Use simple legend for binary ListedColormaps
        if _is_binary_colormap(cmap):
            color = _get_legend_color(cmap)
            colorbar_widget = create_legend_widget(color, params["label"])
        else:
            # Standard gradient colorbar
            colorbar_widget = create_colorbar_widget(
                vmin=params["vmin"],
                vmax=params["vmax"],
                cmap=cmap,
                label=params["label"],
                width=80,
                height=300,
                bar_width=0.1,
                log_scale=params["log_scale"],
            )

        # Add colorbar/legend at topright (will appear below other controls due to add order)
        self.colorbar_control_m1 = WidgetControl(widget=colorbar_widget, position="topright")
        self.m1.add_control(self.colorbar_control_m1)

    def _create_colorbar_m2(self, dataset_name):
        """Create/update colorbar or legend for right map."""
        params = self.param_dict[dataset_name]

        # Remove existing colorbar/legend if present
        if self.colorbar_control_m2 is not None:
            self.m2.remove_control(self.colorbar_control_m2)

        cmap = params["cmap"]

        # Auto-detect: Use simple legend for binary ListedColormaps
        if _is_binary_colormap(cmap):
            color = _get_legend_color(cmap)
            colorbar_widget = create_legend_widget(color, params["label"])
        else:
            # Standard gradient colorbar
            colorbar_widget = create_colorbar_widget(
                vmin=params["vmin"],
                vmax=params["vmax"],
                cmap=cmap,
                label=params["label"],
                width=80,
                height=300,
                bar_width=0.1,
                log_scale=params["log_scale"],
            )

        # Add colorbar/legend at topright (will appear below other controls due to add order)
        self.colorbar_control_m2 = WidgetControl(widget=colorbar_widget, position="topright")
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

    def create_subbasin_selector(self, center, zoom):
        """Create subbasin selection dropdown (to be placed above maps)."""
        self.default_center = center
        self.default_zoom = zoom

        # Get original subbasin names
        original_names = self.subbasins[self.subbasin_column].unique().tolist()

        # Create mapping from formatted display names to original names
        self.subbasin_display_to_original = {}
        display_names = []
        for original_name in original_names:
            display_name = format_subbasin_name(original_name)
            self.subbasin_display_to_original[display_name] = original_name
            display_names.append(display_name)

        # Create dropdown with formatted display names
        self.subbasin_dropdown = widgets.Dropdown(
            options=display_names,
            value=None,
            description="Subbasin:",
            style={"description_width": "initial"},
        )
        self.subbasin_dropdown.observe(self._on_subbasin_change, names="value")

    def _on_subbasin_change(self, change):
        """Handle subbasin selection change - zooms and highlights region."""
        display_name = change["new"]

        # If no selection, do nothing
        if display_name is None:
            return

        # Convert display name back to original name for data lookup
        original_name = self.subbasin_display_to_original.get(display_name, display_name)

        # Remove previous highlight
        if self.current_highlight is not None:
            self.m1.remove_layer(self.current_highlight)
            self.m2.remove_layer(self.current_highlight)
            self.current_highlight = None

        # Find selected subbasin using original name
        selected_subbasin = self.subbasins[self.subbasins[self.subbasin_column] == original_name]

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

        # Update info widget
        self._update_info_widget()

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

        # Update info widget
        self._update_info_widget()

    def create_home_button(self):
        """Create home button for resetting view."""
        home_button = widgets.Button(
            description="Show full California Valley",
            button_style="info",
            tooltip="Reset to full valley view",
            icon="home",
            layout=widgets.Layout(width="auto"),
        )
        home_button.on_click(self._on_home_click)

        return home_button

    def create_info_widget(self):
        """Create info widget to display current selection state."""
        self.info_widget = widgets.HTML(
            value="<i>No selection</i>",
            layout=widgets.Layout(width="auto", padding="5px"),
        )
        return self.info_widget

    def create_stats_widget(self):
        """Create stats widget to display statistics."""
        self.stats_widget = widgets.HTML(
            value="<i>No statistics available</i>",
            layout=widgets.Layout(width="auto", padding="5px", max_width="300px"),
        )
        return self.stats_widget

    def display(self):
        """Display the dual map setup with controls above."""
        # Create top controls
        home_button = self.create_home_button()

        # Build top control row
        top_controls = [home_button]

        # Add subbasin dropdown if it exists
        if self.subbasin_dropdown is not None:
            top_controls.append(self.subbasin_dropdown)

        # Add stats widget if stats_data was provided
        if self.stats_data is not None:
            stats_widget = self.create_stats_widget()
            top_controls.append(stats_widget)

        # Create horizontal box with controls
        top_row = widgets.HBox(
            top_controls,
            layout=widgets.Layout(justify_content="flex-start", align_items="center"),
        )

        # Display everything
        display(self.debug_output)
        display(top_row)
        display(widgets.HBox([self.m1, self.m2]))

        # Initialize stats display
        if self.stats_data is not None:
            # Create info widget for internal use (not displayed)
            self.create_info_widget()
            self._update_info_widget()
