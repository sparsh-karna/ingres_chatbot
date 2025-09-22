"""
Visualization Tools for INGRES AI ChatBot
Defines Plotly chart functions and schemas for LLM-based selection
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional

def create_histogram(data: pd.DataFrame, x_column: str, title: str = "Histogram", color: Optional[str] = None) -> go.Figure:
    """Creates a histogram for distribution analysis."""
    fig = px.histogram(data, x=x_column, color=color, title=title, height=400)
    fig.update_layout(xaxis_title=x_column, yaxis_title="Count")
    return fig

def create_line_chart(data: pd.DataFrame, x_column: str, y_column: str, title: str = "Line Chart", color: Optional[str] = None) -> go.Figure:
    """Creates a line chart for trends over time or sequences."""
    fig = px.line(data, x=x_column, y=y_column, color=color, title=title, height=400)
    fig.update_layout(xaxis_title=x_column, yaxis_title=y_column)
    return fig

def create_pie_chart(data: pd.DataFrame, values_column: str, names_column: str, title: str = "Pie Chart") -> go.Figure:
    """Creates a pie chart for proportions."""
    fig = px.pie(data, values=values_column, names=names_column, title=title, height=400)
    return fig

def create_bar_chart(data: pd.DataFrame, x_column: str, y_column: str, title: str = "Bar Chart", color: Optional[str] = None, orientation: str = 'v') -> go.Figure:
    """Creates a bar chart for comparisons."""
    fig = px.bar(data, x=x_column, y=y_column, color=color, title=title, orientation=orientation, height=400)
    fig.update_layout(xaxis_title=x_column, yaxis_title=y_column)
    fig.update_xaxes(tickangle=45)
    return fig

def create_scatter_plot(data: pd.DataFrame, x_column: str, y_column: str, title: str = "Scatter Plot", color: Optional[str] = None, size: Optional[str] = None) -> go.Figure:
    """Creates a scatter plot for correlations."""
    fig = px.scatter(data, x=x_column, y=y_column, color=color, size=size, title=title, height=400)
    fig.update_layout(xaxis_title=x_column, yaxis_title=y_column)
    return fig

def create_box_plot(data: pd.DataFrame, x_column: str, y_column: str, title: str = "Box Plot", color: Optional[str] = None) -> go.Figure:
    """Creates a box plot for statistical distributions."""
    fig = px.box(data, x=x_column, y=y_column, color=color, title=title, height=400)
    fig.update_layout(xaxis_title=x_column, yaxis_title=y_column)
    return fig

def create_violin_plot(data: pd.DataFrame, x_column: str, y_column: str, title: str = "Violin Plot", color: Optional[str] = None) -> go.Figure:
    """Creates a violin plot for density and distribution."""
    fig = px.violin(data, x=x_column, y=y_column, color=color, title=title, height=400)
    fig.update_layout(xaxis_title=x_column, yaxis_title=y_column)
    return fig

def create_heatmap(data: pd.DataFrame, x_column: str, y_column: str, z_column: str, title: str = "Heatmap") -> go.Figure:
    """Creates a heatmap for intensity or correlations."""
    fig = px.density_heatmap(data, x=x_column, y=y_column, z=z_column, title=title, height=400)
    fig.update_layout(xaxis_title=x_column, yaxis_title=y_column)
    return fig

def create_density_contour(data: pd.DataFrame, x_column: str, y_column: str, title: str = "Density Contour", color: Optional[str] = None) -> go.Figure:
    """Creates a density contour plot."""
    fig = px.density_contour(data, x=x_column, y=y_column, color=color, title=title, height=400)
    fig.update_layout(xaxis_title=x_column, yaxis_title=y_column)
    return fig

def create_density_heatmap(data: pd.DataFrame, x_column: str, y_column: str, title: str = "Density Heatmap", color: Optional[str] = None) -> go.Figure:
    """Creates a density heatmap."""
    fig = px.density_heatmap(data, x=x_column, y=y_column, title=title, height=400)
    fig.update_layout(xaxis_title=x_column, yaxis_title=y_column)
    return fig

def create_area_chart(data: pd.DataFrame, x_column: str, y_column: str, title: str = "Area Chart", color: Optional[str] = None) -> go.Figure:
    """Creates an area chart for cumulative trends."""
    fig = px.area(data, x=x_column, y=y_column, color=color, title=title, height=400)
    fig.update_layout(xaxis_title=x_column, yaxis_title=y_column)
    return fig

def create_funnel_chart(data: pd.DataFrame, values_column: str, names_column: str, title: str = "Funnel Chart") -> go.Figure:
    """Creates a funnel chart for stages or drop-offs."""
    fig = px.funnel(data, x=values_column, y=names_column, title=title, height=400)
    fig.update_layout(xaxis_title=values_column, yaxis_title=names_column)
    return fig

def create_timeline_chart(data: pd.DataFrame, x_start: str, x_end: str, y_column: str, title: str = "Timeline Chart", color: Optional[str] = None) -> go.Figure:
    """Creates a timeline (Gantt-style) chart."""
    fig = px.timeline(data, x_start=x_start, x_end=x_end, y=y_column, color=color, title=title, height=400)
    fig.update_layout(xaxis_title="Time", yaxis_title=y_column)
    return fig

def create_sunburst_chart(data: pd.DataFrame, path_columns: List[str], values_column: str, title: str = "Sunburst Chart") -> go.Figure:
    """Creates a sunburst chart for hierarchical data."""
    fig = px.sunburst(data, path=path_columns, values=values_column, title=title, height=400)
    return fig

def create_treemap_chart(data: pd.DataFrame, path_columns: List[str], values_column: str, title: str = "Treemap Chart") -> go.Figure:
    """Creates a treemap chart for hierarchical data."""
    fig = px.treemap(data, path=path_columns, values=values_column, title=title, height=400)
    return fig

def create_icicle_chart(data: pd.DataFrame, path_columns: List[str], values_column: str, title: str = "Icicle Chart") -> go.Figure:
    """Creates an icicle chart for hierarchical data."""
    fig = px.icicle(data, path=path_columns, values=values_column, title=title, height=400)
    return fig

def create_parallel_coordinates(data: pd.DataFrame, dimensions: List[str], title: str = "Parallel Coordinates Plot") -> go.Figure:
    """Creates a parallel coordinates plot for multi-variable comparisons."""
    fig = px.parallel_coordinates(data, dimensions=dimensions, title=title, height=400)
    return fig

def create_parallel_categories(data: pd.DataFrame, dimensions: List[str], title: str = "Parallel Categories Plot") -> go.Figure:
    """Creates a parallel categories plot for categorical comparisons."""
    fig = px.parallel_categories(data, dimensions=dimensions, title=title, height=400)
    return fig

def create_choropleth(data: pd.DataFrame, locations_column: str, color_column: str, title: str = "Choropleth Map", locationmode: str = "country names") -> go.Figure:
    """Creates a choropleth map for geographic data (e.g., states)."""
    fig = px.choropleth(data, locations=locations_column, color=color_column, locationmode=locationmode, title=title, height=400)
    fig.update_layout(geo_scope="asia")
    return fig

def create_scatter_geo(data: pd.DataFrame, lat_column: str, lon_column: str, size_column: Optional[str] = None, color_column: Optional[str] = None, title: str = "Scatter Geo Plot") -> go.Figure:
    """Creates a scatter geo plot for lat/long data."""
    fig = px.scatter_geo(data, lat=lat_column, lon=lon_column, size=size_column, color=color_column, title=title, height=400)
    fig.update_layout(geo_scope="asia")
    return fig

# Function schemas for LLM
PLOT_FUNCTIONS = [
    {
        "name": "create_histogram",
        "description": "Use for distributions of a single variable (e.g., recharge across districts).",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "List of dicts (DataFrame rows)."},
                "x_column": {"type": "string", "description": "Column for x-axis (numeric or categorical)."},
                "title": {"type": "string", "description": "Chart title."},
                "color": {"type": "string", "description": "Column for color grouping.", "nullable": True}
            },
            "required": ["data", "x_column", "title"]
        }
    },
    {
        "name": "create_line_chart",
        "description": "Use for trends over time (e.g., groundwater levels over years).",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "List of dicts (DataFrame rows)."},
                "x_column": {"type": "string", "description": "Column for x-axis (e.g., year)."},
                "y_column": {"type": "string", "description": "Column for y-axis (numeric)."},
                "title": {"type": "string", "description": "Chart title."},
                "color": {"type": "string", "description": "Column for multiple lines.", "nullable": True}
            },
            "required": ["data", "x_column", "y_column", "title"]
        }
    },
    {
        "name": "create_pie_chart",
        "description": "Use for proportions (e.g., extraction by category).",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "List of dicts (DataFrame rows)."},
                "values_column": {"type": "string", "description": "Column with numeric values."},
                "names_column": {"type": "string", "description": "Column with category names."},
                "title": {"type": "string", "description": "Chart title."}
            },
            "required": ["data", "values_column", "names_column", "title"]
        }
    },
    {
        "name": "create_bar_chart",
        "description": "Use for comparisons across categories (e.g., top states by recharge).",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "List of dicts (DataFrame rows)."},
                "x_column": {"type": "string", "description": "Column for x-axis."},
                "y_column": {"type": "string", "description": "Column for y-axis (numeric)."},
                "title": {"type": "string", "description": "Chart title."},
                "color": {"type": "string", "description": "Column for color grouping.", "nullable": True},
                "orientation": {"type": "string", "description": "'v' for vertical, 'h' for horizontal.", "default": "v"}
            },
            "required": ["data", "x_column", "y_column", "title"]
        }
    },
    {
        "name": "create_scatter_plot",
        "description": "Use for correlations (e.g., rainfall vs. recharge).",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "List of dicts (DataFrame rows)."},
                "x_column": {"type": "string", "description": "Column for x-axis (numeric)."},
                "y_column": {"type": "string", "description": "Column for y-axis (numeric)."},
                "title": {"type": "string", "description": "Chart title."},
                "color": {"type": "string", "description": "Column for color grouping.", "nullable": True},
                "size": {"type": "string", "description": "Column for point sizes.", "nullable": True}
            },
            "required": ["data", "x_column", "y_column", "title"]
        }
    },
    {
        "name": "create_box_plot",
        "description": "Use for statistical distributions (e.g., recharge by state).",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "List of dicts (DataFrame rows)."},
                "x_column": {"type": "string", "description": "Column for x-axis (categorical)."},
                "y_column": {"type": "string", "description": "Column for y-axis (numeric)."},
                "title": {"type": "string", "description": "Chart title."},
                "color": {"type": "string", "description": "Column for color grouping.", "nullable": True}
            },
            "required": ["data", "x_column", "y_column", "title"]
        }
    },
    {
        "name": "create_violin_plot",
        "description": "Use for density and distribution (e.g., extraction by region).",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "List of dicts (DataFrame rows)."},
                "x_column": {"type": "string", "description": "Column for x-axis (categorical)."},
                "y_column": {"type": "string", "description": "Column for y-axis (numeric)."},
                "title": {"type": "string", "description": "Chart title."},
                "color": {"type": "string", "description": "Column for color grouping.", "nullable": True}
            },
            "required": ["data", "x_column", "y_column", "title"]
        }
    },
    {
        "name": "create_heatmap",
        "description": "Use for intensity or correlations (e.g., recharge vs. extraction).",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "List of dicts (DataFrame rows)."},
                "x_column": {"type": "string", "description": "Column for x-axis."},
                "y_column": {"type": "string", "description": "Column for y-axis."},
                "z_column": {"type": "string", "description": "Column for intensity (numeric)."},
                "title": {"type": "string", "description": "Chart title."}
            },
            "required": ["data", "x_column", "y_column", "z_column", "title"]
        }
    },
    {
        "name": "create_density_contour",
        "description": "Use for density of two variables (e.g., rainfall vs. recharge).",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "List of dicts (DataFrame rows)."},
                "x_column": {"type": "string", "description": "Column for x-axis (numeric)."},
                "y_column": {"type": "string", "description": "Column for y-axis (numeric)."},
                "title": {"type": "string", "description": "Chart title."},
                "color": {"type": "string", "description": "Column for color grouping.", "nullable": True}
            },
            "required": ["data", "x_column", "y_column", "title"]
        }
    },
    {
        "name": "create_density_heatmap",
        "description": "Use for density heatmap of two variables.",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "List of dicts (DataFrame rows)."},
                "x_column": {"type": "string", "description": "Column for x-axis (numeric)."},
                "y_column": {"type": "string", "description": "Column for y-axis (numeric)."},
                "title": {"type": "string", "description": "Chart title."},
                "color": {"type": "string", "description": "Column for color grouping.", "nullable": True}
            },
            "required": ["data", "x_column", "y_column", "title"]
        }
    },
    {
        "name": "create_area_chart",
        "description": "Use for cumulative trends (e.g., total recharge over years).",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "List of dicts (DataFrame rows)."},
                "x_column": {"type": "string", "description": "Column for x-axis (e.g., year)."},
                "y_column": {"type": "string", "description": "Column for y-axis (numeric)."},
                "title": {"type": "string", "description": "Chart title."},
                "color": {"type": "string", "description": "Column for color grouping.", "nullable": True}
            },
            "required": ["data", "x_column", "y_column", "title"]
        }
    },
    {
        "name": "create_funnel_chart",
        "description": "Use for stages or drop-offs (e.g., water allocation stages).",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "List of dicts (DataFrame rows)."},
                "values_column": {"type": "string", "description": "Column with numeric values."},
                "names_column": {"type": "string", "description": "Column with category names."},
                "title": {"type": "string", "description": "Chart title."}
            },
            "required": ["data", "values_column", "names_column", "title"]
        }
    },
    {
        "name": "create_timeline_chart",
        "description": "Use for events or milestones (e.g., assessment years).",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "List of dicts (DataFrame rows)."},
                "x_start": {"type": "string", "description": "Column for start dates."},
                "x_end": {"type": "string", "description": "Column for end dates."},
                "y_column": {"type": "string", "description": "Column for y-axis (categorical)."},
                "title": {"type": "string", "description": "Chart title."},
                "color": {"type": "string", "description": "Column for color grouping.", "nullable": True}
            },
            "required": ["data", "x_start", "x_end", "y_column", "title"]
        }
    },
    {
        "name": "create_sunburst_chart",
        "description": "Use for hierarchical data (e.g., recharge by state and district).",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "List of dicts (DataFrame rows)."},
                "path_columns": {"type": "array", "items": {"type": "string"}, "description": "List of columns for hierarchy (e.g., ['state', 'district'])."},
                "values_column": {"type": "string", "description": "Column with numeric values."},
                "title": {"type": "string", "description": "Chart title."}
            },
            "required": ["data", "path_columns", "values_column", "title"]
        }
    },
    {
        "name": "create_treemap_chart",
        "description": "Use for hierarchical data in rectangular layout.",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "List of dicts (DataFrame rows)."},
                "path_columns": {"type": "array", "items": {"type": "string"}, "description": "List of columns for hierarchy."},
                "values_column": {"type": "string", "description": "Column with numeric values."},
                "title": {"type": "string", "description": "Chart title."}
            },
            "required": ["data", "path_columns", "values_column", "title"]
        }
    },
    {
        "name": "create_icicle_chart",
        "description": "Use for hierarchical data in icicle layout.",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "List of dicts (DataFrame rows)."},
                "path_columns": {"type": "array", "items": {"type": "string"}, "description": "List of columns for hierarchy."},
                "values_column": {"type": "string", "description": "Column with numeric values."},
                "title": {"type": "string", "description": "Chart title."}
            },
            "required": ["data", "path_columns", "values_column", "title"]
        }
    },
    {
        "name": "create_parallel_coordinates",
        "description": "Use for multi-variable comparisons (e.g., recharge, extraction, rainfall).",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "List of dicts (DataFrame rows)."},
                "dimensions": {"type": "array", "items": {"type": "string"}, "description": "List of numeric columns."},
                "title": {"type": "string", "description": "Chart title."}
            },
            "required": ["data", "dimensions", "title"]
        }
    },
    {
        "name": "create_parallel_categories",
        "description": "Use for categorical comparisons (e.g., extraction stages by state).",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "List of dicts (DataFrame rows)."},
                "dimensions": {"type": "array", "items": {"type": "string"}, "description": "List of categorical columns."},
                "title": {"type": "string", "description": "Chart title."}
            },
            "required": ["data", "dimensions", "title"]
        }
    },
    {
        "name": "create_choropleth",
        "description": "Use for geographic visualizations by state or district.",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "List of dicts (DataFrame rows)."},
                "locations_column": {"type": "string", "description": "Column with state or district names."},
                "color_column": {"type": "string", "description": "Column for color intensity (numeric)."},
                "title": {"type": "string", "description": "Chart title."},
                "locationmode": {"type": "string", "description": "Geo location mode.", "default": "country names"}
            },
            "required": ["data", "locations_column", "color_column", "title"]
        }
    },
    {
        "name": "create_scatter_geo",
        "description": "Use for point-based geo visualizations with lat/long.",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "List of dicts (DataFrame rows)."},
                "lat_column": {"type": "string", "description": "Column with latitude (numeric)."},
                "lon_column": {"type": "string", "description": "Column with longitude (numeric)."},
                "size_column": {"type": "string", "description": "Column for point sizes.", "nullable": True},
                "color_column": {"type": "string", "description": "Column for color.", "nullable": True},
                "title": {"type": "string", "description": "Chart title."}
            },
            "required": ["data", "lat_column", "lon_column", "title"]
        }
    }
]