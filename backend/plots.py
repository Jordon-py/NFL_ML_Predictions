# ==========================================
# File: backend/plots.py
# Role: Backend helper module.
# Input Data: Function inputs.
# Output Data: Module outputs.
# Dependencies: streamlit, pandas, plotly, numpy
# Notes: Shared utilities.
# ==========================================

"""
Interactive Dataset Visualization Pipeline
Author: Python Data Visualization Expert
Description: A complete, extensible tool for visualizing CSV datasets with 2D and 3D plots
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from io import StringIO
import base64
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

file_content = StringIO('data/heroku-models/game_features_20251218.csv')

class DataVisualizationPipeline:
    """
    Main class for handling data visualization pipeline
    """

    def __init__(self):
        self.df = None
        self.numeric_columns = []
        self.categorical_columns = []
        self.plot_config = {
            'displayModeBar': True,
            'scrollZoom': True,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
        }

    def load_data(self, file_content: StringIO) -> bool:
        """Load CSV data and preprocess"""
        try:
            self.df = pd.read_csv(file_content)

            # Basic data cleaning
            self.df = self.df.dropna(how='all')  # Remove completely empty rows
            self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]  # Remove unnamed columns

            # Classify columns
            self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False

    def validate_plot_requirements(self, plot_type: str, selected_columns: List[str]) -> Tuple[bool, str]:
        """Validate if selected columns are appropriate for the plot type"""
        if not selected_columns:
            return False, "Please select at least one column"

        if plot_type in ['scatter', 'line', '3d_scatter']:
            if len(selected_columns) < 2:
                return False, f"{plot_type.replace('_', ' ').title()} requires at least 2 numeric columns"
            if not all(col in self.numeric_columns for col in selected_columns):
                return False, f"{plot_type.replace('_', ' ').title()} requires numeric columns"

        elif plot_type == 'histogram':
            if len(selected_columns) < 1:
                return False, "Histogram requires at least 1 numeric column"

        elif plot_type == 'box':
            if len(selected_columns) < 1:
                return False, "Box plot requires at least 1 numeric column"

        elif plot_type == '3d_surface':
            if len(selected_columns) != 3:
                return False, "3D Surface plot requires exactly 3 numeric columns"
            # Check if data can be pivoted for surface plot
            if len(self.df) < 9:  # Need enough points for reasonable surface
                return False, "Insufficient data points for surface plot"

        return True, "Valid"

    def create_scatter_plot(self, x_col: str, y_col: str, color_col: Optional[str] = None) -> go.Figure:
        """Create interactive scatter plot"""
        if color_col and color_col in self.categorical_columns:
            fig = px.scatter(self.df, x=x_col, y=y_col, color=color_col,
                           title=f"Scatter Plot: {x_col} vs {y_col}",
                           labels={x_col: x_col, y_col: y_col})
        else:
            fig = px.scatter(self.df, x=x_col, y=y_col,
                           title=f"Scatter Plot: {x_col} vs {y_col}",
                           labels={x_col: x_col, y_col: y_col})

        fig.update_traces(marker=dict(size=8, opacity=0.7),
                         selector=dict(mode='markers'))
        return fig

    def create_line_plot(self, x_col: str, y_cols: List[str]) -> go.Figure:
        """Create interactive line plot"""
        fig = go.Figure()

        for y_col in y_cols:
            fig.add_trace(go.Scatter(x=self.df[x_col], y=self.df[y_col],
                                   mode='lines+markers',
                                   name=y_col,
                                   line=dict(width=2)))

        fig.update_layout(title=f"Line Plot: {x_col} vs Multiple Series",
                         xaxis_title=x_col,
                         yaxis_title="Value",
                         hovermode='x unified')
        return fig

    def create_histogram(self, columns: List[str], bins: int = 30) -> go.Figure:
        """Create interactive histogram"""
        fig = go.Figure()

        for col in columns:
            fig.add_trace(go.Histogram(x=self.df[col],
                                     name=col,
                                     nbinsx=bins,
                                     opacity=0.7))

        fig.update_layout(title=f"Histogram of {', '.join(columns)}",
                         xaxis_title="Value",
                         yaxis_title="Frequency",
                         barmode='overlay')
        fig.update_traces(marker=dict(line=dict(width=1, color='black')))
        return fig

    def create_box_plot(self, columns: List[str], color_col: Optional[str] = None) -> go.Figure:
        """Create interactive box plot"""
        if color_col and color_col in self.categorical_columns:
            fig = px.box(self.df, y=columns, color=color_col,
                        title=f"Box Plot: {', '.join(columns)}")
        else:
            fig = px.box(self.df, y=columns,
                        title=f"Box Plot: {', '.join(columns)}")

        fig.update_layout(xaxis_title="Variables",
                         yaxis_title="Value")
        return fig

    def create_3d_scatter_plot(self, x_col: str, y_col: str, z_col: str,
                              color_col: Optional[str] = None) -> go.Figure:
        """Create interactive 3D scatter plot"""
        if color_col and color_col in self.df.columns:
            fig = px.scatter_3d(self.df, x=x_col, y=y_col, z=z_col, color=color_col,
                              title=f"3D Scatter Plot: {x_col}, {y_col}, {z_col}")
        else:
            fig = px.scatter_3d(self.df, x=x_col, y=y_col, z=z_col,
                              title=f"3D Scatter Plot: {x_col}, {y_col}, {z_col}")

        fig.update_traces(marker=dict(size=5, opacity=0.8))
        return fig

    def create_3d_surface_plot(self, x_col: str, y_col: str, z_col: str) -> go.Figure:
        """Create interactive 3D surface plot"""
        try:
            # Prepare data for surface plot
            pivot_df = self.df.pivot_table(values=z_col, index=x_col, columns=y_col, aggfunc='mean')

            # Create surface plot
            fig = go.Figure(data=[go.Surface(z=pivot_df.values,
                                           x=pivot_df.columns,
                                           y=pivot_df.index)])

            fig.update_layout(title=f"3D Surface Plot: {z_col} = f({x_col}, {y_col})",
                            scene=dict(xaxis_title=x_col,
                                      yaxis_title=y_col,
                                      zaxis_title=z_col))
            return fig
        except Exception as e:
            st.error(f"Could not create surface plot: {str(e)}")
            # Fallback to 3D scatter if surface fails
            return self.create_3d_scatter_plot(x_col, y_col, z_col)

    def export_plot(self, fig: go.Figure, format_type: str) -> str:
        """Export plot in various formats"""
        try:
            if format_type == "HTML":
                return fig.to_html(include_plotlyjs=True)
            elif format_type == "PNG":
                return fig.to_image(format="png", scale=2)
            elif format_type == "JPEG":
                return fig.to_image(format="jpeg", scale=2)
        except Exception as e:
            st.error(f"Export error: {str(e)}")
            return None

    def get_data_summary(self) -> Dict:
        """Get basic data summary"""
        if self.df is None:
            return {}

        return {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'numeric_columns': len(self.numeric_columns),
            'categorical_columns': len(self.categorical_columns),
            'missing_values': self.df.isnull().sum().sum()
        }

def main():
    """Main Streamlit application"""
    st.set_page_config(page_title="Data Visualization Pipeline", page_icon="📊", layout="wide")

    st.title("📊 Interactive Dataset Visualization Pipeline")
    st.markdown("""
    Upload your CSV file and create interactive visualizations with 6 different plot types.
    **Features:** 2D Scatter, Line, Histogram, Box plots | 3D Scatter and Surface plots
    """)

    # Initialize pipeline
    pipeline = DataVisualizationPipeline()

    # Sidebar for file upload and controls
    st.sidebar.header("📁 Data Input")
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])

    if uploaded_file is not None:
        # Load and display data
        if pipeline.load_data(uploaded_file):
            st.sidebar.success("✅ Data loaded successfully!")

            # Display data summary
            summary = pipeline.get_data_summary()
            st.sidebar.subheader("📈 Data Summary")
            st.sidebar.write(f"Rows: {summary['total_rows']} | Columns: {summary['total_columns']}")
            st.sidebar.write(f"Numerical: {summary['numeric_columns']} | Categorical: {summary['categorical_columns']}")
            st.sidebar.write(f"Missing Values: {summary['missing_values']}")

            # Show data preview
            with st.expander("🔍 Data Preview", expanded=False):
                st.dataframe(pipeline.df.head(10), use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Numerical Columns:**", pipeline.numeric_columns)
                with col2:
                    st.write("**Categorical Columns:**", pipeline.categorical_columns)

            # Plot configuration
            st.sidebar.header("🎨 Visualization Settings")
            plot_type = st.sidebar.selectbox(
                "Select Plot Type",
                ["scatter", "line", "histogram", "box", "3d_scatter", "3d_surface"],
                format_func=lambda x: x.replace('_', ' ').title()
            )

            # Column selection based on plot type
            available_columns = pipeline.numeric_columns + pipeline.categorical_columns

            if plot_type in ['scatter', 'line']:
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    x_col = st.selectbox("X-axis", pipeline.numeric_columns)
                with col2:
                    if plot_type == 'scatter':
                        y_col = st.selectbox("Y-axis", pipeline.numeric_columns, index=min(1, len(pipeline.numeric_columns)-1))
                    else:  # line plot
                        y_cols = st.multiselect("Y-axis (multiple)", pipeline.numeric_columns, default=pipeline.numeric_columns[:1])

                color_col = st.sidebar.selectbox("Color by (optional)", [None] + pipeline.categorical_columns)

            elif plot_type == 'histogram':
                hist_cols = st.sidebar.multiselect("Select columns", pipeline.numeric_columns, default=pipeline.numeric_columns[:1])
                bins = st.sidebar.slider("Number of bins", 5, 100, 30)

            elif plot_type == 'box':
                box_cols = st.sidebar.multiselect("Select columns", pipeline.numeric_columns, default=pipeline.numeric_columns[:1])
                box_color = st.sidebar.selectbox("Group by (optional)", [None] + pipeline.categorical_columns)

            elif plot_type == '3d_scatter':
                col1, col2, col3 = st.sidebar.columns(3)
                with col1:
                    x_3d = st.selectbox("X-axis", pipeline.numeric_columns)
                with col2:
                    y_3d = st.selectbox("Y-axis", pipeline.numeric_columns, index=min(1, len(pipeline.numeric_columns)-1))
                with col3:
                    z_3d = st.selectbox("Z-axis", pipeline.numeric_columns, index=min(2, len(pipeline.numeric_columns)-1))
                color_3d = st.sidebar.selectbox("Color by", [None] + available_columns)

            elif plot_type == '3d_surface':
                st.sidebar.info("Surface plots work best with grid-like data")
                col1, col2, col3 = st.sidebar.columns(3)
                with col1:
                    x_surf = st.selectbox("X-axis (index)", pipeline.numeric_columns)
                with col2:
                    y_surf = st.selectbox("Y-axis (columns)", pipeline.numeric_columns, index=min(1, len(pipeline.numeric_columns)-1))
                with col3:
                    z_surf = st.selectbox("Z-axis (values)", pipeline.numeric_columns, index=min(2, len(pipeline.numeric_columns)-1))

            # Generate plot button
            if st.sidebar.button("🚀 Generate Plot", type="primary"):
                try:
                    # Create the selected plot
                    if plot_type == 'scatter':
                        fig = pipeline.create_scatter_plot(x_col, y_col, color_col)

                    elif plot_type == 'line':
                        fig = pipeline.create_line_plot(x_col, y_cols)

                    elif plot_type == 'histogram':
                        fig = pipeline.create_histogram(hist_cols, bins)

                    elif plot_type == 'box':
                        fig = pipeline.create_box_plot(box_cols, box_color)

                    elif plot_type == '3d_scatter':
                        fig = pipeline.create_3d_scatter_plot(x_3d, y_3d, z_3d, color_3d)

                    elif plot_type == '3d_surface':
                        fig = pipeline.create_3d_surface_plot(x_surf, y_surf, z_surf)

                    # Display the plot
                    st.plotly_chart(fig, use_container_width=True, config=pipeline.plot_config)

                    # Export options
                    st.sidebar.header("💾 Export Options")
                    export_format = st.sidebar.selectbox("Export Format", ["HTML", "PNG", "JPEG"])

                    if st.sidebar.button("📥 Export Plot"):
                        export_data = pipeline.export_plot(fig, export_format)
                        if export_data:
                            if export_format == "HTML":
                                b64 = base64.b64encode(export_data.encode()).decode()
                                href = f'<a href="data:file/html;base64,{b64}" download="plot.html">Download HTML Plot</a>'
                                st.sidebar.markdown(href, unsafe_allow_html=True)
                            else:
                                # For image formats
                                b64 = base64.b64encode(export_data).decode()
                                href = f'<a href="data:image/{export_format.lower()};base64,{b64}" download="plot.{export_format.lower()}">Download {export_format} Plot</a>'
                                st.sidebar.markdown(href, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error creating plot: {str(e)}")
                    st.info("Please check your column selections and try again.")

        else:
            st.error("Failed to load data. Please check your CSV file format.")

    else:
        # Demo and instructions
        st.info("👆 Please upload a CSV file to get started")

        # Sample usage instructions
        with st.expander("📖 How to Use This Tool", expanded=True):
            st.markdown("""
            ### Step-by-Step Guide:

            1. **Upload Data**: Click "Browse files" in the sidebar to upload your CSV file
            2. **Explore Data**: Check the data preview and summary statistics
            3. **Choose Plot Type**: Select from 6 different visualization types:
               - **2D Plots**: Scatter, Line, Histogram, Box plots
               - **3D Plots**: 3D Scatter and Surface plots
            4. **Configure**: Select appropriate columns for each axis
            5. **Generate**: Click the "Generate Plot" button
            6. **Export**: Download your plot as HTML, PNG, or JPEG

            ### Tips for Best Results:
            - Ensure your CSV has headers in the first row
            - Clean your data of unnecessary empty rows/columns
            - For 3D Surface plots, use data that forms a grid pattern
            - Use categorical columns for color coding scatter and box plots
            """)

        # Sample data option
        if st.button("Try with Sample Data"):
            # Create sample data
            np.random.seed(42)
            sample_data = pd.DataFrame({
                'x': np.random.randn(100),
                'y': 2 * np.random.randn(100) + 1,
                'z': np.random.randn(100) * 0.5 + 2,
                'category': np.random.choice(['A', 'B', 'C'], 100),
                'value': np.random.exponential(2, 100)
            })

            # Convert to CSV string and create file-like object
            csv_string = file_content.to_csv(index=False)
            file_like_object = StringIO(csv_string)

            if pipeline.load_data(file_like_object):
                st.success("Sample data loaded! Configure your plot in the sidebar.")
                st.dataframe(pipeline.df.head(), use_container_width=True)

if __name__ == "__main__":
    main()
