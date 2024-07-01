import numpy as np
import pandas as pd
import panel as pn
import plotly.express as px
import scipy.sparse
import json
import forecast_display
from forecast_display import get_scaling_factors, get_sparse_value, df
from map_creation import fig, add_airport_marker_departure, add_airport_marker_destination
import airport_check

pn.extension('plotly', 'vega')

# Define CSS class for border styling
raw_css = """
div.panel-column {
    border: 2px solid black;
}
"""

TEXT_INPUT_CSS = """
:host(.validation-error) input.bk-input {
    border-color: red !important;  /* Red border for validation error */
    background-color: rgba(255, 0, 0, 0.3) !important;  /* Red background with transparency for validation error */
}
:host(.validation-success) input.bk-input {
    border-color: green !important;  /* Green border for validation success */
}
"""

# TextInput widgets for entering ICAO codes
icao_departure_input = pn.widgets.TextInput(value='',
                                            description="Enter correct departure ICAO code",
                                            placeholder='ICAO code',
                                            name="Departure",
                                            width=100,
                                            stylesheets=[TEXT_INPUT_CSS])  # Add a CSS class for styling

icao_destination_input = pn.widgets.TextInput(value='',
                                              description="Enter correct destination ICAO code",
                                              placeholder='ICAO code',
                                              name="Destination",
                                              width=100,
                                              stylesheets=[TEXT_INPUT_CSS])  # Add a CSS class for styling

trip_indicator = pn.widgets.Select(name='Legs',
                           options=['One-way', 'Round-trip'],
                           width=100, )

time_of_year = pn.widgets.Select(name='Timeframe',
                           options=['January','February','March','April','May','June','July','August','September','October','November','December', 'Whole year'],
                           width=100, )

load_factor = pn.widgets.FloatSlider(name='Load Factor', start=0, end=1, step=0.01, value=0.8, width=200)


dataframe_pane = pn.pane.HTML(width=400, height=200)

# Create a Panel pane for the Plotly line graph
line_fig = px.line(df, x='Year', y='PAX', title='PAX Forecast', markers=True)
line_graph_pane = pn.pane.Plotly(line_fig, width=800, height=500)

# Callback to update the Seats value based on ICAO codes, load factor, time of year, and trip indicator
@pn.depends(icao_departure_input.param.value, icao_destination_input.param.value, load_factor.param.value, time_of_year.param.value, trip_indicator.param.value, watch=True)
def update_seats(departure_code, destination_code, load_factor_value, time_of_year_value, trip_indicator_value):
    # Set the time_of_year in the forecast_display module
    forecast_display.set_time_of_year(time_of_year_value)
    scaling_factors = get_scaling_factors(departure_code)
    value = get_sparse_value(departure_code, destination_code, time_of_year_value, trip_indicator_value)  

    if value is not None:
        df.at[0, 'Seats'] = value
        df.at[0, 'PAX'] = value * load_factor_value

    if scaling_factors:
        for i in range(1, len(df)):
            if i < len(scaling_factors):
                scaling_factor = scaling_factors[i - 1]  # Adjust index to start from 2024
            else:
                scaling_factor = scaling_factors[-1]  # Use the last available scaling factor
            
            df.at[i, 'Seats'] = df.at[i - 1, 'Seats'] * (1 + scaling_factor)
            df.at[i, 'PAX'] = df.at[i , 'Seats'] * load_factor_value    
            
            # Calculate percentage change
            prev_seats = df.at[i - 1, 'Seats']
            current_seats = df.at[i, 'Seats']
            if prev_seats != 0:
                percentage_change = ((current_seats - prev_seats) / prev_seats) * 100
                df.at[i, 'Percentage Change'] = round(percentage_change, 2)  # Round to 2 decimal places
            else:
                df.at[i, 'Percentage Change'] = 0.0
        

    # Update the DataFrame and line plot
    styled_data = df.style.set_table_styles({
        'Year': [{'selector': 'th', 'props': [('width', '100px')]}],
        'PAX': [{'selector': 'th', 'props': [('width', '100px')]}],
        'Percentage Change': [{'selector': 'th', 'props': [('width', '100px')]}]
    }).hide(axis='index').format({'Percentage Change': '{:.2f}%'}).to_html()  # Format percentage change column
    
    line_fig = px.line(df, x='Year', y='PAX', title='Seats Forecast', markers=True)
    line_graph_pane.object = line_fig
    dataframe_pane.object = styled_data

# Callback to validate and update departure marker on input change
@pn.depends(icao_departure_input.param.value, watch=True)
def validate_departure(value):
    if airport_check.ICAO_check(value):
        icao_departure_input.css_classes = ["validation-success"]
        add_airport_marker_departure(value)
    else:
        icao_departure_input.css_classes = ["validation-error"]

# Callback to validate and update destination marker on input change
@pn.depends(icao_destination_input.param.value, watch=True)
def validate_destination(value):
    if airport_check.ICAO_check(value):
        icao_destination_input.css_classes = ["validation-success"]
        add_airport_marker_destination(value)
    else:
        icao_destination_input.css_classes = ["validation-error"]


# Create the sidebar and template
sidebar = pn.Column(
    pn.widgets.ToggleGroup(name='ToggleGroup', options=['World view', 'Detailed view'], behavior="radio")
)

ACCENT = "#3f51b5"  # Define your accent color if not already defined

template = pn.template.FastGridTemplate(
    title="Aviation Forecast",
    accent_base_color=ACCENT,
    header_background=ACCENT,
    prevent_collision=True,
    save_layout=True,
    theme_toggle=False,
    theme='dark',
    row_height=100,
    sidebar=sidebar,
    collapsed_sidebar=True,
    sidebar_width=300,
)

# Combine all input fields into one panel
input_fields = pn.Row(
    trip_indicator,
    icao_departure_input,
    icao_destination_input,
    time_of_year,
    load_factor,
    sizing_mode='stretch_width'
)


# Input fields in the first row
template.main[0:1, 0:9] = input_fields

# Map in the second row
map_pane = pn.pane.Plotly(fig, css_classes=['panel-column'])  # Apply custom CSS class
template.main[1:8, 0:9] = map_pane

# Dataframe and graph in the third row
template.main[8:13, 0:3] = dataframe_pane
template.main[8:13, 3:9] = line_graph_pane
template.main[0:1, 0:9] = input_fields


# Serve the template
template.servable()
