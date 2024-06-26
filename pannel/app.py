import pandas as pd
import panel as pn
import plotly.graph_objects as go
import plotly.express as px
import scipy.sparse
import numpy as np
import json

import airport_check  # Assuming airport_check contains the ICAO_check and airport_location functions

pn.extension('plotly')

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

pn.extension(raw_css=[raw_css])

# Load the sparse matrix and labels
sparse_matrix = scipy.sparse.load_npz('01-January.npz')
labels = np.load('01-January_labels.npz', allow_pickle=True)

# Load the row and column labels
row_labels = labels['rows']
col_labels = labels['cols']

# Load the country codes from JSON file
with open('CountryCodes.json', 'r') as file:
    country_codes = json.load(file)

# Load the GDP data
gdp_data = pd.read_csv('GDPData.csv')

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

select = pn.widgets.Select(name='Legs', 
                           options=['One-way', 'Round-trip'], 
                           width=100,)

year = pn.widgets.IntSlider(name='Year', start=2023, end=2050, step=1, value=2023, width=200)
load_factor = pn.widgets.FloatSlider(name='Load Factor', start=0, end=1, step=0.01, value=0.8, width=200)

# Create the blank world map
fig = go.Figure(data=go.Choropleth(
    locations=[],  # No data for countries
    z=[],          # No data for color scale
))

# Update the layout for the map
fig.update_layout(
    geo=dict(
        showframe=True,
        projection_type="natural earth",
        showcoastlines=True, coastlinecolor="lightgrey",
        showland=True, landcolor="black",
        showocean=True, oceancolor="dimgrey",
        showlakes=True, lakecolor="black",
        showcountries=True, countrycolor="lightgrey",
    ),
    width=1200,  # Adjust the width of the figure
    height=700,
    margin=dict(l=10, r=10, t=10, b=70),
    legend=dict(
        y=0,  # Position the legend below the map
        x=0.5,
        xanchor='center',
        yanchor='top'
    )
)

# Initialize departure, destination markers, and line
departure_marker = None
destination_marker = None
flight_line = None

# Function to retrieve airport location and add/update marker on map
def add_airport_marker_departure(location):
    global departure_marker, flight_line
    lat, lon = airport_check.airport_location(location)
    if lat is not None and lon is not None:
        # Remove the previous departure marker if it exists
        if departure_marker is not None:
            fig.data = [trace for trace in fig.data if trace != departure_marker]
        
        # Add new departure marker
        departure_marker = go.Scattergeo(
            lon=[lon],
            lat=[lat],
            mode='markers',
            marker=dict(
                size=10,
                color='red',
            ),
            name=f"Departure: {location}",  # Add ICAO code to legend
            legendgroup='departure',
            legendrank=1,
            showlegend=True  # Ensure legend is shown
        )
        fig.add_trace(departure_marker)
        
        # Update flight line if destination exists
        if destination_marker is not None:
            add_flight_line()

# Function to retrieve airport location and add/update marker on map
def add_airport_marker_destination(location):
    global destination_marker, flight_line
    lat, lon = airport_check.airport_location(location)
    if lat is not None and lon is not None:
        # Remove the previous destination marker if it exists
        if destination_marker is not None:
            fig.data = [trace for trace in fig.data if trace != destination_marker]
        
        # Add new destination marker
        destination_marker = go.Scattergeo(
            lon=[lon],
            lat=[lat],
            mode='markers',
            marker=dict(
                size=10,
                color='blue',
            ),
            name=f"Destination: {location}",  # Add ICAO code to legend
            legendgroup='destination',
            legendrank=2,
            showlegend=True  # Ensure legend is shown
        )
        fig.add_trace(destination_marker)
        
        # Update flight line if departure exists
        if departure_marker is not None:
            add_flight_line()

# Function to add/update flight line between departure and destination
def add_flight_line():
    global flight_line
    departure_lat = departure_marker['lat'][0]
    departure_lon = departure_marker['lon'][0]
    destination_lat = destination_marker['lat'][0]
    destination_lon = destination_marker['lon'][0]
    
    # Remove the previous flight line if it exists
    if flight_line is not None:
        fig.data = [trace for trace in fig.data if trace != flight_line]
    
    # Add new flight line
    flight_line = go.Scattergeo(
        lon=[departure_lon, destination_lon],
        lat=[departure_lat, destination_lat],
        mode='lines',
        line=dict(width=2, color='green'),
        showlegend=False  # Do not show flight path in the legend
    )
    fig.add_trace(flight_line)

# Function to get the GDP growth rates for a country
def get_gdp_growth_rates(departure_code):
    country_code = country_codes.get(departure_code[:2])
    if country_code:
        gdp_row = gdp_data[gdp_data.iloc[:, 1] == country_code]
        if not gdp_row.empty:
            growth_rates = gdp_row.loc[:, '2024':].values.flatten() / 100  # Assuming the GDP growth is in percentage
            return growth_rates
    return None

# Function to get the value from the sparse matrix
def get_sparse_value(departure_code, destination_code):
    try:
        departure_idx = np.where(row_labels == departure_code)[0][0]
        destination_idx = np.where(col_labels == destination_code)[0][0]
        return sparse_matrix[departure_idx, destination_idx]
    except IndexError:
        return None

# Function to create a DataFrame for the table with projected values
def create_forecast_dataframe(initial_value, growth_rates):
    years = list(range(2024, 2051))
    seats = [initial_value]
    last_growth_rate = growth_rates[0]
    for i in range(1, len(years)):
        if i < len(growth_rates):
            growth_rate = growth_rates[i-1]
        else:
            growth_rate = last_growth_rate
        seats.append(seats[-1] * (1 + growth_rate))
    data = pd.DataFrame({'Year': years, 'Seats': seats})
    return data

# Callback to update the table and the line graph based on the ICAO codes
@pn.depends(icao_departure_input, icao_destination_input, watch=True)
def update_forecast_table(departure_code, destination_code):
    value = get_sparse_value(departure_code, destination_code)
    growth_rates = get_gdp_growth_rates(departure_code)
    if value is not None and growth_rates is not None:
        data = create_forecast_dataframe(value, growth_rates)
        styled_data = data.style.set_table_styles({
            'Year': [{'selector': 'th', 'props': [('width', '100px')]}],
            'Seats': [{'selector': 'th', 'props': [('width', '100px')]}]
        }).hide(axis='index')
        html_data = styled_data.to_html()
        dataframe_pane.object = html_data
        line_fig = px.line(data, x='Year', y='Seats', title='Seats Forecast', markers=True)
        line_graph_pane.object = line_fig
    else:
        dataframe_pane.object = "Invalid ICAO codes or data not available."
        line_graph_pane.object = None

# Create a Panel pane for the Plotly figure with custom CSS class
map_pane = pn.pane.Plotly(fig, css_classes=['panel-column'], height=700)  # Adjusted height for the map

# Create a Panel HTML pane to display the DataFrame
dataframe_pane = pn.pane.HTML(width=400)

# Create a Panel pane for the Plotly line graph
line_graph_pane = pn.pane.Plotly(width=700)

# Markdown pane for the title
title_pane = pn.pane.Markdown("# Aviation Forecast")

# Layout: arrange vertically with title, input fields, their respective output fields, and the map below
layout = pn.Column(
    title_pane,
    pn.Row(
        select,
        icao_departure_input,
        icao_destination_input,
        year,
        load_factor,
        sizing_mode='stretch_width'
    ),
    pn.Spacer(height=20),  # Add spacer for separation
    pn.Row(
        map_pane,
        sizing_mode='stretch_both'  # Ensure the map pane stretches to fill available space
    ),
    pn.Spacer(height=20),  # Add spacer for separation
    pn.Row(
        dataframe_pane,
        line_graph_pane,
        sizing_mode='stretch_both'  # Ensure the panes stretch to fill available space
    ),
    sizing_mode='stretch_width'  # Ensure the entire layout stretches horizontally
)

layout.servable()
