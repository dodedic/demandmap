import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go
import plotly.express as px
import scipy.sparse
import json
import airport_check  # Assuming airport_check contains the ICAO_check function

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
                           width=100, )

year = pn.widgets.IntSlider(name='Year', start=2023, end=2050, step=1, value=2023, width=200)
load_factor = pn.widgets.FloatSlider(name='Load Factor', start=0, end=1, step=0.01, value=0.8, width=200)

# Create the blank world map
fig = go.Figure(data=go.Choropleth(
    locations=[],  # No data for countries
    z=[],  # No data for color scale
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
    height=650,
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

# Function to get the scaling factors from GDP data based on departure ICAO code
def get_scaling_factors(departure_code):
    scaling_factors = []
    first_two_letters = departure_code[:2]
    country_code = country_codes.get(first_two_letters)
    if country_code is None:
        return scaling_factors
    
    # Find the row corresponding to the country code in GDPData.csv
    country_row = gdp_data[gdp_data['Country'] == country_code]
    if country_row.empty:
        return scaling_factors
    
    # Starting from 2024, get the GDP growth rate until 2050 or until there is not a number anymore
    for year in range(2024, 2051):
        column_name = str(year)
        if column_name in country_row.columns:
            scaling_factors.append(country_row[column_name].values[0] / 100)  # Divide by 100 to get as a percentage
        else:
            break
    
    return scaling_factors

# Function to get the value from the sparse matrix
def get_sparse_value(departure_code, destination_code):
    try:
        departure_idx = np.where(row_labels == departure_code)[0][0]
        destination_idx = np.where(col_labels == destination_code)[0][0]
        return sparse_matrix[departure_idx, destination_idx]
    except IndexError:
        return None

# Create a DataFrame for the additional data
df = pd.DataFrame({
    'Year': list(range(2024, 2051)),
    'Seats': [0] * 27
})

# Callback to update the Seats value based on ICAO codes
@pn.depends(icao_departure_input.param.value, icao_destination_input.param.value, watch=True)
def update_seats(departure_code, destination_code):
    scaling_factors = get_scaling_factors(departure_code)
    value = get_sparse_value(departure_code, destination_code)

    print(f"Initial value from sparse matrix: {value}")  # Debugging line
    if value is not None:
        df.at[0, 'Seats'] = value
    
    if scaling_factors:
        for i in range(1, len(df)):
            if i < len(scaling_factors):
                scaling_factor = scaling_factors[i - 1]  # Adjust index to start from 2024
            else:
                scaling_factor = scaling_factors[-1]  # Use the last available scaling factor
            
            df.at[i, 'Seats'] = df.at[i - 1, 'Seats'] * (1 + scaling_factor)
            
            print(f"Year: {df.at[i, 'Year']}, Seats: {df.at[i, 'Seats']}, Scaling Factor: {scaling_factor}")  # Debugging line

    # Update the DataFrame and line plot
    styled_data = df.style.set_table_styles({
        'Year': [{'selector': 'th', 'props': [('width', '100px')]}],
        'Seats': [{'selector': 'th', 'props': [('width', '100px')]}]
    }).hide(axis='index').to_html()

    dataframe_pane.object = styled_data
    line_fig = px.line(df, x='Year', y='Seats', title='Seats Forecast', markers=True)
    line_graph_pane.object = line_fig

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

# Create the initial line plot using Plotly Express
line_fig = px.line(df, x='Year', y='Seats', title='Seats Forecast', markers=True)

# Convert the DataFrame to an HTML table
styled_data = df.style.set_table_styles({
    'Year': [{'selector': 'th', 'props': [('width', '100px')]}],
    'Seats': [{'selector': 'th', 'props': [('width', '100px')]}]
}).hide(axis='index').to_html()

# Create a Panel pane for the Plotly figure with custom CSS class
map_pane = pn.pane.Plotly(fig, css_classes=['panel-column'])  # Apply custom CSS class

# Create a Panel HTML pane to display the DataFrame
dataframe_pane = pn.pane.HTML(styled_data, width=400, height=200)

# Create a Panel pane for the Plotly line graph
line_graph_pane = pn.pane.Plotly(line_fig, width=700, height=400)

# Markdown pane for the title
title_pane = pn.pane.Markdown("# Aviation Forecast")

# Spacer to separate the map and the table
spacer = pn.Spacer(height=500)

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
    pn.Row(
        map_pane,
        align='start',
        sizing_mode='stretch_both'  # Ensure the map pane stretches to fill available space
    ),
    spacer,
    pn.Row(
        dataframe_pane,
        line_graph_pane,
        align='start',
        sizing_mode='stretch_both'  # Ensure the DataFrame and line plot panes stretch to fill available space
    ),
    sizing_mode='stretch_width'  # Ensure the entire layout stretches horizontally
)

layout.servable()
