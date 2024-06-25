import panel as pn
import plotly.graph_objects as go
import airport_check  # Assuming airport_check contains the ICAO_check function

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

# Callback to validate and update departure marker on input change
@pn.depends(icao_departure_input, watch=True)
def validate_departure(value):
    if airport_check.ICAO_check(value):
        icao_departure_input.css_classes = ["validation-success"]
        add_airport_marker_departure(value)
    else:
        icao_departure_input.css_classes = ["validation-error"]

# Callback to validate and update destination marker on input change
@pn.depends(icao_destination_input, watch=True)
def validate_destination(value):
    if airport_check.ICAO_check(value):
        icao_destination_input.css_classes = ["validation-success"]
        add_airport_marker_destination(value)
    else:
        icao_destination_input.css_classes = ["validation-error"]

# Create a Panel pane for the Plotly figure with custom CSS class
map_pane = pn.pane.Plotly(fig, css_classes=['panel-column'])  # Apply custom CSS class

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
    pn.Row(
        map_pane,
        align='start',
        sizing_mode='stretch_both'  # Ensure the map pane stretches to fill available space
    ),
    sizing_mode='stretch_width',  # Ensure the entire layout stretches horizontally
)

layout.servable()



"""
# Create the template
template = pn.template.MaterialTemplate(title="Aviation Forecast")

# Add the components to the template
template.sidebar.append(select)
template.sidebar.append(icao_departure_input)
template.sidebar.append(icao_destination_input)
template.sidebar.append(year)
template.sidebar.append(load_factor)

template.main.append(title_pane)
template.main.append(map_pane)

# Serve the template
template.servable()
"""