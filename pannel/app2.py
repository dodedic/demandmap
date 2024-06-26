import panel as pn
import plotly.graph_objects as go
import airport_check  # Assuming airport_check contains the ICAO_check function
import flight_connections_with_pannel  # Import the function
import flight_map_utils  # Import the new module

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
fig = go.Figure()

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

# Initial call to add markers (dots) to the map
fig = flight_connections_with_pannel.create_flight_connections_plot(fig, plot_whole_year=True)

# Callback to validate and update departure marker on input change
@pn.depends(icao_departure_input, watch=True)
def validate_departure(value):
    if airport_check.ICAO_check(value):
        icao_departure_input.css_classes = ["validation-success"]
        flight_map_utils.add_airport_marker_departure(fig, value)
    else:
        icao_departure_input.css_classes = ["validation-error"]

# Callback to validate and update destination marker on input change
@pn.depends(icao_destination_input, watch=True)
def validate_destination(value):
    if airport_check.ICAO_check(value):
        icao_destination_input.css_classes = ["validation-success"]
        flight_map_utils.add_airport_marker_destination(fig, value)
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
