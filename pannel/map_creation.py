import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
import airport_check  # Assuming airport_check is a module or script containing airport_location function

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

    margin=dict(l=10, r=10, t=10, b=70),
    legend=dict(
        y=0,  # Position the legend below the map
        x=0.5,
        xanchor='center',
        yanchor='top'
    ),
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
            showlegend=True, # Ensure legend is shown
            hoverinfo='text',  # Display text when hovering
            text=f"Departure: {location}"  # Custom text to display
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
            showlegend=True,  # Ensure legend is shown
            hoverinfo='text',  # Display text when hovering
            text=f"Departure: {location}"  # Custom text to display
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
        showlegend=False,  # Do not show flight path in the legend
        hoverinfo=None
    )
    fig.add_trace(flight_line)
