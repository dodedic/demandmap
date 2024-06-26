import pandas as pd
import plotly.graph_objects as go
import json
import sys
from pathlib import Path

# Ensure the correct module path is added
current_directory = Path(__file__).resolve().parent
api_aerodatabox_path = current_directory.parents[0] / 'api_aerodatabox'
sys.path.insert(0, str(api_aerodatabox_path))

import data_transformation_pandas

def create_flight_connections_plot(fig, plot_whole_year, month=None):
    # File path for storing flight connection data
    file_path = api_aerodatabox_path / "connection_data" / "flight_connections_year.json"

    # Define the list of months
    month_list = ["01-January", "02-February", "03-March", "04-April", "05-May", "06-June", "07-July", "08-August", "09-September", "10-October", "11-November", "12-December"]

    # If the "flight_connections_year" file doesn't exist, generate and save flight connection data
    if not file_path.exists():
        all_connections = []  # List to store all flight connections

        # Process flight connections for January to create initial DataFrame
        connections_df, _ = data_transformation_pandas.process_flight_connections("01-January")

        # Iterate over month_list to gather and concatenate data
        for month in month_list[1:]:
            connections_data, _ = data_transformation_pandas.process_flight_connections(month)
            connections_df = pd.concat([connections_df, connections_data], ignore_index=True)

        # Consolidate connections with the same departure and destination
        connections_df = connections_df.groupby(["icao_departure", "icao_destination"], as_index=False).agg({
            "departure_airport_name": "first",
            "lat_departure": "first",
            "lon_departure": "first",
            "lat_destination": "first",
            "lon_destination": "first",
            "averageDailyFlights": lambda x: x.sum() / 12  # Divide the sum by 12 for annual average
        })

        # Convert DataFrame to dictionary and save as JSON file
        all_connections.append(connections_df.to_dict(orient='records'))
        with open(file_path, 'w') as f:
            json.dump(all_connections, f, indent=4)
        print(f"All connections saved to {file_path}")
    else:
        print(f"The file '{file_path}' already exists. File will not be created again")

    # Plotting
    flight_data_df, daily_flights_df = data_transformation_pandas.process_flight_connections("Year")

    # Calculate the maximum value of 'daily_flights' column
    max_daily_flights = daily_flights_df['number_of_total_flights'].max()

    flight_data_df = flight_data_df[flight_data_df['averageDailyFlights'] > 1].reset_index(drop=True)

    # Define maximum and minimum sizes for markers
    max_size = 20
    min_size = 2

    # Scale the values between min and max sizes
    scaled_sizes = ((daily_flights_df['number_of_total_flights'] - daily_flights_df['number_of_total_flights'].min()) /
                    (daily_flights_df['number_of_total_flights'].max() - daily_flights_df['number_of_total_flights'].min()) *
                    (max_size - min_size)) + min_size

    # Plot the different airport markers
    fig.add_trace(go.Scattergeo(
        lat=daily_flights_df["lat_departure"],
        lon=daily_flights_df["lon_departure"],
        text=daily_flights_df["departure_airport_name"] + '<br>'
        + "Number of average departing daily flights: "
        + daily_flights_df["number_of_total_flights"].apply(lambda x: round(x, 2)).astype(str),  # Round to 2 decimal places
        hoverinfo='text',  # Set hoverinfo to 'text'
        mode='markers',
        showlegend=False,
        marker=dict(
            size=scaled_sizes,  # Scale the sizes between min and max sizes
            opacity=0.9,
            line_width=0,
            autocolorscale=False,
            colorscale='Bluered',
            cmin=0,
            color=daily_flights_df['number_of_total_flights'],
            cmax=max_daily_flights,
            colorbar_title="Average flights per day",
        )
    ))

    # Define maximum and minimum opacities for flight connections
    max_opacity = 1
    min_opacity = 0.0

    # Scale the opacities between min and max opacities
    scaled_opacities = ((flight_data_df['averageDailyFlights'] - flight_data_df['averageDailyFlights'].min()) /
                        (flight_data_df['averageDailyFlights'].max() - flight_data_df['averageDailyFlights'].min()) *
                        (max_opacity - min_opacity)) + min_opacity

    # Plot all the different connections
    for i in range(len(flight_data_df)):
        fig.add_trace(
            go.Scattergeo(
                lon=[flight_data_df['lon_departure'][i], flight_data_df['lon_destination'][i]],
                lat=[flight_data_df['lat_departure'][i], flight_data_df['lat_destination'][i]],
                mode='lines',
                line=dict(width=1, color='white'),
                opacity=scaled_opacities[i],  # Use scaled opacities
                hoverinfo='skip',
                showlegend=False,
            )
        )

    return fig
