#######################################
# IMPORTS #############################
#######################################

import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from pathlib import Path


#######################################
# Function to prepare airport data
#######################################

"""
    This function reads a list with all airports with available data from the
    available_airports.json, extracts detailed information for each of those
    single airports from their respective JSON files, and compiles the data
    into a GeoDataFrame.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing information about
                          the airports, including ICAO code, name, latitude,
                          longitude, and geometry. Returns None if there is
                          an error in reading the JSON data.


    Each airport_data/airports_detail_data/{ICAO_CODE}.json file contains the
    detailed information for the corresponding airport.

    Example JSON Structure:

        available_airports.json:
        {
            "items": ["ICAO1", "ICAO2", "ICAO3"]
        }

        {ICAO_CODE}.json:
        {
            "icao":	    "CYWK"
            "fullName": "Airport Name",
            "location": {
                "lat": 12.34,
                "lon": 56.78
            }
        }

        The function will return a DataFrame of the form:
    +--------------+-----------+-------------------------+----------+------------+------+-----------+
    | UID (=index) | icao      | airport_name            | lat      | lon        | geometry         |
    +==============+===========+=========================+==========+============+==================+
    | 123          | CYWK      | Wabush                  | 52.9219  | -66.8644   | -66.8644 52.9219 |
    +--------------+-----------+-------------------------+----------+------------+------+------------
    | 456          | LEST      | Santiago de Compostela  | 42.8963  | -8.41514   | -8.41514 42.8963 |
    +--------------+-----------+-------------------------+----------+------------+------+------------
    """


def prepare_airport_data():
    # Determine the current directory
    current_directory = Path(__file__).resolve().parent

    # Add the path to the api_aerodatabox folder to sys.path
    api_aerodatabox_path = current_directory.parents[0] / 'api_aerodatabox'

    # Read airports data
    file_path = api_aerodatabox_path / "airport_data/Available_Airports.json"
    try:
        with open(file_path, 'r') as f:
            airports_icao = json.load(f)
    except json.decoder.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {file_path}: {e}")
        return None

    # Extract airport information
    airport_list = [{'icao': airport_loc} for airport_loc in airports_icao['items']]
    airport_list_df = pd.DataFrame(airport_list)

    # Create airport DataFrame
    airport_info_list = []
    for icao_airport in airport_list_df['icao']:
        file_path = api_aerodatabox_path / f"airport_data/airports_detail_data/{icao_airport}.json"
        if file_path.exists():
            with open(file_path, 'r',encoding='utf-8') as f:
                airport_info = json.load(f)
            airport = {
                'icao': icao_airport,
                'airport_name': airport_info['fullName'],
                'lat': airport_info['location']['lat'],
                'lon': airport_info['location']['lon'],
            }
            airport_info_list.append(airport)
        else:
            print(f"Data for {icao_airport} not found. Skipping...")
    airports_info_df = pd.DataFrame(airport_info_list)

    # Create GeoDataFrame
    departure_airports_geodf = gpd.GeoDataFrame(
        airports_info_df,
        geometry=gpd.points_from_xy(
            x=airports_info_df["lon"],
            y=airports_info_df["lat"],
            crs='EPSG:4326'
        )
    )
    return departure_airports_geodf
