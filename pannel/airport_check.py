import json
from pathlib import Path
import pandas as pd
import sys
import data_preperation

# Determine the current directory
current_directory = Path(__file__).resolve().parent

# Add the path to the api_aerodatabox folder to sys.path
api_aerodatabox_path = current_directory.parents[0] / 'api_aerodatabox'

def ICAO_check(argument):
    """
    Check if the given ICAO code exists in the list of airports.

    Args:
        argument (str): The ICAO code to check.

    Returns:
        bool: True if the ICAO code exists in the airport list, False otherwise.
    """
    # Read airports data
    file_path = api_aerodatabox_path / "airport_data" / "Available_Airports.json"
    try:
        with open(file_path, 'r') as f:
            airports_icao = json.load(f)
    except json.decoder.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {file_path}: {e}")
        return False  # Return False if there's an error reading JSON
    
    # Extract airport information
    airport_list = [{'icao': airport_loc} for airport_loc in airports_icao['items']]
    airport_list_df = pd.DataFrame(airport_list)

    # Check if argument exists in the DataFrame
    check = False
    for index, row in airport_list_df.iterrows():
        airport = row['icao']
        if argument == airport:
            check = True
            break  # Stop iteration once a match is found
    
    return check

airport_df = data_preperation.prepare_airport_data()
def airport_location(location):
    for index, row in airport_df.iterrows():
        airport = row['icao']
        if location == airport:
            return row['lat'], row['lon']

