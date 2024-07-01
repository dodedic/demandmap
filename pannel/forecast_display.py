import numpy as np
import pandas as pd
import plotly.express as px
import panel as pn
import scipy.sparse
import json
import airport_check  

time_of_year = 'January'

def set_time_of_year(time_of_year_value):
    global time_of_year
    print(f"Setting time_of_year to: {time_of_year_value}")  # Debug print
    time_of_year = time_of_year_value

# Load the country codes from JSON file
with open('CountryCodes.json', 'r') as file:
    country_codes = json.load(file)

# Load the GDP data
gdp_data = pd.read_csv('GDPData.csv')

# Create a DataFrame for the additional data
df = pd.DataFrame({
    'Year': list(range(2024, 2051)),
    'Seats': [0] * 27,
    'Percentage Change': [0.0] * 27,  # Initialize percentage change column
    'PAX': [0] * 27  # Initialize PAX column
})

# Function to get the scaling factors from GDP data based on departure ICAO code
def get_scaling_factors(departure_code):
    scaling_factors = []
    if departure_code.startswith('K'):
        country_code = "USA"
    else:
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
def get_sparse_value(departure_code, destination_code, time_of_year_value, trip_indicator_value):
    def get_value_for_route(departure_code, destination_code, time_of_year_value):
        if time_of_year_value == "Whole year":
            total_value = 0
            months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
            for month in months:
                sparse_matrix = scipy.sparse.load_npz(f'seat_matrices/{month}.npz')
                labels = np.load(f'seat_matrices/{month}_labels.npz', allow_pickle=True)
                row_labels = labels['rows']
                col_labels = labels['cols']
                try:
                    departure_idx = np.where(row_labels == departure_code)[0][0]
                    destination_idx = np.where(col_labels == destination_code)[0][0]
                    total_value += sparse_matrix[departure_idx, destination_idx] * 30
                except IndexError:
                    continue  # Skip if index not found for the month
            return total_value
        else:
            # Load the sparse matrix and labels for the specified time_of_year_value
            sparse_matrix = scipy.sparse.load_npz(f'seat_matrices/{time_of_year_value}.npz')
            labels = np.load(f'seat_matrices/{time_of_year_value}_labels.npz', allow_pickle=True)
            row_labels = labels['rows']
            col_labels = labels['cols']
            try:
                departure_idx = np.where(row_labels == departure_code)[0][0]
                destination_idx = np.where(col_labels == destination_code)[0][0]
                return sparse_matrix[departure_idx, destination_idx]
            except IndexError:
                return 0

    if trip_indicator_value == "One-way":
        return get_value_for_route(departure_code, destination_code, time_of_year_value)
    elif trip_indicator_value == "Round-trip":
        one_way_value = get_value_for_route(departure_code, destination_code, time_of_year_value)
        return_trip_value = get_value_for_route(destination_code, departure_code, time_of_year_value)
        return one_way_value + return_trip_value


