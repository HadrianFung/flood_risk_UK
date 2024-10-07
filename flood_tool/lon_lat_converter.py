import pandas as pd
from pyproj import Proj, transform

def en_to_latlon(easting, northing):
    # British National Grid Projection
    bng = Proj('epsg:27700')
    # WGS84 Projection
    wgs84 = Proj('epsg:4326')
    
    lon, lat = transform(bng, wgs84, easting, northing)
    return lat, lon

def convert_coordinates(input_file_path, output_file_path):
    '''Convert coordinates from British National Grid (BNG) projection to
        latitude and longitude in WGS84 projection and save the results to a CSV file.

    Parameters
    ----------
    input_file_path (str):
        The file path to the input CSV file containing the BNG coordinates.

    output_file_path (str): 
        The file path to save the output CSV file with latitude and longitude.

    Returns
    -------
    DataFrame:
        Returns a DataFrame containing updated dataset with longitude and latitude

    '''

    # Read the dataset
    print("Reading the dataset...")
    df = pd.read_csv(input_file_path)

    
    print("Starting conversion of coordinates...")
    converted_coords = []
    for index, row in df.iterrows():
        lat, lon = en_to_latlon(row['easting'], row['northing'])  
        converted_coords.append((lat, lon))
        if index % 100 == 0:  
            print(f"Converted {index + 1} rows...")

    # Assigning new converted values to new columns in the dataframe
    df['latitude'], df['longitude'] = zip(*converted_coords)
    print("Conversion completed.")

    print("Saving the updated dataset...")
    df.to_csv(output_file_path, index=False)
    print(f"Dataset saved to {output_file_path}")

    return df

#input_file_path = '/home/ls2523/ads-deluge-thames/flood_tool/resources/postcodes_labelled.csv'
#output_file_path = '/home/ls2523/ads-deluge-thames/flood_tool/resources/lon_lat_post_labelled.csv'
#convert_coordinates(input_file_path, output_file_path)

# To run this converter you need to create two variables called 'input_file_path' containing the address of the input csv
# and 'output_file_path' with the desired address of the new .csv file.

# Lastly call 'converted_coordinates(input_ file_path, output_file_path)' to run the function

