import pandas as pd
import os
import sys

sys.path.append('..') # Add parent directory to path to always find flood_tool
                      # This is not best practice, but it works for this example
                      
import matplotlib.pyplot as plt
from ..tool import _data_dir
from live import *
import folium
from folium.plugins import HeatMap
import ipywidgets as widgets
from IPython.display import display, clear_output

# Function to create a heatmap for a given time
def create_heatmap(time):
    # Filter data for the selected time
    time_data = typical_level_data[typical_level_data['dateTime'].dt.time == time]

    # Create a list of tuples with latitude, longitude, and value
    heat_data = [[row['latitude'], row['longitude'], row['value']] for index, row in time_data.iterrows()]

    # Create a map centered around the average coordinates
    avg_lat, avg_lon = time_data['latitude'].mean(), time_data['longitude'].mean()
    map_ = folium.Map(location=[avg_lat, avg_lon], zoom_start=6)

    # Add the heatmap to the folium map
    HeatMap(heat_data).add_to(map_)

    return map_

# Function to update the map based on the slider's value
def update_map(change):
    time = change.new
    map_ = create_heatmap(time)
    with map_display:
        clear_output(wait=True)
        display(map_)

# Update the heatmap display
def update_heatmap_display(time_index):
    with map_output:
        clear_output(wait=True)
        display(generate_heatmap(time_index))
    time_display.value = unique_timestamps[time_index].strftime('%Y-%m-%d %H:%M:%S')

# Update slider range based on selected times
def update_slider_range(*args):
    start_time = start_time_picker.value
    end_time = end_time_picker.value

# Function to generate a heatmap
def generate_heatmap(time_index):
    time_filter = merged_rainfall['dateTime'] == unique_timestamps[time_index]
    data = merged_rainfall[time_filter][['latitude', 'longitude', 'value']].values.tolist()
    m = folium.Map(location=[merged_rainfall['latitude'].mean(), merged_rainfall['longitude'].mean()], zoom_start=6)
    HeatMap(data, radius=10, blur=15, max_zoom=1).add_to(m)
    return m


# The data on measurement stations
stations_df = pd.read_csv(os.path.join(_data_dir, 'stations.csv'))
# The data for a wet day
typical_day_df = pd.read_csv(os.path.join(_data_dir, 'wet_day.csv'))

# The data for a more typical day
wet_day_df = pd.read_csv(os.path.join(_data_dir, 'typical_day.csv'))

merged_typical = pd.merge(typical_day_df, stations_df, on='stationReference', how='left')

typical_level_data = merged_typical[merged_typical['parameter'] == 'level']
typical_level_data['value'] = pd.to_numeric(typical_level_data['value'], errors='coerce')
typical_level_data = typical_level_data.dropna(subset=['latitude', 'longitude','value'])

typical_level_data.to_csv('typical_level_data.csv', index=False) 

# Convert the 'dateTime' column to datetime objects
typical_level_data['dateTime'] = pd.to_datetime(typical_level_data['dateTime'])
# Filter the data to include only points at 00:00, 00:15, 00:30, 00:45, etc., and seconds equal to 0
typical_level_data = typical_level_data[(typical_level_data['dateTime'].dt.minute % 15 == 0) & (typical_level_data['dateTime'].dt.second == 0)]

# Extract and sort the unique times (excluding dates)
unique_typical_times = typical_level_data['dateTime'].dt.time.unique()
sorted_typical_times = sorted(unique_typical_times)

# Create a container for displaying the map
map_display = widgets.Output()

# Create the slider widget
slider = widgets.SelectionSlider(
    options=sorted_typical_times,
    value=sorted_typical_times[0],
    description='Time',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True
)

# Listen for changes in the slider value
slider.observe(update_map, names='value')

# Display the initial map and the slider
initial_map = create_heatmap(sorted_typical_times[0])
with map_display:
    display(initial_map)

display(slider)
display(map_display)

typical_rainfall_data = merged_typical[merged_typical['parameter'] == 'rainfall']
typical_rainfall_data['value'] = pd.to_numeric(typical_rainfall_data['value'], errors='coerce')
typical_rainfall_data = typical_rainfall_data[typical_rainfall_data['value'] >= 0]
typical_rainfall_data = typical_rainfall_data.dropna(subset=['latitude', 'longitude','value'])

typical_rainfall_data.to_csv('typical_rainfall_data.csv', index=False) 

# Convert the 'dateTime' column to datetime objects
typical_rainfall_data['dateTime'] = pd.to_datetime(typical_rainfall_data['dateTime'])
# Filter the data to include only points at 00:00, 00:15, 00:30, 00:45, etc., and seconds equal to 0
typical_rainfall_data = typical_rainfall_data[(typical_rainfall_data['dateTime'].dt.minute % 15 == 0) & (typical_rainfall_data['dateTime'].dt.second == 0)]

# Extract and sort the unique times (excluding dates)
unique_rainfall_times = typical_rainfall_data['dateTime'].dt.time.unique()
sorted_rainfall_times = sorted(unique_rainfall_times)

# Create a container for displaying the map
map_display = widgets.Output()

# Create the slider widget
slider = widgets.SelectionSlider(
    options=sorted_rainfall_times,
    value=sorted_rainfall_times[0],
    description='Time',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True
)

# Listen for changes in the slider value
slider.observe(update_map, names='value')

# Display the initial map and the slider
initial_map = create_heatmap(sorted_rainfall_times[0])
with map_display:
    display(initial_map)

display(slider)
display(map_display)

merged_wet = pd.merge(wet_day_df, stations_df, on='stationReference', how='left')

wet_level_data = merged_wet[merged_wet['parameter'] == 'level']
wet_level_data['value'] = pd.to_numeric(wet_level_data['value'], errors='coerce')
wet_level_data = wet_level_data.dropna(subset=['latitude', 'longitude','value'])

wet_level_data.to_csv('wet_level_data.csv', index=False) 

wet_rainfall_data = merged_wet[merged_wet['parameter'] == 'rainfall']
wet_rainfall_data['value'] = pd.to_numeric(wet_rainfall_data['value'], errors='coerce')
wet_rainfall_data = wet_rainfall_data[wet_rainfall_data['value'] >= 0]
wet_rainfall_data = wet_rainfall_data.dropna(subset=['latitude', 'longitude','value'])

wet_rainfall_data.to_csv('wet_rainfall_data.csv', index=False) 

# Convert the 'dateTime' column to datetime objects
wet_rainfall_data['dateTime'] = pd.to_datetime(wet_rainfall_data['dateTime'])
# Filter the data to include only points at 00:00, 00:15, 00:30, 00:45, etc., and seconds equal to 0
wet_rainfall_data = wet_rainfall_data[(wet_rainfall_data['dateTime'].dt.minute % 15 == 0) & (wet_rainfall_data['dateTime'].dt.second == 0)]

# Extract and sort the unique times (excluding dates)
unique_wet_times = wet_rainfall_data['dateTime'].dt.time.unique()
sorted_wet_times = sorted(unique_wet_times)

# Create a container for displaying the map
map_display = widgets.Output()

# Create the slider widget
slider = widgets.SelectionSlider(
    options=sorted_wet_times,
    value=sorted_wet_times[0],
    description='Time',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True
)

# Listen for changes in the slider value
slider.observe(update_map, names='value')

# Display the initial map and the slider
initial_map = create_heatmap(sorted_wet_times[0])
with map_display:
    display(initial_map)

display(slider)
display(map_display)

latest_rainfall = get_latest_rainfall_readings()
latest_rainfall = latest_rainfall.reset_index()

latest_rainfall = latest_rainfall[latest_rainfall.stationReference != ''].drop(columns=["qualifier", "unitName", "parameter"])
latest_rainfall.loc[:, 'stationReference']

latest_rainfall.to_csv('latest_rainfall.csv', index=False)

stations = pd.read_csv(os.path.join(_data_dir, 'stations.csv'))
merged_rainfall = pd.merge(latest_rainfall, stations, on='stationReference', how='left')
merged_rainfall["dateTime"] = pd.to_datetime(merged_rainfall["dateTime"])
merged_rainfall["value"] = pd.to_numeric(merged_rainfall["value"], errors="coerce")

merged_rainfall.to_csv('merged_rainfall.csv', index=False) 

# # Load the data
# file_path = 'merged_rainfall.csv'
# merged_rainfall_df = pd.read_csv(file_path)
merged_rainfall['dateTime'] = pd.to_datetime(merged_rainfall['dateTime'])
merged_rainfall.sort_values('dateTime', inplace=True)

# Get unique timestamps for the current day
current_day = pd.Timestamp.now().date()
unique_timestamps = merged_rainfall[merged_rainfall['dateTime'].dt.date == current_day]['dateTime'].unique()

# Time pickers for start and end time
start_time_picker = widgets.TimePicker(description='Start Time')
end_time_picker = widgets.TimePicker(description='End Time')

# Time slider widget
time_slider = widgets.IntSlider(
    value=0,
    min=0,
    max=len(unique_timestamps) - 1,
    step=1,
    description='Time:',
    continuous_update=False
)

# Text widget to display the selected time
time_display = widgets.Text(value='', description='Selected Time:', disabled=True)

# Output container for storing and displaying the map
map_output = widgets.Output()


  