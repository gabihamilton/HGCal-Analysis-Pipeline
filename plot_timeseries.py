# plot_timeseries.py

# Author: Gabi Hamilton
# Date: October 2025
# Description: This script connects to InfluxDB instances to retrieve temperature data
#              in time intervals and generates a time-series line plot for all sensors.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import matplotlib.dates as mdates
import warnings
from influxdb_client.client.warnings import MissingPivotFunction
warnings.simplefilter("ignore", MissingPivotFunction)
from datetime import datetime
import pytz
import os

# --- Step 1: Load Our Custom Sensor Map ---
# (Not needed for this plot, but good to be aware of)
# from cassette_map import sensor_positions

# --- Step 2: Configure SINGLE InfluxDB Connection ---
INFLUX_URL = "http://localhost:8087" 
INFLUX_TOKEN = os.environ.get("INFLUXDB_TOKEN", "4SiGtuPRS4xtdOaqTNBO8-Y_MGZrah9hjEXd9Pq77Vlg1gWx2R7iMHSMprVe7mpEF2yHkp1R0iDt7SA3GKmJdQ==")
INFLUX_ORG = "fnal"
INFLUX_BUCKET = "RTDs_2025Oct"

# List of all trains found in your DB check earlier
TRAINS = ["HD1", "HD2", "LD1", "LD1+HD1", "LD2", "LD2+HD2", "LD3", "LD4", "LD4+LD5", "LD5"]

# --- Updated Time Configuration ---
# Look at the last 24 hours to ensure we catch today's runs
TIME_WINDOW_START = "-24h"
TIME_WINDOW_STOP = "now()"
TIME_INTERVAL = "1m"

# --- Step 3: Query Data ---
all_sensor_data = []
print(f"--- Querying data from {INFLUX_URL} ---")

client = influxdb_client.InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
query_api = client.query_api()

flux_train_list = '["' + '", "'.join(TRAINS) + '"]'

query = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: {TIME_WINDOW_START}, stop: {TIME_WINDOW_STOP})
  |> filter(fn: (r) => r._measurement == "cassette_readings")
  |> filter(fn: (r) => r.meas == "temperature")
  |> filter(fn: (r) => contains(value: r.train, set: {flux_train_list}))
  |> aggregateWindow(every: {TIME_INTERVAL}, fn: mean, createEmpty: false)
  |> yield(name: "mean")
'''

try:
    result_df = query_api.query_data_frame(query=query)
    # Handle both single DataFrame and list of DataFrames
    if isinstance(result_df, list):
        all_sensor_data.extend(result_df)
    else:
        all_sensor_data.append(result_df)
except Exception as e:
    print(f"FAILED to query InfluxDB. Error: {e}")
finally:
    client.close()

# --- Step 4: Process All Data ---
if not all_sensor_data:
    print("\n--- No data found for any sensors. Exiting. ---")
    exit()

print("\n--- Processing all time-series data ---")
full_df = pd.concat(all_sensor_data, ignore_index=True)

# --- Step 5: Pivot Data for Plotting ---
print("--- Pivoting data for plotting ---")

# Create a unique key for each sensor
full_df['sensor_key'] = full_df['train'] + '_' + full_df['module']

# Filter out 'tj_user' and 'therm' sensors
print("--- Filtering out 'tj_user' and 'therm' sensors ---")
full_df = full_df[
    ~full_df['sensor_key'].str.contains('tj_user', na=False) &
    ~full_df['sensor_key'].str.contains('therm', na=False)
]
# --- NEW: Filter out 'rtd_rtd' and other noise sensors ---
print("--- Removing duplicate 'rtd_rtd' sensors and noise tags ---")
full_df = full_df[
    ~full_df['sensor_key'].str.contains('rtd_rtd', na=False) &
    ~full_df['sensor_key'].str.contains('tj_user', na=False) &
    ~full_df['sensor_key'].str.contains('therm', na=False)
]

# Pivot the data: 
# index = time, columns = sensor names, values = temperature
pivot_df = full_df.pivot_table(index='_time', columns='sensor_key', values='_value')

# Convert time index to be in your local timezone for plotting
pivot_df.index = pivot_df.index.tz_convert('America/Chicago')

print("--- Interpolating missing data points to create continuous lines ---")
pivot_df = pivot_df.interpolate(method='time')


# --- Step 6: Create the Plot with Mar 3 Cold Room Runs ---
print("--- Generating plot with Cold Room run markers ---")

fig, ax = plt.subplots(figsize=(15, 8))

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

for i, sensor_name in enumerate(pivot_df.columns):
    color = colors[i % len(colors)] # Cycle through colors
    
    # Assign linestyle based on name
    if 'lpgbt_internal' in sensor_name:
        style = '--' # Dashed for lpGBT
    elif '_w' in sensor_name:
        style = ':'  # Dotted for West
    elif '_e' in sensor_name:
        style = '-.' # Dash-dot for East
    else:
        style = '-'  # Solid for all others (like 'm' modules)

    ax.plot(pivot_df.index, pivot_df[sensor_name], linestyle=style, label=sensor_name, color=color)
# Plot all columns (sensors) against the index (time)
#pivot_df.plot(ax=ax)
print("--- Adding data event markers ---")
# 1. Update the dates and times based on your latest log
tz = pytz.timezone('America/Chicago')
event_markers = {
    #datetime(2026, 3, 3, 9, 4, tzinfo=tz):  "Run 166 (9:04)",
    datetime(2026, 3, 3, 9, 5, tzinfo=tz):  "Run 167 (9:05)",
    datetime(2026, 3, 3, 12, 5, tzinfo=tz): "Run 168 (12:05)",
    datetime(2026, 3, 3, 14, 45, tzinfo=tz): "Run 169 (14:45)",
    datetime(2026, 3, 3, 15, 10, tzinfo=tz): "Run 170 (15:10)"
}

# Get the top of the plot to position the text
y_top = ax.get_ylim()[1]
y_min, y_max = ax.get_ylim()
text_y_position = y_max - (0.2 * (y_max - y_min))

# Loop through the events to add vertical lines
for i, (time, label) in enumerate(event_markers.items()):
    # Only add the "Data Point" label to the legend once
    legend_label = 'Run Event' if i == 0 else '_nolegend_'
    
    ax.axvline(time, color='gray', linestyle='--', linewidth=1.5, label=legend_label, alpha=0.8)
    
    # Position the text vertically at the top of the current Y-limit
    ax.text(
        x=time, 
        y=text_y_position, # Now using the calculated internal position
        #y=y_top, 
        s=label, 
        rotation=90, 
        color='black',
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        alpha=0.6
    )

# Set the window title to reflect the Cold Room test
ax.set_title(f'HGCal Cassette Temperatures - COLD TEST (2026 Mar 3)')
ax.set_xlabel('Time (America/Chicago)')
ax.set_ylabel('Temperature (°C)')

# Set Y-axis limits to match the Grafana plot
ax.set_ylim(-40, 40)

# Add gridlines
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pivot_df.index.tz))

# --- Format the legend ---
# This makes the legend smaller and places it outside the plot
ax.legend(
    title='Sensors',
    loc='center left', 
    bbox_to_anchor=(1, 0.5), # Position to the right of the plot
    fontsize='small',
    ncol=2 # Arrange in 2 columns
)

# Adjust layout to make room for the legend
plt.tight_layout()

print("--- Displaying plot... ---")
plt.show()

print("\n--- Analysis complete! ---")