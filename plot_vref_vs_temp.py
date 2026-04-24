#!/usr/bin/env python3
# plot_vref_vs_temp.py

# Author: Gabi Hamilton
# Date: October 2025
# Description: This script queries lpGBT VREF_TUNE values and the internal
#              junction temperature (tj_user) used for calibration, then plots
#              VREF_TUNE vs. Temperature for each train to study stability.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import influxdb_client
import argparse
import matplotlib.dates as mdates
import warnings
from influxdb_client.client.warnings import MissingPivotFunction
warnings.simplefilter("ignore", MissingPivotFunction)

# --- 1. Configure InfluxDB Connections ---
INFLUX_URL_LABC2 = "http://localhost:8086"
INFLUX_URL_LABC3 = "http://localhost:8087"
INFLUX_TOKEN_LABC2 = "duCMZjGG4kJ2S8V_rSzGJ08pOy6_huzwjZTDWsXmE7QpBfk8YupVVlSJ8ICdR3sO_nSBPqvZwxWffVPTTCLQXg=="
INFLUX_TOKEN_LABC3 = "iC7xTSadQLaWH1nDkhkNeyEEysRXdT2AK078VckGaaWVLzbfG9tQFo7LupUuZtxjBh60RWBzwioyYNi75oLhsQ=="
INFLUX_ORG = "fnal"
INFLUX_BUCKET = "RTDs_2025Oct"
# This map determines which server holds the lpGBT data for each train
# Adjust if needed, but likely matches your other scripts
train_to_server_map = {
    "LD1": INFLUX_URL_LABC3, "LD2": INFLUX_URL_LABC3, "LD3": INFLUX_URL_LABC2,
    "LD4": INFLUX_URL_LABC2, "LD5": INFLUX_URL_LABC2,
    "HD1": INFLUX_URL_LABC3, "HD2": INFLUX_URL_LABC3,
}
token_map = {
    INFLUX_URL_LABC2: INFLUX_TOKEN_LABC2,
    INFLUX_URL_LABC3: INFLUX_TOKEN_LABC3,
}

# --- Time Configuration (Match your run) ---
TIME_WINDOW_START = "2025-10-17T07:15:00Z"
TIME_WINDOW_STOP = "2025-10-17T19:00:00Z"
TIME_INTERVAL = "1m" # Average over 1 minute intervals

# --- 2. Query Data (VREF_TUNE and TJ_USER Temp) ---
all_dataframes = []
print("--- Starting to query lpGBT calibration data from InfluxDB ---")

for server_url in [INFLUX_URL_LABC2, INFLUX_URL_LABC3]:
    print(f"Querying data from {server_url}...")
    
    trains_on_this_server = [train for train, url in train_to_server_map.items() if url == server_url]
    if not trains_on_this_server:
        continue
    
    flux_train_list = '["' + '", "'.join(trains_on_this_server) + '"]'
    
    client = influxdb_client.InfluxDBClient(
        url=server_url, token=token_map[server_url], org=INFLUX_ORG
    )
    query_api = client.query_api()

    # Query VREF_TUNE (meas=calibration, type=vref_tune, module=lpgbt)
    # AND TJ_USER Temp (meas=temperature, type=junction_temp, module=lpgbt_tj_user)
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {TIME_WINDOW_START}, stop: {TIME_WINDOW_STOP})
      |> filter(fn: (r) => r._measurement == "cassette_readings")
      |> filter(fn: (r) => contains(value: r.train, set: {flux_train_list}))
      |> filter(fn: (r) => 
           (r.meas == "calibration" and r.type == "vref_tune" and r.module == "lpgbt") or
           (r.meas == "temperature" and r.type == "junction_temp" and r.module == "lpgbt_tj_user")
         )
      |> aggregateWindow(every: {TIME_INTERVAL}, fn: mean, createEmpty: false)
      |> yield(name: "mean")
    '''
    
    try:
        # Result might be a list if both types are found
        result = query_api.query_data_frame(query=query)
        if isinstance(result, list):
            all_dataframes.extend(result)
        elif not result.empty:
            all_dataframes.append(result)
            
    except Exception as e:
        print(f"  > FAILED to query {server_url}. Error: {e}")
    finally:
        client.close()

# --- 3. Process and Pivot Data ---
if not all_dataframes:
    print("--- No VREF or TJ_USER data found. Exiting. ---")
    exit()

print("--- Processing and pivoting data ---")
full_df = pd.concat(all_dataframes, ignore_index=True)

# Create a combined key based on measurement type and train
# We need to distinguish VREF from Temp for the pivot
full_df['pivot_key'] = full_df['meas'] + '_' + full_df['train']

# Pivot: index=time, columns=['calibration_LD1', 'temperature_LD1', ...], values=value
pivot_df = full_df.pivot_table(index='_time', columns='pivot_key', values='_value')

# Interpolate missing points
print("--- Interpolating missing data points ---")
pivot_df = pivot_df.interpolate(method='time')

# --- 4. Create the Plot ---
print("--- Generating VREF vs. Temperature plot ---")
fig, ax = plt.subplots(figsize=(12, 8))

# Get a color cycle
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
markers = ['o', 's', '^', 'v', 'D', '<', '>', 'p', '*', 'h'] # Different marker per train

trains = sorted(list(train_to_server_map.keys())) # Get unique train names

for i, train in enumerate(trains):
    temp_col = f'temperature_{train}' # Column name for tj_user temperature
    vref_col = f'calibration_{train}' # Column name for vref_tune
    
    # Check if both columns exist for this train
    if temp_col in pivot_df.columns and vref_col in pivot_df.columns:
        ax.scatter(
            pivot_df[temp_col], 
            pivot_df[vref_col], 
            label=train, 
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            alpha=0.7 # Make points slightly transparent
        )
    else:
        print(f"Warning: Missing Temperature or VREF data for train {train}.")

# Set labels and title
ax.set_title(f'lpGBT VREF_TUNE vs. TJ_USER Temperature ({TIME_WINDOW_START} to {TIME_WINDOW_STOP})')
ax.set_xlabel('TJ_USER Temperature (°C)')
ax.set_ylabel('VREF_TUNE (ADC Counts)')
ax.grid(True, linestyle='--')
ax.legend(title='Train', loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
print("--- Displaying plot... ---")
plt.show()

print("\n--- Analysis complete! ---")