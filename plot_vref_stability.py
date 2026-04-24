#!/usr/bin/env python3
# plot_vref_stability.py

# Description: Plots lpGBT VREF_TUNE and TJ_USER Temperature vs. Time for each train.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import influxdb_client
import matplotlib.dates as mdates
import pytz
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
train_to_server_map = {
    "LD1": INFLUX_URL_LABC3, "LD2": INFLUX_URL_LABC3, "LD3": INFLUX_URL_LABC2,
    "LD4": INFLUX_URL_LABC2, "LD5": INFLUX_URL_LABC2,
    "HD1": INFLUX_URL_LABC3, "HD2": INFLUX_URL_LABC3,
}
token_map = {
    INFLUX_URL_LABC2: INFLUX_TOKEN_LABC2,
    INFLUX_URL_LABC3: INFLUX_TOKEN_LABC3,
}

# --- Time Configuration ---
TIME_WINDOW_START = "2025-10-17T07:15:00Z"
TIME_WINDOW_STOP = "2025-10-17T19:00:00Z"
TIME_INTERVAL = "1m" 

# --- 2. Query Data ---
all_dataframes = []
print("--- Starting to query lpGBT calibration data from InfluxDB ---")
for server_url in [INFLUX_URL_LABC2, INFLUX_URL_LABC3]:
    print(f"Querying data from {server_url}...")
    trains_on_this_server = [train for train, url in train_to_server_map.items() if url == server_url]
    if not trains_on_this_server: continue
    flux_train_list = '["' + '", "'.join(trains_on_this_server) + '"]'
    client = influxdb_client.InfluxDBClient(url=server_url, token=token_map[server_url], org=INFLUX_ORG)
    query_api = client.query_api()
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
        result = query_api.query_data_frame(query=query)
        if isinstance(result, list): all_dataframes.extend(result)
        elif not result.empty: all_dataframes.append(result)
    except Exception as e: print(f"  > FAILED to query {server_url}. Error: {e}")
    finally: client.close()

# --- 3. Process and Pivot Data ---
if not all_dataframes:
    print("--- No VREF or TJ_USER data found. Exiting. ---")
    exit()
print("--- Processing and pivoting data ---")
full_df = pd.concat(all_dataframes, ignore_index=True)
full_df['pivot_key'] = full_df['meas'] + '_' + full_df['train']
pivot_df = full_df.pivot_table(index='_time', columns='pivot_key', values='_value')
pivot_df = pivot_df.interpolate(method='time')
pivot_df.index = pivot_df.index.tz_convert('America/Chicago') # Use Chicago time

# --- 4. Create the Plot ---
print("--- Generating VREF_TUNE and TJ_USER vs. Time plot ---")
fig, ax1 = plt.subplots(figsize=(15, 8))

# Get a color cycle
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
trains = sorted(list(train_to_server_map.keys())) 
all_lines = [] # For legend
all_lines1 = [] # For VREF lines
all_lines2 = [] # For Temp lines

# Plot VREF_TUNE on left axis
ax1.set_xlabel('Time (America/Chicago)')
ax1.set_ylabel('VREF_TUNE (ADC Counts)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True, linestyle='--')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pivot_df.index.tz))

# Plot TJ_USER on right axis
ax2 = ax1.twinx()
ax2.set_ylabel('TJ_USER Temperature (°C)', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

for i, train in enumerate(trains):
    vref_col = f'calibration_{train}'
    temp_col = f'temperature_{train}'
    color = colors[i % len(colors)]
    
    if vref_col in pivot_df.columns:
        line1 = ax1.plot(pivot_df.index, pivot_df[vref_col], color=color, linestyle='-', label=f'{train} VREF')
        all_lines1.extend(line1) # Add VREF line to legend list
    else: print(f"Warning: Missing VREF data for train {train}.")
        
    if temp_col in pivot_df.columns:
        line2 = ax2.plot(pivot_df.index, pivot_df[temp_col], color=color, linestyle='--', label=f'{train} Temp') 
        all_lines2.extend(line2) # Add Temp line to list 2
    else: print(f"Warning: Missing TJ_USER Temp data for train {train}.")

ax1.set_title(f'lpGBT VREF_TUNE and TJ_USER vs. Time ({TIME_WINDOW_START} to {TIME_WINDOW_STOP})')
# --- REPLACE the old fig.legend and plt.tight_layout/fig.subplots_adjust lines ---
# Combine lines and labels from both axes
all_lines = all_lines1 + all_lines2
all_labels = [l.get_label() for l in all_lines]

# Create a combined legend BELOW the plot using ax.legend()
ax1.legend(all_lines, all_labels, 
           loc='upper center', 
           bbox_to_anchor=(0.5, -0.15), # Position it below the axes
           ncol=len(trains), # Use number of trains for columns (or adjust as needed)
           title="Train VREF (Solid) / Temp (Dashed)",
           fontsize='small') 

# Adjust layout to make room for the legend at the bottom
plt.tight_layout() # Use standard tight_layout first
fig.subplots_adjust(bottom=0.25) # Add manual space at the bottom (increased space)
# --- END REPLACEMENT ---
print("--- Displaying plot... ---")
plt.show()

print("\n--- Analysis complete! ---")