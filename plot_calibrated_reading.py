#!/usr/bin/env python3
# plot_calibrated_reading.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import influxdb_client
import argparse
import os
import warnings
from influxdb_client.client.warnings import MissingPivotFunction

warnings.simplefilter("ignore", MissingPivotFunction)

# --- 1. Argument Parsing ---
parser = argparse.ArgumentParser(description="Plot stable source reading vs. lpGBT temperature.")
parser.add_argument('--meas', type=str, required=True, help="Measurement tag (e.g., 'voltage')")
parser.add_argument('--type', type=str, required=True, help="Type tag (e.g., 'external_monitor')")
parser.add_argument('--module', type=str, required=True, help="Module tag (e.g., 'stable_ref')")
args = parser.parse_args()

# --- 2. Configure SINGLE InfluxDB Connection (Aligned with your write scripts) ---
INFLUX_URL = "http://localhost:8087"
# This will look for the token in your shell environment, falling back to a placeholder if not set
INFLUX_TOKEN = os.environ.get("INFLUXDB_TOKEN", "4SiGtuPRS4xtdOaqTNBO8-Y_MGZrah9hjEXd9Pq77Vlg1gWx2R7iMHSMprVe7mpEF2yHkp1R0iDt7SA3GKmJdQ==")
INFLUX_ORG = "fnal"
INFLUX_BUCKET = "RTDs_2025Oct"

# --- Updated Train List from your DB check ---
TRAINS = ["HD1", "HD2", "LD1", "LD1+HD1", "LD2", "LD2+HD2", "LD3", "LD4", "LD4+LD5", "LD5"]

# --- Time Configuration (Updated for March 2026 Testing) ---
TIME_WINDOW_START = "-24h"
TIME_WINDOW_STOP = "now()"
TIME_INTERVAL = "1m"

# --- 3. Query Data ---
print(f"--- Querying data from {INFLUX_URL} ---")
client = influxdb_client.InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
query_api = client.query_api()

flux_train_list = '["' + '", "'.join(TRAINS) + '"]'

query = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: {TIME_WINDOW_START}, stop: {TIME_WINDOW_STOP})
  |> filter(fn: (r) => r._measurement == "cassette_readings")
  |> filter(fn: (r) => contains(value: r.train, set: {flux_train_list}))
  |> filter(fn: (r) => 
       (r.meas == "temperature" and r.type == "lpgbt" and r.module == "lpgbt_internal") or
       (r.meas == "{args.meas}" and r.type == "{args.type}" and r.module == "{args.module}")
     )
  |> aggregateWindow(every: {TIME_INTERVAL}, fn: mean, createEmpty: false)
  |> yield(name: "mean")
'''

try:
    result = query_api.query_data_frame(query=query)
except Exception as e:
    print(f"FAILED to query InfluxDB. Error: {e}")
    exit()
finally:
    client.close()

# --- 4. Process and Pivot Data ---
if result is None or (isinstance(result, pd.DataFrame) and result.empty):
    print("--- No data found. Check your TIME_WINDOW or INFLUX_TOKEN. ---")
    exit()

full_df = pd.concat(result) if isinstance(result, list) else result
full_df['pivot_key'] = full_df['meas'] + '_' + full_df['train'] 
pivot_df = full_df.pivot_table(index='_time', columns='pivot_key', values='_value')
pivot_df = pivot_df.interpolate(method='time')

# --- 5. Create the Plot ---
fig, ax = plt.subplots(figsize=(12, 8))
# Cycle through distinct colors for the 7 trains
colors = plt.cm.tab10(np.linspace(0, 1, len(TRAINS)))
markers = ['o', 's', '^', 'v', 'D', '<', '>'] 

for i, train in enumerate(TRAINS):
    temp_col = f'temperature_{train}'
    source_col = f'{args.meas}_{train}'
    
    if temp_col in pivot_df.columns and source_col in pivot_df.columns:
        ax.scatter(
            pivot_df[temp_col], 
            pivot_df[source_col], 
            label=train, 
            color=colors[i],
            marker=markers[i % len(markers)],
            alpha=0.6
        )

ax.set_title(f'Thermal Stability: {args.meas} ({args.module}) vs. lpGBT Temp', fontsize=14)
ax.set_xlabel('lpGBT Junction Temperature [°C]', fontsize=12)
ax.set_ylabel(f'Calibrated {args.meas} Reading', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend(title='Train', loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.show()