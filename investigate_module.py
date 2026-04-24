#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import influxdb_client
import argparse
import os
import matplotlib.dates as mdates
import warnings
from influxdb_client.client.warnings import MissingPivotFunction
warnings.simplefilter("ignore", MissingPivotFunction)

# --- 1. Parse Command-Line Arguments ---
parser = argparse.ArgumentParser(description="Investigate a single module's HV, current, and temperature.")
parser.add_argument('--train', type=str, required=True, help="Train name (e.g., HD1)")
parser.add_argument('--module', type=str, required=True, help="Module name (e.g., m1)")
parser.add_argument('--output', type=str, help="Save plot to file instead of showing it.")
args = parser.parse_args()

# --- 2. Configure Single InfluxDB Connection ---
INFLUX_URL = "http://localhost:8087"
INFLUX_TOKEN = os.environ.get("INFLUXDB_TOKEN", "4SiGtuPRS4xtdOaqTNBO8-Y_MGZrah9hjEXd9Pq77Vlg1gWx2R7iMHSMprVe7mpEF2yHkp1R0iDt7SA3GKmJdQ==")
INFLUX_ORG = "fnal"
INFLUX_BUCKET = "RTDs_2025Oct"

# --- Time Configuration (Last 24 hours for current shifts) ---
TIME_WINDOW_START = "2026-03-03T15:00:00Z" # 09:00 AM Chicago Time
TIME_WINDOW_STOP = "2026-03-03T23:30:00Z"  # 05:00 PM Chicago Time
TIME_INTERVAL = "1m"

# --- 3. Query Data (Consolidated for March 2026 Cold Test) ---
all_dataframes = [] # Initialized here to prevent NameError
print(f"--- Querying data for Train: {args.train}, Module: {args.module} ---")

client = influxdb_client.InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
query_api = client.query_api()

# This query pulls RTDs, lpGBT chips, and MPOD Bias data in one go
# This query handles the prefix difference between the RTD sensors and the MPOD supply
query = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: {TIME_WINDOW_START}, stop: {TIME_WINDOW_STOP})
  |> filter(fn: (r) => r._measurement == "cassette_readings")
  |> filter(fn: (r) => r.train == "{args.train}")
  |> filter(fn: (r) => 
       (r.module == "rtd_{args.module}" and r.type == "rtd") or 
       (r.module == "lpgbt_internal" and r.type == "lpgbt") or
       (r.module == "{args.module}" and r.type == "bias_voltage")
     )
  |> aggregateWindow(every: {TIME_INTERVAL}, fn: mean, createEmpty: false)
  |> yield(name: "mean")
'''

try:
    result = query_api.query_data_frame(query=query)
    if isinstance(result, list):
        all_dataframes.extend(result)
    elif result is not None and not result.empty:
        all_dataframes.append(result)
except Exception as e:
    print(f"  > FAILED to query InfluxDB. Error: {e}")
finally:
    client.close()

# --- 4. Process and Pivot Data ---
if not all_dataframes:
    print(f"--- No data found for {args.train} {args.module}. ---")
    exit()

full_df = pd.concat(all_dataframes, ignore_index=True)

# Create a unique key to separate the two different temperature sources and the HV data
full_df['pivot_key'] = full_df['meas'] + '_' + full_df['module'] + '_' + full_df['type']

# Pivot the data
pivot_df = full_df.pivot_table(index='_time', columns='pivot_key', values='_value')
pivot_df.index = pivot_df.index.tz_convert('America/Chicago')
pivot_df = pivot_df.interpolate(method='time', limit=2)

# --- 5. Create Diagnostic Plot ---
fig, ax = plt.subplots(figsize=(15, 7))
fig.suptitle(f"Investigation: Train {args.train}, Module {args.module}", fontsize=14)

# 1. Plot Leakage Current (Left Y-Axis)
col_curr = f'current_{args.module}_bias_voltage'
if col_curr in pivot_df.columns:
    ax.plot(pivot_df.index, pivot_df[col_curr], color='tab:blue', label='Leakage Current (uA)')
    ax.set_ylabel('Current (uA)', color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
else:
    print(f"  > Warning: No Current data found for {args.module}")

# 2. Plot Temperature (Right Y-Axis)
ax_temp = ax.twinx()
ax_temp.set_ylabel('Temperature (C)', color='tab:red')

# THE MODULE RTD (The external one you wanted)
# ADDED 'rtd_' PREFIX HERE:
col_rtd = f'temperature_rtd_{args.module}_rtd'
if col_rtd in pivot_df.columns:
    ax_temp.plot(pivot_df.index, pivot_df[col_rtd], color='tab:red', linewidth=2.5, label=f'Module {args.module} RTD')
else:
    print(f"  > Warning: No RTD data found for rtd_{args.module}")

# THE LPGBT CHIP (The internal baseline)
col_chip = 'temperature_lpgbt_internal_lpgbt'
if col_chip in pivot_df.columns:
    ax_temp.plot(pivot_df.index, pivot_df[col_chip], color='darkred', linestyle='--', alpha=0.4, label='lpGBT Chip (Train Baseline)')
else:
    print("  > Warning: No lpGBT Chip data found.")

# 3. Plot HV Voltage (Third Y-Axis)
col_volt = f'voltage_{args.module}_bias_voltage'
if col_volt in pivot_df.columns:
    ax_volt = ax.twinx()
    ax_volt.spines['right'].set_position(('outward', 60))
    ax_volt.plot(pivot_df.index, pivot_df[col_volt], color='tab:green', linestyle=':', label='HV Bias Voltage (V)')
    ax_volt.set_ylabel('HV Voltage (V)', color='tab:green')
    ax_volt.tick_params(axis='y', labelcolor='tab:green')
else:
    print(f"  > Warning: No HV Voltage data found for {args.module}")

ax.set_xlabel('Time (Chicago)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pivot_df.index.tz))
ax.grid(True, alpha=0.3)
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))

plt.tight_layout()
if args.output:
    plt.savefig(args.output, dpi=150)
else:
    plt.show()