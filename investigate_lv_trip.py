#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import influxdb_client
import os
import matplotlib.dates as mdates
import warnings
from influxdb_client.client.warnings import MissingPivotFunction
warnings.simplefilter("ignore", MissingPivotFunction)

#INFLUX_URL = "http://localhost:8087"
#INFLUX_TOKEN = os.environ.get("INFLUXDB_TOKEN", "4SiGtuPRS4xtdOaqTNBO8-Y_MGZrah9hjEXd9Pq77Vlg1gWx2R7iMHSMprVe7mpEF2yHkp1R0iDt7SA3GKmJdQ==")
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN= os.environ.get("INFLUX_TOKEN", "7OCEYNtM_xoMu3aW4Cu8R0lIfBJSl54GA5urmdnMdavWKiUovAbvj8Lln-zK_99yg5SVp3KKrzDpRJiNPYDFEw==")
INFLUX_ORG = "fnal"
INFLUX_BUCKET = "RTDs_2025Oct"

#TIME_WINDOW_START = "2026-03-03T15:00:00Z" # 09:00 AM
#TIME_WINDOW_STOP = "2026-03-03T23:00:00Z"  # 05:00 PM
TIME_WINDOW_START = "2026-03-11T18:10:00Z"
TIME_WINDOW_STOP = "2026-03-11T19:25:00Z"
TARGET_TRAIN = "LD1+HD1"

print(f"--- Querying LV Data for {TARGET_TRAIN} ---")
client = influxdb_client.InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
query_api = client.query_api()

query = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: {TIME_WINDOW_START}, stop: {TIME_WINDOW_STOP})
  |> filter(fn: (r) => r._measurement == "cassette_readings")
  |> filter(fn: (r) => r.type == "low_voltage")
  |> filter(fn: (r) => r.train == "{TARGET_TRAIN}")
  |> aggregateWindow(every: 30s, fn: mean, createEmpty: false)
  |> yield(name: "mean")
'''

try:
    result = query_api.query_data_frame(query=query)
    if isinstance(result, list):
        df = pd.concat(result, ignore_index=True)
    else:
        df = result
except Exception as e:
    print(f"Failed to query: {e}")
    exit()
finally:
    client.close()

if df is None or df.empty:
    print("No LV data found.")
    exit()

# Pivot to separate voltage and current
pivot_df = df.pivot_table(index='_time', columns='meas', values='_value')
pivot_df.index = pivot_df.index.tz_convert('America/Chicago')
pivot_df = pivot_df.interpolate(method='time', limit=2)

fig, ax1 = plt.subplots(figsize=(12, 6))
fig.suptitle(f"Low Voltage Trip Investigation: {TARGET_TRAIN}", fontsize=14)

# Plot LV Current (Amps)
if 'current' in pivot_df.columns:
    ax1.plot(pivot_df.index, pivot_df['current'], color='red', linewidth=2, label='LV Current (A)')
    ax1.set_ylabel('LV Current (Amperes)', color='red')
    ax1.tick_params(axis='y', labelcolor='red')

# Plot LV Voltage (Volts)
ax2 = ax1.twinx()
if 'voltage' in pivot_df.columns:
    ax2.plot(pivot_df.index, pivot_df['voltage'], color='blue', linestyle='--', label='LV Voltage (V)')
    ax2.set_ylabel('LV Voltage (Volts)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

ax1.set_xlabel('Time (Chicago)')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pivot_df.index.tz))
ax1.grid(True, alpha=0.3)

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.tight_layout()
plt.show()
