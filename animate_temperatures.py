#!/usr/bin/env python3
# animate_temperatures.py

# Author: Gabi Hamilton
# Date: October 2025 / Updated March 2026
# Description: This script retrieves temperature data and generates an animated 
#              visualization of the temperature distribution across the cassette.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import influxdb_client
import os
import matplotlib.patches as patches
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import matplotlib.animation as animation
from matplotlib.lines import Line2D 
import warnings
from influxdb_client.client.warnings import MissingPivotFunction
warnings.simplefilter("ignore", MissingPivotFunction)

# --- Step 1: Load Our Custom Sensor Map ---
from cassette_map import sensor_positions

# --- Step 2: Configure Single InfluxDB Connection ---
INFLUX_URL = "http://localhost:8087"
INFLUX_TOKEN = os.environ.get("INFLUXDB_TOKEN", "4SiGtuPRS4xtdOaqTNBO8-Y_MGZrah9hjEXd9Pq77Vlg1gWx2R7iMHSMprVe7mpEF2yHkp1R0iDt7SA3GKmJdQ==")
INFLUX_ORG = "fnal"
INFLUX_BUCKET = "RTDs_2025Oct"

# --- Animation Configuration (March 3 Cold Test) ---
TIME_WINDOW_START = "2026-03-03T15:00:00Z" # 09:00 AM Chicago Time
TIME_WINDOW_STOP = "2026-03-03T23:00:00Z"  # 05:00 PM Chicago Time
TIME_INTERVAL = "1m"                      
OUTPUT_FILENAME = "cassette_animation.mp4"
FRAMES_PER_SECOND = 10                    


# --- Step 3: Query ALL Time-Series Data ---
print("--- Querying all temperature data from InfluxDB ---")

client = influxdb_client.InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
query_api = client.query_api()

# Simplified query: Grabs all RTD and lpGBT temperatures at once
query = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: {TIME_WINDOW_START}, stop: {TIME_WINDOW_STOP})
  |> filter(fn: (r) => r._measurement == "cassette_readings")
  |> filter(fn: (r) => r.type == "rtd" or r.type == "lpgbt")
  |> aggregateWindow(every: {TIME_INTERVAL}, fn: mean, createEmpty: false)
  |> yield(name: "mean")
'''

try:
    full_df = query_api.query_data_frame(query=query)
    # If the query returns a list of dataframes, concatenate them
    if isinstance(full_df, list):
        full_df = pd.concat(full_df, ignore_index=True)
except Exception as e:
    print(f"  > FAILED to query. Error: {e}")
    exit()
finally:
    client.close()

# --- Step 4: Process All Data ---
if full_df is None or full_df.empty:
    print("\n--- No data found. Check your token and time window. ---")
    exit()

# Get a sorted list of all the unique timestamps (our "frames")
unique_timestamps = sorted(full_df['_time'].unique())
num_frames = len(unique_timestamps)
print(f"--- Found {len(full_df)} total data points across {num_frames} time frames ---")

# --- Step 5: Setup the Plot Elements ---
print("--- Initializing plot for animation ---")
fig, ax = plt.subplots(figsize=(12, 8))

# Global colormap normalization (Locked between -30C and +50C)
vmin = -30 
vmax = 50 

# Create the custom granular colormap: Coldest (Purple) to Warmest (Red)
custom_colors = ['indigo', 'mediumblue', 'yellow', 'orange', 'red', 'darkred']
cmap = LinearSegmentedColormap.from_list('granular_temp', custom_colors)
norm = Normalize(vmin=vmin, vmax=vmax)

radius = 1 / np.sqrt(3)

def get_plot_coords(grid_x, grid_y, radius):
    center_x = grid_x * (radius * np.sqrt(3)) + ((grid_y % 2) * (radius * np.sqrt(3)) / 2)
    center_y = grid_y * (radius * 1.5)
    return center_x, center_y

def get_hexagon_vertices(center_x, center_y, radius, shape_type="full"):
    angles = np.array([np.pi/2, 5*np.pi/6, 7*np.pi/6, 3*np.pi/2, 11*np.pi/6, np.pi/6])
    vertices_x = center_x + radius * np.cos(angles)
    vertices_y = center_y + radius * np.sin(angles)
    full_vertices = np.array(list(zip(vertices_x, vertices_y)))
    v_top, v_top_left, v_bottom_left, v_bottom, v_bottom_right, v_top_right = full_vertices
    if shape_type == "full":
        return full_vertices
    elif shape_type == "left_cut":
        return [v_top, (1, v_top_left[1]), (1, v_bottom_left[1]), v_bottom, v_bottom_right, v_top_right]
    elif shape_type == "top_left_cut":
        return [v_top, v_top_right, v_bottom_right, v_bottom, v_bottom_left]
    elif shape_type == "half_left_cut":
        return [(v_top+v_top_right)/2, v_top_right, v_bottom_right, v_bottom, (v_bottom+v_bottom_left)/2]
    return full_vertices

hex_patches = {}
lpgbt_circles = []

# Draw Hexagons
for key, (grid_x, grid_y) in sensor_positions.items():
    if "lpgbt_internal" in key: continue
        
    center_x, center_y = get_plot_coords(grid_x, grid_y, radius)
    shape_type = "full"
    if key in ["LD2_w3"]: shape_type = "left_cut"
    elif key == "LD5_w2": shape_type = "half_left_cut"
    elif key == "LD5_e3": shape_type = "top_left_cut"
        
    vertices = get_hexagon_vertices(center_x, center_y, radius, shape_type=shape_type)
    polygon = patches.Polygon(vertices, closed=True, facecolor='black', edgecolor='white', linewidth=1.0)
    ax.add_patch(polygon)
    hex_patches[key] = polygon 

# Draw Circles
for key, (grid_x, grid_y) in sensor_positions.items():
    if "lpgbt_internal" not in key: continue
        
    center_x, center_y = get_plot_coords(grid_x, grid_y, radius)
    circle = ax.scatter(center_x, center_y, s=100, facecolor='black', edgecolor='white', linewidth=1.5, zorder=10)
    lpgbt_circles.append((key, circle)) 

# Axes limits
ax.set_aspect('equal')
all_plot_x = [get_plot_coords(x, y, radius)[0] for x, y in sensor_positions.values()]
all_plot_y = [get_plot_coords(x, y, radius)[1] for x, y in sensor_positions.values()]
ax.set_xlim(min(all_plot_x) - radius*1.5, max(all_plot_x) + radius*1.5)
ax.set_ylim(min(all_plot_y) - radius*1.5, max(all_plot_y) + radius*1.5)
ax.axis('off') 

# Colorbar & Legend
sm = ScalarMappable(cmap=cmap, norm=norm)
fig.colorbar(sm, ax=ax, label='Temperature (°C)')
legend_elements = [Line2D([0], [0], marker='o', color='w', label='lpGBT Location', markerfacecolor='grey', markeredgecolor='white', markersize=10)]
ax.legend(handles=legend_elements, loc='upper left')

# Title setup
plot_title = ax.set_title('Cassette Temperature', fontsize=16)


# --- Step 6: Define Animation Function ---
def update(frame):
    timestamp = unique_timestamps[frame]
    frame_data = full_df[full_df['_time'] == timestamp]
    
    # Update Hexagons (The External RTDs)
    for key, patch in hex_patches.items():
        train_tag, module_tag = key.split('_', 1)
        data_row = frame_data[
            (frame_data['train'] == train_tag) &
            (frame_data['module'] == f"rtd_{module_tag}")
        ]
        
        # Check if the row exists AND the temperature is a valid number
        if not data_row.empty and not pd.isna(data_row['_value'].iloc[0]):
            temp = data_row['_value'].iloc[0]
            patch.set_facecolor(cmap(norm(temp)))
        else:
            # If data is missing or NaN, turn the module black
            patch.set_facecolor('black')
            
    # Update lpGBT Circles
    for key, circle in lpgbt_circles:
        train_tag, module_tag = key.split('_', 1) 
        data_row = frame_data[
            (frame_data['train'] == train_tag) &
            (frame_data['module'] == module_tag)
        ]
        
        # Check if the row exists AND the temperature is a valid number
        if not data_row.empty and not pd.isna(data_row['_value'].iloc[0]):
            temp = data_row['_value'].iloc[0]
            circle.set_facecolor(cmap(norm(temp))) 
        else:
             # If data is missing or NaN, turn the chip black
            circle.set_facecolor('black')

    # Convert timestamp to Chicago time for display
    local_time = timestamp.tz_convert('America/Chicago')
    plot_title.set_text(f'Cassette Temperature at {local_time.strftime("%H:%M")} CT')

    if frame % 20 == 0:
        print(f"Processing frame {frame}/{num_frames}...")
    
    return list(hex_patches.values()) + [c[1] for c in lpgbt_circles] + [plot_title]

# --- Step 7: Create and Save Animation ---
print("\n--- Creating animation, this will take some time... ---")
ani = animation.FuncAnimation(fig, update, frames=num_frames, blit=False)
ani.save(OUTPUT_FILENAME, writer='ffmpeg', fps=FRAMES_PER_SECOND, dpi=150)
print(f"\n--- Animation saved to {OUTPUT_FILENAME}! ---")