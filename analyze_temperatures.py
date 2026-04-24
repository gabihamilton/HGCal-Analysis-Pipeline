# analyze_temperatures.py

# Author: Gabi Hamilton
# Date: October 2025
# Description: This script connects to InfluxDB instances to retrieve temperature data
#              from RTD sensors in cassette modules, processes the data, and generates
#              visualizations of the temperature distribution across the cassette.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
# --- Add this line to create a custom legend ---
from matplotlib.lines import Line2D 
# --- Add these lines to hide the pivot warning ---
import warnings
from influxdb_client.client.warnings import MissingPivotFunction
warnings.simplefilter("ignore", MissingPivotFunction)


# --- Step 1: Load Our Custom Sensor Map ---
from cassette_map import sensor_positions

# --- Step 2: Configure InfluxDB Connections ---
# (This section is identical to your script)
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


# --- Step 3: Query Data for All Sensors ---
# (This section is identical to your script)
all_sensor_data = []
print("--- Starting to query sensor data from InfluxDB ---")
for key, position in sensor_positions.items():
    train_tag, module_tag = key.split('_', 1)
    server_url = train_to_server_map.get(train_tag)
    if not server_url:
        print(f"Warning: No server mapping found for train '{train_tag}'. Skipping {key}.")
        continue
    print(f"Querying {key} from {server_url}...")
    client = influxdb_client.InfluxDBClient(
        url=server_url, token=token_map[server_url], org=INFLUX_ORG
    )
    query_api = client.query_api()
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: 2025-10-17T12:00:00Z, stop: 2025-10-17T17:00:00Z)
      |> filter(fn: (r) => r._measurement == "cassette_readings")
      |> filter(fn: (r) => r.meas == "temperature")
      |> filter(fn: (r) => r.train == "{train_tag}")
      |> filter(fn: (r) => r.module == "{module_tag}")
      |> mean()
    '''
    try:
        result_df = query_api.query_data_frame(query=query)
        if not result_df.empty:
            result_df['x_pos'] = position[0]
            result_df['y_pos'] = position[1]
            all_sensor_data.append(result_df)
        else:
            print(f"  > No data returned for {key}.")
    except Exception as e:
        print(f"  > FAILED to query {key}. Error: {e}")
    finally:
        client.close()


# --- Step 4: Process Data for Plotting ---
# (This section is identical to your script)
if not all_sensor_data:
    print("\n--- No data found for any sensors. Exiting. ---")
    exit()
print(f"\n--- Processing {len(all_sensor_data)} data points for visualization ---")
try:
    final_df = pd.concat(all_sensor_data, ignore_index=True)
    x_coords = final_df['x_pos'].values
    y_coords = final_df['y_pos'].values
    temps = final_df['_value'].values
    print("\nFinal Data Points (first 5):")
    print(final_df[['train', 'module', 'x_pos', 'y_pos', '_value']].head())
except Exception as e:
    print(f"\n--- ERROR during data processing (pd.concat): {e} ---")
    exit()

# --- Helper function to get hexagon vertices (flat-top) ---
# (This section is identical to your script)
def get_hexagon_vertices(center_x, center_y, radius, shape_type="full"):
    angles = np.array([np.pi/2, 5*np.pi/6, 7*np.pi/6, 3*np.pi/2, 11*np.pi/6, np.pi/6])
    vertices_x = center_x + radius * np.cos(angles)
    vertices_y = center_y + radius * np.sin(angles)
    full_vertices = np.array(list(zip(vertices_x, vertices_y)))
    v_top = full_vertices[0]
    v_top_left = full_vertices[1]
    v_bottom_left = full_vertices[2]
    v_bottom = full_vertices[3]
    v_bottom_right = full_vertices[4]
    v_top_right = full_vertices[5]
    if shape_type == "full":
        return full_vertices
    elif shape_type == "left_cut":
        v_new_top_left = (1, v_top_left[1])
        v_new_bottom_left = (1, v_bottom_left[1])
        return [v_top, v_new_top_left, v_new_bottom_left, v_bottom, v_bottom_right, v_top_right]
    elif shape_type == "top_left_cut":
        return [v_top, v_top_right, v_bottom_right, v_bottom, v_bottom_left]
    elif shape_type == "half_left_cut":
        return [(v_top+v_top_right)/2, v_top_right, v_bottom_right, v_bottom, (v_bottom+v_bottom_left)/2]
    return full_vertices


# --- Step 5: Separate Sensors and Create Plot Base ---
print("\n--- Separating sensor types ---")
module_sensors = {k: v for k, v in sensor_positions.items() if "lpgbt_internal" not in k}
lpgbt_sensors = {k: v for k, v in sensor_positions.items() if "lpgbt_internal" in k}
print(f"--- Found {len(module_sensors)} module RTDs and {len(lpgbt_sensors)} lpGBT sensors ---")
print("\n--- Generating hexagonal plot (flat-top) ---")
try:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # --- CHANGED: Use ALL temps for normalization ---
    if not final_df.empty:
        temps = final_df['_value'] # Use all temps
        vmin = temps.min()
        vmax = temps.max()
    else:
        vmin, vmax = -20, 30 # Default if no data

    cmap = plt.get_cmap('coolwarm') # Changed colormap to coolwarm
    norm = Normalize(vmin=vmin, vmax=vmax)

    radius = 1 / np.sqrt(3)

    # --- Step 6: Draw Hexagons (Modules) ---
    # (This section is identical to your script)
    print(f"--- Plotting {len(module_sensors)} module hexagons ---")
    def get_plot_coords(grid_x, grid_y, radius):
        center_x = grid_x * (radius * np.sqrt(3)) + ((grid_y % 2) * (radius * np.sqrt(3)) / 2)
        center_y = grid_y * (radius * 1.5)
        return center_x, center_y
    for key, (grid_x, grid_y) in module_sensors.items():
        center_x, center_y = get_plot_coords(grid_x, grid_y, radius)
        shape_type = "full"
        if key in ["LD2_w3"]:
            shape_type = "left_cut"
        elif key == "LD5_w2":
            shape_type = "half_left_cut"
        elif key == "LD5_e3":
            shape_type = "top_left_cut"
        train_tag, module_tag = key.split('_', 1)
        data_row = final_df[
            (final_df['train'] == train_tag) &
            (final_df['module'] == module_tag)
        ]
        if not data_row.empty:
            temp = data_row['_value'].iloc[0]
            color = cmap(norm(temp))
        else:
            color = 'black'
        vertices = get_hexagon_vertices(center_x, center_y, radius, shape_type=shape_type)
        polygon = patches.Polygon(
            vertices, closed=True, facecolor=color, edgecolor='white', linewidth=1.0
        )
        ax.add_patch(polygon)

    # --- Step 7: Draw Circles (lpGBTs) ---
    print(f"--- Plotting {len(lpgbt_sensors)} lpGBT sensor markers ---")
    for key, (grid_x, grid_y) in lpgbt_sensors.items():
        center_x, center_y = get_plot_coords(grid_x, grid_y, radius)
        
        # --- CHANGED: Look up temp and color the circle ---
        train_tag, module_tag = key.split('_', 1)
        data_row = final_df[
            (final_df['train'] == train_tag) &
            (final_df['module'] == module_tag)
        ]
        
        if not data_row.empty:
            temp = data_row['_value'].iloc[0]
            color = cmap(norm(temp))
        else:
            color = 'grey' # Fallback color if lpGBT data is missing
            
        ax.scatter(
            center_x, center_y,
            s=100,
            facecolor=color, # Use the mapped color
            edgecolor='black',
            linewidth=1.5,
            zorder=10
        )

    # --- Step 8: Final Plot Setup ---
    print("--- Setting up final plot axes and colorbar ---")
    ax.set_aspect('equal')
    all_plot_x = []
    all_plot_y = []
    for key, (grid_x, grid_y) in sensor_positions.items():
        x, y = get_plot_coords(grid_x, grid_y, radius)
        all_plot_x.append(x)
        all_plot_y.append(y)
    ax.set_xlim(min(all_plot_x) - radius*1.5, max(all_plot_x) + radius*1.5)
    ax.set_ylim(min(all_plot_y) - radius*1.5, max(all_plot_y) + radius*1.5)

    
    # --- CHANGED: Updated colorbar label ---
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Temperature (°C)')
    
    # --- CHANGED: Manually create a consistent legend ---
    legend_elements = [Line2D([0], [0], marker='o', color='w',
                            label='lpGBT Location',
                            markerfacecolor='white', # Use white just for the legend
                            markeredgecolor='black', markersize=10)]
    ax.legend(handles=legend_elements, loc='upper left')
    
    ax.set_title('Cassette Temperature (Oct 17, 8AM-1PM Avg)')
    #ax.set_xlabel('X Position (Grid Units)')
    #ax.set_ylabel('Y Position (Grid Units)')
    ax.axis("off")  # Hide axes for cleaner look
    
    print("--- Displaying plot... ---")
    plt.show()

except Exception as e:
    print(f"\n--- ERROR during plotting (matplotlib): {e} ---")

print("\n--- Analysis complete! ---")