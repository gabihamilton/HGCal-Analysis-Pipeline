import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import os
import glob
from matplotlib.ticker import ScalarFormatter, NullFormatter

# ===================================================================
# Analysis and Grouping Script for IV Curve Data
# ===================================================================

# --- Configuration ---
BASE_DATA_DIRECTORY = '.'
OUTPUT_PLOT_DIRECTORY = './iv_plots_grouped'

# --- NEW: Grouping Thresholds (in µA) ---
# Define the current thresholds for sorting channels at the max voltage.
# - A channel is "Leaky" if its final current is above this value.
# - A channel is in "Breakdown" if its final current is above this value.
LEAKY_THRESHOLD_UA = 10.0
BREAKDOWN_THRESHOLD_UA = 100.0

# Channel Mapping March 3rd
CHANNEL_MAP = {
    '301': 'HD1 m3', '302': 'HD1 m2', '303': 'HD1 m1',
    '304': 'LD1 e3', '305': 'LD1 e2', '306': 'LD1 e1', '307': 'LD1 w1', '308': 'LD1 w2', '309': 'LD1 w3',
    '311': 'HD2 m3', '312': 'HD2 m2', '313': 'HD2 m1',
    '314': 'LD2 e3', '315': 'LD2 e2', '316': 'LD2 e1', '317': 'LD2 w1', '318': 'LD2 w2', '319': 'LD2 w3',
    '321': 'LD3 unknown', '322': 'LD3 e3', '323': 'LD3 e2', '324': 'LD3 e1', '325': 'LD3 w1', '326': 'LD3 w2', '327': 'LD3 w3',
    '329': 'LD4 e2', '330': 'LD4 e1', '331': 'LD4 w1', '332': 'LD4 w2', '333': 'LD4 w3',
    '335': 'LD5 e2', '336': 'LD5 e1', '337': 'LD5 w1', '338': 'LD5 w2', '339': 'LD5 e3'
}

# --- Plotting Function for Groups ---
def create_group_plot(csv_files, group_name, output_dir):
    """
    Reads a list of IV data CSVs for a specific group and generates a combined plot.
    """
    if not csv_files:
        print(f"No channels found for the '{group_name}' group. Skipping plot.")
        return

    try:
        fig, ax = plt.subplots(figsize=(12, 9))
        hep.style.use("CMS")

        colors = plt.cm.viridis(np.linspace(0, 1, len(csv_files)))

        for i, csv_file in enumerate(csv_files):
            channel_name = os.path.basename(csv_file).replace('.csv', '').replace('channel_', '')
            data = pd.read_csv(csv_file)
            
            voltage = pd.to_numeric(data['Voltage(V)'], errors='coerce')
            current_col = 'Current(uA)' if 'Current(uA)' in data.columns else 'Current(A)'
            current = pd.to_numeric(data[current_col], errors='coerce')

            valid_data = pd.DataFrame({'V': voltage, 'I': current}).dropna()
            if valid_data.empty: continue

            origin_point = pd.DataFrame({'V': [0], 'I': [0]})
            plot_data = pd.concat([origin_point, valid_data]).sort_values(by='V').reset_index(drop=True)

            min_positive_current = plot_data[plot_data['I'] > 0]['I'].min()
            small_value = min_positive_current / 10 if pd.notna(min_positive_current) and min_positive_current > 0 else 1e-3
            plot_data['I'] = plot_data['I'].replace(0, small_value)

            legend_label = CHANNEL_MAP.get(channel_name, f'Channel {channel_name}')
            ax.plot(plot_data['V'], plot_data['I'], 'o-', label=legend_label,
                    color=colors[i], linewidth=1.5, markersize=6)

        ax.set_yscale('log')
        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Leakage Current (µA)')
        ax.set_xlim(0,305)
        ax.set_ylim(1e-3, 2000)

        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())

        ax.legend(loc='upper left', title=f"{group_name} Sensors", ncol=3, fontsize='small')
        ax.grid(True, which="both", linestyle='--', linewidth=0.5)
        hep.cms.label(ax=ax, label="HGCal Cassete Testing", data=True)

        output_file = os.path.join(output_dir, f'iv_curve_group_{group_name.lower().replace(" ", "_")}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Group plot saved to '{output_file}'")
        plt.close(fig)

    except Exception as e:
        print(f"An error occurred while creating the '{group_name}' plot: {e}")

# --- Main Analysis Script ---
if __name__ == "__main__":
    # Find the latest data directory
    data_dirs = sorted(glob.glob(os.path.join(BASE_DATA_DIRECTORY, 'iv_curve_data_*')))
    if not data_dirs:
        print("Error: No 'iv_curve_data_*' directories found.")
        exit()
    
    latest_data_dir = data_dirs[-1]
    csv_files = sorted(glob.glob(os.path.join(latest_data_dir, 'channel_*.csv')))
    
    if not csv_files:
        print(f"Error: No 'channel_*.csv' files found in '{latest_data_dir}'.")
        exit()

    # Create dictionaries to hold the categorized channel files
    groups = {
        "Good": [],
        "Leaky": [],
        "Breakdown": []
    }

    print("\n--- Analyzing and Categorizing Channels ---")
    for csv_file in csv_files:
        try:
            channel_name = os.path.basename(csv_file).replace('.csv', '').replace('channel_', '')
            data = pd.read_csv(csv_file)
            if data.empty: continue

            current_col = 'Current(uA)' if 'Current(uA)' in data.columns else 'Current(A)'
            final_current = pd.to_numeric(data[current_col], errors='coerce').iloc[-1]
            
            if final_current >= BREAKDOWN_THRESHOLD_UA:
                groups["Breakdown"].append(csv_file)
            elif final_current >= LEAKY_THRESHOLD_UA:
                groups["Leaky"].append(csv_file)
            else:
                groups["Good"].append(csv_file)
        except Exception as e:
            print(f"Could not process {csv_file}: {e}")

    print("\n--- Summary ---")
    for group_name, files in groups.items():
        channel_names = [CHANNEL_MAP.get(os.path.basename(f).replace('.csv','').replace('channel_',''), os.path.basename(f)) for f in files]
        print(f"{group_name} Channels ({len(files)}): {', '.join(channel_names)}")
    print("-----------------\n")

    # Create the output directory
    os.makedirs(OUTPUT_PLOT_DIRECTORY, exist_ok=True)
    
    # Generate a plot for each group
    for group_name, files in groups.items():
        create_group_plot(files, group_name, OUTPUT_PLOT_DIRECTORY)
