import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import os
import glob
from matplotlib.ticker import ScalarFormatter, NullFormatter

# ===================================================================
# Plotting Script for IV Curve Comparison
# ===================================================================

# --- Configuration ---
BASE_DATA_DIRECTORY = '.'
OUTPUT_PLOT_DIRECTORY = './iv_plots_comparison'

# --- Channel & Reference Mapping ---
#CHANNEL_MAP = {
#    '304': 'LD1 e3', '305': 'LD1 e2', '306': 'LD1 e1', '307': 'LD1 w1',
#    '308': 'LD1 w2', '309': 'LD1 w3', '314': 'LD2 e3', '315': 'LD2 e2',
#    '316': 'LD2 e1', '317': 'LD2 w1', '318': 'LD2 w2', '322': 'LD3 e3',
#    '323': 'LD3 e2', '324': 'LD3 e1', '312': 'LD3 w1', '326': 'LD3 w2',
#    '327': 'LD3 w3', '329': 'LD4 e3', '330': 'LD4 e2', '331': 'LD4 e1',
#    '311': 'LD4 w1', '333': 'LD4 w2', '335': 'LD5 e2', '336': 'LD5 e1',
#    '337': 'LD5 w1', '338': 'LD5 w2'
#}
# March 3rd:
CHANNEL_MAP = {
    '301': 'HD1 m3', '302': 'HD1 m2', '303': 'HD1 m1',
    '304': 'LD1 e3', '305': 'LD1 e2', '306': 'LD1 e1', '307': 'LD1 w1', '308': 'LD1 w2', '309': 'LD1 w3',
    '311': 'HD2 m3', '312': 'HD2 m2', '313': 'HD2 m1',
    '314': 'LD2 e3', '315': 'LD2 e2', '316': 'LD2 e1', '317': 'LD2 w1', '318': 'LD2 w2', '319': 'LD2 w3',
    '321': 'LD3 unknown', '322': 'LD3 e3', '323': 'LD3 e2', '324': 'LD3 e1', '325': 'LD3 w1', '326': 'LD3 w2', '327': 'LD3 w3',
    '329': 'LD4 e2', '330': 'LD4 e1', '331': 'LD4 w1', '332': 'LD4 w2', '333': 'LD4 w3',
    '335': 'LD5 e2', '336': 'LD5 e1', '337': 'LD5 w1', '338': 'LD5 w2', '339': 'LD5 e3'
}

REFERENCE_MAP = {
    'LD1 e3': '320MLF3TCTT0202', 'LD1 e2': '320MLF3TCCM0103', 'LD1 e1': '320MLF3TCCM0101',
    'LD1 w1': '320MLF3TCCM0113', 'LD1 w2': '320MLF3TCCM0108', 'LD2 e3': '320MLF3CXTT0021',
    'LD2 e2': '320MLF3TCTT0206', 'LD2 e1': '320MLF3TCTT0209', 'LD2 w1': '320MLF3TCTT0208',
    'LD2 w2': '320MLF3TCTT0203', 'LD4 e3': '320MLF3TCTT0212', 'LD4 w2': '320MLF3TCTT0207'
}

# --- Plotting Function ---
def create_comparison_plot(csv_file, output_dir):
    """
    Reads a single IV data CSV and generates a plot with a comparison title.
    """
    try:
        channel_num = os.path.basename(csv_file).replace('.csv', '').replace('channel_', '')
        
        data = pd.read_csv(csv_file)
        voltage = pd.to_numeric(data['Voltage(V)'], errors='coerce')
        current_col = 'Current(uA)' if 'Current(uA)' in data.columns else 'Current(A)'
        current = pd.to_numeric(data[current_col], errors='coerce')

        valid_data = pd.DataFrame({'V': voltage, 'I': current}).dropna()
        if valid_data.empty:
            print(f"Warning: No valid numeric data found in {csv_file}. Skipping.")
            return

        origin_point = pd.DataFrame({'V': [0], 'I': [0]})
        plot_data = pd.concat([origin_point, valid_data]).sort_values(by='V').reset_index(drop=True)

        min_positive_current = plot_data[plot_data['I'] > 0]['I'].min()
        small_value = min_positive_current / 10 if pd.notna(min_positive_current) and min_positive_current > 0 else 1e-3
        plot_data['I'] = plot_data['I'].replace(0, small_value)

        channel_name = CHANNEL_MAP.get(channel_num, f'Channel {channel_num}')
        reference_id = REFERENCE_MAP.get(channel_name)
        
        plot_title = f'IV Curve for {channel_name}'
        if reference_id:
            plot_title += f' (Ref: {reference_id})'

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # --- CHANGE: Use standard Matplotlib style, not CMS ---
        # hep.style.use("CMS") # This line is commented out

        ax.set_title(plot_title, fontsize=18)

        ax.plot(plot_data['V'], plot_data['I'], 'o-', label=f'Sensor Data ({channel_name})',
                color='tab:blue', linewidth=1.5, markersize=6)

        ax.set_yscale('log')
        ax.set_xlabel('Reverse Bias [V]') # Changed for consistency
        ax.set_ylabel('Leakage Current [µA]') # Changed for consistency
        
        # --- CHANGE: Set fixed axis limits to match reference plot ---
        ax.set_xlim(0, 305)
        ax.set_ylim(1e-3, 2000)

        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())

        ax.legend(loc='upper left')
        ax.grid(True, which="both", linestyle='--', linewidth=0.5)

        # --- CHANGE: Removed the hep.cms.label line completely ---
        # hep.cms.label(ax=ax, label="HGCal Cassete Testing", data=True, rlabel='')

        output_filename = f"iv_curve_{channel_name.replace(' ', '_')}.png"
        output_file = os.path.join(output_dir, output_filename)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to '{output_file}'")
        plt.close(fig)

    except Exception as e:
        print(f"An error occurred while plotting {csv_file}: {e}")

# --- Main Script Execution ---
if __name__ == "__main__":
    data_dirs = sorted(glob.glob(os.path.join(BASE_DATA_DIRECTORY, 'iv_curve_data_*')))
    if not data_dirs:
        print("Error: No 'iv_curve_data_*' directories found.")
        exit()
    
    latest_data_dir = data_dirs[-1]
    csv_files = sorted(glob.glob(os.path.join(latest_data_dir, 'channel_*.csv')))
    
    if not csv_files:
        print(f"Error: No 'channel_*.csv' files found in '{latest_data_dir}'.")
        exit()

    os.makedirs(OUTPUT_PLOT_DIRECTORY, exist_ok=True)
    
    for file in csv_files:
        create_comparison_plot(file, OUTPUT_PLOT_DIRECTORY)

