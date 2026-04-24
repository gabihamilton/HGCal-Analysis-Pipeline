import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import os
import glob
import argparse
from matplotlib.ticker import LogFormatterSciNotation, NullFormatter

# --- Configuration ---
BASE_DATA_DIRECTORY = '.'
OUTPUT_PLOT_DIRECTORY = './iv_plots'

# --- Channel & Reference Mapping ---
CHANNEL_MAP = {
    '301': 'HD1 m3', '302': 'HD1 m2', '303': 'HD1 m1',
    '304': 'LD1 e3', '305': 'LD1 e2', '306': 'LD1 e1', '307': 'LD1 w1', '308': 'LD1 w2', '309': 'LD1 w3',
    '311': 'HD2 m3', '312': 'HD2 m2', '313': 'HD2 m1',
    '314': 'LD2 e3', '315': 'LD2 e2', '316': 'LD2 e1', '317': 'LD2 w1', '318': 'LD2 w2', '319': 'LD2 w3',
    '321': 'LD3 unknown', '322': 'LD3 e3', '323': 'LD3 e2', '324': 'LD3 e1', '325': 'LD3 w1', '326': 'LD3 w2', '327': 'LD3 w3',
    '329': 'LD4 e2', '330': 'LD4 e1', '331': 'LD4 w1', '332': 'LD4 w2', '333': 'LD4 w3',
    '335': 'LD5 e2', '336': 'LD5 e1', '337': 'LD5 w1', '338': 'LD5 w2', '339': 'LD5 e3'
}

def get_color(label):
    if "HD1" in label: return "tab:red"
    if "HD2" in label: return "tab:orange"
    if "LD1" in label: return "tab:blue"
    if "LD2" in label: return "tab:cyan"
    if "LD3" in label: return "tab:green"
    if "LD4" in label: return "tab:olive"
    if "LD5" in label: return "gold"
    return "black"

def create_individual_plot(csv_file, output_dir):
    try:
        channel_num = os.path.basename(csv_file).replace('.csv', '').replace('channel_', '')
        data = pd.read_csv(csv_file)
        voltage = pd.to_numeric(data['Voltage(V)'], errors='coerce')
        current_col = 'Current(uA)' if 'Current(uA)' in data.columns else 'Current(A)'
        current = pd.to_numeric(data[current_col], errors='coerce')
        
        valid_data = pd.DataFrame({'V': voltage, 'I': current}).dropna()
        if valid_data.empty: return

        plot_data = valid_data.sort_values(by='V').reset_index(drop=True)
        plot_data['I'] = plot_data['I'].apply(lambda x: x if x > 1e-4 else 1e-4)
            
        channel_name = CHANNEL_MAP.get(channel_num, f'Channel {channel_num}')
        fig, ax = plt.subplots(figsize=(10, 8))
        hep.style.use("CMS")

        ax.plot(plot_data['V'], plot_data['I'], 'o-', label=channel_name,
                color=get_color(channel_name), linewidth=1.5, markersize=6)

        ax.set_yscale('log')
        ax.set_xlabel('Bias Voltage [V]')
        ax.set_ylabel('Leakage Current [µA]')
        ax.set_xlim(0, 305)
        ax.set_ylim(1e-3, 2000)
        ax.yaxis.set_major_formatter(LogFormatterSciNotation())
        ax.legend(loc='upper left')
        ax.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.5)
        hep.cms.label(ax=ax, label="HGCal Cassette Testing", data=True, rlabel='')

        plt.savefig(os.path.join(output_dir, f"iv_indiv_{channel_name.replace(' ', '_')}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"Error plotting {csv_file}: {e}")

def create_combined_plot(csv_files, output_dir):
    try:
        fig, ax = plt.subplots(figsize=(14, 10))
        hep.style.use("CMS")

        marker_map = {'e1': 'o', 'e2': 's', 'e3': '^', 'w1': 'v', 'w2': '<', 'w3': '>', 'm1': 'D', 'm2': 'P', 'm3': 'X'}

        for csv_file in csv_files:
            # --- RESTORED DATA LOADING LOGIC ---
            channel_num = os.path.basename(csv_file).replace('.csv', '').replace('channel_', '')
            data = pd.read_csv(csv_file)
            voltage = pd.to_numeric(data['Voltage(V)'], errors='coerce')
            current_col = 'Current(uA)' if 'Current(uA)' in data.columns else 'Current(A)'
            current = pd.to_numeric(data[current_col], errors='coerce')
            
            valid_data = pd.DataFrame({'V': voltage, 'I': current}).dropna()
            if valid_data.empty: continue

            # Lowered noise floor to 1e-4 to capture V < 25 points
            plot_data = valid_data.sort_values(by='V').reset_index(drop=True)
            plot_data['I'] = plot_data['I'].apply(lambda x: x if x > 1e-4 else 1e-4) 

            channel_name = CHANNEL_MAP.get(channel_num, f'CH {channel_num}')
            module_pos = channel_name.split()[-1] 
            marker_type = marker_map.get(module_pos, 'o') 

            ax.plot(plot_data['V'], plot_data['I'], label=channel_name,
                    color=get_color(channel_name), marker=marker_type, 
                    linestyle='-', linewidth=1.0, markersize=4, alpha=0.7)

        ax.set_yscale('log')
        ax.set_xlabel('Bias Voltage [V]')
        ax.set_ylabel('Leakage Current [µA]')
        ax.set_xlim(0, 305)
        ax.set_ylim(1e-3, 2000) # Updated Y-limit to show low-current points

        ax.yaxis.set_major_formatter(LogFormatterSciNotation())
        ax.legend(loc='upper left', title="Modules", bbox_to_anchor=(1.02, 1), ncol=2, fontsize=9)
        ax.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.3)
        hep.cms.label(ax=ax, label="HGCal Cassette Testing", data=True, rlabel='Fermilab')

        plt.savefig(os.path.join(output_dir, 'iv_combined_all.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Done! Combined plot saved.")
    except Exception as e:
        print(f"Error in combined plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=None, help='Path to a specific iv_curve_data_* directory')
    parser.add_argument('--output-dir', default=None, help='Where to save plots (default: ./iv_plots)')
    args = parser.parse_args()

    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dirs = sorted(glob.glob(os.path.join(BASE_DATA_DIRECTORY, 'iv_curve_data_*')))
        if not data_dirs:
            print("No iv_curve_data_* directories found.")
            exit(1)
        data_dir = data_dirs[-1]

    output_dir = args.output_dir if args.output_dir else OUTPUT_PLOT_DIRECTORY

    csv_files = sorted(glob.glob(os.path.join(data_dir, 'channel_*.csv')))
    if not csv_files:
        print(f"No channel_*.csv files found in {data_dir}")
        exit(1)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Reading data from: {data_dir}")
    print(f"Saving plots to:   {output_dir}")
    for f in csv_files:
        create_individual_plot(f, output_dir)
    create_combined_plot(csv_files, output_dir)