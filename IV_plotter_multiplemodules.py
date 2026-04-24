import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import os # Added to check if files exist

# ===================================================================
# Plotting Script for IV Curve Data (CMS HGCAL Style)
# ===================================================================

# Apply the CMS plotting style.
hep.style.use("CMS")

# --- Configuration for Multiple Modules ---
# 1. Use a dictionary to define the files.
# The key is the label that will appear in the legend.
# The value is the filename for that dataset.
# NOTE: Replace these with your actual filenames.
INPUT_FILES = {
    'Module 1': 'hgcal_iv_data0.csv',
    'Module 2': 'hgcal_iv_data1.csv',
    'Module 3': 'hgcal_iv_data3.csv',
}

OUTPUT_PLOT_FILE = 'iv_curve_plot_multi_module.png'

# --- Plotting Function ---
def create_multi_iv_plot(files_dict, output_file):
    """
    Reads data for multiple modules from a dictionary of CSV files
    and generates a single, styled comparison plot.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # --- Prepare for Plotting ---
    # 2. Define lists of colors and markers to cycle through for each module.
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(files_dict)))
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
    
    # Keep track of the overall data range to set axis limits correctly
    max_voltage_overall = 0
    min_current_overall = float('inf')
    max_current_overall = 0
    
    # --- Loop Through and Plot Each File ---
    # 3. Loop over the dictionary of files.
    for i, (label, file_path) in enumerate(files_dict.items()):
        try:
            if not os.path.exists(file_path):
                print(f"Warning: Data file not found for '{label}': {file_path}. Skipping.")
                continue

            data = pd.read_csv(file_path)
            voltage = data['Voltage(V)']
            current = data['Current(mA)']

            # Plot this module's data with a unique style
            ax.plot(voltage, current,
                    marker=markers[i % len(markers)], # Cycle through markers
                    linestyle='-',
                    label=label,
                    color=colors[i], # Assign a unique color
                    linewidth=1.5,
                    markersize=6)
            
            # Update the overall min/max values for axis scaling
            if not voltage.empty:
                max_voltage_overall = max(max_voltage_overall, voltage.max())
            if not current.empty:
                min_current_overall = min(min_current_overall, current.min())
                max_current_overall = max(max_current_overall, current.max())

        except Exception as e:
            print(f"An error occurred while processing '{file_path}': {e}")

    # --- Configure Final Plot ---
    # 4. Configure labels and limits AFTER plotting all data.
    ax.set_yscale('log')
    ax.set_xlabel('Eff Voltage (V)')
    ax.set_ylabel('Leakage Current (mA)')

    # Set limits based on all datasets plotted
    if max_voltage_overall > 0:
        ax.set_xlim(0, max_voltage_overall * 1.05)
        ax.set_ylim(bottom=min_current_overall * 0.8, top=max_current_overall * 1.2)

    # Add legend for all plotted lines
    ax.legend(loc='upper left', fontsize=12)

    # Add CMS label
    hep.cms.label(ax=ax, loc=0, label="Preliminary", data=False, com=None, year="2025")

    # Save the final plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to '{output_file}'")
    plt.close(fig)


# --- Run the Script ---
if __name__ == "__main__":
    create_multi_iv_plot(INPUT_FILES, OUTPUT_PLOT_FILE)