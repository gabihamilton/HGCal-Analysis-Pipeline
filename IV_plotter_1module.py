import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

# ===================================================================
# Plotting Script for IV Curve Data (CMS HGCAL Style)
# ===================================================================

# Apply the CMS plotting style.
hep.style.use("CMS")

# --- Configuration ---
INPUT_CSV_FILE = 'hgcal_iv_data.csv'
OUTPUT_PLOT_FILE = 'iv_curve_plot_hgcal_style.png'

# --- Plotting Function ---
def create_iv_plot(csv_file, output_file):
    """
    Reads a clean IV data CSV and generates a plot styled like the HGCAL example.
    """
    try:
        data = pd.read_csv(csv_file)
        voltage_base = data['Voltage(V)']
        current_base = data['Current(mA)']

        fig, ax = plt.subplots(figsize=(10, 7))

        ax.plot(voltage_base, current_base, 'o-', label='Sensor Data',
                color='tab:blue', linewidth=1.5, markersize=6)

        # --- Configure Axes and Labels ---
        ax.set_yscale('log')
        ax.set_xlabel('Eff Voltage (V)')
        ax.set_ylabel('Leakage Current (mA)')

        # --- Fine-tune the Plot Appearance ---
        ax.set_xlim(0, max(voltage_base) * 1.05)
        ax.set_ylim(bottom=min(current_base) * 0.8, top=max(current_base) * 1.2)
        ax.legend(loc='upper left', fontsize=12)

        # CMS HGCAL Preliminary label
        hep.cms.label(
            ax=ax,
            loc=0,
            label="Preliminary",
            data=False,
            com=None,
            year=""
        )


        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot successfully saved to '{output_file}'")
        plt.close(fig)

    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Run the Script ---
if __name__ == "__main__":
    create_iv_plot(INPUT_CSV_FILE, OUTPUT_PLOT_FILE)