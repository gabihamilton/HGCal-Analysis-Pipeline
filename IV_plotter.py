import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np # Needed for generating dummy data

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

        # --- Plotting Multiple Data Series (Simulated for demonstration) ---
        # You would replace these with your actual data series.
        # For demonstration, I'm creating slight variations of your single data.

        # Line 1: sensor@1.0e16 neq/cm2, Oxide B
        ax.plot(voltage_base, current_base, 'o-', label='sensor@1.0e16 neq/cm2, Oxide B',
                color='tab:blue', linewidth=1.5, markersize=5)

        # Line 2: module@1.0e16 neq/cm2, Oxide B, CF (dashed orange)
        current_cf = current_base * 1.5 # Example variation
        ax.plot(voltage_base, current_cf, 's--', label='module@1.0e16 neq/cm2, Oxide B, CF',
                color='tab:orange', linewidth=1.5, markersize=5)

        # Line 3: module@1.0e16 neq/cm2, Oxide C, PCB (solid green)
        current_pcb = current_base * 1.8 # Example variation
        ax.plot(voltage_base, current_pcb, '^:', label='module@1.0e16 neq/cm2, Oxide C, PCB',
                color='tab:green', linewidth=1.5, markersize=5)

        # Line 4: module@1.0e16 neq/cm2, Oxide D, CuW (dotted pink)
        current_cuw = current_base * 2.0 # Example variation
        ax.plot(voltage_base, current_cuw, 'D-', label='module@1.0e16 neq/cm2, Oxide D, CuW',
                color='tab:pink', linewidth=1.5, markersize=5)

        # --- Configure Axes and Labels ---
        ax.set_yscale('log')
        ax.set_xlabel('Eff Voltage (V)', fontsize=14) # mplhep handles font sizes, but you can override
        ax.set_ylabel('Leakage Current (mA)', fontsize=14)

        # Main Plot Title
        ax.set_title('Leakage current vs Bias Voltage', fontsize=20, fontweight='bold', ha='center', y=1.03)

        # --- Fine-tune the Plot Appearance ---
        ax.set_xlim(0, max(voltage_base) * 1.05)
        # Set y-axis limits to match the reference plot's scale
        ax.set_ylim(5e-2, 5e1) # Adjust these based on your actual data range if needed

        # Legend styling, positioned inside the plot as in the reference
        ax.legend(loc='upper left', fontsize=12, frameon=True, edgecolor='black', fancybox=False)

        # CMS HGCAL Preliminary label, adjusted to match the reference plot's position and text
        hep.cms.label(
            ax=ax,
            loc=0, # Loc 0 puts "CMS" on the left, "HGCAL Preliminary" on the right of it.
            label="HGCAL Preliminary", # This becomes the right part of the label
            data=False, # Do not show (13.6 TeV)
            com=None, # Do not show center of mass energy
            year="" # Do not show year
        )
        
        # Add "Chuck T" annotation
        # Adjust x, y coordinates as needed to position it correctly on your plot
        ax.text(350, 25, 'Chuck T = -40°C', fontsize=16, fontweight='bold', ha='left',
                transform=ax.transData)


        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot successfully saved to '{output_file}'")
        
        # plt.show() # Uncomment to display the plot
        plt.close(fig) # Close the figure to free memory

    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Run the Script ---
if __name__ == "__main__":
    create_iv_plot(INPUT_CSV_FILE, OUTPUT_PLOT_FILE)