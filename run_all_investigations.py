#!/usr/bin/env python3
# run_all_investigations.py

# Author: Gabi Hamilton
# Date: October 2025
# Description: This script loops through all known modules, runs the
#              investigation script for each, and saves the resulting plot.

import os
import subprocess
from cassette_map import sensor_positions # We'll get the module list from our map

# --- Configuration ---
OUTPUT_DIR = "investigation_plots"
PYTHON_CMD = "python" # or "python3" if needed

# --- Main Script ---
def main():
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"--- Batch investigation started. Plots will be saved to '{OUTPUT_DIR}' ---")
    
    total_modules = 0
    
    # Loop through every sensor in our map
    for key in sensor_positions.keys():
        
        # We only want to run this for module RTDs, not the lpGBTs
        if "lpgbt_internal" in key:
            continue
            
        # Parse the train and module from the key (e.g., "LD1_e1")
        try:
            train_tag, module_tag = key.split('_', 1)
        except ValueError:
            print(f"Skipping malformed key: {key}")
            continue
            
        total_modules += 1
        print(f"\n--- Processing: Train {train_tag}, Module {module_tag} ---")
        
        # Define the output filename
        filename = f"investigation_{train_tag}_{module_tag}.png"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # Build the command to run
        command = [
            PYTHON_CMD,
            "investigate_module.py",
            "--train", train_tag,
            "--module", module_tag,
            "--output", output_path
        ]
        
        # Run the command
        try:
            # We use check=True to make sure it stops if one script fails
            subprocess.run(command, check=True) 
        except subprocess.CalledProcessError as e:
            print(f"!!! FAILED to process {train_tag} {module_tag}. Error: {e} !!!")
        
    print(f"\n--- Batch investigation complete! Processed {total_modules} modules. ---")

if __name__ == "__main__":
    main()