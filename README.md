# HGCAL Data Analysis Pipeline

This script processes raw HGCAL data from text files, unpacks the binary packets, and generates plots for ADC, ADC-1, and noise distributions for each module.

---

## Prerequisites

Before you begin, ensure you have a Conda-based package manager installed, such as **Miniconda**, **Anaconda**, or **Micromamba**.

---

## Installation

Follow these steps to set up the environment and install the required packages.

1.  **Clone the Repository (if applicable)**:
    If you have this project in a Git repository, clone it first:
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create the Environment from File**:
    This repository includes an `environment.yml` file that defines all necessary packages. Open your terminal and use the command that matches your package manager.

    * **For Micromamba users:**
        ```bash
        micromamba create -f environment.yml
        ```

    * **For Conda / Anaconda users:**
        ```bash
        conda env create -f environment.yml
        ```
    This will create a new environment named `hgcal-analysis`.

3.  **Activate the Environment**:
    You must activate the environment every time you want to run the script.
    ```bash
    micromamba activate hgcal-analysis
    # OR for Conda users:
    # conda activate hgcal-analysis
    ```
    Your terminal prompt should now show `(hgcal-analysis)`.

---

## Usage

The main script `analysis_script.py` is run from the command line and takes the name of your data folder as an input.

### Project Structure

Organize your data files in a folder. The script will look for this folder in the same directory where it is located.
```
your-project-directory/
├── analysis_script.py
└── data/
├── data_file_1.txt
├── data_file_2.txt
└── ...
```

### Running the Analysis

To run the script, use the following command, replacing `data` with the name of your data folder.

```bash
python analysis_script.py data
```

### Optional Arguments

* **`--marker_link`**: You can specify which link column contains the start-of-packet marker. It defaults to `link6`. If your marker is in a different link (e.g., `link0`), use this option:

    ```bash
    python analysis_script.py data --marker_link link0
    ```

### Output

The script will generate two new folders:

* `Unpacked_data/`: Contains the processed and decoded data in `.pkl` format for faster re-loading.
* `Plots/`: Contains the output plots in `.pdf` format.

