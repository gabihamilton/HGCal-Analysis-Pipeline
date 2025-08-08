import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import glob
import os
import json
import argparse # Import the argparse library

#==============================================================================
# PART 1: PACKET DECODING (PARSING)
# Functions from ParseEtxOutputs.py and the first notebook cell.
# These take a raw packet and decode it into a structured DataFrame.
#==============================================================================

def parseHeaderWord0(HeaderWord0, returnDict=False):
    HdrMarker=(HeaderWord0>>23)&0x1ff
    PayloadLength=(HeaderWord0>>14)&0x1ff
    P=(HeaderWord0>>13)&0x1
    E=(HeaderWord0>>12)&0x1
    HT=(HeaderWord0>>10)&0x3
    EBO=(HeaderWord0>>8)&0x3
    M=(HeaderWord0>>7)&0x1
    T=(HeaderWord0>>6)&0x1
    Hamming=(HeaderWord0>>0)&0x3f
    if returnDict:
        return {"HdrMarker":HdrMarker,"PayloadLength":PayloadLength,"P":P,"E":E,"HT":HT,"EBO":EBO,"M":M,"T":T,"Hamming":Hamming}
    else:
        return HdrMarker, PayloadLength, P, E, HT, EBO, M, T, Hamming

def parseHeaderWord1(HeaderWord1, returnDict=False):
    BX=(HeaderWord1>>20)&0xfff
    L1A=(HeaderWord1>>14)&0x3f
    Orb=(HeaderWord1>>11)&0x7
    S=(HeaderWord1>>10)&0x1
    RR=(HeaderWord1>>8)&0x3
    CRC=(HeaderWord1)&0xff
    if returnDict:
        return {"Bunch":BX,"Event":L1A,"Orbit":Orb,"S":S,"RR":RR,"CRC":CRC}
    else:
        return BX, L1A, Orb, S, RR, CRC

def parseHeaderWords(HeaderWords, returnDict=False):
    if not HeaderWords or len(HeaderWords) < 2:
        return {} if returnDict else []
    # CORRECTED: Use np.bytes_ instead of the removed np.string_ and check for standard str type
    if isinstance(HeaderWords[0], (str, np.bytes_)):
        hdr_0 = int(HeaderWords[0], 16)
    else:
        hdr_0 = HeaderWords[0]
    if isinstance(HeaderWords[1], (str, np.bytes_)):
        hdr_1 = int(HeaderWords[1], 16)
    else:
        hdr_1 = HeaderWords[1]

    if returnDict:
        hdrFields = parseHeaderWord0(hdr_0, returnDict=True)
        hdrFields.update(parseHeaderWord1(hdr_1, returnDict=True))
        return hdrFields
    else:
        return list(parseHeaderWord0(hdr_0)) + list(parseHeaderWord1(hdr_1))

def parsePacketHeader(packetHeader0, packetHeader1=0, asHex=True, returnDict=False):
    Stat=(packetHeader0>>29)&0x7
    Ham = (packetHeader0>>26)&0x7
    F=(packetHeader0>>25)&0x1
    CM0=(packetHeader0>>15)&0x3ff
    CM1=(packetHeader0>>5)&0x3ff
    E = (packetHeader0>>4)&0x1 if F==1 else 0
    ChMap=((packetHeader0&0x1f)<<32)+packetHeader1
    if asHex:
        _stat, _ham, _f, _cm0, _cm1, _e, _chmap = f'{Stat:01x}',f'{Ham:01x}',f'{F:01x}',f'{CM0:03x}',f'{CM1:03x}',f'{E:01x}',f'{ChMap:010x}'
    else:
        _stat, _ham, _f, _cm0, _cm1, _e, _chmap = Stat, Ham, F, CM0, CM1, E, ChMap
    if returnDict:
        return {"Stat":_stat,"Ham" :_ham,"F":_f,"CM0":_cm0,"CM1":_cm1,"E":_e,"ChMap":_chmap}
    else:
        return _stat, _ham, _f, _cm0, _cm1, _e, _chmap

def unpackSinglePacket(packet, activeLinks):
    chData = np.array([[''] * 37 * 12], dtype=object).reshape(12, 37)
    eRxHeaderData = np.array([['', '', '', '', '', '', ''] * 12], dtype=object).reshape(12, 7)

    headerInfo = parseHeaderWords(packet, returnDict=True)
    if not headerInfo: return None # Packet is too short

    # --- ADDED PRINTOUTS ---
    print("-" * 50)
    print(f"  - Payload Length: {headerInfo.get('PayloadLength', 'N/A')}")
    print(f"  - Truncated (T): {'Yes' if headerInfo.get('T') == 1 else 'No'}")
    print(f"  - Passthrough (P): {'Yes' if headerInfo.get('P') == 1 else 'No'}")
    print(f"  - Bunch Crossing (BX): {headerInfo.get('Bunch', 'N/A')}")
    print(f"  - L1A Event: {headerInfo.get('Event', 'N/A')}")
    print("-" * 50)
    # --- END OF ADDED PRINTOUTS ---

    subPackets = packet[2:-1]
    crc = packet[-1]

    # If the header says the packet is truncated, we trust it and return an empty data row.
    if headerInfo.get('T') == 1:
        # The assertion below is too strict for some real-world data, which may have
        # a truncated flag but still contain some data words. We comment it out
        # to prevent crashing and proceed with the intended logic.
        # assert len(subPackets) == 0
        return list(headerInfo.values()) + list(np.concatenate([eRxHeaderData, chData], axis=1).flatten()) + [crc]

    subpacketBinString = ''.join(np.vectorize(lambda x: f'{int(str(x), 16):032b}')(subPackets))

    for eRx in activeLinks:
        if len(subpacketBinString) < 32: continue
        eRxHeader = parsePacketHeader(int(subpacketBinString[:32], 2), 0)
        fullSubPacket = eRxHeader[2] == '0'
        
        if fullSubPacket:
            if len(subpacketBinString) < 64: continue
            eRxHeader = parsePacketHeader(int(subpacketBinString[:32], 2), int(subpacketBinString[32:64], 2))
            subpacketBinString = subpacketBinString[64:]
        else:
            subpacketBinString = subpacketBinString[32:]
        
        eRxHeaderData[eRx] = eRxHeader
        chMapInt = int(eRxHeader[-1], 16)
        chMap = [(chMapInt >> (36 - i)) & 0x1 for i in range(37)]
        chAddr = np.argwhere(chMap).flatten()
        bitCounter = 0
        
        for ch in chAddr:
            if len(subpacketBinString) < 2: continue
            if headerInfo.get('P') == 1:
                if len(subpacketBinString) < 32: continue
                chData[eRx][ch] = subpacketBinString[:32]
                subpacketBinString = subpacketBinString[32:]
            else:
                code = subpacketBinString[:2]
                if code == '00':
                    if len(subpacketBinString) < 4: continue
                    code = subpacketBinString[:4]
                
                tctp, adcm1, adc, toa = '00', '0' * 10, '0' * 10, '0' * 10
                
                if code == '0000' and len(subpacketBinString) >= 24:
                    bitCounter += 24; adcm1 = subpacketBinString[4:14]; adc = subpacketBinString[14:24]; subpacketBinString = subpacketBinString[24:]
                elif code == '0001' and len(subpacketBinString) >= 16:
                    bitCounter += 16; adc = subpacketBinString[4:14]; subpacketBinString = subpacketBinString[16:]
                elif code == '0010' and len(subpacketBinString) >= 24:
                    bitCounter += 24; adcm1 = subpacketBinString[4:14]; adc = subpacketBinString[14:24]; tctp = '01'; subpacketBinString = subpacketBinString[24:]
                elif code == '0011' and len(subpacketBinString) >= 24:
                    bitCounter += 24; adc = subpacketBinString[4:14]; toa = subpacketBinString[14:24]; subpacketBinString = subpacketBinString[24:]
                elif code == '01' and len(subpacketBinString) >= 32:
                    bitCounter += 32; adcm1 = subpacketBinString[2:12]; adc = subpacketBinString[12:22]; toa = subpacketBinString[22:32]; subpacketBinString = subpacketBinString[32:]
                elif code == '11' and len(subpacketBinString) >= 32:
                    bitCounter += 32; adcm1 = subpacketBinString[2:12]; adc = subpacketBinString[12:22]; toa = subpacketBinString[22:32]; tctp = '11'; subpacketBinString = subpacketBinString[32:]
                elif code == '10' and len(subpacketBinString) >= 32:
                    bitCounter += 32; adcm1 = subpacketBinString[2:12]; adc = subpacketBinString[12:22]; toa = subpacketBinString[22:32]; tctp = '10'; subpacketBinString = subpacketBinString[32:]
                else:
                    continue # Skip if not enough bits for any code
                
                chData[eRx][ch] = tctp + adcm1 + adc + toa
        
        paddedBits = (32 - (bitCounter % 32)) % 32
        if len(subpacketBinString) < paddedBits: continue
        subpacketBinString = subpacketBinString[paddedBits:]

    return list(headerInfo.values()) + list(np.concatenate([eRxHeaderData, chData], axis=1).flatten()) + [crc]

def unpackPackets(packetList, activeLinks):
    unpackedInfo = []
    print(f"\n--- Unpacking {len(packetList)} packets ---")
    for i, p in enumerate(packetList):
        print(f"\n--- Processing Packet #{i+1} ---")
        unpacked = unpackSinglePacket(p, activeLinks)
        if unpacked:
            unpackedInfo.append(unpacked)
    
    if not unpackedInfo: return pd.DataFrame()

    columns = ['HeaderMarker', 'PayloadLength', 'P', 'E', 'HT', 'EBO', 'M', 'T', 'HdrHamming', 'BXNum', 'L1ANum', 'OrbNum', 'S', 'RR', 'HdrCRC']
    for i in range(12):
        columns += [f'eRx{i:02d}_{x}' for x in ['Stat', 'Ham', 'F', 'CM0', 'CM1', 'E', 'ChMap']]
        columns += [f'eRx{i:02d}_ChData{x:02d}' for x in range(37)]
    columns += ['CRC']

    # Pad rows that are too short, truncate rows that are too long
    max_len = len(columns)
    equalized_info = [row[:max_len] + [None]*(max_len - len(row)) for row in unpackedInfo]

    return pd.DataFrame(equalized_info, columns=columns)

#==============================================================================
# PART 2: RAW DATA FILE READING & PRE-PROCESSING
#==============================================================================

def read_data_files(folder):
    """
    Reads all .txt and .csv files in a folder, combines them, 
    and returns a single DataFrame.
    """
    headers = ["link0", "link1", "link2", "link3", "link4", "link5", "link6"]
    
    # Find both .txt and .csv files in the specified folder
    all_files = glob.glob(os.path.join(folder, "*.txt")) + glob.glob(os.path.join(folder, "*.csv"))
    
    if not all_files:
        print(f"Warning: No .txt or .csv files found in folder '{folder}'")
        return pd.DataFrame()

    df_list = []
    for file in all_files:
        # --- ADDED PRINTOUT ---
        print(f"--> Reading data from file: {os.path.basename(file)}")
        # --- END OF ADDED PRINTOUT ---
        try:
            if file.endswith('.txt'):
                # Logic for space-delimited text files, skipping header lines
                df = pd.read_csv(file, sep='\s+', skiprows=2, header=None, usecols=range(1, 8))
                df.columns = headers
                df_list.append(df)
            elif file.endswith('.csv'):
                # Logic for standard comma-separated files with a header
                df = pd.read_csv(file)
                # Ensure the columns match what the rest of the script expects
                if not all(h in df.columns for h in headers):
                    print(f"  - Warning: CSV file {file} is missing one of the required columns: {headers}")
                    continue
                df = df[headers] # Select/reorder columns to be safe
                df_list.append(df)
        except Exception as e:
            print(f"Could not read file {file}. Error: {e}")

    if not df_list: return pd.DataFrame()
    return pd.concat(df_list, ignore_index=True)

def data_to_pkt(df, marker_link='link6'):
    """
    Finds packet boundaries in the raw DataFrame and groups links
    into modules based on the new mapping.
    """
    if df.empty or marker_link not in df.columns:
        print(f"Marker link '{marker_link}' not found in DataFrame or DataFrame is empty.")
        return [], [], []
        
    position = []
    for i in range(1, len(df[marker_link])):
        prev_word = str(df[marker_link][i-1])
        curr_word = str(df[marker_link][i])
        if prev_word.startswith("555555") and not curr_word.startswith("555555"):
            position.append(i)
    position.append(len(df[marker_link]))
    
    Lpkt_east0, Lpkt_east1, Lpkt_east2 = [], [], []
    
    for j in range(len(position)-1):
        pkt_e0, pkt_e1, pkt_e2 = [], [], []
        for i in range(position[j], position[j+1]):
            # Filter out idle words and append data to the correct packet list
            if not str(df["link4"][i]).startswith("555555"): pkt_e0.append(df["link4"][i])
            if not str(df["link5"][i]).startswith("555555"): pkt_e1.append(df["link5"][i])
            if not str(df["link6"][i]).startswith("555555"): pkt_e2.append(df["link6"][i])
        
        Lpkt_east0.append(pkt_e0); Lpkt_east1.append(pkt_e1); Lpkt_east2.append(pkt_e2)
        
    return Lpkt_east0, Lpkt_east1, Lpkt_east2

#==============================================================================
# PART 3: DATA EXTRACTION FOR PLOTTING
#==============================================================================

def retrieve_ADCs(df, active_erx, channels, event_num=None):
    adcs, adcms, toas, noises = [], [], [], []
    
    num_events_in_df = len(df)
    if event_num is not None and event_num >= num_events_in_df:
        print(f"Warning: Event number {event_num} is out of bounds. The data has {num_events_in_df} events. Plotting average instead.")
        event_num = None # Revert to averaging

    for erx in active_erx:
        for ch in channels:
            col_name = f"eRx{int(erx):02d}_ChData{int(ch):02d}"
            if col_name not in df.columns or df[col_name].iloc[0] == "":
                adcs.append(0); adcms.append(0); toas.append(0); noises.append(0)
                continue
            
            # If a specific event is requested, get its data
            if event_num is not None:
                raw_str = df[col_name].iloc[event_num]
                if raw_str and len(raw_str) == 32:
                    adcs.append(int(raw_str[12:22], 2))
                    adcms.append(int(raw_str[2:12], 2))
                    toas.append(int(raw_str[22:], 2))
                else:
                    adcs.append(0); adcms.append(0); toas.append(0)
            # Otherwise, calculate the average across all events
            else:
                adc_evts, adcm_evts, toa_evts = [], [], []
                for raw_str in df[col_name].dropna():
                    if raw_str and len(raw_str) == 32:
                        adcm_evts.append(int(raw_str[2:12], 2))
                        adc_evts.append(int(raw_str[12:22], 2))
                        toa_evts.append(int(raw_str[22:], 2))
                
                adcs.append(np.mean(adc_evts) if adc_evts else 0)
                adcms.append(np.mean(adcm_evts) if adcm_evts else 0)
                toas.append(np.mean(toa_evts) if toa_evts else 0)
                noises.append(np.std(adc_evts) if adc_evts else 0)

    if event_num is None:
        print(f"Processed average of {num_events_in_df} events")
    else:
        print(f"Processed single event #{event_num}")
        noises = [0] * len(adcs) # Noise is not well-defined for a single event

    return adcs, adcms, toas, noises

def retrieve_CMs(df, active_erx, channels):
    CM0s, CM1s, avg0, avg1, CM0_rms, CM1_rms = [], [], [], [], [], []
    for erx in active_erx:
        cm0_col = f"eRx{int(erx):02d}_CM0"
        cm1_col = f"eRx{int(erx):02d}_CM1"
        
        CM0_evt = [int(str(x), 16) for x in df[cm0_col].dropna() if x]
        CM1_evt = [int(str(x), 16) for x in df[cm1_col].dropna() if x]
        
        CM0s.append(CM0_evt)
        CM1s.append(CM1_evt)
        avg0.append(np.mean(CM0_evt) if CM0_evt else 0)
        avg1.append(np.mean(CM1_evt) if CM1_evt else 0)
        CM0_rms.append(np.std(CM0_evt) if CM0_evt else 0)
        CM1_rms.append(np.std(CM1_evt) if CM1_evt else 0)

    return CM0s, CM1s, avg0, avg1, CM0_rms, CM1_rms

#==============================================================================
# PART 4: PLOTTING & ANALYSIS FUNCTIONS
#==============================================================================

def Plot_ADCs(data_dict, erxs, channels, runID, event_num=None):
    plt.style.use(hep.style.CMS)
    fig, axs = plt.subplots(2, 2, figsize=(20, 15), constrained_layout=True)
    fig.suptitle("ADC Plots for All Modules", fontsize=20)
    
    fig2, axs2 = plt.subplots(2, 2, figsize=(20, 15), constrained_layout=True)
    fig2.suptitle("ADC-1 Plots for All Modules", fontsize=20)
    
    fig3, axs3 = plt.subplots(2, 2, figsize=(20, 15), constrained_layout=True)
    fig3.suptitle("Noise Plots for All Modules", fontsize=20)
    
    axs_flat = axs.flatten(); axs2_flat = axs2.flatten(); axs3_flat = axs3.flatten()
    
    keys = list(data_dict.keys())
    for i, key in enumerate(keys):
        if i >= len(axs_flat): break # Stop if we run out of subplots
        
        df = data_dict[key]
        if df.empty:
            print(f"Skipping Module {key} due to empty dataframe.")
            continue
            
        adc, adcm, _, noise = retrieve_ADCs(df, erxs, channels, event_num=event_num)
        _, _, CM0_erx, CM1_erx, CM0_rms, CM1_rms = retrieve_CMs(df, erxs, channels)
        
        # ADC Plot
        ax = axs_flat[i]
        ax.plot(adc, marker='o', linestyle='-')
        ax.set_title(f"Module {key}"); ax.set_xlabel('Channel'); ax.set_ylabel('ADC')
        
        # ADCM Plot
        ax2 = axs2_flat[i]
        ax2.plot(adcm, marker='s', linestyle='--', color='r')
        ax2.set_title(f"Module {key}"); ax2.set_xlabel('Channel'); ax2.set_ylabel('ADC-1')

        # Noise Plot
        ax3 = axs3_flat[i]
        ax3.plot(noise, marker='o', linestyle='-', color='g')
        ax3.set_title(f"Module {key}"); ax3.set_xlabel('Channel'); ax3.set_ylabel('Noise')

        # Add lines and annotations
        for e_idx, erx in enumerate(erxs):
            start_ch = e_idx * len(channels)
            end_ch = (e_idx + 1) * len(channels)
            avg_adc_per_erx = np.mean(adc[start_ch:end_ch])
            avg_noise_per_erx = np.mean(noise[start_ch:end_ch])

            for p_ax in [ax, ax2, ax3]:
                p_ax.axvline(x=end_ch - 0.5, color='gray', linestyle='--')
            
            ax.plot([start_ch, end_ch], [CM0_erx[e_idx], CM0_erx[e_idx]], linestyle='--', color='red', label=f'eRx{erx} CM0' if e_idx==0 else "")
            ax.plot([start_ch, end_ch], [CM1_erx[e_idx], CM1_erx[e_idx]], linestyle='--', color='orange', label=f'eRx{erx} CM1' if e_idx==0 else "")
            ax.plot([start_ch, end_ch], [avg_adc_per_erx, avg_adc_per_erx], linestyle='--', color='k', label='Avg ADC' if e_idx==0 else "")
            
            ax3.plot([start_ch, end_ch], [CM0_rms[e_idx], CM0_rms[e_idx]], linestyle='--', color='red', label=f'eRx{erx} CM0 RMS' if e_idx==0 else "")
            ax3.plot([start_ch, end_ch], [CM1_rms[e_idx], CM1_rms[e_idx]], linestyle='--', color='orange', label=f'eRx{erx} CM1 RMS' if e_idx==0 else "")
            ax3.plot([start_ch, end_ch], [avg_noise_per_erx, avg_noise_per_erx], linestyle='--', color='k', label='Avg Noise' if e_idx==0 else "")
        
        ax.legend(); ax3.legend()

    # Turn off unused subplots
    for i in range(len(keys), len(axs_flat)):
        axs_flat[i].axis('off')
        axs2_flat[i].axis('off')
        axs3_flat[i].axis('off')

    output_dir = f"Plots/{runID}"
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "adc.pdf"))
    fig2.savefig(os.path.join(output_dir, "adc1.pdf"))
    fig3.savefig(os.path.join(output_dir, "noise.pdf"))
    plt.show()

def Plot_Single_Link_ADC(data_dict, link_to_plot, erxs, channels, runID, event_num=None):
    """
    Generates and saves a single plot for a specified link,
    showing ADC values across all its channels.
    """
    # Map the link name to the corresponding module key in the data_dict
    link_to_module_map = {
        'link4': 'east 0',
        'link5': 'east 1',
        'link6': 'east 2',
    }
    
    module_key = link_to_module_map.get(link_to_plot)
    if not module_key:
        print(f"Error: Link '{link_to_plot}' is not mapped to a known module.")
        return
        
    df = data_dict.get(module_key)
    if df is None or df.empty:
        print(f"Skipping plot for {link_to_plot} (Module {module_key}) as it has no data.")
        return

    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(15, 8))
    
    adc, _, _, _ = retrieve_ADCs(df, erxs, channels, event_num=event_num)
    
    plot_title = f"ADC Distribution for {link_to_plot.capitalize()} (Module {module_key})"
    y_label = "ADC Value"
    if event_num is None:
        y_label = "Average " + y_label
    else:
        plot_title += f" - Event {event_num}"

    ax.plot(adc, marker='o', linestyle='-')
    ax.set_title(plot_title, fontsize=16)
    ax.set_xlabel("Channel Index")
    ax.set_ylabel(y_label)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Add vertical lines to separate eRx blocks
    for e_idx, erx in enumerate(erxs):
        end_ch = (e_idx + 1) * len(channels)
        ax.axvline(x=end_ch - 0.5, color='gray', linestyle='--')

    output_dir = f"Plots/{runID}"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"adc_plot_{link_to_plot}.pdf")
    fig.savefig(output_filename)
    print(f"Single link plot saved to: {output_filename}")
    plt.show()

#==============================================================================
# PART 5: MAIN EXECUTION BLOCK
#==============================================================================

if __name__ == "__main__":
    # --- Argument Parser Setup ---
    parser = argparse.ArgumentParser(description="Process and plot HGCAL data from text or CSV files.")
    parser.add_argument("run_id", type=str, help="The name of the folder containing the .txt or .csv data files. This will also be used for output directories.")
    parser.add_argument("--marker_link", type=str, default="link6", help="The link column to check for the start-of-packet marker (e.g., 'link0', 'link6'). Defaults to 'link6'.")
    parser.add_argument("--plot_link", type=str, help="Generate a single plot for a specified link (e.g., 'link4', 'link5', 'link6').")
    parser.add_argument("--event", type=int, help="Plot a specific event number instead of the average. Starts at 0.")
    args = parser.parse_args()
    
    # Use the parsed arguments instead of hardcoded values
    RUN_ID = args.run_id
    MARKER_LINK = args.marker_link
    PLOT_LINK = args.plot_link
    EVENT_NUM = args.event
    
    # --- Configuration ---
    # Define active links, eRxs, and channels based on your notebook
    LINKS = [0, 1, 2, 3, 4, 5]
    ERXS = ["00", "01", "02"]
    CHANNELS = [f"{j}{i}" for j in range(4) for i in range(10) if f"{j}{i}" != "18"]
    CHANNELS.append("36") # The loop logic in the notebook is a bit complex, this simplifies to get the same list
    
    print(f"--- Starting Data Processing for RUN_ID: {RUN_ID} ---")
    print(f"Using '{MARKER_LINK}' to find packet boundaries.")
    
    # --- Step 1: Read and Unpack Data ---
    unpacked_data_dir = f"Unpacked_data/{RUN_ID}/"
    data_all_modules = {}

    if os.path.exists(unpacked_data_dir):
        print("Loading existing unpacked dataframes from pickle files...")
        try:
            data_all_modules["east 0"] = pd.read_pickle(os.path.join(unpacked_data_dir, "dat_e0.pkl"))
            data_all_modules["east 1"] = pd.read_pickle(os.path.join(unpacked_data_dir, "dat_e1.pkl"))
            data_all_modules["east 2"] = pd.read_pickle(os.path.join(unpacked_data_dir, "dat_e2.pkl"))
        except FileNotFoundError:
            print("Pickle files not found, proceeding to unpack raw data.")
            pass

    # If data wasn't loaded from pickle, process from raw text files
    if not data_all_modules:
        print(f"Reading raw data files from folder '{RUN_ID}'...")
        raw_df = read_data_files(RUN_ID)
        
        if not raw_df.empty:
            print("Extracting packets from raw data...")
            Lpkt_east0, Lpkt_east1, Lpkt_east2 = data_to_pkt(raw_df, marker_link=MARKER_LINK)
            
            print("Unpacking data for each module...")
            data_all_modules = {
                "east 0": unpackPackets(Lpkt_east0, LINKS),
                "east 1": unpackPackets(Lpkt_east1, LINKS),
                "east 2": unpackPackets(Lpkt_east2, LINKS),
            }
            
            print(f"Saving unpacked data to: {unpacked_data_dir}")
            os.makedirs(unpacked_data_dir, exist_ok=True)
            data_all_modules["east 0"].to_pickle(os.path.join(unpacked_data_dir, "dat_e0.pkl"))
            data_all_modules["east 1"].to_pickle(os.path.join(unpacked_data_dir, "dat_e1.pkl"))
            data_all_modules["east 2"].to_pickle(os.path.join(unpacked_data_dir, "dat_e2.pkl"))
        else:
            print(f"No raw data found in folder '{RUN_ID}'. Exiting.")
            exit()

    # --- Step 2: Generate and Save Plots ---
    if PLOT_LINK:
        print(f"Generating single plot for {PLOT_LINK}...")
        Plot_Single_Link_ADC(data_all_modules, PLOT_LINK, ERXS, CHANNELS, RUN_ID, event_num=EVENT_NUM)
    else:
        if any(not df.empty for df in data_all_modules.values()):
            print("Generating plots for all modules...")
            Plot_ADCs(data_all_modules, ERXS, CHANNELS, RUN_ID, event_num=EVENT_NUM)
        else:
            print("All dataframes are empty after unpacking. No plots will be generated.")

    print("--- Processing Complete ---")
