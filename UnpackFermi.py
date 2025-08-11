import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os
import argparse

#==============================================================================
# PART 1: ECON-D PACKET DECODING (FROM PREVIOUS SCRIPT)
# This section contains the functions to decode a standard 32-bit ECON-D packet.
# This logic remains the same.
#==============================================================================

def parseHeaderWord0(HeaderWord0, returnDict=False):
    HdrMarker=(HeaderWord0>>23)&0x1ff; PayloadLength=(HeaderWord0>>14)&0x1ff; P=(HeaderWord0>>13)&0x1; E=(HeaderWord0>>12)&0x1; HT=(HeaderWord0>>10)&0x3; EBO=(HeaderWord0>>8)&0x3; M=(HeaderWord0>>7)&0x1; T=(HeaderWord0>>6)&0x1; Hamming=(HeaderWord0>>0)&0x3f
    if returnDict: return {"HdrMarker":HdrMarker,"PayloadLength":PayloadLength,"P":P,"E":E,"HT":HT,"EBO":EBO,"M":M,"T":T,"Hamming":Hamming}
    else: return HdrMarker, PayloadLength, P, E, HT, EBO, M, T, Hamming

def parseHeaderWord1(HeaderWord1, returnDict=False):
    BX=(HeaderWord1>>20)&0xfff; L1A=(HeaderWord1>>14)&0x3f; Orb=(HeaderWord1>>11)&0x7; S=(HeaderWord1>>10)&0x1; RR=(HeaderWord1>>8)&0x3; CRC=(HeaderWord1)&0xff
    if returnDict: return {"Bunch":BX,"Event":L1A,"Orbit":Orb,"S":S,"RR":RR,"CRC":CRC}
    else: return BX, L1A, Orb, S, RR, CRC

def parseHeaderWords(HeaderWords, returnDict=False):
    if not HeaderWords or len(HeaderWords) < 2: return {} if returnDict else []
    if isinstance(HeaderWords[0], (str, np.bytes_)): hdr_0 = int(HeaderWords[0], 16)
    else: hdr_0 = HeaderWords[0]
    if isinstance(HeaderWords[1], (str, np.bytes_)): hdr_1 = int(HeaderWords[1], 16)
    else: hdr_1 = HeaderWords[1]
    if returnDict:
        hdrFields = parseHeaderWord0(hdr_0, returnDict=True); hdrFields.update(parseHeaderWord1(hdr_1, returnDict=True)); return hdrFields
    else: return list(parseHeaderWord0(hdr_0)) + list(parseHeaderWord1(hdr_1))

def parsePacketHeader(packetHeader0,packetHeader1=0,asHex=True,returnDict=False):
    Stat=(packetHeader0>>29)&0x7; Ham = (packetHeader0>>26)&0x7; F=(packetHeader0>>25)&0x1; CM0=(packetHeader0>>15)&0x3ff; CM1=(packetHeader0>>5)&0x3ff
    E = (packetHeader0>>4)&0x1 if F==1 else 0
    ChMap=((packetHeader0&0x1f)<<32)+packetHeader1
    if asHex: _stat, _ham, _f, _cm0, _cm1, _e, _chmap = f'{Stat:01x}',f'{Ham:01x}',f'{F:01x}',f'{CM0:03x}',f'{CM1:03x}',f'{E:01x}',f'{ChMap:010x}'
    else: _stat, _ham, _f, _cm0, _cm1, _e, _chmap = Stat, Ham, F, CM0, CM1, E, ChMap
    if returnDict: return {"Stat":_stat,"Ham" :_ham,"F":_f,"CM0":_cm0,"CM1":_cm1,"E":_e,"ChMap":_chmap}
    else: return _stat, _ham, _f, _cm0, _cm1, _e, _chmap

def unpackSinglePacket(packet, activeLinks):
    chData = np.array([[''] * 37 * 12], dtype=object).reshape(12, 37)
    eRxHeaderData = np.array([['', '', '', '', '', '', ''] * 12], dtype=object).reshape(12, 7)
    headerInfo = parseHeaderWords(packet, returnDict=True)
    if not headerInfo: return None

    # --- ADDED PRINTOUTS ---
    print("-" * 50)
    print(f"  - Payload Length: {headerInfo.get('PayloadLength', 'N/A')}")
    print(f"  - Truncated (T): {'Yes' if headerInfo.get('T') == 1 else 'No'}")
    print(f"  - Passthrough (P): {'Yes' if headerInfo.get('P') == 1 else 'No'}")
    print(f"  - Bunch Crossing (BX): {headerInfo.get('Bunch', 'N/A')}")
    print(f"  - L1A Event: {headerInfo.get('Event', 'N/A')}")
    print("-" * 50)
    # --- END OF ADDED PRINTOUTS ---

    subPackets = packet[2:-1]; crc = packet[-1]
    if headerInfo.get('T') == 1: return list(headerInfo.values()) + list(np.concatenate([eRxHeaderData, chData], axis=1).flatten()) + [crc]
    subpacketBinString = ''.join(np.vectorize(lambda x: f'{int(str(x), 16):032b}')(subPackets))
    for eRx in activeLinks:
        if len(subpacketBinString) < 32: continue
        eRxHeader = parsePacketHeader(int(subpacketBinString[:32], 2), 0)
        fullSubPacket = eRxHeader[2] == '0'
        if fullSubPacket:
            if len(subpacketBinString) < 64: continue
            eRxHeader = parsePacketHeader(int(subpacketBinString[:32], 2), int(subpacketBinString[32:64], 2))
            subpacketBinString = subpacketBinString[64:]
        else: subpacketBinString = subpacketBinString[32:]
        eRxHeaderData[eRx] = eRxHeader
        chMapInt = int(eRxHeader[-1], 16); chMap = [(chMapInt >> (36 - i)) & 0x1 for i in range(37)]; chAddr = np.argwhere(chMap).flatten(); bitCounter = 0
        for ch in chAddr:
            if len(subpacketBinString) < 2: continue
            if headerInfo.get('P') == 1:
                if len(subpacketBinString) < 32: continue
                chData[eRx][ch] = subpacketBinString[:32]; subpacketBinString = subpacketBinString[32:]
            else:
                code = subpacketBinString[:2]
                if code == '00':
                    if len(subpacketBinString) < 4: continue
                    code = subpacketBinString[:4]
                tctp, adcm1, adc, toa = '00', '0' * 10, '0' * 10, '0' * 10
                if code == '01' and len(subpacketBinString) >= 32:
                    bitCounter += 32; adcm1 = subpacketBinString[2:12]; adc = subpacketBinString[12:22]; toa = subpacketBinString[22:32]; subpacketBinString = subpacketBinString[32:]
                else: continue
                chData[eRx][ch] = tctp + adcm1 + adc + toa
        paddedBits = (32 - (bitCounter % 32)) % 32
        if len(subpacketBinString) < paddedBits: continue
        subpacketBinString = subpacketBinString[paddedBits:]
    return list(headerInfo.values()) + list(np.concatenate([eRxHeaderData, chData], axis=1).flatten()) + [crc]

def unpackPackets(packetList, activeLinks):
    unpackedInfo = [unpackSinglePacket(p, activeLinks) for p in packetList if p]
    if not unpackedInfo: return pd.DataFrame()
    columns = ['HeaderMarker', 'PayloadLength', 'P', 'E', 'HT', 'EBO', 'M', 'T', 'HdrHamming', 'BXNum', 'L1ANum', 'OrbNum', 'S', 'RR', 'HdrCRC']
    for i in range(12):
        columns += [f'eRx{i:02d}_{x}' for x in ['Stat', 'Ham', 'F', 'CM0', 'CM1', 'E', 'ChMap']]
        columns += [f'eRx{i:02d}_ChData{x:02d}' for x in range(37)]
    columns += ['CRC']
    max_len = len(columns)
    equalized_info = [row[:max_len] + [None]*(max_len - len(row)) for row in unpackedInfo]
    return pd.DataFrame(equalized_info, columns=columns)

#==============================================================================
# PART 2: NEW RAW DATA PARSING
# This function reads the new file format and extracts a single capture block.
#==============================================================================

def read_and_extract_capture_block(filepath):
    """
    Reads the new 64-bit data format, splits words, and extracts the first
    full ECON-D packet from the first capture block it finds.
    """
    print(f"Reading and parsing file: {filepath}")
    all_32bit_words = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split(': 0x')
                if len(parts) != 2: continue
                
                word64_str = parts[1]
                if len(word64_str) != 16: continue
                
                # Split the 64-bit word into two 32-bit words
                word1_32 = word64_str[0:8]
                word2_32 = word64_str[8:16]
                all_32bit_words.extend([word1_32, word2_32])
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

    # Find the start of the first capture block (marker 0xfe...)
    capture_block_start_idx = -1
    for i, word in enumerate(all_32bit_words):
        if word.startswith('fe'):
            capture_block_start_idx = i
            break
            
    if capture_block_start_idx == -1:
        print("Error: Could not find a capture block header (starting with 0xfe).")
        return None
        
    print(f"Found capture block header at 32-bit word index {capture_block_start_idx}.")

    # Find the start of the ECON-D packet within that capture block (marker 0xaa...)
    econ_packet_start_idx = -1
    for i in range(capture_block_start_idx, len(all_32bit_words)):
        if all_32bit_words[i].startswith('aa'):
            econ_packet_start_idx = i
            break

    if econ_packet_start_idx == -1:
        print("Error: Could not find an ECON-D packet header (starting with 0xaa) within the capture block.")
        return None
        
    print(f"Found ECON-D packet header at 32-bit word index {econ_packet_start_idx}.")

    # --- CORRECTED LOGIC ---
    # Instead of guessing the end, parse the header to get the real payload length.
    if len(all_32bit_words) < econ_packet_start_idx + 2:
        print("Error: Not enough data to read packet header.")
        return None
        
    header_words = all_32bit_words[econ_packet_start_idx : econ_packet_start_idx + 2]
    header_info = parseHeaderWords(header_words, returnDict=True)
    payload_length = header_info.get('PayloadLength')

    if payload_length is None:
        print("Error: Could not determine payload length from packet header.")
        return None

    # The full packet length is 2 (header) + payload + 1 (CRC)
    full_packet_length = 2 + payload_length + 1
    econ_packet_end_idx = econ_packet_start_idx + full_packet_length
    
    if len(all_32bit_words) < econ_packet_end_idx:
        print(f"Warning: File ends before full packet length of {full_packet_length} words is reached.")
        econ_packet_end_idx = len(all_32bit_words)

    # The final packet is the slice of 32-bit words
    econ_packet = all_32bit_words[econ_packet_start_idx:econ_packet_end_idx]
    print(f"Extracted ECON-D packet with {len(econ_packet)} 32-bit words based on PayloadLength.")
    
    return econ_packet

#==============================================================================
# PART 3: DATA EXTRACTION FOR PLOTTING
#==============================================================================

def retrieve_ADCs(df, active_erx, channels, event_num=0):
    adcs, adcms, toas, noises = [], [], [], []
    
    if df.empty or event_num >= len(df):
        print(f"Warning: DataFrame is empty or event {event_num} is out of bounds.")
        return [], [], [], []

    print(f"Extracting data for event #{event_num}")
    for erx in active_erx:
        for ch in channels:
            col_name = f"eRx{int(erx):02d}_ChData{int(ch):02d}"
            if col_name not in df.columns or df.iloc[event_num][col_name] == "":
                adcs.append(0); adcms.append(0); toas.append(0)
                continue
            
            raw_str = df[col_name].iloc[event_num]
            if raw_str and len(raw_str) == 32:
                adcm_val = int(raw_str[2:12], 2)
                adc_val = int(raw_str[12:22], 2)
                toa_val = int(raw_str[22:], 2)
                adcs.append(adc_val); adcms.append(adcm_val); toas.append(toa_val)
            else:
                adcs.append(0); adcms.append(0); toas.append(0)
    
    # Noise is not well-defined for a single event, so we return zeros
    noises = [0] * len(adcs)
    return adcs, adcms, toas, noises

#==============================================================================
# PART 4: PLOTTING
#==============================================================================

def plot_single_capture_block(df, erxs, channels, runID, event_num=0):
    """
    Generates and saves a single plot for the first event in the DataFrame.
    """
    if df.empty:
        print("DataFrame is empty. No plot will be generated.")
        return

    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(15, 8))
    
    adc, _, _, _ = retrieve_ADCs(df, erxs, channels, event_num=event_num)
    
    plot_title = f"ADC Distribution for Run {runID} (Event {event_num})"
    y_label = "ADC Value"

    ax.plot(adc, marker='o', linestyle='-')
    ax.set_title(plot_title, fontsize=16)
    ax.set_xlabel("Channel Index")
    ax.set_ylabel(y_label)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Add vertical lines to separate eRx blocks
    num_channels_per_erx = 37 # Based on the unpacking logic
    for e_idx, erx in enumerate(erxs):
        end_ch = (e_idx + 1) * num_channels_per_erx
        ax.axvline(x=end_ch - 0.5, color='gray', linestyle='--')

    output_dir = f"Plots/{runID}"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"adc_plot_event_{event_num}.pdf")
    fig.savefig(output_filename)
    print(f"Plot saved to: {output_filename}")
    plt.show()

#==============================================================================
# PART 5: MAIN EXECUTION BLOCK
#==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a new data format and plot ADC distribution for one capture block.")
    parser.add_argument("filepath", type=str, help="The path to the raw data file (e.g., 'forGabi.txt').")
    args = parser.parse_args()
    
    RUN_ID = os.path.splitext(os.path.basename(args.filepath))[0]
    
    # --- Configuration ---
    # According to the unpacking logic, there are 12 possible eRx links (0-11)
    ACTIVE_LINKS = list(range(12))
    # We will plot data for the first 6 eRx's, similar to the previous script
    ERXS_TO_PLOT = ["00", "01", "02", "03", "04", "05"]
    CHANNELS_TO_PLOT = list(range(37))
    
    print(f"--- Starting Data Processing for Run: {RUN_ID} ---")
    
    # 1. Read the raw data file and extract the first ECON-D packet
    econ_packet = read_and_extract_capture_block(args.filepath)
    
    if econ_packet:
        # 2. Unpack the single packet into a DataFrame.
        #    We put it in a list because unpackPackets expects a list of packets.
        unpacked_df = unpackPackets([econ_packet], ACTIVE_LINKS)
        
        # 3. Generate and save the plot for the first event (event 0)
        if not unpacked_df.empty:
            plot_single_capture_block(unpacked_df, ERXS_TO_PLOT, CHANNELS_TO_PLOT, RUN_ID, event_num=0)
        else:
            print("Unpacking resulted in an empty DataFrame. Cannot generate plot.")
    else:
        print("Failed to extract a valid packet from the file.")

    print("--- Processing Complete ---")
