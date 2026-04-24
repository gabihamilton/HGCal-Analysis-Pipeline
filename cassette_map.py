# The complete physical positions of ALL sensors on the cassette.
sensor_positions = {
    # --- Module RTDs (from the hexagons) ---
    # LD1 Row (y=1)
    "LD1_w3": (1, 1), "LD1_w2": (2, 1), "LD1_w1": (3, 1), "LD1_e1": (4, 1), "LD1_e2": (5, 1), "LD1_e3": (6, 1),
    # LD2 Row (y=2)
    "LD2_w3": (1, 2), "LD2_w2": (2, 2), "LD2_w1": (3, 2), "LD2_e1": (4, 2), "LD2_e2": (5, 2), "LD2_e3": (6, 2), "LD2_e4": (7, 2),
    # LD3 Row (y=3)
    "LD3_w3": (1, 3), "LD3_w2": (2, 3), "LD3_w1": (3, 3), "LD3_e1": (4, 3), "LD3_e2": (5, 3), "LD3_e3": (6, 3), "LD3_e4": (7, 3),
    # LD4 Row (y=4)
    "LD4_w3": (2, 4), "LD4_w2": (3, 4), "LD4_w1": (4, 4), "LD4_e1": (5, 4), "LD4_e2": (6, 4),
    # LD5 Row (y=5)
    "LD5_w2": (1, 5), "LD5_w1": (2, 5), "LD5_e1": (3, 5), "LD5_e2": (4, 5),
    # Top-most e3 module
    "LD5_e3": (3, 6), # Assuming this is part of the LD5 train
    
    # --- HD1 & HD2 Modules (right side) ---
    # HD1 (bottom row)
    "HD1_m1": (7, 1), "HD1_m2": (8, 1), "HD1_m3": (9, 1),
    # HD2 (top rows)
    "HD2_m1": (8, 2), "HD2_m2": (9, 2),
    
    # --- lpGBT Internal Sensors (from the red dots) ---
    "LD1_lpgbt_internal": (3.5, 1),    # Located on module e1
    "LD2_lpgbt_internal": (3.5, 2),    # Located on module e1
    "LD3_lpgbt_internal": (3.5, 3),    # Located on module e1
    "LD4_lpgbt_internal": (3.5, 4),    # Located on module w1
    "LD5_lpgbt_internal": (2.5, 5),    # Located on module w1
    "HD1_lpgbt_internal": (3.5, 1),  # Located on module m1, between the rows
}