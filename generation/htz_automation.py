import pyautogui
import time
import pandas as pd
import numpy as np
from math import cos, radians, sqrt
import os
import glob

# Setup for safe operation
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.5

# ============================================================================
# COORDINATE CALCULATION FUNCTIONS - SUPER-ROBUST VERSION
# ============================================================================

def calculate_coordinate_offsets(lat, km_offset):
    """
    Calculate latitude and longitude offsets for a given distance in km.
    
    Args:
        lat: Latitude in degrees
        km_offset: Distance offset in kilometers
    
    Returns:
        tuple: (lat_offset, lon_offset) in degrees
    """
    # 1 degree of latitude ≈ 111 km everywhere
    lat_offset = km_offset / 111.0
    
    # 1 degree of longitude varies with latitude
    # At the equator: 1° ≈ 111 km
    # At latitude φ: 1° ≈ 111 * cos(φ) km
    lon_offset = km_offset / (111.0 * cos(radians(lat)))
    
    return lat_offset, lon_offset

def calculate_distance_km(lat1, lon1, lat2, lon2):
    """
    Calculate approximate distance in km between two points.
    Uses simplified formula for short distances.
    """
    center_lat = (lat1 + lat2) / 2
    lat_diff_km = (lat2 - lat1) * 111.0
    lon_diff_km = (lon2 - lon1) * 111.0 * cos(radians(center_lat))
    return sqrt(lat_diff_km**2 + lon_diff_km**2)

def calculate_bounding_rectangle(csv_file_path, target_size_km=1.0, safety_margin_km=0.05):
    """
    SUPER-ROBUST version that calculates bounding rectangle coordinates.
    
    This version:
    1. Creates a square of EXACTLY target_size_km × target_size_km
    2. Adds a small safety margin to ensure all points are comfortably inside
    3. Performs extensive validation
    4. Provides detailed diagnostics
    
    Args:
        csv_file_path: Path to the CSV file
        target_size_km: Target size for the square in kilometers (default: 1.0)
        safety_margin_km: Additional safety margin in km (default: 0.05 = 50 meters)
    
    Returns:
        dict: Dictionary containing min/max longitude and latitude
    """
    print(f"\n📊 Reading CSV file: {csv_file_path}")
    
    # Check if file exists
    if not os.path.exists(csv_file_path):
        print(f"❌ Error: File not found: {csv_file_path}")
        return None
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Check if required columns exist
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        print("❌ Error: CSV file must contain 'latitude' and 'longitude' columns")
        return None
    
    # Get unique locations
    unique_locations = df.groupby('location')[['latitude', 'longitude']].first().reset_index()
    
    print(f"✓ Found {len(unique_locations)} unique location(s):")
    for idx, row in unique_locations.iterrows():
        print(f"  • {row['location']}: ({row['latitude']:.6f}, {row['longitude']:.6f})")
    
    # Calculate min and max coordinates
    min_lat = unique_locations['latitude'].min()
    max_lat = unique_locations['latitude'].max()
    min_lon = unique_locations['longitude'].min()
    max_lon = unique_locations['longitude'].max()
    
    # Calculate center point for accurate offset calculation
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    
    # Calculate natural span
    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon
    
    # Calculate current size in km
    current_height_km = lat_span * 111.0
    current_width_km = lon_span * 111.0 * cos(radians(center_lat))
    
    print(f"\n📍 Original bounding box (NO margins):")
    print(f"  Latitude:  {min_lat:.6f} to {max_lat:.6f} (span: {current_height_km:.4f} km)")
    print(f"  Longitude: {min_lon:.6f} to {max_lon:.6f} (span: {current_width_km:.4f} km)")
    
    # Calculate maximum distance between any two points
    max_distance = 0
    furthest_pair = None
    for i in range(len(unique_locations)):
        for j in range(i+1, len(unique_locations)):
            loc1 = unique_locations.iloc[i]
            loc2 = unique_locations.iloc[j]
            dist = calculate_distance_km(loc1['latitude'], loc1['longitude'],
                                        loc2['latitude'], loc2['longitude'])
            if dist > max_distance:
                max_distance = dist
                furthest_pair = (loc1['location'], loc2['location'])
    
    print(f"\n📏 Maximum distance between any two locations: {max_distance:.4f} km")
    if furthest_pair:
        print(f"   Between: {furthest_pair[0]} ↔ {furthest_pair[1]}")
    
    # Add safety margin to target size to ensure comfortable fit
    effective_target_km = target_size_km + (2 * safety_margin_km)
    
    print(f"\n🎯 Target: {target_size_km:.3f} km × {target_size_km:.3f} km SQUARE")
    print(f"   + Safety margin: {safety_margin_km:.3f} km on each side")
    print(f"   = Effective target: {effective_target_km:.3f} km × {effective_target_km:.3f} km")
    
    # Verify that target size can accommodate all points
    if max_distance > effective_target_km * 0.85:  # Use 85% as threshold (diagonal)
        print(f"\n⚠️  WARNING: Max distance ({max_distance:.4f} km) is close to target size!")
        print(f"   Increasing effective target size to ensure fit...")
        effective_target_km = max_distance * 1.2  # 20% extra for safety
        print(f"   New effective target: {effective_target_km:.3f} km × {effective_target_km:.3f} km")
    
    # Calculate how much extra space we need
    extra_height_needed = max(0, effective_target_km - current_height_km)
    extra_width_needed = max(0, effective_target_km - current_width_km)
    
    print(f"\n🔧 Extra padding needed to reach {effective_target_km:.3f} km:")
    print(f"  Height: {extra_height_needed:.4f} km (add {extra_height_needed/2:.4f} km per side)")
    print(f"  Width:  {extra_width_needed:.4f} km (add {extra_width_needed/2:.4f} km per side)")
    
    # Convert to degrees and split evenly on both sides
    lat_padding_per_side = (extra_height_needed / 2) / 111.0
    lon_padding_per_side = (extra_width_needed / 2) / (111.0 * cos(radians(center_lat)))
    
    print(f"\n📐 Padding per side (in degrees):")
    print(f"  Latitude:  ±{lat_padding_per_side:.6f}°")
    print(f"  Longitude: ±{lon_padding_per_side:.6f}°")
    
    # Apply padding to create square
    min_lat_with_margin = min_lat - lat_padding_per_side
    max_lat_with_margin = max_lat + lat_padding_per_side
    min_lon_with_margin = min_lon - lon_padding_per_side
    max_lon_with_margin = max_lon + lon_padding_per_side
    
    # Verify the final size
    final_lat_diff = max_lat_with_margin - min_lat_with_margin
    final_lon_diff = max_lon_with_margin - min_lon_with_margin
    
    final_height_km = final_lat_diff * 111.0
    final_width_km = final_lon_diff * 111.0 * cos(radians(center_lat))
    
    print(f"\n✓ Final bounding rectangle:")
    print(f"  Min Longitude (left):   {min_lon_with_margin:.6f}")
    print(f"  Max Latitude (top):     {max_lat_with_margin:.6f}")
    print(f"  Max Longitude (right):  {max_lon_with_margin:.6f}")
    print(f"  Min Latitude (bottom):  {min_lat_with_margin:.6f}")
    
    print(f"\n📐 Verified rectangle size: {final_width_km:.4f} km × {final_height_km:.4f} km")
    
    # COMPREHENSIVE VALIDATION - Check all points are inside with margin
    print(f"\n🔍 VALIDATION: Checking all locations are inside...")
    all_inside = True
    min_margin_km = float('inf')
    worst_location = None
    
    for idx, row in unique_locations.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        
        # Check if inside
        lat_check = min_lat_with_margin <= lat <= max_lat_with_margin
        lon_check = min_lon_with_margin <= lon <= max_lon_with_margin
        
        # Calculate margins from edges
        margin_north = (max_lat_with_margin - lat) * 111.0
        margin_south = (lat - min_lat_with_margin) * 111.0
        margin_east = (max_lon_with_margin - lon) * 111.0 * cos(radians(center_lat))
        margin_west = (lon - min_lon_with_margin) * 111.0 * cos(radians(center_lat))
        
        min_margin_this_point = min(margin_north, margin_south, margin_east, margin_west)
        
        if min_margin_this_point < min_margin_km:
            min_margin_km = min_margin_this_point
            worst_location = row['location']
        
        if lat_check and lon_check:
            print(f"  ✓ {row['location']}")
            print(f"     Margins: N:{margin_north:.4f}km S:{margin_south:.4f}km E:{margin_east:.4f}km W:{margin_west:.4f}km")
        else:
            all_inside = False
            print(f"  ✗ {row['location']} is OUTSIDE bounds!")
            if not lat_check:
                print(f"     ❌ Latitude issue: {lat:.6f} not in [{min_lat_with_margin:.6f}, {max_lat_with_margin:.6f}]")
                print(f"        Over by: {min(abs(lat - min_lat_with_margin), abs(lat - max_lat_with_margin)) * 111.0:.4f} km")
            if not lon_check:
                print(f"     ❌ Longitude issue: {lon:.6f} not in [{min_lon_with_margin:.6f}, {max_lon_with_margin:.6f}]")
                print(f"        Over by: {min(abs(lon - min_lon_with_margin), abs(lon - max_lon_with_margin)) * 111.0 * cos(radians(center_lat)):.4f} km")
    
    if all_inside:
        print(f"\n  ✅ All {len(unique_locations)} location(s) verified INSIDE bounds!")
        print(f"  📏 Minimum margin from edge: {min_margin_km:.4f} km (at {worst_location})")
        if min_margin_km < 0.02:  # Less than 20 meters
            print(f"  ⚠️  WARNING: Very tight fit! Consider increasing safety_margin_km")
    else:
        print(f"\n  ❌ ERROR: Some locations are OUTSIDE the bounds!")
        print(f"  This should never happen - please report this as a bug.")
    
    # Calculate actual output size (subtract safety margins for reporting)
    output_width_km = final_width_km - (2 * safety_margin_km)
    output_height_km = final_height_km - (2 * safety_margin_km)
    
    print(f"\n📊 SUMMARY:")
    print(f"  • Requested size: {target_size_km:.3f} km × {target_size_km:.3f} km")
    print(f"  • Actual size (with safety margins): {final_width_km:.4f} km × {final_height_km:.4f} km")
    print(f"  • All locations inside: {'YES ✓' if all_inside else 'NO ✗'}")
    print(f"  • Minimum clearance: {min_margin_km:.4f} km")
    
    return {
        'min_lon': min_lon_with_margin,
        'max_lat': max_lat_with_margin,
        'max_lon': max_lon_with_margin,
        'min_lat': min_lat_with_margin,
        'width_km': final_width_km,
        'height_km': final_height_km,
        'all_inside': all_inside,
        'min_margin_km': min_margin_km
    }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def move_and_click(x, y, description, duration=2, wait_after=1):
    print(f"\n--- {description} ---")
    start_pos = pyautogui.position()
    print(f"Starting from: {start_pos}")
    print(f"Moving to ({x}, {y})...")
    pyautogui.moveTo(x, y, duration=duration)
    current_pos = pyautogui.position()
    print(f"Positioned at: {current_pos}")
    time.sleep(0.5)
    print("Clicking...")
    pyautogui.click(button='left')
    print("✓ Clicked!")
    time.sleep(wait_after)
    return current_pos

def move_click_and_type(x, y, text, description, duration=2, wait_after=1):
    print(f"\n--- {description} ---")
    start_pos = pyautogui.position()
    print(f"Starting from: {start_pos}")
    print(f"Moving to ({x}, {y})...")
    pyautogui.moveTo(x, y, duration=duration)
    current_pos = pyautogui.position()
    print(f"Positioned at: {current_pos}")
    time.sleep(0.5)
    print("Clicking...")
    pyautogui.click(button='left')
    print("✓ Clicked!")
    time.sleep(0.5)
    print(f"Typing path...")
    pyautogui.typewrite(text)
    print("✓ Path entered!")
    time.sleep(wait_after)
    return current_pos

def move_and_double_click(x, y, description, duration=2, wait_after=1):
    print(f"\n--- {description} ---")
    start_pos = pyautogui.position()
    print(f"Starting from: {start_pos}")
    print(f"Moving to ({x}, {y})...")
    pyautogui.moveTo(x, y, duration=duration)
    current_pos = pyautogui.position()
    print(f"Positioned at: {current_pos}")
    time.sleep(0.5)
    print("Double clicking...")
    pyautogui.doubleClick(button='left')
    print("✓ Double clicked!")
    time.sleep(wait_after)
    return current_pos

def enter_coordinate_value(x, y, value, description):
    """Helper function to enter a coordinate value with proper text replacement"""
    print(f"\n--- {description} ---")
    print(f"Moving to ({x}, {y})...")
    pyautogui.moveTo(x, y, duration=1.5)
    time.sleep(0.5)
    pyautogui.click(button='left')
    time.sleep(0.3)
    print("  - Selecting all text (Ctrl+A)...")
    pyautogui.hotkey('ctrl', 'a')
    time.sleep(0.2)
    print("  - Deleting (Backspace)...")
    pyautogui.press('backspace')
    time.sleep(0.2)
    print(f"  - Entering value: {value}")
    pyautogui.typewrite(str(value), interval=0.05)
    print(f"✓ Entered {value} at ({x}, {y})")
    time.sleep(0.5)

def move_only(x, y, description, duration=2, wait_after=1):
    print(f"\n--- {description} ---")
    start_pos = pyautogui.position()
    print(f"Starting from: {start_pos}")
    print(f"Moving to ({x}, {y})...")
    pyautogui.moveTo(x, y, duration=duration)
    current_pos = pyautogui.position()
    print(f"Positioned at: {current_pos}")
    print("✓ Moved (no click)")
    time.sleep(wait_after)
    return current_pos

# ============================================================================
# AUTOMATION SEQUENCES - ALL 12 SEQUENCES
# ============================================================================

def run_all_sequences(csv_file_path, directory_path, csv_path, antenna_path, antenna_name, 
                      target_size_km=1.0, safety_margin_km=0.05, 
                      is_very_first_file=False,is_absolute_first_file=False, is_first_power_in_antenna=False):
    """
    Run all 12 automation sequences for a single CSV file.
    
    Args:
        csv_file_path: Full path to the CSV file
        directory_path: Directory path for Sequence 2
        csv_path: CSV path for Sequence 4
        antenna_path: Antenna path for Sequence 5
        antenna_name: Name of the antenna folder (e.g., 'Ant1', 'Ant2')
        target_size_km: Target size for the square in kilometers (default: 1.0)
        safety_margin_km: Additional safety margin in km (default: 0.05)
        is_very_first_file: Boolean - True only for Ant1/p10 (very first file overall)
        is_first_power_in_antenna: Boolean - True for the first power level in each antenna
    """
    
    # Derive paths from CSV file location
    final_path = os.path.dirname(csv_file_path)
    patches_path = final_path
    automation_path = final_path
    
    print(f"\n📂 Working with:")
    print(f"   • Antenna: {antenna_name}")
    print(f"   • CSV File: {csv_file_path}")
    print(f"   • Final Path: {final_path}")
    print(f"   • Target Size: {target_size_km:.3f} km × {target_size_km:.3f} km")
    print(f"   • Safety Margin: {safety_margin_km:.3f} km")
    print(f"   • Very First File: {'YES' if is_very_first_file else 'NO'}")
    print(f"   • First Power in Antenna: {'YES' if is_first_power_in_antenna else 'NO'}")
    
    # Define antenna-specific coordinates for Sequence 5
    antenna_coords = {
        'Ant1': (212, 137),
        'Ant2': (212,160),
        'Ant3': (217,181),
        'Ant4': (217, 200)
    }
    
    ant_coord = antenna_coords.get(antenna_name, (794, 402))  # Default to Ant1 if not found
    
    # ============================================================================
    # CALCULATE COORDINATES FROM CSV - SUPER-ROBUST VERSION
    # ============================================================================
    
    print("\n" + "=" * 70)
    print("🔍 CALCULATING BOUNDING RECTANGLE FROM CSV (SUPER-ROBUST VERSION)...")
    print("=" * 70)
    
    coords = calculate_bounding_rectangle(csv_file_path, 
                                         target_size_km=target_size_km,
                                         safety_margin_km=safety_margin_km)
    
    if coords is None:
        print("\n❌ ERROR: Could not calculate coordinates from CSV file!")
        print("Skipping this file...")
        return False
    
    if not coords['all_inside']:
        print("\n❌ CRITICAL ERROR: Some locations are outside the bounding rectangle!")
        print("This should never happen. Skipping this file for safety...")
        return False
    
    # Extract coordinate values with 6 decimal places for precision
    coord1 = f"{coords['min_lon']:.6f}"
    
    coord2 =f"{coords['min_lat']:.6f}"  # Max Latitude (top)
    coord3 =f"{coords['max_lon']:.6f}"  # Max Longitude (right)
    coord4 =f"{coords['max_lat']:.6f}"   # Min Latitude (bottom)
    
    print("\n✅ Coordinates calculated successfully!")
    print(f"  1. Min Longitude (left):   {coord1}")
    print(f"  2. Max Latitude (top):     {coord2}")
    print(f"  3. Max Longitude (right):  {coord3}")
    print(f"  4. Min Latitude (bottom):  {coord4}")
    print(f"\n  📐 Final size: {coords['width_km']:.4f} km × {coords['height_km']:.4f} km")
    print(f"  📏 Minimum clearance: {coords['min_margin_km']:.4f} km")
    
    # ============================================================================
    # RUN ALL SEQUENCES (SAME AS BEFORE)
    # ============================================================================
    
    print("\n" + "=" * 70)
    print("🚀 STARTING ALL 12 AUTOMATION SEQUENCES")
    print("=" * 70)
    

    
    # SEQUENCE 2: DIRECTORY AND FILE OPERATIONS
    print("\n" + "=" * 70)
    print("🚀 STARTING SEQUENCE 2: DIRECTORY AND FILE OPERATIONS")
    print("=" * 70)
    move_and_click(14, 31, "First Click", duration=2, wait_after=1.5)
    move_and_click(9, 81, "Second Click", duration=2, wait_after=1.5)
    move_click_and_type(868, 47, directory_path, "Enter Directory Path", duration=2, wait_after=1)
    pyautogui.press('enter')
    time.sleep(2)
    move_and_double_click(217, 264, "Double Click", duration=2, wait_after=2)
    print("\n✅ SEQUENCE 2 COMPLETED!")
    time.sleep(7)

    # [ALL 12 SEQUENCES - EXACT SAME CODE AS BEFORE]
    # SEQUENCE 1: MAP LAYER AUTOMATION
    print("\n" + "=" * 70)
    print("🚀 STARTING SEQUENCE 1: MAP LAYER AUTOMATION")
    print("=" * 70)
    move_and_click(1673, 62, "Opening HTZ Map Layer", duration=2, wait_after=1.5)
    move_and_click(1423,386, "Selecting Map Layer Option", duration=2, wait_after=1.5)
    move_and_click(1870, 36, "Closing Map Layer Panel", duration=2, wait_after=2)
    print("\n✅ SEQUENCE 1 COMPLETED!")
    time.sleep(7)
    
    # SEQUENCE 3: COORDINATE INPUT AND NAVIGATION (CONDITIONAL)
    print("\n" + "=" * 70)
    if is_very_first_file:
        print("🚀 STARTING SEQUENCE 3: COORDINATE INPUT AND NAVIGATION - FULL VERSION")
    else:
        print("🚀 STARTING SEQUENCE 3: COORDINATE INPUT AND NAVIGATION - SKIP COORDINATES")
    print("=" * 70)
    print("📍 Using dynamically calculated coordinates from CSV file!")
    
    pyautogui.moveTo(955, 65, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    
    pyautogui.moveTo(1051, 100, duration=1.5)
    time.sleep(1)
    
    pyautogui.moveTo(1066, 184, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    
    # Only enter coordinates for the VERY FIRST FILE (Ant1/p10)
    if is_very_first_file:
        print("  📍 Entering coordinates (very first file only)...")
        enter_coordinate_value(1310, 478, coord1, f"Coordinate 1 - Min Longitude (left): {coord1}")
        enter_coordinate_value(1313, 505, coord2, f"Coordinate 2 - Max Latitude (top): {coord2}")
        enter_coordinate_value(1307, 528, coord3, f"Coordinate 3 - Max Longitude (right): {coord3}")
        enter_coordinate_value(1312, 552, coord4, f"Coordinate 4 - Min Latitude (bottom): {coord4}")
    else:
        print("  ⏭️  Skipping coordinate entry (not the very first file)...")
    
    pyautogui.moveTo(1227, 615, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    
    pyautogui.moveTo(1297, 8, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    
    pyautogui.moveTo(1316,574, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    
    print("\n✅ SEQUENCE 3 COMPLETED!")
    time.sleep(10)
    
    # SEQUENCE 4: CSV PATH AND FINAL OPERATIONS
    print("\n" + "=" * 70)
    print("🚀 STARTING SEQUENCE 4: CSV PATH AND FINAL OPERATIONS")
    print("=" * 70)
    move_and_click(14, 31, "First Click", duration=2, wait_after=1.5)
    move_and_click(10, 390, "Second Click", duration=2, wait_after=1.5)
    move_and_click(325, 375, "Third Click", duration=2, wait_after=1.5)
    move_and_click(433, 576, "Fourth Click", duration=2, wait_after=1.5)
    move_and_click(910, 912, "Fifth Click", duration=2, wait_after=1.5)
    move_click_and_type(859,45, csv_path, "Enter CSV Path", duration=2, wait_after=1)
    #move_click_and_type(619, 88, csv_path, "Enter CSV Path", duration=2, wait_after=1)
    pyautogui.press('enter')
    time.sleep(2)
    move_and_double_click(217,198, "Double Click", duration=2, wait_after=2)
    move_and_click(1265,526, "Click", duration=2, wait_after=1.5)
    move_click_and_type(934,46, final_path, "Enter Final Path", duration=2, wait_after=1)
    pyautogui.press('enter')
    time.sleep(2)
    move_and_double_click(222,161, "Double Click", duration=2, wait_after=2)
    move_and_click(133,584, "Click", duration=2, wait_after=1.5)
    move_and_click(1422, 780, "Click", duration=2, wait_after=1.5)
    move_and_click(1225, 710, "Final Click", duration=2, wait_after=2)
    print("\n✅ SEQUENCE 4 COMPLETED!")
    time.sleep(2)
    
    # SEQUENCE 5: ANTENNA PATH OPERATIONS (WITH ANTENNA-SPECIFIC COORDINATES)
    print("\n" + "=" * 70)
    print(f"🚀 STARTING SEQUENCE 5: ANTENNA PATH OPERATIONS ({antenna_name})")
    print("=" * 70)
    print(f"  📍 Using antenna-specific coordinates: {ant_coord}")
    move_and_click(410, 66, "First Click", duration=2, wait_after=1.5)
    move_and_click(524, 102, "Second Click", duration=2, wait_after=1.5)
    move_and_click(926, 415, "Third Click", duration=2, wait_after=1.5)
    move_and_click(714, 722, "Fourth Click", duration=2, wait_after=1.5)
    move_click_and_type(931,46, antenna_path, "Enter Antenna Path", duration=2, wait_after=1)
    pyautogui.press('enter')
    time.sleep(2)
    # Use antenna-specific coordinates here
    move_and_double_click(ant_coord[0], ant_coord[1], f"Double Click for {antenna_name}", duration=2, wait_after=2)
    move_and_click(1206, 823, "Click", duration=2, wait_after=1.5)
    move_and_click(892, 583, "Click", duration=2, wait_after=2)
    move_and_click(970, 560, "Final Click", duration=2, wait_after=2)
    print("\n✅ SEQUENCE 5 COMPLETED!")
    time.sleep(2)
    
    # SEQUENCE 6: SIX-CLICK NAVIGATION
    print("\n" + "=" * 70)
    print("🚀 STARTING SEQUENCE 6: SIX-CLICK NAVIGATION")
    print("=" * 70)
    move_and_click(926, 33, "Click 1", duration=2, wait_after=1.5)
    move_and_click(950,193, "Click 2", duration=2, wait_after=1.5)
    move_and_click(855,193, "Click 3", duration=2, wait_after=1.5)
    move_and_click(658, 179, "Click 4", duration=2, wait_after=1.5)
    move_and_click(1402, 702, "Click 5", duration=2, wait_after=1.5)
    move_and_click(1038, 261, "Click 6", duration=2, wait_after=2)
    print("\n✅ SEQUENCE 6 COMPLETED!")
    time.sleep(2)
    
    # SEQUENCE 7: BACKSPACE OPERATIONS
    print("\n" + "=" * 70)
    print("🚀 STARTING SEQUENCE 7: BACKSPACE OPERATIONS")
    print("=" * 70)
    move_and_click(101, 31, "Click 1", duration=2, wait_after=1.5)
    move_and_click(162, 55, "Click 2", duration=2, wait_after=1.5)
    move_and_click(421, 52, "Click 3", duration=2, wait_after=1.5)
    move_and_click(457, 96, "Click 4", duration=2, wait_after=0.5)
    pyautogui.press('backspace')
    time.sleep(0.3)
    pyautogui.press('backspace')
    time.sleep(1)
    move_and_click(459, 307, "Final Click", duration=2, wait_after=2)
    print("\n✅ SEQUENCE 7 COMPLETED!")
    time.sleep(50)
    
    # SEQUENCE 8: COVERAGE PATH OPERATIONS
    print("\n" + "=" * 70)
    print("🚀 STARTING SEQUENCE 8: COVERAGE PATH OPERATIONS")
    print("=" * 70)
    print("📍 Using automatically derived patches_path from CSV file location!")
    move_and_click(12, 30, "Click 1", duration=2, wait_after=1.5)
    move_and_click(54, 406, "Click 2", duration=2, wait_after=1.5)
    move_only(361, 409, "Move Only", duration=2, wait_after=1.5)
    move_and_click(372, 495, "Click 3", duration=2, wait_after=2)
    move_and_click(1227, 586, "Click 4", duration=2, wait_after=2)
    move_click_and_type(719,50, patches_path, "Enter Patches Path", duration=2, wait_after=1)
    pyautogui.press('enter')
    time.sleep(2)
    move_and_click(218,934, "Click", duration=2, wait_after=0.5)
    pyautogui.press('backspace')
    time.sleep(0.3)
    pyautogui.typewrite('coverage')
    time.sleep(1)
    move_and_click(1756,1006, "Final Click", duration=2, wait_after=2)
    print("\n✅ SEQUENCE 8 COMPLETED!")
    time.sleep(5)
    
    # SEQUENCE 9: COMPLETE COORDINATE SEQUENCE (LAYOUT) - CONDITIONAL
    print("\n" + "=" * 70)
    if is_very_first_file:
        print("🚀 STARTING SEQUENCE 9: COMPLETE COORDINATE SEQUENCE (LAYOUT) - FULL VERSION")
    else:
        print("🚀 STARTING SEQUENCE 9: COMPLETE COORDINATE SEQUENCE (LAYOUT) - SHORTENED VERSION")
    print("=" * 70)
    
    pyautogui.moveTo(12, 33, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(50, 449, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(298, 445, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(724, 402, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(724, 427, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    
    # Only execute these checkbox clicks for the VERY FIRST FILE (Ant1/p10)
    if is_absolute_first_file:
        print("  🔘 Clicking additional checkboxes (very first file only)...")
        pyautogui.moveTo(724, 536, duration=1.5)
        pyautogui.click()
        time.sleep(1)
        pyautogui.moveTo(725, 558, duration=1.5)
        pyautogui.click()
        time.sleep(1)
    else:
        print("  ⏭️  Skipping additional checkboxes (not the very first file)...")
    
    pyautogui.moveTo(884, 414, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(907, 706, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(1057, 712, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(1042,53, duration=1.5)
    pyautogui.click()
    time.sleep(0.3)
    pyautogui.press('backspace')
    time.sleep(0.2)
    pyautogui.typewrite(automation_path, interval=0.03)
    time.sleep(0.3)
    pyautogui.press('enter')
    time.sleep(1)
    pyautogui.moveTo(290,983, duration=1.5)
    pyautogui.click()
    time.sleep(0.3)
    pyautogui.press('backspace')
    time.sleep(0.2)
    pyautogui.typewrite('layout', interval=0.05)
    time.sleep(1)
    pyautogui.moveTo(1745,1004, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(1111, 466, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(1133, 838, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(1069, 703, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    print("\n✅ SEQUENCE 9 COMPLETED!")
    time.sleep(2)
    
    # SEQUENCE 10: SIMPLIFIED COORDINATE SEQUENCE (COVERAGE)
    print("\n" + "=" * 70)
    print("🚀 STARTING SEQUENCE 10: SIMPLIFIED COORDINATE SEQUENCE (COVERAGE)")
    print("=" * 70)
    pyautogui.moveTo(12, 33, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(50, 449, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(298, 445, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(884, 414, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(907, 706, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(1057, 712, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(728, 46, duration=1.5)
    pyautogui.click()
    time.sleep(0.3)
    pyautogui.press('backspace')
    time.sleep(0.2)
    pyautogui.typewrite(automation_path, interval=0.03)
    time.sleep(0.3)
    pyautogui.press('enter')
    time.sleep(1)
    pyautogui.moveTo(278,977, duration=1.5)
    pyautogui.click()
    time.sleep(0.3)
    pyautogui.press('backspace')
    time.sleep(0.2)
    pyautogui.typewrite('coverage', interval=0.05)
    time.sleep(1)
    pyautogui.moveTo(1744,1005, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(1111, 466, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(1133, 838, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(1069, 703, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    print("\n✅ SEQUENCE 10 COMPLETED!")
    time.sleep(2)
    
    # SEQUENCE 11: SITE TEXT SEQUENCE
    print("\n" + "=" * 70)
    print("🚀 STARTING SEQUENCE 11: SITE TEXT SEQUENCE")
    print("=" * 70)
    pyautogui.moveTo(12, 33, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(50, 449, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(298, 445, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(884, 414, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(1061, 400, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(907, 706, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(1057, 712, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(781, 46, duration=1.5)
    pyautogui.click()
    time.sleep(0.3)
    pyautogui.press('backspace')
    time.sleep(0.2)
    pyautogui.typewrite(automation_path, interval=0.03)
    time.sleep(0.3)
    pyautogui.press('enter')
    time.sleep(1)
    pyautogui.moveTo(274,977, duration=1.5)
    pyautogui.click()
    time.sleep(0.3)
    pyautogui.press('backspace')
    time.sleep(0.2)
    pyautogui.typewrite('site', interval=0.05)
    time.sleep(1)
    pyautogui.moveTo(1734,1011, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(1111, 466, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(1133, 838, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(1069, 703, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    print("\n✅ SEQUENCE 11 COMPLETED!")
    time.sleep(100)
    
    # SEQUENCE 12: OBJECT TEXT SEQUENCE (FINAL)
    print("\n" + "=" * 70)
    print("🚀 STARTING SEQUENCE 12: OBJECT TEXT SEQUENCE (FINAL)")
    print("=" * 70)
    
    pyautogui.moveTo(12, 33, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    
    pyautogui.moveTo(49, 450, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    
    pyautogui.moveTo(294, 452, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    
    pyautogui.moveTo(724, 379, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    
    pyautogui.moveTo(727, 400, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    
    pyautogui.moveTo(895, 702, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    
    pyautogui.moveTo(1059, 711, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    
    pyautogui.moveTo(730, 49, duration=1.5)
    pyautogui.click()
    time.sleep(0.3)
    pyautogui.press('backspace')
    time.sleep(0.2)
    pyautogui.typewrite(automation_path, interval=0.03)
    time.sleep(0.3)
    pyautogui.press('enter')
    time.sleep(1)
    
    pyautogui.moveTo(242,976, duration=1.5)
    pyautogui.click()
    time.sleep(0.3)
    pyautogui.press('backspace')
    time.sleep(0.2)
    pyautogui.typewrite('object', interval=0.05)
    time.sleep(1)
    
    pyautogui.moveTo(1744,1007, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    
    pyautogui.moveTo(1127, 842, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    
    pyautogui.moveTo(1079, 700, duration=1.5)
    pyautogui.click()
    time.sleep(1)
    
    print("\n✅ SEQUENCE 12 COMPLETED!")
    
    # CLOSING THE PROJECT
    print("\n" + "=" * 70)
    print("🚀 CLOSING THE PROJECT")
    print("=" * 70)
    move_and_click(12, 33, "Opening Menu", duration=2, wait_after=1.5)
    move_and_click(34, 187, "Selecting Close Option", duration=2, wait_after=1.5)
    move_and_click(918, 583, "Confirming Close", duration=2, wait_after=2)
    move_and_click(996, 550, "Final Confirmation", duration=2, wait_after=2)
    print("\n✅ PROJECT CLOSED!")
    time.sleep(2)
    
    return True

# ============================================================================
# MAIN FUNCTION - ITERATE OVER ALL ANTENNAS AND POWER LEVELS
# ============================================================================

def main():
    # ============================================================================
    # CONFIGURATION - EDIT BASE PATH
    # ============================================================================
    
    # Base directory structure (everything before the cluster folder)
    BASE_PATH = r"C:\Users\pariamdz\Desktop\csv\filtered\patches\l61\f4"
    
    # Cluster folders to process
    CLUSTER_FOLDERS = ['cluster_15']
    
    
    
    # Antenna folders to proc
    ANTENNA_FOLDERS = ['Ant4']
    
    # Power level folders to process
    POWER_FOLDERS = ['pdef']
   # CLUSTER_FOLDERS = ['cluster_9','cluster_10','cluster_11','cluster_12','cluster_13']
    
    # Antenna folders t'o process
   # ANTENNA_FOLDERS = ['Ant1','Ant2','Ant3', 'Ant4']
    
    # Power level folders to process
   # POWER_FOLDERS = ['p10', 'p50', 'pdef']
    
    # Other paths (constant across all iterations)
    directory_path = r"C:\Users\pariamdz\Desktop\csv\filtered\patches\freqency_count"
    csv_path = r"C:\Users\pariamdz\Desktop\csv"
    antenna_path = r"C:\Users\pariamdz\Desktop\csv\filtered\patches\Antenna"
    
    # Target size for the bounding square (in kilometers)
    TARGET_SIZE_KM = 1.0
    
    # Safety margin (in kilometers) - extra buffer to ensure all points are comfortably inside
    # Default: 0.05 km (50 meters) provides good safety without making the box too large
    SAFETY_MARGIN_KM = 0.05
    
    # ============================================================================
    # MAIN LOOP - PROCESS EACH CLUSTER, ANTENNA AND POWER FOLDER
    # ============================================================================
    
    print("=" * 70)
    print("=== HTZ AUTOMATION: MULTI-CLUSTER, ANTENNA & POWER PROCESSING ===")
    print("=== SUPER-ROBUST VERSION WITH COMPREHENSIVE VALIDATION ===")
    print("=" * 70)
    print(f"\nBase Path: {BASE_PATH}")
    print(f"Cluster Folders: {', '.join(CLUSTER_FOLDERS)}")
    print(f"Antenna Folders: {', '.join(ANTENNA_FOLDERS)}")
    print(f"Power Folders: {', '.join(POWER_FOLDERS)}")
    print(f"Target Square Size: {TARGET_SIZE_KM:.3f} km × {TARGET_SIZE_KM:.3f} km")
    print(f"Safety Margin: {SAFETY_MARGIN_KM:.3f} km")
    print(f"\nTotal Files to Process: {len(CLUSTER_FOLDERS)} × {len(ANTENNA_FOLDERS)} × {len(POWER_FOLDERS)} = {len(CLUSTER_FOLDERS) * len(ANTENNA_FOLDERS) * len(POWER_FOLDERS)}")
    print("=" * 70)
    
    # Countdown before starting
    print("\nStarting automation in 5 seconds...")
    for i in range(5, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    print("\nStarting NOW!")
    time.sleep(0.5)
    
    successful = 0
    failed = 0
    total_files = len(CLUSTER_FOLDERS) * len(ANTENNA_FOLDERS) * len(POWER_FOLDERS)
    current_file = 0
    
    for cluster_idx, cluster_folder in enumerate(CLUSTER_FOLDERS, 1):
        print("\n" + "=" * 70)
        print(f"🏢 STARTING {cluster_folder.upper()} ({cluster_idx}/{len(CLUSTER_FOLDERS)})")
        print("=" * 70)
        
        for ant_idx, antenna_folder in enumerate(ANTENNA_FOLDERS, 1):
            for power_idx, power_folder in enumerate(POWER_FOLDERS, 1):
                current_file += 1
                
                print("\n" + "=" * 70)
                print(f"📁 PROCESSING {current_file}/{total_files}: {cluster_folder}/{antenna_folder}/{power_folder}")
                print("=" * 70)
                
                # Determine flags for conditional execution
                #is_very_first = (cluster_idx == 1 and ant_idx == 1 and power_idx == 1)
                is_absolute_first = (current_file == 1) 
                is_very_first = (ant_idx == 1 and power_idx == 1)
                is_first_power_in_antenna = (power_idx == 1)
                
                if is_very_first:
                    print("  🌟 This is the VERY FIRST FILE (cluster_1/Ant1/p10)")
                if is_absolute_first:
                    print("  ⭐ This is the ABSOLUTE FIRST FILE overall")
                if not is_very_first:
                    print("  ℹ️  This is NOT the first file in cluster")
                    
                
                # Construct full path
                folder_path = os.path.join(BASE_PATH, cluster_folder, antenna_folder, power_folder)
                csv_file_path = os.path.join(folder_path, f"{power_folder}_data.csv")
                
                if os.path.exists(csv_file_path):
                    print(f"✓ Found CSV file: {csv_file_path}")
                    
                    # Run all 12 sequences for this CSV file
                    success = run_all_sequences(
                        csv_file_path=csv_file_path,
                        directory_path=directory_path,
                        csv_path=csv_path,
                        antenna_path=antenna_path,
                        antenna_name=antenna_folder,
                        target_size_km=TARGET_SIZE_KM,
                        safety_margin_km=SAFETY_MARGIN_KM,
                        is_very_first_file=is_very_first,
                        is_absolute_first_file=is_absolute_first,
                        is_first_power_in_antenna=is_first_power_in_antenna
                    )
                    
                    if success:
                        successful += 1
                        print(f"\n✅ Successfully completed {cluster_folder}/{antenna_folder}/{power_folder}")
                    else:
                        failed += 1
                        print(f"\n❌ Failed to process {cluster_folder}/{antenna_folder}/{power_folder}")
                    
                    # Pause between iterations (except after the last one)
                    if current_file < total_files:
                        print(f"\n⏸️  Pausing 5 seconds before next iteration...")
                        time.sleep(5)
                else:
                    print(f"❌ CSV file not found: {csv_file_path}")
                    print("Skipping this file...")
                    failed += 1
        
        # Summary after each cluster
        print("\n" + "=" * 70)
        print(f"✅ COMPLETED {cluster_folder.upper()}")
        print("=" * 70)
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    
    print("\n" + "=" * 70)
    print("🎉🎉🎉 ALL PROCESSING COMPLETED! 🎉🎉🎉")
    print("=" * 70)
    print("\n📊 FINAL SUMMARY:")
    print(f"\n✅ Successfully processed: {successful}/{total_files} files")
    print(f"❌ Failed: {failed}/{total_files} files")
    print(f"\n🎯 Target size per file: {TARGET_SIZE_KM:.3f} km × {TARGET_SIZE_KM:.3f} km")
    print(f"🛡️  Safety margin: {SAFETY_MARGIN_KM:.3f} km")
    print("\n🔍 Special features:")
    print("   • Comprehensive validation of all locations")
    print("   • Automatic size adjustment if locations too spread out")
    print("   • Detailed diagnostics and margin calculations")
    print("   • Safety checks prevent processing invalid rectangles")
    print("=" * 70)
    print("\n🏆 ALL AUTOMATION FINISHED! 🏆")
    print("=" * 70)
    
    input("\n👉 Press Enter to exit...")

# ============================================================================
# RUN THE AUTOMATION
# ============================================================================

if __name__ == "__main__":
    main()