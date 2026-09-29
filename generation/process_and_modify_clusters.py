import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import os
import sys

# --- Configuration ---
MAX_DIAMETER_KM = 1.0  # Max distance between any two points in a cluster
ANTENNA_FOLDERS = ['Ant1', 'Ant2', 'Ant3', 'Ant4']
POWER_FOLDERS_MAP = {
    'p10': 10,
    'p50': 50,
    'pdef': None  # None means keep the original value
}
# ---------------------

def modify_dataframe_columns(df, has_header=True):
    """
    Apply the column modifications to a dataframe:
    1. Create 'height' column (structure_height + tx_ant_height)
    2. Duplicate 'bandwidth' column (2 copies total)
    3. Drop unwanted columns
    4. Reorder columns to place 'height' after 'structure_height'
    
    Args:
        df: Input dataframe
        has_header: Whether the dataframe has headers (affects column handling)
    
    Returns:
        Modified dataframe
    """
    if not has_header:
        # For files without headers, we need to work with column indices
        # Assuming the structure matches the original CSV format
        # We'll add headers temporarily, modify, then return without headers
        
        # Standard column order based on the original file
        standard_columns = [
            'area_name*', 'location', 'station_type', 'technology', 'latitude', 
            'longitude', 'site_type', 'structure_height', 'structure_type', 
            'tx_frequency', 'rx_frequency', 'bandwidth', 'tx_power', 
            'downlink_allocation', 'tx_number_antennas', 'rx_number_antennas',
            'tx_ant_manufacturer', 'rx_ant_manufacturer', 'tx_ant_height', 
            'rx_ant_height', 'tx_ant_omni_indicator', 'rx_ant_omni_indicator',
            'tx_ant_horiz_beamwidth', 'rx_ant_horiz_beamwidth', 
            'tx_ant_vert_beamwidth', 'rx_ant_vert_beamwidth', 'tx_ant_azimuth',
            'rx_ant_azimuth', 'tx_ant_elevation_angle', 'rx_ant_elevation_angle',
            'tx_ant_gain', 'rx_ant_gain', 'tx_line_loss', 'rx_line_loss'
        ]
        
        # Assign column names temporarily
        df.columns = standard_columns[:len(df.columns)]
    
    # --- 1. Create 'height' column ---
    df['height'] = df['structure_height'] + df['tx_ant_height']
    
    # --- 2. Handle 'bandwidth' columns ---
    all_cols = df.columns.tolist()
    
    # Find the first column that starts with 'bandwidth'
    first_bw_col_name = next((col for col in all_cols if col.startswith('bandwidth')), None)
    
    if first_bw_col_name:
        # Save its index and data
        bw_index = all_cols.index(first_bw_col_name)
        bw_data = df.iloc[:, bw_index].copy()
        
        # Columns to drop
        cols_to_drop = [
            'structure_type',
            'tx_ant_manufacturer',
            'rx_ant_manufacturer'
        ]
        
        # Add all bandwidth columns to drop list
        bw_cols_to_drop = [col for col in all_cols if col.startswith('bandwidth')]
        
        # Combine and drop
        final_drop_list = [col for col in cols_to_drop if col in all_cols] + bw_cols_to_drop
        df = df.drop(columns=final_drop_list)
        
        # Get updated column list after dropping
        all_cols = df.columns.tolist()
        
        # Adjust index if needed
        if bw_index > len(all_cols):
            bw_index = len(all_cols)
        
        # Insert TWO bandwidth columns
        df.insert(bw_index, 'bandwidth', bw_data)
        df.insert(bw_index + 1, 'bandwidth_temp', bw_data)
        
        # Rename the temporary column
        df.rename(columns={'bandwidth_temp': 'bandwidth'}, inplace=True)
        
    else:
        # Drop other columns even if bandwidth not found
        cols_to_drop = [
            'structure_type',
            'tx_ant_manufacturer',
            'rx_ant_manufacturer'
        ]
        existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        df = df.drop(columns=existing_cols_to_drop)
    
    # --- 3. Reorder columns to place 'height' after 'structure_height' ---
    if 'structure_height' in df.columns and 'height' in df.columns:
        # Find the position of structure_height
        sh_idx = df.columns.tolist().index('structure_height')
        
        # Extract the height column
        height_col = df['height'].copy()
        
        # Drop height from the dataframe
        df = df.drop(columns=['height'])
        
        # Insert height right after structure_height
        df.insert(sh_idx + 1, 'height', height_col)
    
    return df

def cluster_dataframe(df):
    """
    Takes a dataframe and returns it with a new 'cluster' column
    using the 1km max diameter (complete linkage) method.
    """
    print("Calculating average coordinates for each unique location...")
    location_coords = df.groupby('location')[['latitude', 'longitude']].mean().reset_index()

    print("Converting coordinates to radians...")
    location_coords_radians = np.radians(location_coords[['latitude', 'longitude']])
    
    print("Manually calculating all pairwise haversine distances...")
    # Calculate distance matrix in KM
    dist_matrix_km = haversine_distances(location_coords_radians, location_coords_radians) * 6371 # Earth's radius in km

    # Convert to condensed format for linkage
    condensed_dist_matrix = squareform(dist_matrix_km)

    print("Running hierarchical clustering...")
    # Method 'complete' ensures the 1km max diameter rule (prevents chaining)
    Z = linkage(condensed_dist_matrix, method='complete')

    # Form flat clusters based on our distance threshold
    clusters = fcluster(Z, t=MAX_DIAMETER_KM, criterion='distance')
    
    location_coords['cluster'] = clusters
    num_clusters = len(set(clusters))
    print(f"Clustering complete. Found {num_clusters} unique clusters.")

    # Map cluster results back to the main dataframe
    df_clustered = df.merge(location_coords[['location', 'cluster']], on='location', how='left')
    return df_clustered, num_clusters

def create_output_files(cluster_df, base_path):
    """
    Creates the Ant1-4 and p10/p50/pdef subfolders and files
    for a single cluster's dataframe.
    Applies column modifications to each generated file.
    """
    for ant_folder in ANTENNA_FOLDERS:
        ant_folder_path = os.path.join(base_path, ant_folder)
        os.makedirs(ant_folder_path, exist_ok=True)
        
        for power_name, power_value in POWER_FOLDERS_MAP.items():
            power_folder_path = os.path.join(ant_folder_path, power_name)
            os.makedirs(power_folder_path, exist_ok=True)
            
            # Create the modified dataframe for this power level
            output_df = cluster_df.copy()
            if power_value is not None:
                output_df['tx_power'] = power_value
            
            # --- APPLY COLUMN MODIFICATIONS ---
            output_df = modify_dataframe_columns(output_df, has_header=True)
                
            # Define file paths
            file_name_base = f'{power_name}_data'
            header_file_path = os.path.join(power_folder_path, f'{file_name_base}.csv')
            no_header_file_path = os.path.join(power_folder_path, f'{file_name_base}_noheader.csv')
            
            # Save file with header
            output_df.to_csv(header_file_path, index=False)
            
            # Save file without header
            output_df.to_csv(no_header_file_path, index=False, header=False)

def main():
    # --- 1. Get Input from Command Line ---
    if len(sys.argv) != 3:
        print("\nError: Invalid arguments.")
        print("Usage: python process_and_modify_clusters.py \"<path_to_directory>\" \"<csv_filename>\"")
        print("Example:")
        print("python process_and_modify_clusters.py \"C:\\Users\\pariamdz\\Desktop\\my_data\" \"p50.csv\"")
        sys.exit(1)

    INPUT_DIRECTORY = sys.argv[1]
    INPUT_CSV_NAME = sys.argv[2]
    input_file_path = os.path.join(INPUT_DIRECTORY, INPUT_CSV_NAME)

    # --- 2. Check if file exists ---
    if not os.path.exists(input_file_path):
        print(f"Error: The file '{input_file_path}' was not found.")
        sys.exit(1)
        
    print(f"Loading '{input_file_path}'...")
    df = pd.read_csv(input_file_path)

    # --- 3. Run Clustering ---
    df_clustered, num_clusters = cluster_dataframe(df)
    
    print(f"\nStarting file and folder generation with column modifications in '{INPUT_DIRECTORY}'...")

    # --- 4. Loop, Create Folders, and Save Modified Files ---
    for cluster_id in sorted(list(df_clustered['cluster'].unique())):
        print(f"  Processing Cluster {cluster_id}...")
        
        # Get all rows for this cluster
        cluster_df = df_clustered[df_clustered['cluster'] == cluster_id].copy()
        # Remove the cluster column, it's not needed in the output
        cluster_df.drop(columns=['cluster'], inplace=True) 
        
        # Define the new folder for this cluster (e.g., "C:\...\f1\cluster_1")
        cluster_folder_name = f'cluster_{cluster_id}'
        cluster_folder_path = os.path.join(INPUT_DIRECTORY, cluster_folder_name)
        os.makedirs(cluster_folder_path, exist_ok=True)
        
        # Run the file generation function with modifications
        create_output_files(cluster_df, cluster_folder_path)

    print("\n--- All Done! ---")
    print(f"Successfully processed {num_clusters} clusters.")
    print(f"All files saved with column modifications in subfolders within '{INPUT_DIRECTORY}'.")
    print("\nModifications applied:")
    print("  - Added 'height' column (structure_height + tx_ant_height)")
    print("  - Duplicated 'bandwidth' column (2 copies total)")
    print("  - Removed: structure_type, tx_ant_manufacturer, rx_ant_manufacturer")
    print("  - Reordered: 'height' placed after 'structure_height'")

# This makes the script runnable
if __name__ == "__main__":
    main()