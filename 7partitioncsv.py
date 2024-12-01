import pandas as pd
import os
import math

def partition_csv(input_file, output_folder, rows_per_file=7):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Get the total number of rows (excluding header)
    total_rows = len(df)
    
    # Calculate number of files needed
    num_files = math.ceil(total_rows / rows_per_file)
    
    # Split the dataframe into chunks and save them
    for i in range(num_files):
        start_idx = i * rows_per_file
        end_idx = min((i + 1) * rows_per_file, total_rows)
        
        # Get the chunk of data
        chunk = df.iloc[start_idx:end_idx]
        
        # Create the output filename
        output_file = os.path.join(output_folder, f'signals_part_{i+1}.csv')
        
        # Save the chunk to a CSV file
        chunk.to_csv(output_file, index=False)
        print(f'Created file: {output_file} with {len(chunk)} rows')

if __name__ == "__main__":
    # Input file path
    input_file = 'signals_output.csv'
    
    # Output folder name
    output_folder = 'partitioned_signals'
    
    # Call the function to partition the CSV
    partition_csv(input_file, output_folder)