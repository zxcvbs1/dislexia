import pandas as pd
import glob
import os

# Path to your CSV files
path = 'partitioned_signals'

# Get all CSV files in the directory
all_files = glob.glob(os.path.join(path, "*.csv"))

# Create an empty list to store individual dataframes
df_list = []

# Read each CSV file and append to the list
for file in sorted(all_files):  # sorted() to ensure files are processed in order
    df = pd.read_csv(file)
    df_list.append(df)

# Concatenate all dataframes
combined_df = pd.concat(df_list, ignore_index=True)

# Save the combined dataframe to a new CSV file
combined_df.to_csv('combined_signals.csv', index=False)

print(f"Combined {len(all_files)} files into combined_signals.csv")