import pandas as pd

# Read the CSV file
df = pd.read_csv('combined_signals.csv')

# Create a list to store the transformed data
transformed_data = []

# Process each row
for _, row in df.iterrows():
    # Split the fixed_signals string into individual signals
    signals = row['fixed_signals'].split(';')
    
    # Process each signal
    for signal in signals:
        # Split the signal into its components
        token, position, leverage = signal.split(':')
        
        # Add to transformed data
        transformed_data.append({
            'date': row['date'],
            'time': row['time'],
            'token': token,
            'position': position,
            'leverage': int(leverage)
        })

# Create new dataframe from transformed data
new_df = pd.DataFrame(transformed_data)

# Save to new CSV file
new_df.to_csv('transformed_signals.csv', index=False)

print("Transformation complete. Data saved to transformed_signals.csv")