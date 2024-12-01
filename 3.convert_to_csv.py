import json
import csv
from datetime import datetime

def convert_signals_to_csv(json_data, output_file):
    # Open the output CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        # Create CSV writer
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['date', 'time', 'token', 'position', 'leverage'])
        
        # Process each message
        for message in json_data:
            date = message.get('date', '')
            time = message.get('time', '')
            
            # Process each signal in the message
            for signal in message.get('signals', []):
                # Split the signal into its components
                try:
                    token, position, leverage = signal.split(':')
                    writer.writerow([date, time, token, position, leverage])
                except ValueError:
                    print(f"Skipping malformed signal: {signal}")

# Read the JSON file
with open('processed_signals.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# Convert to CSV
convert_signals_to_csv(json_data, 'signals.csv')