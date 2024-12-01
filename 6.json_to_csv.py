import json
import csv
import re

def fix_leverage(leverage):
    leverage = int(leverage)
    # If number has 3 digits, remove the last one by integer division by 10
    if leverage >= 100:
        return leverage // 10
    return leverage

def process_signal(signal):
    # Split the signal into parts
    parts = signal.split(':')
    if len(parts) == 3:
        symbol, direction, leverage = parts
        # Fix the leverage
        fixed_leverage = fix_leverage(leverage)
        # Return the reconstructed signal
        return f"{symbol}:{direction}:{fixed_leverage}"
    return signal

def convert_json_to_csv(input_file='processed_signals.json', output_file='signals_output.csv'):
    # Read JSON file
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Ensure data is a list
    if not isinstance(data, list):
        data = [data]
    
    # Process each entry
    for entry in data:
        if 'signals' in entry and isinstance(entry['signals'], list):
            # Process each signal in the signals list
            entry['signals'] = [process_signal(signal) for signal in entry['signals']]
    
    # Define the fieldnames
    fieldnames = ['date', 'time', 'trade_section', 'signals', 'processed_at']
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header
        writer.writeheader()
        
        # Write each row
        for item in data:
            # Convert signals list to a string with semicolon separator
            if isinstance(item['signals'], list):
                item['signals'] = ';'.join(item['signals'])
            writer.writerow(item)

# Execute the conversion
try:
    convert_json_to_csv()
    print("Conversion completed successfully. Output saved to 'signals_output.csv'")
except FileNotFoundError:
    print("Error: processed_signals.json file not found in the current directory")
except json.JSONDecodeError:
    print("Error: Invalid JSON format in the input file")
except Exception as e:
    print(f"An error occurred: {str(e)}")