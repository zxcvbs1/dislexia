import pandas as pd

# Read the CSV file
df = pd.read_csv('signals.csv')

def fix_leverage(leverage):
    leverage = int(leverage)
    # If number has 3 digits, remove the last one by integer division by 10
    if leverage >= 100:
        return leverage // 10
    return leverage

# Apply the fix to the leverage column
df['leverage'] = df['leverage'].apply(fix_leverage)

# Save the corrected data
df.to_csv('signals_fixed.csv', index=False)

print("\nFixed data:")
print(df)