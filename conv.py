import pandas as pd
import json
from ast import literal_eval
from datetime import datetime
import math # For checking NaN

# Load CSV
try:
    df = pd.read_csv('data.csv')
except FileNotFoundError:
    print("Error: data.csv not found. Please make sure it's in the same directory.")
    exit()

# Convert timestamp to Unix timestamp (seconds)
# Ensure the 'timestamp' column is parsed as datetime objects first for robust conversion
try:
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
    df['unix_timestamp'] = df['timestamp_dt'].apply(lambda x: int(x.timestamp()))
except KeyError:
    print("Error: 'timestamp' column not found in data.csv.")
    exit()
except Exception as e:
    print(f"Error converting 'timestamp' to datetime: {e}")
    exit()


# Helper function to safely parse string lists
def parse_list_string(s):
    if pd.isna(s) or s == '':
        return []
    try:
        parsed_list = literal_eval(s)
        if isinstance(parsed_list, list):
            # Filter out non-numeric values if any, or handle them as needed
            return [x for x in parsed_list if isinstance(x, (int, float)) and not math.isnan(x)]
        return [] # If not a list after eval, return empty
    except (ValueError, SyntaxError):
        return [] # If parsing fails, return empty list

df['Support_parsed'] = df['Support'].apply(parse_list_string)
df['Resistance_parsed'] = df['Resistance'].apply(parse_list_string)

candles_data = []
markers_data = []
support_bands_data = []
resistance_bands_data = []

for _, row in df.iterrows():
    timestamp = row['unix_timestamp']

    # Add candlestick data
    candles_data.append({
        "time": timestamp,
        "open": row["open"],
        "high": row["high"],
        "low": row["low"],
        "close": row["close"]
    })

    # Add arrow marker based on 'direction'
    # Check for NaN in direction column too
    direction = row["direction"]
    if pd.notna(direction) and direction != '': # Check for non-empty string as well
        if direction == "LONG":
            markers_data.append({
                "time": timestamp,
                "position": "belowBar",
                "color": "green",
                "shape": "arrowUp",
                "text": "L" # Short text for marker
            })
        elif direction == "SHORT":
            markers_data.append({
                "time": timestamp,
                "position": "aboveBar",
                "color": "red",
                "shape": "arrowDown",
                "text": "S" # Short text for marker
            })
        # If direction has other values, no marker is added from here.
        # Could add an 'else' for a default marker if needed.
    else: # Handles NaN, empty strings, or if the cell is truly empty
        markers_data.append({
            "time": timestamp,
            "position": "inBar", # Or 'aboveBar' / 'belowBar' if preferred for neutral
            "color": "yellow",
            "shape": "circle",
            "text": "N" # Short text for Neutral
        })


    # Add support band if valid data exists
    if row["Support_parsed"]: # Check if the list is not empty
        support_bands_data.append({
            "time": timestamp,
            "low": min(row["Support_parsed"]),
            "high": max(row["Support_parsed"])
        })

    # Add resistance band if valid data exists
    if row["Resistance_parsed"]: # Check if the list is not empty
        resistance_bands_data.append({
            "time": timestamp,
            "low": min(row["Resistance_parsed"]),
            "high": max(row["Resistance_parsed"])
        })

# Consolidate into the final JSON structure
output_json = {
    "candles": candles_data,
    "markers": markers_data,
    "support_bands": support_bands_data,
    "resistance_bands": resistance_bands_data
}

# Save to JSON
try:
    with open("data.json", "w") as f:
        json.dump(output_json, f, indent=2)
    print("✅ data.json created successfully.")
except IOError:
    print("Error: Could not write to data.json.")
