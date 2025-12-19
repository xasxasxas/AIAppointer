import pandas as pd
import re
import sys
import os

sys.path.append(os.getcwd())
from src.data_processor import DataProcessor

def analyze_units():
    print("Loading Data...")
    df = pd.read_csv('data/hr_star_trek_v4c_modernized_clean_modified_v4.csv')
    dp = DataProcessor()
    df_trans = dp.create_transition_dataset(df)
    
    # Heuristic to extract unit
    def extract_unit(role):
        role = str(role)
        # Common prefixes
        prefixes = ['USS', 'SS', 'RS', 'CS', 'ISS', 'Starbase', 'Dockyard', 'Field HQ']
        
        for p in prefixes:
            if p in role:
                # Extract "USS Name"
                # Regex: prefix + space + word
                match = re.search(f"({p}\s+\w+)", role)
                if match:
                    return match.group(1).strip()
                # Or just split?
                parts = role.split(p)
                if len(parts) > 1:
                    # Take the first word after prefix?
                    # "Div Officer USS Vanguard" -> " Vanguard"
                    # "Starbase 12" -> " 12"
                    suffix = parts[1].strip()
                    first_word = suffix.split(' ')[0]
                    return f"{p} {first_word}".strip()
        
        # If no prefix, check for "Wing", "Group"
        if "Wing" in role: return "Wing " + role.split("Wing")[1].strip().split(' ')[0]
        
        return "Unknown"

    units = df_trans['Target_Next_Role_Raw'].apply(extract_unit)
    
    print(f"\nTotal Transitions: {len(units)}")
    print(f"Unique Units: {units.nunique()}")
    print("\nTop 20 Units:")
    print(units.value_counts().head(20))
    
    # Check "Stay Probability"
    stay_count = 0
    valid_trans = 0
    
    for idx, row in df_trans.iterrows():
        prev = extract_unit(row['last_role_title']) # Assuming extracted by fe? No, need to re-extract
        curr = extract_unit(row['Target_Next_Role_Raw'])
        
        if prev != "Unknown" and curr != "Unknown":
            valid_trans += 1
            if prev == curr:
                stay_count += 1
                
    print(f"\nUnit Stay Probability: {stay_count/valid_trans:.2%}")

if __name__ == "__main__":
    analyze_units()
