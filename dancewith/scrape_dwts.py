import requests
import pandas as pd
import os

url = "https://en.wikipedia.org/wiki/Dancing_with_the_Stars_(American_TV_series)_season_12"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    # Parse tables
    dfs = pd.read_html(response.text)
    
    print(f"Found {len(dfs)} tables.")
    
    # Save all found tables to CSVs for inspection
    output_dir = "dwts_season_12_data"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, df in enumerate(dfs):
        csv_path = os.path.join(output_dir, f"table_{i}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved table {i} to {csv_path}")
        
    print("Scraping complete.")

except Exception as e:
    print(f"An error occurred: {e}")
