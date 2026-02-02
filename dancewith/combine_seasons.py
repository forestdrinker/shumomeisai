import pandas as pd
import os

excel_file = r'd:\shumomeisai\dancewith\DWTS_All_Seasons_1-34.xlsx'
xl = pd.ExcelFile(excel_file)
print(f"Sheet names: {xl.sheet_names}")
print(f"Total sheets: {len(xl.sheet_names)}")

# Read all sheets and combine
all_data = []
for sheet in xl.sheet_names:
    df = pd.read_excel(xl, sheet_name=sheet)
    df['Season_Sheet'] = sheet  # Add source sheet name
    all_data.append(df)
    print(f"Sheet '{sheet}': {len(df)} rows, columns: {list(df.columns)}")

# Combine all
combined = pd.concat(all_data, ignore_index=True)
print(f"\nCombined: {len(combined)} total rows, {len(combined.columns)} columns")
print(f"Columns: {list(combined.columns)}")

# Save to CSV
output_path = r'd:\shumomeisai\dancewith\DWTS_All_Seasons_Combined.csv'
combined.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\nSaved to: {output_path}")
