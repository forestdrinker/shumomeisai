
import pandas as pd
import re

csv_path = r'd:\shumomeisai\dancewith\2026_MCM_Problem_C_Data.csv'
xlsx_path = r'd:\shumomeisai\dancewith\DWTS_All_Seasons_1-34.xlsx'

result_file = 'comparison_report.md'

with open(result_file, 'w', encoding='utf-8') as f:
    f.write("# Data File Comparison Report\n\n")

    # Load CSV
    try:
        df_csv = pd.read_csv(csv_path)
        f.write(f"## CSV File: `2026_MCM_Problem_C_Data.csv`\n")
        f.write(f"- Shape: {df_csv.shape}\n")
        f.write(f"- Columns: {len(df_csv.columns)} columns (Detailed scores and placements)\n")
        f.write(f"- Seasons Covered: {sorted(df_csv['season'].unique())}\n\n")
    except Exception as e:
        f.write(f"Error loading CSV: {e}\n")
        df_csv = None

    # Load Excel (Header on row 2, index 2)
    try:
        df_xlsx = pd.read_excel(xlsx_path, header=2)
        f.write(f"## Excel File: `DWTS_All_Seasons_1-34.xlsx`\n")
        f.write(f"- Shape: {df_xlsx.shape}\n")
        f.write(f"- Columns: {df_xlsx.columns.tolist()}\n")
        
        # Clean Season column (S1 -> 1)
        df_xlsx['Season_Num'] = df_xlsx['Season'].astype(str).str.extract(r'(\d+)').astype(int)
        f.write(f"- Seasons Covered: {sorted(df_xlsx['Season_Num'].unique())}\n\n")
    except Exception as e:
        f.write(f"Error loading Excel: {e}\n")
        df_xlsx = None

    if df_csv is not None and df_xlsx is not None:
        f.write("## Detailed Consistency Check\n\n")
        
        # Check Season Coverage
        csv_seasons = set(df_csv['season'].unique())
        xlsx_seasons = set(df_xlsx['Season_Num'].unique())
        
        missing_in_csv = xlsx_seasons - csv_seasons
        missing_in_xlsx = csv_seasons - xlsx_seasons
        
        if missing_in_csv:
            f.write(f"### Seasons missing in CSV:\n- {sorted(list(missing_in_csv))}\n\n")
        if missing_in_xlsx:
            f.write(f"### Seasons missing in Excel:\n- {sorted(list(missing_in_xlsx))}\n\n")
            
        # Check Winners Consistency
        f.write("### Winner Consistency Check\n")
        f.write("| Season | Excel Winner | CSV Winner (Placement=1) | Match? |\n")
        f.write("|---|---|---|---|\n")
        
        matches = 0
        total_checks = 0
        
        for index, row in df_xlsx.iterrows():
            season_num = row['Season_Num']
            excel_winner = row['Winner']
            
            # Find winner in CSV
            csv_winner_row = df_csv[(df_csv['season'] == season_num) & (df_csv['placement'] == 1)]
            
            if not csv_winner_row.empty:
                # CSV formatting: name might be just celebrity or "Celebrity & Pro"
                # Excel formatting: "Celebrity & Pro"
                csv_cel_name = csv_winner_row.iloc[0]['celebrity_name']
                csv_partner = csv_winner_row.iloc[0]['ballroom_partner']
                
                # Normalize excel winner name for loose matching
                # Excel: "Kelly & Alec"
                # CSV: "Kelly Monaco", "Alec Mazo"
                
                cel_first = csv_cel_name.split()[0]
                
                is_match = cel_first in excel_winner
                match_str = "✅" if is_match else "❌"
                if is_match: matches += 1
                
                f.write(f"| {season_num} | {excel_winner} | {csv_cel_name} & {csv_partner} | {match_str} |\n")
                total_checks += 1
            else:
                if season_num in csv_seasons:
                    f.write(f"| {season_num} | {excel_winner} | **No Placement=1 found** | ❌ |\n")
                else:
                    # Season not in CSV, already reported
                    pass

        f.write(f"\n**Consistency Score**: {matches}/{total_checks} matches found for overlapping seasons.\n")

print("Comparison complete. Report generated.")
