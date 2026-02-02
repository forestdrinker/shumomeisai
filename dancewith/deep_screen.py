
import pandas as pd
import numpy as np
import re

csv_path = r'd:\shumomeisai\dancewith\2026_MCM_Problem_C_Data.csv'
xlsx_path = r'd:\shumomeisai\dancewith\DWTS_All_Seasons_1-34.xlsx'
report_file = 'deep_data_quality_report.md'

def normalize_name(name):
    if not isinstance(name, str):
        return ""
    # Remove accents/special chars for comparison
    name = name.lower()
    name = re.sub(r'[^a-z0-9\s]', '', name)
    return name.strip()

with open(report_file, 'w', encoding='utf-8') as f:
    f.write("# 深度数据质量与差异筛查报告\n\n")
    
    # 1. Load Data
    try:
        df_csv = pd.read_csv(csv_path)
        # Fix column names to be consistent
        df_csv.columns = [c.strip().lower() for c in df_csv.columns]
    except Exception as e:
        f.write(f"CRITICAL: Failed to load CSV: {e}\n")
        exit()
        
    try:
        df_xlsx = pd.read_excel(xlsx_path, header=2)
        df_xlsx['Season_Num'] = df_xlsx['Season'].astype(str).str.extract(r'(\d+)').astype(int)
    except Exception as e:
        f.write(f"CRITICAL: Failed to load Excel: {e}\n")
        exit()

    # 2. Top 3 Consistency Check (Winner, Runner-up, Third Place)
    f.write("## 1. 前三名一致性筛查 (Top 3 Consistency)\n")
    f.write("对比 Excel (汇总表) 与 CSV (详细表) 中的前三名归属。\n\n")
    f.write("| 赛季 | 名次 | Excel 记录 | CSV 记录 | 匹配状态 | 备注 |\n")
    f.write("|---|---|---|---|---|---|\n")

    mismatches = []
    
    for _, row in df_xlsx.iterrows():
        season_num = row['Season_Num']
        
        # Mapping Excel columns to rank
        targets = [
            (1, row['Winner']),
            (2, row['Runner-up']),
            (3, row['Third Place'])
        ]
        
        for rank, excel_name in targets:
            if pd.isna(excel_name):
                continue
                
            # Find in CSV
            csv_record = df_csv[(df_csv['season'] == season_num) & (df_csv['placement'] == rank)]
            
            status = "✅"
            note = ""
            csv_name_display = "❌ Not Found"
            
            if len(csv_record) == 0:
                status = "❌"
                note = "CSV中无此排名的记录"
            else:
                csv_cel = csv_record.iloc[0]['celebrity_name']
                csv_pro = csv_record.iloc[0]['ballroom_partner']
                csv_name_display = f"{csv_cel} & {csv_pro}"
                
                # Loose matching: Check if excel name part is in csv name
                # Excel: "Kelly & Alec"
                # CSV: "Kelly Monaco"
                norm_excel = normalize_name(excel_name)
                norm_csv_cel = normalize_name(csv_cel)
                norm_csv_pro = normalize_name(csv_pro)
                
                # Split excel name by '&'
                parts = [p.strip() for p in excel_name.split('&')]
                if len(parts) >= 1:
                    match_cel = normalize_name(parts[0]) in norm_csv_cel
                    match_pro = False
                    if len(parts) > 1:
                         match_pro = normalize_name(parts[1]) in norm_csv_pro
                    else:
                        match_pro = True # Assume true if not listed
                    
                    if not (match_cel):
                        status = "⚠️"
                        note = "名字拼写不完全匹配"
                        mismatches.append(f"S{season_num} Rank {rank}: {excel_name} vs {csv_name_display}")
            
            f.write(f"| {season_num} | {rank} | {excel_name} | {csv_name_display} | {status} | {note} |\n")

    if not mismatches:
        f.write("\n**结论**: 所有赛季的前三名在模糊匹配下均一致。\n")
    else:
        f.write(f"\n**结论**: 发现 {len(mismatches)} 处潜在差异（详见上表）。\n")

    # 3. CSV Internal Data Quality Check
    f.write("\n## 2. CSV 内部数据质量自查 (Data Quality Check)\n")
    
    # 3.1 Missing Score Values
    score_cols = [c for c in df_csv.columns if 'score' in c]
    f.write("### 3.1 缺失评分检测\n")
    
    missing_scores = df_csv[score_cols].isnull().sum().sum()
    if missing_scores > 0:
        f.write(f"- **总计缺失评分数**: {missing_scores}\n")
        # Breakdown by season
        missing_by_season = df_csv.groupby('season')[score_cols].apply(lambda x: x.isnull().sum().sum())
        missing_by_season = missing_by_season[missing_by_season > 0]
        if not missing_by_season.empty:
            f.write("- **按赛季缺失情况**:\n")
            for s, count in missing_by_season.items():
                f.write(f"  - Season {s}: 缺失 {count} 个评分\n")
    else:
        f.write("- ✅ 所有评分列均无缺失值 (NaN)。\n")

    # 3.2 Score Range Validity
    f.write("\n### 3.2 评分数值合理性检测\n")
    # Usually scores are out of 10. Check for scores > 10 or < 0
    # Note: Some earlier seasons might have scores out of 30 total, but individual judge scores are usually <= 10.
    # Let's check max value per column
    
    invalid_scores = []
    for col in score_cols:
        col_max = df_csv[col].max()
        col_min = df_csv[col].min()
        if col_max > 10:
            invalid_scores.append(f"{col}: Max {col_max}")
        if col_min < 0:
            invalid_scores.append(f"{col}: Min {col_min}")
            
    if invalid_scores:
        f.write("- ⚠️ 发现异常高/低的单项评分 (通常应在0-10之间):\n")
        for inv in invalid_scores:
            f.write(f"  - {inv}\n")
        f.write("  > 注意：如果某些赛季裁判总分作为单列记录，可能会超过10分，需结合业务理解。\n")
    else:
        f.write("- ✅ 所有单项评分均在 0-10 范围内（假设均为单个裁判打分）。\n")
        
    # 3.3 Placement vs Results Consistency
    f.write("\n### 3.3 排名与结果描述的一致性\n")
    # Check if placement 1 has result 'Winner'
    winners = df_csv[df_csv['placement'] == 1]
    non_winner_label = winners[~winners['results'].str.contains('Winner', case=False, na=False)]
    
    if not non_winner_label.empty:
        f.write("- ⚠️ 发现排名第1但 `results` 字段不包含 'Winner' 的记录:\n")
        for _, row in non_winner_label.iterrows():
            f.write(f"  - S{row['season']} {row['celebrity_name']}: Result='{row['results']}'\n")
    else:
        f.write("- ✅ 所有排名第1的选手 `results` 字段均标记为 Winner。\n")

    # 3.4 Duplicate Records
    f.write("\n### 3.4 重复记录检测\n")
    dupes = df_csv.duplicated(subset=['season', 'celebrity_name'], keep=False)
    if dupes.any():
        f.write("- ⚠️ 发现重复的 (Season, Celebrity) 记录:\n")
        f.write(df_csv[dupes][['season', 'celebrity_name']].to_markdown(index=False))
    else:
        f.write("- ✅ 无重复参赛记录。\n")

print("Deep screening complete.")
