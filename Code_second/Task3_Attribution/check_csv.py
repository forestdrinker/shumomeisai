import pandas as pd
try:
    df = pd.read_csv(r'd:\shumomeisai\Code_second\Data\2026_MCM_Problem_C_Data.csv', encoding='utf-8-sig')
    print("Columns (utf-8-sig):")
    for c in df.columns:
        print(f" - {c}")
except:
    try:
        df = pd.read_csv(r'd:\shumomeisai\Code_second\Data\2026_MCM_Problem_C_Data.csv', encoding='ISO-8859-1')
        print("Columns (latin1):")
        for c in df.columns:
            print(f" - {c}")
    except Exception as e:
        print("Error:", e)
