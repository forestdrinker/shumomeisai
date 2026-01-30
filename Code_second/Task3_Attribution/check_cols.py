
import pandas as pd
try:
    df = pd.read_parquet(r'd:\shumomeisai\Code_second\Data\panel.parquet')
    print("Columns:", list(df.columns))
    print("Example Row:\n", df.iloc[0])
except Exception as e:
    print("Error:", e)
