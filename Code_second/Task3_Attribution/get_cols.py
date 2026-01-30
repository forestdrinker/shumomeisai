
import pandas as pd
try:
    df = pd.read_csv(r'd:\shumomeisai\Code_second\Data\2026_MCM_Problem_C_Data.csv', encoding='ISO-8859-1') # Try latin1
    with open(r'd:\shumomeisai\Code_second\Task3_Attribution\cols.txt', 'w') as f:
        for c in df.columns:
            f.write(c + '\n')
except:
    # Try utf-8
    df = pd.read_csv(r'd:\shumomeisai\Code_second\Data\2026_MCM_Problem_C_Data.csv', encoding='utf-8')
    with open(r'd:\shumomeisai\Code_second\Task3_Attribution\cols.txt', 'w') as f:
        for c in df.columns:
            f.write(c + '\n')
