
import pandas as pd

def check_data_quality(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {file_path}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    if 'celebrity_industry' in df.columns:
        print("\n--- Unique Industry Values ---")
        industries = df['celebrity_industry'].unique()
        for i in sorted([str(x) for x in industries]):
            print(f"'{i}'")
            
        print("\n--- Typos / Inconsistencies Check ---")
        # Check for Beauty Pagent vs Pageant
        typos = [i for i in industries if 'Pagent' in str(i)]
        if typos:
            print(f"FOUND TYPO: {typos}")
        else:
            print("No 'Pagent' typo found.")
            
        # Check for case sensitivity duplicates
        lower_map = {}
        duplicates = []
        for i in industries:
            s = str(i).strip().lower()
            if s in lower_map:
                duplicates.append((lower_map[s], i))
            else:
                lower_map[s] = i
        
        if duplicates:
            print(f"FOUND CASE DUPLICATES: {duplicates}")
        else:
            print("No case sensitivity duplicates found.")

    else:
        print("Column 'celebrity_industry' not found!")

if __name__ == "__main__":
    check_data_quality(r'd:\shumomeisai\Code_second\processed\panel.csv')
