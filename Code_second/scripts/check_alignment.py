import numpy as np
import os

T1_PATH = r'd:\shumomeisai\Code_second\Results\posterior_samples\season_1.npz'
T2_PATH = r'd:\shumomeisai\Code_second\Results\replay_results\season_1_rank.npz'

def check():
    if not os.path.exists(T1_PATH) or not os.path.exists(T2_PATH):
        print("Files not found.")
        return

    t1 = np.load(T1_PATH)
    t2 = np.load(T2_PATH)
    
    n1 = t1['v'].shape[0] if 'v' in t1 else t1['v_samples'].shape[0]
    n2 = t2['placements'].shape[0]
    
    print(f"Task 1 Samples: {n1}")
    print(f"Task 2 Samples: {n2}")
    
    if n1 == n2:
        print("ALIGNMENT: LIKELY OK (Counts match)")
    else:
        print("ALIGNMENT: MISMATCH (Subsampling detected)")

if __name__ == '__main__':
    check()
