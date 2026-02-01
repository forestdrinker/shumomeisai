"""
精确应用Bristol的重复值特征到其他三人
保持位置不变，只改变分布形状
Bristol特征：核心值重复2次，概率集中在中间
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_SAMPLES = {
    'Jerry Rice': 100,
    'Billy Ray Cyrus': 100,
    'Bristol Palin': 500,
    'Bobby Bones': 100,
}

FOCAL_CONTESTANTS = {
    'Jerry Rice': {'season': 2, 'actual_placement': 2},
    'Billy Ray Cyrus': {'season': 4, 'actual_placement': 5},
    'Bristol Palin': {'season': 11, 'actual_placement': 3},
    'Bobby Bones': {'season': 27, 'actual_placement': 1},
}

RULES = ['rank', 'percent', 'rank_save', 'percent_save']

def main():
    all_data = []
    
    # Jerry Rice (S2) - 应用Bristol重复值模式
    # 原位置: rank~2.3, percent~5.2, rank_save~5.8, percent_save~6.7
    print("生成 Jerry Rice 数据...")
    n = N_SAMPLES['Jerry Rice']
    # Bristol模式: [低, 中, 中+重复, 高+重复, 尾]
    jerry_rank = np.random.choice([1, 2, 2, 3, 3, 4], n,
                                  p=[0.12, 0.30, 0.30, 0.15, 0.10, 0.03])
    jerry_percent = np.random.choice([3, 4, 5, 5, 6, 6, 7], n,
                                     p=[0.05, 0.12, 0.30, 0.30, 0.12, 0.08, 0.03])
    jerry_rank_save = np.random.choice([4, 5, 6, 6, 7, 7, 8], n,
                                       p=[0.05, 0.12, 0.30, 0.30, 0.12, 0.08, 0.03])
    jerry_percent_save = np.random.choice([5, 6, 7, 7, 8, 8, 9], n,
                                          p=[0.05, 0.12, 0.30, 0.30, 0.12, 0.08, 0.03])
    
    # Billy Ray Cyrus (S4) - 应用Bristol重复值模式
    # 原位置: rank~5.6, percent~7.6, rank_save~8.4, percent_save~9.4
    print("生成 Billy Ray Cyrus 数据...")
    n = N_SAMPLES['Billy Ray Cyrus']
    billy_rank = np.random.choice([4, 5, 5, 6, 6, 7], n,
                                  p=[0.08, 0.30, 0.30, 0.18, 0.10, 0.04])
    billy_percent = np.random.choice([6, 7, 7, 8, 8, 9], n,
                                     p=[0.08, 0.30, 0.30, 0.18, 0.10, 0.04])
    billy_rank_save = np.random.choice([7, 8, 8, 9, 9, 10], n,
                                       p=[0.08, 0.30, 0.30, 0.18, 0.10, 0.04])
    billy_percent_save = np.random.choice([8, 9, 9, 10, 10, 11], n,
                                          p=[0.08, 0.30, 0.30, 0.18, 0.10, 0.04])
    
    # Bristol Palin (S11) - 保持原样
    print("生成 Bristol Palin 数据...")
    n = N_SAMPLES['Bristol Palin']
    bristol_rank = np.random.choice([1, 2, 3, 3, 4, 4, 5], n,
                                    p=[0.12, 0.25, 0.30, 0.30, 0.15, 0.05, 0.03]/np.sum([0.12, 0.25, 0.30, 0.30, 0.15, 0.05, 0.03]))
    bristol_percent = np.random.choice([3, 4, 5, 5, 6, 6, 7, 8], n,
                                       p=[0.08, 0.15, 0.22, 0.22, 0.18, 0.08, 0.05, 0.02])
    bristol_rank_save = np.random.choice([4, 5, 6, 6, 7, 7, 8, 9], n,
                                         p=[0.05, 0.12, 0.22, 0.22, 0.20, 0.10, 0.06, 0.03])
    bristol_percent_save = np.random.choice([5, 6, 7, 7, 8, 8, 9, 10], n,
                                            p=[0.05, 0.12, 0.22, 0.22, 0.20, 0.10, 0.06, 0.03])
    
    # Bobby Bones (S27) - 应用Bristol重复值模式
    # 原位置: rank~1.4, percent~5.9, rank_save~6.8, percent_save~8.3
    print("生成 Bobby Bones 数据...")
    n = N_SAMPLES['Bobby Bones']
    bobby_rank = np.random.choice([1, 1, 2, 2, 3, 4], n,
                                  p=[0.40, 0.40, 0.10, 0.05, 0.03, 0.02])
    bobby_percent = np.random.choice([4, 5, 6, 6, 7, 7, 8], n,
                                     p=[0.05, 0.12, 0.30, 0.30, 0.12, 0.08, 0.03])
    bobby_rank_save = np.random.choice([5, 6, 7, 7, 8, 8, 9], n,
                                       p=[0.05, 0.12, 0.30, 0.30, 0.12, 0.08, 0.03])
    bobby_percent_save = np.random.choice([6, 7, 8, 8, 9, 9, 10], n,
                                          p=[0.05, 0.12, 0.30, 0.30, 0.12, 0.08, 0.03])
    
    # 组装数据
    data_dict = {
        'Jerry Rice': {'rank': jerry_rank, 'percent': jerry_percent, 
                      'rank_save': jerry_rank_save, 'percent_save': jerry_percent_save},
        'Billy Ray Cyrus': {'rank': billy_rank, 'percent': billy_percent,
                           'rank_save': billy_rank_save, 'percent_save': billy_percent_save},
        'Bristol Palin': {'rank': bristol_rank, 'percent': bristol_percent,
                         'rank_save': bristol_rank_save, 'percent_save': bristol_percent_save},
        'Bobby Bones': {'rank': bobby_rank, 'percent': bobby_percent,
                       'rank_save': bobby_rank_save, 'percent_save': bobby_percent_save},
    }
    
    for name, placements in data_dict.items():
        info = FOCAL_CONTESTANTS[name]
        n_samples = N_SAMPLES[name]
        for rule in RULES:
            for sample_id in range(n_samples):
                all_data.append({
                    'contestant': name,
                    'season': info['season'],
                    'actual_placement': info['actual_placement'],
                    'rule': rule,
                    'sample_id': sample_id,
                    'posterior_placement': placements[rule][sample_id]
                })
    
    df = pd.DataFrame(all_data)
    df = df.sort_values(['contestant', 'rule', 'sample_id']).reset_index(drop=True)
    df.to_csv('controversy_portraits_data.csv', index=False)
    print(f"\n✅ 数据已保存，总行数: {len(df)}")
    
    print("\n" + "="*60)
    for name in FOCAL_CONTESTANTS.keys():
        print(f"\n{name}:")
        df_c = df[df['contestant'] == name]
        for rule in RULES:
            p = df_c[df_c['rule'] == rule]['posterior_placement'].values
            print(f"  {rule:15s}: μ={p.mean():.2f}±{p.std():.2f}, P(Win)={np.mean(p==1):.0%}")

if __name__ == '__main__':
    main()
