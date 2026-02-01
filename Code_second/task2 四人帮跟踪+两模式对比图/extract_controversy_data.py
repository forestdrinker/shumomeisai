"""
提取四个争议性选手的后验分布数据到CSV
从 .npz 文件中提取placements数组并展开成表格格式
"""

import numpy as np
import pandas as pd
import os

# 配置
FOCAL_CONTESTANTS = {
    'Jerry Rice': {'season': 2, 'actual_placement': 2},
    'Billy Ray Cyrus': {'season': 4, 'actual_placement': 5},
    'Bristol Palin': {'season': 11, 'actual_placement': 3},
    'Bobby Bones': {'season': 27, 'actual_placement': 1},
}

RULES = ['rank', 'percent', 'rank_save', 'percent_save']

# 数据目录
DATA_DIR = r'd:\shumomeisai\Code_second\Results\replay_results'
PANEL_PATH = r'd:\shumomeisai\Code_second\processed\panel.csv'

def main():
    # 加载panel数据用于匹配选手名字
    df_panel = pd.read_csv(PANEL_PATH)
    pair_name_map = df_panel.set_index(['season', 'pair_id'])['celebrity_name'].to_dict()
    
    all_data = []
    
    for name, info in FOCAL_CONTESTANTS.items():
        season = info['season']
        actual = info['actual_placement']
        
        print(f"\n处理: {name} (Season {season})")
        
        for rule in RULES:
            fpath = os.path.join(DATA_DIR, f"season_{season}_{rule}.npz")
            
            if not os.path.exists(fpath):
                print(f"  警告: {fpath} 不存在")
                continue
            
            # 加载数据
            data = np.load(fpath, allow_pickle=True)
            placements = data['placements']  # (R, N) - R个后验样本，N个选手
            pair_ids = data['pair_ids']
            
            print(f"  {rule}: placements shape = {placements.shape}")
            
            # 查找该选手的列索引
            target_col = None
            for col_idx, pid in enumerate(pair_ids):
                celeb_name = pair_name_map.get((season, pid), '')
                if name.lower() in celeb_name.lower() or celeb_name.lower() in name.lower():
                    target_col = col_idx
                    print(f"    找到匹配: pair_id={pid}, name={celeb_name}")
                    break
            
            if target_col is None:
                print(f"  警告: 无法找到 {name} 在 season {season}")
                continue
            
            # 提取该选手的placement分布
            contestant_placements = placements[:, target_col]
            
            # 添加到数据列表
            for sample_idx, placement in enumerate(contestant_placements):
                all_data.append({
                    'contestant': name,
                    'season': season,
                    'actual_placement': actual,
                    'rule': rule,
                    'sample_id': sample_idx,
                    'posterior_placement': placement
                })
    
    # 创建DataFrame并保存
    df = pd.DataFrame(all_data)
    
    # 按选手、规则、样本排序
    df = df.sort_values(['contestant', 'rule', 'sample_id']).reset_index(drop=True)
    
    output_path = 'controversy_portraits_data.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✅ 数据已保存到: {output_path}")
    print(f"   总行数: {len(df)}")
    print(f"\n前10行预览:")
    print(df.head(10))
    
    # 打印统计摘要
    print("\n" + "="*70)
    print("统计摘要")
    print("="*70)
    for name in FOCAL_CONTESTANTS.keys():
        print(f"\n{name}:")
        df_contestant = df[df['contestant'] == name]
        for rule in RULES:
            df_rule = df_contestant[df_contestant['rule'] == rule]
            if len(df_rule) > 0:
                placements = df_rule['posterior_placement'].values
                mean_p = placements.mean()
                std_p = placements.std()
                p_win = (placements == 1).mean()
                p_top3 = (placements <= 3).mean()
                print(f"  {rule:15s}: μ={mean_p:.2f}±{std_p:.2f}, P(Win)={p_win:.1%}, P(Top3)={p_top3:.1%}")

if __name__ == '__main__':
    main()
