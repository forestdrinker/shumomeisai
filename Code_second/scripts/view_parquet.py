#!/usr/bin/env python
"""
查看 parquet 文件的脚本
使用方法: python view_parquet.py
"""

try:
    import pandas as pd

    # 读取 parquet 文件
    df = pd.read_parquet('panel.parquet')

    print("=" * 80)
    print("📊 Panel.parquet 文件信息")
    print("=" * 80)

    print(f"\n✅ 数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")

    print(f"\n📋 列名:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")

    print(f"\n📈 数据类型:")
    print(df.dtypes)

    print(f"\n🔍 前 10 行数据:")
    print(df.head(10).to_string())

    print(f"\n📊 统计摘要:")
    print(df.describe())

    print(f"\n✨ 缺失值统计:")
    print(df.isnull().sum())

except ImportError:
    print("❌ 错误: 需要安装 pandas 和 pyarrow")
    print("\n请运行: pip install pandas pyarrow")
except FileNotFoundError:
    print("❌ 错误: 找不到 panel.parquet 文件")
except Exception as e:
    print(f"❌ 错误: {e}")
