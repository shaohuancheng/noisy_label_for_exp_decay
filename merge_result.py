"""
-*- coding: utf-8 -*-
2023/11/2 12:30 merge_result.py
"""

# 用于合并某一文件夹下的所有seed*.csv的结果文件

import os
import sys
import pandas as pd


def merge_csv_files(path):
    # 确保指定的路径存在
    if not os.path.exists(path):
        print(f"指定的路径 '{path}' 不存在")
        return

    # 列出目录下的所有文件
    files = [f for f in os.listdir(path) if f.startswith("seed") and f.endswith(".csv")]  # f.startswith("seed") and

    if not files:
        print("未找到匹配的CSV文件")
        return

    # 创建一个空的DataFrame，用于存储合并后的数据
    merged_df = pd.DataFrame()

    # 逐个读取CSV文件并合并到DataFrame
    for file in files:
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path)
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    # 计算所有行的平均值并添加到最后一行
    avg_row = merged_df.mean()
    # avg_row["YourColumnName"] = "Average"  # 可以修改列名
    merged_df = merged_df.append(avg_row, ignore_index=True)

    # 保存合并后的DataFrame为一个新的CSV文件
    merged_file_path = os.path.join(path, "result.csv")
    merged_df.to_csv(merged_file_path, index=False)

    print(f"合并完成，结果保存在 '{merged_file_path}'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_csv.py <path>")
    else:
        path = sys.argv[1]
        merge_csv_files(path)
