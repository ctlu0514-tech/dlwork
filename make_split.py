import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold

# 1. 配置路径
csv_path = '/nfs/zc1/qianliexian/dataset/qianliexian_clinical_isup.csv'
save_path = './dataset_split.csv'  # 生成的名单保存在当前目录

# 2. 读取数据
df = pd.read_csv(csv_path)

# 假设你的ID列叫 'id'，标签列叫 'isup2' (二分类标签)
# 如果是其他名字请修改
ids = df['id'].values
labels = df['isup2'].values 

# 3. 创建 Stratified K-Fold (分层随机抽样，保证训练/测试集的正负样本比例一致)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4. 分配 Fold 编号 (0, 1, 2, 3, 4)
df['fold'] = -1
for fold_idx, (train_index, test_index) in enumerate(skf.split(ids, labels)):
    # 在这一折里被当作“测试集”的样本，标记为该折编号
    # 例如：标记为 0 的样本，就是第 0 折的测试集
    df.loc[test_index, 'fold'] = fold_idx

# 5. 保存名单
# 只保存 id 和 fold 即可，其他列不需要
df_split = df[['id', 'fold']]
df_split.to_csv(save_path, index=False)

print(f"名单已生成！共 {len(df_split)} 个病人。")
print(df_split.head())
print(f"文件已保存至: {save_path}")