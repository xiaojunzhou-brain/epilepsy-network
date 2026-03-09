import pandas as pd
import numpy as np
import os

# 读取数据
data = pd.read_excel('connectome/mouse.xlsx', header=0, index_col=0, sheet_name=None, engine='openpyxl')
metadate = pd.read_excel('connectome/mouse_meta.xlsx', sheet_name='Voxel Count_295 Structures')

# 只选用参与建模的区域
m = metadate.loc[metadate['Represented in Linear Model Matrix'] == 'Yes']

# 构建 column 顺序和 cortices 切片信息
columns = []
cortices = [[0, 0]]
for region in m['Major Region'].unique():
    acronyms = m.loc[m['Major Region'] == region, 'Acronym'].values
    for acronym in acronyms:
        columns.append(acronym.replace(' ', ''))
    cortices.append([cortices[-1][-1], cortices[-1][-1] + len(acronyms)])
cortices.remove([0, 0])
regions = m['Major Region'].unique()

# 构造矩阵并过滤不显著连接
d = data['W_ipsi']
p = data['PValue_ipsi']
d = d[columns].reindex(columns)
p = p[columns].reindex(columns)
d = d.values
p = p.values
p[np.isnan(p)] = 1
d[p > 0.01] = 0
matrix = np.zeros_like(d)

for i in [1e-4, 1e-2, 1]:
    matrix[d >= i] += 1

# ⬇️ 构建 category dataframe，包括子区域名字（Acronym）、大类（Major Region）和 Node index
categories = []
node_id = 0
for region in m['Major Region'].unique():
    region_rows = m.loc[m['Major Region'] == region]
    for _, row in region_rows.iterrows():
        categories.append({
            'Node': node_id,
            'Category': region,
            'Acronym': row['Acronym'].replace(' ', ''),
            'Full Name': row['Name']
        })
        node_id += 1

# 写入 Excel 多表文件
df_matrix = pd.DataFrame(matrix)
df_categories = pd.DataFrame(categories)

filename = os.path.join('connectome', 'matrix_with_labels.xlsx')
with pd.ExcelWriter(filename) as writer:
    df_matrix.to_excel(writer, sheet_name='Adjacency Matrix', index=False, header=False)
    df_categories.to_excel(writer, sheet_name='Categories', index=False)
