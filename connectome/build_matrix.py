import pandas as pd
import numpy as np

# 假设文件路径
file_path = 'budapest_connectome_3.0_209_0_median.csv'  # 替换为实际文件路径
df = pd.read_csv(file_path, delimiter=';')

# 1. 获取父节点标签（父区名称）
parent_labels = pd.unique(df['parent name node1'])

# 2. 创建标签列表：包含 idx 和 parent node name
labels = []
node_to_index = {}  # 标签与索引的映射
index = 0

for parent_name in parent_labels:
    labels.append([index, parent_name])  # 每个父节点
    node_to_index[parent_name] = index  # 保存父节点到索引的映射
    index += 1

# 转换为 DataFrame 方便查看
labels_df = pd.DataFrame(labels, columns=["Idx", "Parent Node Name"])

# 3. 创建权重矩阵：初始化一个大小为 parent node 数量 x parent node 数量的矩阵
n = len(parent_labels)
adj_matrix = np.zeros((n, n))

# 4. 填充权重矩阵：计算每一对父节点之间的连接权重
for _, row in df.iterrows():
    # 获取父节点1和父节点2
    parent_name_1 = row['parent name node1']
    parent_name_2 = row['parent name node2']
    weight = row['edge weight(med nof)']

    # 获取父节点的索引
    idx_1 = node_to_index[parent_name_1]
    idx_2 = node_to_index[parent_name_2]

    # 累加连接权重到矩阵中
    adj_matrix[idx_1, idx_2] += weight
    adj_matrix[idx_2, idx_1] += weight  # 如果是无向连接，反向也要累加

# 5. 将标签保存到文件
labels_df.to_csv('parent_node_labels.csv', index=False)

# 6. 将权重矩阵保存到文件
adj_matrix_df = pd.DataFrame(adj_matrix, index=parent_labels, columns=parent_labels)
adj_matrix_df.to_csv('parent_weight_matrix.csv', index=True)

# 打印结果查看
print("\nParent Node Labels:")
print(labels_df)

print("\nParent Node Weight Matrix:")
print(adj_matrix_df)