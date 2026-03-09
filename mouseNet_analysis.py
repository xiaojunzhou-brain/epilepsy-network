import os
import numpy as np
import networkx as nx
from tools.network import networkAnalysis as netAna
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from scipy.stats import linregress

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

plt.figure()
adjacent_matrix = matrix.copy()
adjacent_matrix[adjacent_matrix==0] = np.nan
cmap = plt.colormaps['winter'].copy()
cmap.set_bad(color='white')
plt.matshow(adjacent_matrix, cmap=cmap)
plt.xticks([cortex[1] - 0.5 for cortex in cortices[:-1]], [cortex[1] for cortex in cortices[:-1]],
           rotation="vertical")
plt.yticks([cortex[1] - 0.5 for cortex in cortices[:-1]], [cortex[1] for cortex in cortices[:-1]])
plt.ylabel("From")
plt.xlabel("To")
plt.gca().xaxis.set_label_position("top")
plt.grid()
plt.colorbar(ticks=[1, 2, 3])
plt.gcf().set_size_inches(4.5, 4.5)
plt.title('Mouse connectome matrix', fontsize=15)
figname = os.path.join('data/figures/mouse_net/' + 'net.eps')
plt.savefig(figname)
plt.close()


# --------------------------------------------------------------------------------------

matrix_b = np.zeros_like(matrix)
matrix_b[matrix != 0] = 1
dist = []
cluster = []
c_matrix = np.zeros((len(regions), len(regions)))
c_matrix_display = np.zeros_like(matrix)

graph_b = nx.from_numpy_array(matrix_b)
dist_b = round(netAna(graph_b).MSPL(), 3)
cluster_b = round(netAna(graph_b).ACC(), 3)
print('binary matrix MSPL: ', dist_b)
print('binary matrix ACC: ', cluster_b)

for i, region in enumerate(regions):
    start_i = cortices[i][0]
    end_i = cortices[i][1]
    for j, region2 in enumerate(regions):
        start_j = cortices[j][0]
        end_j = cortices[j][1]
        c_matrix[i, j] = np.mean(matrix_b[start_i:end_i, start_j:end_j])
        c_matrix_display[start_i:end_i, start_j:end_j] = c_matrix[i, j]
    region_graph = nx.from_numpy_array(matrix_b[start_i:end_i, start_i:end_i])
    if not nx.is_connected(region_graph):
        print(region, ' is not connected')
        print('***********************')
        continue
    else:
        d = round(netAna(region_graph).MSPL(), 3)
        c = round(netAna(region_graph).ACC(), 3)
        # c_in = round(c_in, 3)
        # c_out = round(c_out, 3)
        print(region, ':')
        print('MSPL: ', d)
        print('cluster: ', c)
        print('************************')
        dist.append(d)
        cluster.append(c)

c_graph = nx.from_numpy_array(c_matrix)
c_dist = round(netAna(c_graph).MSPL(), 3)
c_cluster = round(netAna(c_graph).ACC(), 3)
print('convergence matrix MSPL: ', c_dist)
print('convergence matrix ACC: ', c_cluster)


# --------------------------------------------------------------------------------------

degrees_in = np.sum(matrix, axis=0)
SLEre_degrees_in = np.sum(matrix[70:77, :], axis=0)
degrees_out = np.sum(matrix, axis=1)
SLEre_degrees_out = np.zeros(matrix.shape[0])
SLEre_degrees_out[70:77] = np.sum(matrix[70:77, :], axis=1)

plt.figure(figsize=(12, 10))
plt.subplot(2, 1, 1)
plt.bar(range(len(degrees_in)), degrees_in, color='dodgerblue', alpha=0.7, label='Original Degrees')
plt.bar(range(len(degrees_in)), SLEre_degrees_in, color='lightcoral', alpha=0.7, label='SLE Degrees')
plt.xticks([cortex[1] - 0.5 for cortex in cortices[:-1]], [cortex[1] for cortex in cortices[:-1]],
           rotation="vertical")
plt.ylabel('Degree', fontsize=15)
plt.xticks(fontsize=15)  # 调整横轴刻度的大小
plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
plt.legend(fontsize=15, loc='upper right')
plt.title('In-degrees statistics(weight)', fontsize=18)
plt.subplot(2, 1, 2)
plt.bar(range(len(degrees_out)), degrees_out, color='dodgerblue', alpha=0.7, label='Original Degrees')
plt.bar(range(len(degrees_out)), SLEre_degrees_out, color='lightcoral', alpha=0.7, label='SLE Degrees')
plt.xticks([cortex[1] - 0.5 for cortex in cortices[:-1]], [cortex[1] for cortex in cortices[:-1]],
           rotation="vertical")
plt.xlabel('Channel', fontsize=15)
plt.ylabel('Degree', fontsize=15)
plt.legend(fontsize=15, loc='upper right')
plt.xticks(fontsize=15)  # 调整横轴刻度的大小
plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
plt.title('Out-degrees statistics(weight)', fontsize=18)
figname = os.path.join('data/figures/mouse_net/' + 'statistics.eps')
plt.savefig(figname)
#
# plt.figure(figsize=(12, 10))
# plt.subplot(2, 1, 1)
# degrees_in = np.sum(matrix, axis=0)
# SLEre_degrees_in = matrix[70, :]
# plt.bar(range(len(degrees_in)), degrees_in, color='dodgerblue', alpha=0.7, label='Original Degrees')
# plt.bar(range(len(degrees_in)), SLEre_degrees_in, color='lightcoral', alpha=0.7, label='SLE Degrees')
# plt.xticks([cortex[1] - 0.5 for cortex in cortices[:-1]], [cortex[1] for cortex in cortices[:-1]],
#            rotation="vertical")
# plt.ylabel('Degree')
# plt.legend()
# plt.title('Node In Degrees(70)')
# plt.subplot(2, 1, 2)
# SLEre_degrees_in = matrix[76, :]
# plt.bar(range(len(degrees_in)), degrees_in, color='dodgerblue', alpha=0.7, label='Original Degrees')
# plt.bar(range(len(degrees_in)), SLEre_degrees_in, color='lightcoral', alpha=0.7, label='SLE Degrees')
# plt.xticks([cortex[1] - 0.5 for cortex in cortices[:-1]], [cortex[1] for cortex in cortices[:-1]],
#            rotation="vertical")
# plt.xlabel('Channel')
# plt.ylabel('Degree')
# plt.legend()
# plt.title('Node Out Degrees(76)')
# figname = os.path.join('data/figures/mouse_net/' + '70_76_distribution.eps')
# plt.savefig(figname)
#
# plt.figure()
# plt.matshow(matrix_b, cmap='binary')
# plt.xticks([cortex[1] - 0.5 for cortex in cortices[:-1]], [cortex[1] for cortex in cortices[:-1]],
#            rotation="vertical")
# plt.yticks([cortex[1] - 0.5 for cortex in cortices[:-1]], [cortex[1] for cortex in cortices[:-1]])
# plt.ylabel("From")
# plt.xlabel("To")
# plt.gca().xaxis.set_label_position("top")
# plt.grid()
# plt.gcf().set_size_inches(4.5, 4.5)
# plt.title('mouse binary connectome', fontsize=10)
# figname = os.path.join('data/figures/mouse_net/' + 'binary_net.eps')
# plt.savefig(figname)
#

degrees_in_b = np.sum(matrix_b, axis=0)
SLEre_degrees_in_b = np.sum(matrix_b[70:77, :], axis=0)
degrees_out_b = np.sum(matrix_b, axis=1)
SLEre_degrees_out_b = np.zeros(matrix_b.shape[0])
SLEre_degrees_out_b[70:77] = np.sum(matrix_b[70:77, :], axis=1)

plt.figure(figsize=(12, 10))
plt.subplot(2, 1, 1)
plt.bar(range(len(degrees_in_b)), degrees_in_b, color='dodgerblue', alpha=0.7, label='Original Degrees')
plt.bar(range(len(degrees_in_b)), SLEre_degrees_in_b, color='lightcoral', alpha=0.7, label='SLE Degrees')
plt.xticks([cortex[1] - 0.5 for cortex in cortices[:-1]], [cortex[1] for cortex in cortices[:-1]],
           rotation="vertical")
plt.ylabel('Degree', fontsize=15)
plt.legend(fontsize=14, loc='upper right')
plt.xticks(fontsize=12)  # 调整横轴刻度的大小
plt.yticks(fontsize=12)  # 调整纵轴刻度的大小
plt.title('In-degrees statistics(binary)', fontsize=18)
plt.subplot(2, 1, 2)
plt.bar(range(len(degrees_out_b)), degrees_out_b, color='dodgerblue', alpha=0.7, label='Original Degrees')
plt.bar(range(len(degrees_out_b)), SLEre_degrees_out_b, color='lightcoral', alpha=0.7, label='SLE Degrees')
plt.xticks([cortex[1] - 0.5 for cortex in cortices[:-1]], [cortex[1] for cortex in cortices[:-1]],
           rotation="vertical")
plt.xlabel('Channel', fontsize=15)
plt.ylabel('Degree', fontsize=15)
plt.xticks(fontsize=15)  # 调整横轴刻度的大小
plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
plt.legend(fontsize=15, loc='upper right')
plt.title('Out-degrees statistics(binary)', fontsize=18)
figname = os.path.join('data/figures/mouse_net/' + 'b_statistics.eps')
plt.savefig(figname)
plt.close()

# --------------------------------------------------------------------------------------

out_sort = sorted(degrees_out, reverse=False)
out_sort_b = sorted(degrees_out_b, reverse=False)
in_sort = sorted(degrees_in, reverse=False)
in_sort_b = sorted(degrees_in_b, reverse=False)

plt.figure()
out_sort_count = np.bincount(out_sort)
out_sort_degrees = np.arange(len(out_sort_count))
out_count = np.sum(out_sort_count)
out_sort_prob = out_sort_count / out_count
plt.bar(out_sort_degrees, out_sort_prob, color='dodgerblue', alpha=0.7)
plt.xlabel('x', fontsize=15)
plt.ylabel('p(x)', fontsize=15)
plt.xticks(fontsize=15)  # 调整横轴刻度的大小
plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
plt.title('Out-degrees distribution(weight)', fontsize=18)
figname = os.path.join('data/figures/mouse_net/' + 'out_degrees.eps')
plt.savefig(figname)

plt.figure()
x = out_sort_degrees[out_sort_count > 0]
y = out_sort_count[out_sort_count > 0]
log_x = np.log10(x + 1e-10)
log_y = np.log10(y)
slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
print(f'slope: {slope}, intercept: {intercept}, R-squared: {r_value**2}')
eq_text = f'$p(x)={slope}x + {intercept}, R^2={r_value**2}$'
plt.figure()
plt.scatter(log_x, log_y, color='dodgerblue')
plt.plot(log_x, intercept + slope*log_x, 'r', linestyle='--', label=eq_text)
plt.title('Out-degrees log fit(weight)', fontsize=18)
plt.xlabel('log p(x)', fontsize=15)
plt.ylabel('log x', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
figname = os.path.join('data/figures/mouse_net/' + 'out_fit.eps')
plt.savefig(figname)

plt.figure()
out_sort_b_count = np.bincount(out_sort_b)
out_sort_b_degrees = np.arange(len(out_sort_b_count))
out_b_count = np.sum(out_sort_b_count)
out_sort_b_prob = out_sort_b_count / out_b_count
plt.bar(out_sort_b_degrees, out_sort_b_prob, color='dodgerblue', alpha=0.7)
plt.xlabel('x', fontsize=15)
plt.ylabel('p(x)', fontsize=15)
plt.xticks(fontsize=15)  # 调整横轴刻度的大小
plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
plt.title('Out-degrees distribution(binary)', fontsize=18)
figname = os.path.join('data/figures/mouse_net/' + 'out_degrees_b.eps')
plt.savefig(figname)

plt.figure()
x_b = out_sort_b_degrees[out_sort_b_count > 0]
y_b = out_sort_b_count[out_sort_b_count > 0]
log_x_b = np.log10(x_b + 1e-10)
log_y_b = np.log10(y_b)
slope_b, intercept_b, r_value_b, p_value_b, std_err_b = linregress(log_x_b, log_y_b)
print(f'slope: {slope_b}, intercept: {intercept_b}, R-squared: {r_value_b**2}')
eq_text_b = f'$p(x)={slope_b}x + {intercept_b}, R^2={r_value_b**2}$'
plt.figure()
plt.scatter(log_x_b, log_y_b, color='dodgerblue')
plt.plot(log_x_b, intercept_b + slope_b*log_x_b, 'r', linestyle='--', label=eq_text_b)
plt.title('Out-degrees log fit(binary)', fontsize=18)
plt.xlabel('log p(x)', fontsize=15)
plt.ylabel('x', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
figname = os.path.join('data/figures/mouse_net/' + 'out_b_fit.eps')
plt.savefig(figname)

plt.figure()
in_sort_count = np.bincount(in_sort)
in_sort_degrees = np.arange(len(in_sort_count))
in_count = np.sum(in_sort_count)
in_sort_prob = in_sort_count / in_count
plt.bar(in_sort_degrees, in_sort_prob, color='dodgerblue', alpha=0.7)
plt.xlabel('x', fontsize=15)
plt.ylabel('p(x)', fontsize=15)
plt.xticks(fontsize=15)  # 调整横轴刻度的大小
plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
plt.title('In-degrees distribution(weight)', fontsize=18)
figname = os.path.join('data/figures/mouse_net/' + 'in_degrees.eps')
plt.savefig(figname)

plt.figure()
in_sort_b_count = np.bincount(in_sort_b)
in_sort_b_degrees = np.arange(len(in_sort_b_count))
in_b_count = np.sum(in_sort_b_count)
in_sort_b_prob = in_sort_b_count / in_b_count
plt.bar(in_sort_b_degrees, in_sort_b_prob, color='dodgerblue', alpha=0.7)
plt.xlabel('x', fontsize=15)
plt.ylabel('p(x)', fontsize=15)
plt.xticks(fontsize=15)  # 调整横轴刻度的大小
plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
plt.title('In-degrees distribution(binary)', fontsize=18)
figname = os.path.join('data/figures/mouse_net/' + 'in_degrees_b.eps')
plt.savefig(figname)
plt.close()

# --------------------------------------------------------------------------------------

HPF_local = np.sum(matrix[70:81, 70:81], axis=1)
HPF_local_b = np.sum(matrix_b[70:81, 70:81], axis=1)
HPF_global = np.sum(matrix[70:81, :], axis=1)
HPF_global_b = np.sum(matrix_b[70:81, :], axis=1)
HPF_local_sort = sorted(HPF_local, reverse=False)
HPF_local_sort_b = sorted(HPF_local_b, reverse=False)
HPF_global_sort = sorted(HPF_global, reverse=False)
HPF_global_sort_b = sorted(HPF_global_b, reverse=False)

plt.figure()
plt.bar(range(len(HPF_local_sort)), HPF_local_sort[::-1], color='dodgerblue', alpha=0.7)
plt.xlabel('Channel', fontsize=15)
plt.ylabel('Out-degrees', fontsize=15)
plt.xticks(fontsize=15)  # 调整横轴刻度的大小
plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
plt.title('HPF Order(local weight)', fontsize=18)
figname = os.path.join('data/figures/mouse_net/' + 'HPF_sort_local.eps')
plt.savefig(figname)

plt.figure()
plt.bar(range(len(HPF_local_sort_b)), HPF_local_sort_b[::-1], color='dodgerblue', alpha=0.7)
plt.xlabel('Channel', fontsize=15)
plt.ylabel('Out-degrees', fontsize=15)
plt.xticks(fontsize=15)  # 调整横轴刻度的大小
plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
plt.title('HPF Order(local binary)', fontsize=18)
figname = os.path.join('data/figures/mouse_net/' + 'HPF_sort_local_b.eps')
plt.savefig(figname)

plt.figure()
plt.bar(range(len(HPF_global_sort)), HPF_global_sort[::-1], color='dodgerblue', alpha=0.7)
plt.xlabel('Channel', fontsize=15)
plt.ylabel('Out-degrees', fontsize=15)
plt.xticks(fontsize=15)  # 调整横轴刻度的大小
plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
plt.title('HPF Order(global weight)', fontsize=18)
figname = os.path.join('data/figures/mouse_net/' + 'HPF_sort_global.eps')
plt.savefig(figname)

plt.figure()
plt.bar(range(len(HPF_global_sort_b)), HPF_global_sort_b[::-1], color='dodgerblue', alpha=0.7)
plt.xlabel('Channel', fontsize=15)
plt.ylabel('Out-degrees', fontsize=15)
plt.xticks(fontsize=15)  # 调整横轴刻度的大小
plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
plt.title('HPF Order(global binary)', fontsize=18)
figname = os.path.join('data/figures/mouse_net/' + 'HPF_sort_global_b.eps')
plt.savefig(figname)

plt.figure()
plt.bar(range(len(HPF_local)), HPF_local, color='dodgerblue', alpha=0.7)
plt.xlabel('Channel', fontsize=15)
plt.ylabel('Out-degrees', fontsize=15)
plt.xticks([0, 2, 4, 6, 8, 10], ['70', '72', '74', '76', '78', '80'], fontsize=15)  # 调整横轴刻度的大小
plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
plt.title('HPF Statistics(local weight)', fontsize=18)
figname = os.path.join('data/figures/mouse_net/' + 'HPF_stat_local.eps')
plt.savefig(figname)

plt.figure()
HPF_local_count = np.bincount(HPF_local_sort)
HPF_local_degrees = np.arange(len(HPF_local_count))
plt.bar(HPF_local_degrees, HPF_local_count, color='dodgerblue', alpha=0.7)
plt.xlabel('x', fontsize=15)
plt.ylabel('p(x)', fontsize=15)
plt.xticks(fontsize=15)  # 调整横轴刻度的大小
plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
plt.title('HPF Distribution(local weight)', fontsize=18)
figname = os.path.join('data/figures/mouse_net/' + 'HPF_degree_local.eps')
plt.savefig(figname)

plt.figure()
plt.bar(range(len(HPF_local_b)), HPF_local_b, color='dodgerblue', alpha=0.7)
plt.xlabel('Channel', fontsize=15)
plt.ylabel('Out-degrees', fontsize=15)
plt.xticks([0, 2, 4, 6, 8, 10], ['70', '72', '74', '76', '78', '80'], fontsize=15)  # 调整横轴刻度的大小
plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
plt.title('HPF Statistics(local binary)', fontsize=18)
figname = os.path.join('data/figures/mouse_net/' + 'HPF_stat_local_b.eps')
plt.savefig(figname)

plt.figure()
HPF_local_count_b = np.bincount(HPF_local_sort_b)
HPF_local_degrees_b = np.arange(len(HPF_local_count_b))
plt.bar(HPF_local_degrees_b, HPF_local_count_b, color='dodgerblue', alpha=0.7)
plt.xlabel('x', fontsize=15)
plt.ylabel('p(x)', fontsize=15)
plt.xticks(fontsize=15)  # 调整横轴刻度的大小
plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
plt.title('HPF Distribution(local binary)', fontsize=18)
figname = os.path.join('data/figures/mouse_net/' + 'HPF_degree_local_b.eps')
plt.savefig(figname)

plt.figure()
plt.bar(range(len(HPF_global)), HPF_global, color='dodgerblue', alpha=0.7)
plt.xlabel('Channel', fontsize=15)
plt.ylabel('Out-degrees', fontsize=15)
plt.xticks([0, 2, 4, 6, 8, 10], ['70', '72', '74', '76', '78', '80'], fontsize=15)  # 调整横轴刻度的大小
plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
plt.title('HPF Statistics(global weight)', fontsize=18)
figname = os.path.join('data/figures/mouse_net/' + 'HPF_stat_global.eps')
plt.savefig(figname)

HPF_global_count = np.bincount(HPF_global_sort)
HPF_global_degrees = np.arange(len(HPF_global_count))
plt.bar(HPF_global_degrees, HPF_global_count, color='dodgerblue', alpha=0.7)
plt.xlabel('x', fontsize=15)
plt.ylabel('p(x)', fontsize=15)
plt.xticks(fontsize=15)  # 调整横轴刻度的大小
plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
plt.title('HPF Distribution(global weight)', fontsize=18)
figname = os.path.join('data/figures/mouse_net/' + 'HPF_degree_global.eps')
plt.savefig(figname)

plt.figure()
plt.bar(range(len(HPF_global_b)), HPF_global_b, color='dodgerblue', alpha=0.7)
plt.xlabel('Channel', fontsize=15)
plt.ylabel('Out-degrees', fontsize=15)
plt.xticks([0, 2, 4, 6, 8, 10], ['70', '72', '74', '76', '78', '80'], fontsize=15)  # 调整横轴刻度的大小
plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
plt.title('HPF Statistics(global binary)', fontsize=18)
figname = os.path.join('data/figures/mouse_net/' + 'HPF_stat_global_b.eps')
plt.savefig(figname)

HPF_global_count_b = np.bincount(HPF_global_sort_b)
HPF_global_degrees_b = np.arange(len(HPF_global_count_b))
plt.bar(HPF_global_degrees_b, HPF_global_count_b, color='dodgerblue', alpha=0.7)
plt.xlabel('x', fontsize=15)
plt.ylabel('p(x)', fontsize=15)
plt.xticks(fontsize=15)  # 调整横轴刻度的大小
plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
plt.title('HPF Distribution(global binary)', fontsize=18)
figname = os.path.join('data/figures/mouse_net/' + 'HPF_degree_global_b.eps')
plt.savefig(figname)



df_HPF = pd.DataFrame(matrix[70:81, 70:81])
df_HPF_b = pd.DataFrame(matrix_b[70:81, 70:81])
df_HPF_global = pd.DataFrame(HPF_global)
df_HPF_global_b = pd.DataFrame(HPF_global_b)
filename_HPF = os.path.join('connectome', 'HPF.xlsx')
with pd.ExcelWriter(filename_HPF) as writer:
    df_HPF.to_excel(writer, sheet_name='HPF Matrix weight')
    df_HPF_b.to_excel(writer, sheet_name='HPF Matrix Binary')
    df_HPF_global.to_excel(writer, sheet_name='HPF Degrees Weight')
    df_HPF_global_b.to_excel(writer, sheet_name='HPF Degrees Binary')

plt.figure()
HCP_adjacent_matrix = matrix[70:81, 70:81].copy()
HCP_adjacent_matrix[HCP_adjacent_matrix==0] = np.nan
# colors = ['dodgerblue', 'skyblue', 'c']
# cmap = mcolors.ListedColormap(colors)
cmap = plt.colormaps['Blues'].copy()
cmap.set_bad(color='white')
norm = mcolors.Normalize(vmin=0, vmax=3)
plt.matshow(HCP_adjacent_matrix, cmap=cmap, norm=norm)
plt.xticks([0, 4, 6], ['70', '74', '76'])
plt.yticks([0, 4, 6], ['70', '74', '76'])
plt.ylabel("From")
plt.xlabel("To")
plt.gca().xaxis.set_label_position("top")
# plt.grid()
plt.colorbar(ticks=[0, 1, 2, 3])
plt.gcf().set_size_inches(4.5, 4.5)
plt.title('HCP connectome matrix', fontsize=15)
figname = os.path.join('data/figures/mouse_net/' + 'HCP_net.eps')
plt.savefig(figname)
# plt.figure()
# matrix[matrix==0] = np.nan
# plt.matshow(matrix)
# plt.xticks([cortex[1] - 0.5 for cortex in cortices[:-1]], [cortex[1] for cortex in cortices[:-1]],
#            rotation="vertical")
# plt.yticks([cortex[1] - 0.5 for cortex in cortices[:-1]], [cortex[1] for cortex in cortices[:-1]])
# plt.ylabel("From")
# plt.xlabel("To")
# plt.gca().xaxis.set_label_position("top")
# plt.grid()
# plt.colorbar(ticks=[1, 2, 3])
# plt.title('mouse connectome', fontsize=10)
# plt.show()

# plt.figure()
# c_matrix_display[c_matrix_display==0] = np.nan
# plt.matshow(c_matrix_display)
# plt.xticks([cortex[1] - 0.5 for cortex in cortices[:-1]], [cortex[1] for cortex in cortices[:-1]],
#            rotation="vertical")
# plt.yticks([cortex[1] - 0.5 for cortex in cortices[:-1]], [cortex[1] for cortex in cortices[:-1]])
# plt.ylabel("From")
# plt.xlabel("To")
# plt.gca().xaxis.set_label_position("top")
# plt.grid()
# plt.colorbar()
# plt.gcf().set_size_inches(4.5, 4.5)
# plt.title('Global Connectome(mean shortest path:' + str(c_dist) + '  mean cluster:' + str(c_cluster) + ')', fontsize=10)
# figname = os.path.join('data/figures/mouse_net/' + 'globalNet.eps')
# plt.savefig(figname)
