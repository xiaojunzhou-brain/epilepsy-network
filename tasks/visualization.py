# coding:utf-8
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
from tools.methods import butter_filter


class base_visualization:
    def __init__(self, config):
        self.config = config
        self.exp = config['exp']
        self.netType = config['netType']
        self.n = config['n']
        self.n_SLE = config['n_SLE']
        self.start = config['start']
        self.net_names = {
            'random': 'Random',
            'small_world': 'Small-world',
            'scale_free': 'Scale-free'
        }

    def make_data_folder(self):
        dataFolder = self.config['figFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def load_data(self, dataIndex, type):
        dataFolder = self.config[dataIndex + 'Folder']
        expFolder = os.path.join(dataFolder, self.exp)
        fileName = os.path.join(expFolder, type + '.npy')
        return np.load(fileName, allow_pickle=True).item()

    def run(self):
        self.make_data_folder()
        for type in self.netType:
            outputData = self.load_data('output', type)
            analysisData = self.load_data('analysis', type)

            # Vol = butter_filter(outputData['Vol'])
            Vol = outputData['Vol']
            T = outputData['T']
            graph = outputData['graph']
            matrix = outputData['matrix']
            centrality = analysisData['centrality']
            freq = analysisData['freq']
            energy = analysisData['energy']
            mean_path = analysisData['mean_path']
            mean_cluster = analysisData['mean_cluster']
            sync = analysisData['sync']
            energy_index = analysisData['energy_index']

            index = np.zeros(self.n)
            index[0:self.n_SLE] = 1
            SLE_list = np.where(index == 1)[0]
            degrees = np.sum(matrix, axis=0)
            SLEre_degrees = np.sum(matrix[SLE_list, :], axis=0)

            plt.figure(figsize=(6, 6))
            pos = nx.circular_layout(graph)
            nx.draw_networkx_nodes(graph, pos, node_size=5, node_color='dodgerblue')
            nx.draw_networkx_nodes(graph, pos, nodelist=SLE_list, node_color='lightcoral', node_size=15)
            nx.draw_networkx_edges(graph, pos, alpha=0.5, width=0.2)
            plt.title('L=' + str(mean_path) + '   ' + 'C=' + str(mean_cluster), fontsize=18)
            plt.axis('off')
            figname = os.path.join(self.expFolder, type + '_net.pdf')
            plt.savefig(figname)
            plt.figure(figsize=(6, 6))
            plt.imshow(matrix, cmap='binary')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            figname = os.path.join(self.expFolder, type + '_matrix.pdf')
            plt.savefig(figname)

            plt.figure(figsize=(12, 6))
            plt.bar(range(len(degrees)), degrees, color='dodgerblue', alpha=0.7, label='NS')
            plt.bar(range(len(degrees)), SLEre_degrees, color='lightcoral', alpha=0.7, label='SLE')
            plt.title(self.net_names[type] + '  Degrees statistics', fontsize=18)
            plt.xlabel('Channel', fontsize=15)
            plt.ylabel('Degree', fontsize=15)
            plt.xticks(fontsize=15)  # 调整横轴刻度的大小
            plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
            plt.legend(fontsize=15, loc='upper right')
            figname = os.path.join(self.expFolder, type + '_degrees.pdf')
            plt.savefig(figname)

            plt.figure(figsize=(12, 6))
            node_labels = list(centrality.keys())
            node_betweenness = list(centrality.values())
            # 创建柱状图
            plt.bar(node_labels, node_betweenness,
                    color=['lightcoral' if node in SLE_list else 'dodgerblue' for node in node_labels], alpha=0.7)
            handles = [
                plt.Line2D([0], [0], color='lightcoral', lw=12, label='SLE'),
                plt.Line2D([0], [0], color='dodgerblue', lw=12, label='NS')
            ]
            # 设置图表标题和标签
            plt.title(self.net_names[type] + '  Betweenness statistics', fontsize=18)
            plt.xlabel('Channel', fontsize=15)
            plt.ylabel('Betweenness centrality', fontsize=15)
            plt.xticks(fontsize=15)  # 调整横轴刻度的大小
            plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
            plt.legend(handles=handles, handlelength=1.5, fontsize=15, loc='upper right')
            figname = os.path.join(self.expFolder, type + '_centrality.pdf')
            plt.savefig(figname)

            plt.figure(figsize=(20, 5))
            channel_ticks = []
            channel_labels = []
            for i in range(6):
                plt.plot(T[10000:], Vol[9 + 30 * i, 10000:] - (i * 4), color='black')
                channel_ticks.append(-2-(i*4))
                channel_labels.append(str(10+30*i))
            # plt.legend(fontsize=15, loc='upper right')
            plt.title(self.net_names[type] + ' ' + 'Vol-T', fontsize=18)
            plt.xlabel('Time', fontsize=15)
            plt.ylabel('Channel', fontsize=15)
            ax = plt.gca()
            # ax.axes.get_yaxis().set_visible(False)
            # plt.xticks([500, 2500, 4500, 6500, 8500, 10500], ['500', '2500', '4500', '6500', '8500', '10500'],
                       # fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(channel_ticks, channel_labels, fontsize=15)
            [ax.get_yticklabels()[idx].set_color('lightcoral') for idx in [0, 1]]
            [ax.get_yticklabels()[idx].set_color('dodgerblue') for idx in [2, 3, 4, 5]]
            # plt.axis('off')
            figname = os.path.join(self.expFolder, type + '_vol.pdf')
            plt.savefig(figname)

            plt.figure(figsize=(14, 3))
            plt.subplot(1, 2, 1)
            plt.imshow(freq[:, self.start:], cmap='ocean_r', aspect=4.5)
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=15)  # Set the fontsize here
            plt.xlabel('Time', fontsize=15)
            plt.ylabel('Channel index', fontsize=15)
            plt.title(self.net_names[type] + '  Frequency  S=' + str(sync), fontsize=15)
            plt.subplot(1, 2, 2)
            plt.imshow(energy[:, self.start:], cmap='gist_heat_r', aspect=4.5)
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=15)  # Set the fontsize here
            plt.xlabel('Time', fontsize=15)
            plt.ylabel('Channel index', fontsize=15)
            plt.title(self.net_names[type] + '  Energy  E=' + str(energy_index), fontsize=15)
            figname = os.path.join(self.expFolder, type + '_FE.pdf')
            plt.savefig(figname)


class number_visualization:
    """
    Figure:
        S-p curve
        E-p curve
    """
    def __init__(self, config):
        self.config = config
        self.exp = config['exp']
        self.netType = config['netType']
        self.n = config['n']
        self.start = config['start']
        self.label_map = {
            'random': 'random',
            'small_world': 'small-world',
            'scale_free': 'scale-free'
        }

    def make_data_folder(self):
        dataFolder = self.config['figFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def load_data(self, dataIndex, type, idx):
        dataFolder = self.config[dataIndex + 'Folder']
        expFolder = os.path.join(dataFolder, self.exp)
        fileName = os.path.join(expFolder, type, idx + '.npy')
        return np.load(fileName, allow_pickle=True).item()

    def run(self):
        self.make_data_folder()
        color_line = {'random': 'royalblue', 'small_world': 'orange', 'scale_free': 'green'}
        color_zone = {'random': 'lightskyblue', 'small_world': 'wheat', 'scale_free': 'lightgreen'}
        for type in self.netType:
            analysisData = self.load_data('analysis', type, 'index')

            nSLE_range = analysisData['nSLE_range']
            percent_range = [f"{percent * 100}%" for percent in nSLE_range/self.n]
            S = analysisData['S']
            E = analysisData['E']

            plt.figure('S_curve', figsize=(7.5, 5.5))
            S_mean = np.mean(S, axis=0)
            S_min = np.min(S, axis=0)
            S_max = np.max(S, axis=0)
            plt.fill_between(nSLE_range, S_min, S_max, color=color_zone[type], alpha=0.5)
            plt.plot(nSLE_range, S_mean, color=color_line[type], label=self.label_map[type])
            plt.title('Sync index curve', fontsize=18)
            plt.legend(fontsize=15, loc='upper left')
            plt.xlabel('p range', fontsize=15)
            plt.ylabel('S', fontsize=15)
            plt.xticks([0, 20, 40, 60, 80, 100], ['0%', '10%', '20%', '30%', '40%', '50%'], fontsize=15)
            plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
            # plt.show()
            figname = os.path.join(self.expFolder, 'number_curve_S.pdf')
            plt.savefig(figname)

            plt.figure('E_curve', figsize=(7.5, 5.5))
            E_mean = np.mean(E, axis=0)
            E_min = np.min(E, axis=0)
            E_max = np.max(E, axis=0)
            plt.fill_between(nSLE_range, E_min, E_max, color=color_zone[type], alpha=0.5)
            plt.plot(nSLE_range, E_mean, color=color_line[type], label=self.label_map[type])
            plt.title('Energy index curve', fontsize=18)
            plt.legend(fontsize=15, loc='upper left')
            plt.xlabel('p range', fontsize=15)
            plt.ylabel('E', fontsize=15)
            plt.xticks([0, 20, 40, 60, 80, 100], ['0%', '10%', '20%', '30%', '40%', '50%'], fontsize=15)
            plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
            # plt.show()
            figname = os.path.join(self.expFolder, 'number_curve_E.pdf')
            plt.savefig(figname)

            # plt.figure(figsize=(14, 3))
            # plt.subplot(1, 2, 1)
            # plt.imshow(FREQ[1][39][:, self.start:], cmap='ocean_r', aspect=4.5)
            # cbar = plt.colorbar()
            # cbar.ax.tick_params(labelsize=15)  # Set the fontsize here
            # plt.xlabel('Time', fontsize=15)
            # plt.ylabel('channel index', fontsize=15)
            # plt.xticks(fontsize=15)  # 调整横轴刻度的大小
            # plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
            # plt.title('Frequency  S=' + str(S[1, 39]), fontsize=18)
            # plt.subplot(1, 2, 2)
            # plt.imshow(ENERGY[1][39][:, self.start:], cmap='gist_heat_r', aspect=4.5)
            # cbar = plt.colorbar()
            # cbar.ax.tick_params(labelsize=15)  # Set the fontsize here
            # plt.xlabel('Time', fontsize=15)
            # plt.ylabel('channel index', fontsize=15)
            # plt.xticks(fontsize=15)  # 调整横轴刻度的大小
            # plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
            # plt.title('Energy  E=' + str(E[1, 39]), fontsize=18)
            # figname = os.path.join(self.expFolder, type + '_freq_and energy.eps')
            # plt.savefig(figname)


class scalefree_dist_number_visualization:
    def __init__(self, config):
        self.config = config
        self.exp = config['exp']
        self.n = config['n']
        self.SLE_dist = config['SLE_dist']
        self.start = config['start']

    def make_data_folder(self):
        dataFolder = self.config['figFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def load_data(self, dataIndex, type, idx):
        dataFolder = self.config[dataIndex + 'Folder']
        expFolder = os.path.join(dataFolder, self.exp)
        fileName = os.path.join(expFolder, type, idx + '.npy')
        return np.load(fileName, allow_pickle=True).item()

    def run(self):
        self.make_data_folder()
        color_line = {0: 'purple', 25: 'orange', 50: 'green', 75: 'red', 100: 'royalblue'}
        color_zone = {0: 'plum', 25: 'wheat', 50: 'lightgreen', 75: 'pink', 100: 'lightskyblue'}
        for dist in self.SLE_dist:
            analysisData = self.load_data('analysis', str(dist), 'index')
            nSLE_range = analysisData['nSLE_range']
            percent_range = [f"{percent * 100}%" for percent in nSLE_range/self.n]
            S = analysisData['S']
            E = analysisData['E']

            plt.figure('S_curve', figsize=(7.5, 5.5))
            S_mean = np.mean(S, axis=0)
            S_min = np.min(S, axis=0)
            S_max = np.max(S, axis=0)
            plt.fill_between(nSLE_range, S_min, S_max, color=color_zone[dist], alpha=0.5)
            plt.plot(nSLE_range, S_mean, color=color_line[dist], label=dist)
            plt.title('Sync index curve', fontsize=18)
            plt.legend(fontsize=15, loc='upper left')
            plt.xlabel('p range', fontsize=15)
            plt.ylabel('S', fontsize=15)
            plt.xticks([0, 20, 40, 60, 80, 100], ['0%', '10%', '20%', '30%', '40%', '50%'], fontsize=15)
            plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
            figname = os.path.join(self.expFolder, 'scalefreeNum_curve_S.pdf')
            plt.savefig(figname)

            plt.figure('E_curve', figsize=(7.5, 5.5))
            E_mean = np.mean(E, axis=0)
            E_min = np.min(E, axis=0)
            E_max = np.max(E, axis=0)
            plt.fill_between(nSLE_range, E_min, E_max, color=color_zone[dist], alpha=0.5)
            plt.plot(nSLE_range, E_mean, color=color_line[dist], label=dist)
            plt.title('Energy index curve', fontsize=18)
            plt.legend(fontsize=15, loc='upper left')
            plt.xlabel('p range', fontsize=15)
            plt.ylabel('E', fontsize=15)
            plt.xticks([0, 20, 40, 60, 80, 100], ['0%', '10%', '20%', '30%', '40%', '50%'], fontsize=15)
            plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
            figname = os.path.join(self.expFolder, 'scalefreeNum_curve_E.pdf')
            plt.savefig(figname)


class randomIdx_visualization:
    def __init__(self, config):
        self.config = config
        self.exp = config['exp']
        self.netType = config['netType']
        self.n = config['n']
        self.n_SLE = config['n_SLE']
        self.start = config['start']

    def make_data_folder(self):
        dataFolder = self.config['figFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def load_data(self, dataIndex, type):
        dataFolder = self.config[dataIndex + 'Folder']
        expFolder = os.path.join(dataFolder, self.exp)
        fileName = os.path.join(expFolder, type + '.npy')
        return np.load(fileName, allow_pickle=True).item()

    def run(self):
        self.make_data_folder()
        for type in self.netType:
            outputData = self.load_data('output', type)
            analysisData = self.load_data('analysis', type)

            Vol = outputData['Vol']
            T = outputData['T']
            graph = outputData['graph']
            index = outputData['index']
            freq = analysisData['freq']
            energy = analysisData['energy']
            sync = analysisData['sync']
            energy_index = analysisData['energy_index']
            SLE_list = np.where(index == 1)[0]

            plt.figure(figsize=(6, 6))
            pos = nx.circular_layout(graph)
            nx.draw_networkx_nodes(graph, pos, node_size=5, node_color='skyblue')
            nx.draw_networkx_nodes(graph, pos, nodelist=SLE_list, node_color='red', node_size=15)
            nx.draw_networkx_edges(graph, pos, alpha=0.5, width=0.5)
            plt.axis('off')
            figname = os.path.join(self.expFolder, type + '_net.eps')
            plt.savefig(figname)

            plt.figure(figsize=(20, 5))
            channel_ticks = []
            channel_labels = []
            for i in range(6):
                plt.plot(T[10000:], Vol[9 + 30 * i, 10000:] - (i * 4), color='black')
                channel_ticks.append(-2-(i*4))
                channel_labels.append(str(10+30*i))
            plt.title(type + ' (k=7.8) ' + 'Vol-T', fontsize=15)
            plt.xlabel('Time', fontsize=15)
            plt.ylabel('Channel', fontsize=15)
            ax = plt.gca()
            plt.xticks([500, 5500, 10500, 15500, 20500, 25500, 35500], ['500', '5500', '10500', '15500', '20500', '25500', '30500'],
                       fontsize=15)
            plt.yticks(channel_ticks, channel_labels, fontsize=15)
            [ax.get_yticklabels()[idx].set_color('lightcoral') for idx in [0, 1]]
            [ax.get_yticklabels()[idx].set_color('dodgerblue') for idx in [2, 3, 4, 5]]
            # plt.axis('off')
            figname = os.path.join(self.expFolder, type + '_check_vol2.eps')
            plt.savefig(figname)

            plt.figure(figsize=(14, 3))
            plt.subplot(1, 2, 1)
            plt.imshow(freq[:, self.start:], cmap='ocean_r', aspect=4.5)
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=15)
            plt.xlabel('Time', fontsize=15)
            plt.ylabel('Channel index', fontsize=15)
            plt.xticks(fontsize=15)  # 调整横轴刻度的大小
            plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
            plt.title('Frequency  S=' + str(sync), fontsize=18)
            plt.subplot(1, 2, 2)
            plt.imshow(energy[:, self.start:], cmap='gist_heat_r', aspect=4.5)
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=15)  # Set the fontsize here
            plt.xlabel('Time', fontsize=15)
            plt.ylabel('Channel index', fontsize=15)
            plt.xticks(fontsize=15)  # 调整横轴刻度的大小
            plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
            plt.title('Energy  E=' + str(energy_index), fontsize=18)
            figname = os.path.join(self.expFolder, type + '_FE.pdf')
            plt.savefig(figname)


class coupling_strength_visualization:
    def __init__(self, config):
        self.config = config
        self.exp = config['exp']
        self.netType = config['netType']
        self.n = config['n']
        self.n_SLE = config['n_SLE']
        self.start = config['start']
        self.label_map = {
            'random': 'random',
            'small_world': 'small-world',
            'scale_free': 'scale-free'
        }

    def make_data_folder(self):
        dataFolder = self.config['figFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def load_data(self, dataIndex, type, idx):
        dataFolder = self.config[dataIndex + 'Folder']
        expFolder = os.path.join(dataFolder, self.exp)
        fileName = os.path.join(expFolder, type, idx + '.npy')
        return np.load(fileName, allow_pickle=True).item()

    def run(self):
        self.make_data_folder()
        color_line = {'random': 'royalblue', 'small_world': 'orange', 'scale_free': 'green'}
        color_zone = {'random': 'lightskyblue', 'small_world': 'wheat', 'scale_free': 'lightgreen'}
        for type in self.netType:
            analysisData = self.load_data('analysis', type, 'index')

            k_range = analysisData['k_range']
            S = analysisData['S']
            E = analysisData['E']
            index = np.zeros(self.n)
            index[0:self.n_SLE] = 1
            SLE_list = np.where(index == 1)[0]

            # plt.figure(figsize=(12, 6))
            # plt.subplot(1, 2, 1)
            # pos = nx.circular_layout(graph)
            # nx.draw_networkx_nodes(graph, pos, node_size=3, node_color='skyblue')
            # nx.draw_networkx_nodes(graph, pos, nodelist=SLE_list, node_color='red', node_size=6)
            # nx.draw_networkx_edges(graph, pos, alpha=0.5)
            # plt.axis('off')
            # plt.subplot(1, 2, 2)
            # plt.imshow(matrix, cmap='binary')
            # # plt.axis('off')
            # figname = os.path.join(self.expFolder, type + '_net.eps')
            # plt.savefig(figname)

            plt.figure('S_curve', figsize=(7.5, 5.5))
            S_mean = np.mean(S, axis=0)
            S_min = np.min(S, axis=0)
            S_max = np.max(S, axis=0)
            plt.fill_between(k_range, S_min, S_max, color=color_zone[type], alpha=0.5)
            plt.plot(k_range, S_mean, color=color_line[type], label=self.label_map[type])
            plt.title('Sync index curve', fontsize=18)
            plt.legend(fontsize=15, loc='upper left')
            plt.xlabel('k range', fontsize=15)
            plt.ylabel('S', fontsize=15)
            plt.xticks(fontsize=15)  # 调整横轴刻度的大小
            plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
            figname = os.path.join(self.expFolder, 'couplingStrength_curve_S.pdf')
            plt.savefig(figname)

            plt.figure('E_curve', figsize=(7.5, 5.5))
            E_mean = np.mean(E, axis=0)
            E_min = np.min(E, axis=0)
            E_max = np.max(E, axis=0)
            plt.fill_between(k_range, E_min, E_max, color=color_zone[type], alpha=0.5)
            plt.plot(k_range, E_mean, color=color_line[type], label=self.label_map[type])
            plt.title('Energy index curve', fontsize=18)
            plt.legend(fontsize=15, loc='upper right')
            plt.xlabel('k range', fontsize=15)
            plt.ylabel('E', fontsize=15)
            plt.xticks(fontsize=15)  # 调整横轴刻度的大小
            plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
            figname = os.path.join(self.expFolder, 'couplingStrength_curve_E.pdf')
            plt.savefig(figname)

            # plt.figure(figsize=(14, 3))
            # plt.subplot(1, 2, 1)
            # plt.imshow(FREQ[1][9][:, self.start:], cmap='ocean_r', aspect=4.5)
            # cbar = plt.colorbar()
            # cbar.ax.tick_params(labelsize=15)  # Set the fontsize here
            # plt.xlabel('Time', fontsize=15)
            # plt.ylabel('channel index', fontsize=15)
            # plt.xticks(fontsize=15)  # 调整横轴刻度的大小
            # plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
            # plt.title('Frequency  S=' + str(S[1, 9]), fontsize=18)
            # plt.subplot(1, 2, 2)
            # plt.imshow(ENERGY[1][9][:, self.start:], cmap='gist_heat_r', aspect=4.5)
            # cbar = plt.colorbar()
            # cbar.ax.tick_params(labelsize=15)  # Set the fontsize here
            # plt.xlabel('Time', fontsize=15)
            # plt.ylabel('channel index', fontsize=15)
            # plt.xticks(fontsize=15)  # 调整横轴刻度的大小
            # plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
            # plt.title('Energy  E=' + str(E[1, 9]), fontsize=18)
            # figname = os.path.join(self.expFolder, type + '_freq_and energy.eps')
            # plt.savefig(figname)

class random_num_visualization:
    def __init__(self, config):
        self.config = config
        self.exp = config['exp']
        self.n = config['n']
        self.n_SLE = config['n_SLE']
        self.start = config['start']

    def make_data_folder(self):
        dataFolder = self.config['figFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def load_data(self, dataIndex, type, idx):
        dataFolder = self.config[dataIndex + 'Folder']
        expFolder = os.path.join(dataFolder, self.exp)
        fileName = os.path.join(expFolder, type, idx + '.npy')
        return np.load(fileName, allow_pickle=True).item()

    def run(self):
        self.make_data_folder()
        color_line = {20: 'royalblue', 40: 'orange', 60: 'green', 80: 'red'}
        color_zone = {20: 'lightskyblue', 40: 'wheat', 60: 'lightgreen', 80: 'pink'}
        label_line = {20: '10%', 40: '20%', 60: '30%', 80: '40%'}
        for SLE in self.n_SLE:
            analysisData = self.load_data('analysis', str(SLE), 'index')
            k_range = analysisData['k_range']
            S = analysisData['S']
            E = analysisData['E']
            FEData = self.load_data('analysis', str(SLE), '0')
            Freq = FEData['Freq']
            Energy = FEData['Energy']

            plt.figure('S_curve', figsize=(7.5, 5.5))
            S_mean = np.mean(S, axis=0)
            S_min = np.min(S, axis=0)
            S_max = np.max(S, axis=0)
            plt.fill_between(k_range, S_min, S_max, color=color_zone[SLE], alpha=0.5)
            plt.plot(k_range, S_mean, color=color_line[SLE], label=label_line[SLE])
            plt.title('Sync index curve (random)', fontsize=18)
            plt.legend(fontsize=15, loc='upper right')
            plt.xlabel('k range', fontsize=15)
            plt.ylabel('S', fontsize=15)
            plt.xticks(fontsize=15)  # 调整横轴刻度的大小
            plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
            figname = os.path.join(self.expFolder, 'randomNum_curve_S.pdf')
            plt.savefig(figname)

            plt.figure('E_curve', figsize=(7.5, 5.5))
            E_mean = np.mean(E, axis=0)
            E_min = np.min(E, axis=0)
            E_max = np.max(E, axis=0)
            plt.fill_between(k_range, E_min, E_max, color=color_zone[SLE], alpha=0.5)
            plt.plot(k_range, E_mean, color=color_line[SLE], label=label_line[SLE])
            plt.title('Energy index curve (random)', fontsize=18)
            plt.legend(fontsize=15, loc='upper right')
            plt.xlabel('k range', fontsize=15)
            plt.ylabel('E', fontsize=15)
            plt.xticks(fontsize=15)  # 调整横轴刻度的大小
            plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
            figname = os.path.join(self.expFolder, 'randomNum_curve_E.pdf')
            plt.savefig(figname)

            plt.figure(figsize=(14, 3))
            plt.subplot(1, 2, 1)
            plt.imshow(Freq[14][:, self.start:], cmap='ocean_r', aspect=4.5)
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=15)  # Set the fontsize here
            plt.xlabel('Time', fontsize=15)
            plt.ylabel('Channel index', fontsize=15)
            plt.xticks(fontsize=10)  # 调整横轴刻度的大小
            plt.yticks(fontsize=10)  # 调整纵轴刻度的大小
            plt.title('Frequency k=3  S=' + str(S[0, 14]), fontsize=15)
            plt.subplot(1, 2, 2)
            plt.imshow(Energy[14][:, self.start:], cmap='gist_heat_r', aspect=4.5)
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=15)  # Set the fontsize here
            plt.xlabel('Time', fontsize=15)
            plt.ylabel('Channel index', fontsize=15)
            plt.xticks(fontsize=10)  # 调整横轴刻度的大小
            plt.yticks(fontsize=10)  # 调整纵轴刻度的大小
            plt.title('Energy k=3  E=' + str(E[0, 14]), fontsize=15)
            figname = os.path.join(self.expFolder, 'FE_3.eps')
            plt.savefig(figname)


class scalefree_dist_visualization:
    def __init__(self, config):
        self.config = config
        self.exp = config['exp']
        self.n = config['n']
        self.n_SLE = config['n_SLE']
        self.SLE_dist = config['SLE_dist']
        self.start = config['start']

    def make_data_folder(self):
        dataFolder = self.config['figFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def load_data(self, dataIndex, type, idx):
        dataFolder = self.config[dataIndex + 'Folder']
        expFolder = os.path.join(dataFolder, self.exp)
        fileName = os.path.join(expFolder, type, idx + '.npy')
        return np.load(fileName, allow_pickle=True).item()

    def run(self):
        self.make_data_folder()
        base_scale_free_data = np.load('data/output/base/scale_free.npy', allow_pickle=True).item()
        graph = base_scale_free_data['graph']
        color_line = {0: 'purple', 25: 'orange', 50: 'green', 75: 'red', 100: 'royalblue'}
        color_zone = {0: 'plum', 25: 'wheat', 50: 'lightgreen', 75: 'pink', 100: 'lightskyblue'}
        for dist in self.SLE_dist:
            analysisData = self.load_data('analysis', str(dist), 'index')
            k_range = analysisData['k_range']
            S = analysisData['S']
            E = analysisData['E']
            index = np.zeros(self.n)
            index[dist: dist + self.n_SLE] = 1
            SLE_list = np.where(index == 1)[0]

            plt.figure(figsize=(8, 8))
            pos = nx.circular_layout(graph)
            nx.draw_networkx_nodes(graph, pos, node_size=5, node_color='skyblue')
            nx.draw_networkx_nodes(graph, pos, nodelist=SLE_list, node_color='red', node_size=15)
            nx.draw_networkx_edges(graph, pos, alpha=0.5, width=0.5)
            plt.axis('off')
            plt.title('place: ' + str(dist), fontsize=20)
            figname = os.path.join(self.expFolder, str(dist) + '_net.eps')
            plt.savefig(figname)

            plt.figure('S_curve', figsize=(7.5, 5.5))
            S_mean = np.mean(S, axis=0)
            S_min = np.min(S, axis=0)
            S_max = np.max(S, axis=0)
            plt.fill_between(k_range, S_min, S_max, color=color_zone[dist], alpha=0.5)
            plt.plot(k_range, S_mean, color=color_line[dist], label=dist)
            plt.title('Sync index curve (scale-free)', fontsize=18)
            plt.legend(fontsize=15, loc='upper left')
            plt.xlabel('k range', fontsize=15)
            plt.ylabel('S', fontsize=15)
            plt.xticks(fontsize=15)  # 调整横轴刻度的大小
            plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
            figname = os.path.join(self.expFolder, 'scalefreeDist_curve_S.pdf')
            plt.savefig(figname)

            plt.figure('E_curve', figsize=(7.5, 5.5))
            E_mean = np.mean(E, axis=0)
            E_min = np.min(E, axis=0)
            E_max = np.max(E, axis=0)
            plt.fill_between(k_range, E_min, E_max, color=color_zone[dist], alpha=0.5)
            plt.plot(k_range, E_mean, color=color_line[dist], label=dist)
            plt.title('Energy index curve (scale-free)', fontsize=18)
            plt.legend(fontsize=15, loc='upper right')
            plt.xlabel('k range', fontsize=15)
            plt.ylabel('E', fontsize=15)
            plt.xticks(fontsize=15)  # 调整横轴刻度的大小
            plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
            figname = os.path.join(self.expFolder, 'scalefreeDist_curve_E.pdf')
            plt.savefig(figname)

            # plt.figure(figsize=(14, 3))
            # plt.subplot(1, 2, 1)
            # plt.imshow(FREQ[1][4][:, self.start:], cmap='ocean_r', aspect=4.5)
            # cbar = plt.colorbar()
            # cbar.ax.tick_params(labelsize=15)  # Set the fontsize here
            # plt.xlabel('Time', fontsize=15)
            # plt.ylabel('channel index', fontsize=15)
            # plt.xticks(fontsize=15)  # 调整横轴刻度的大小
            # plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
            # plt.title('Frequency  S=' + str(S[1, 4]), fontsize=18)
            # plt.subplot(1, 2, 2)
            # plt.imshow(ENERGY[1][4][:, self.start:], cmap='gist_heat_r', aspect=4.5)
            # cbar = plt.colorbar()
            # cbar.ax.tick_params(labelsize=15)  # Set the fontsize here
            # plt.xlabel('Time', fontsize=15)
            # plt.ylabel('channel index', fontsize=15)
            # plt.xticks(fontsize=15)  # 调整横轴刻度的大小
            # plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
            # plt.title('Energy  E=' + str(E[1, 4]), fontsize=18)
            # figname = os.path.join(self.expFolder, type + '_freq_and energy.eps')
            # plt.savefig(figname)


class mouse_chimera_visualization:
    def __init__(self, config):
        self.config = config
        self.exp = config['exp']

    def make_data_folder(self):
        dataFolder = self.config['figFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def load_data(self, dataIndex, type):
        dataFolder = self.config[dataIndex + 'Folder']
        expFolder = os.path.join(dataFolder, self.exp)
        fileName = os.path.join(expFolder, type + '.npy')
        return np.load(fileName, allow_pickle=True).item()

    def run(self):
        self.make_data_folder()
        outputData = self.load_data('output', self.exp)
        analysisData = self.load_data('analysis', self.exp)

        graph = outputData['graph']
        cortices = outputData['cortices']
        matrix = outputData['matrix']
        Vol = outputData['Vol']
        T = outputData['T']
        freq = analysisData['freq']
        energy = analysisData['energy']
        phi = analysisData['phi']
        regions = outputData['regions']
        y_ticks = [cortex[1] for cortex in cortices[:-1]]
        p_ticks = [cortex[0] + (cortex[1] - cortex[0])/2 for cortex in cortices]
        # xticks = [cortex[1] + i + 0.5 for i, cortex in enumerate(cortices)][:-1]
        # matrix[matrix == 0] = np.nan

        # plt.figure(figsize=(12, 10))
        # plt.subplot(2, 1, 1)
        # plt.imshow(freq, origin='lower', cmap='ocean_r', aspect=4.5)
        # plt.yticks(p_ticks, regions, rotation="horizontal")
        # for y in y_ticks:
        #     plt.axhline(y=y, color='black', linewidth=2)
        # # plt.hlines(y=y_ticks, color='black', linewidth=2.0)
        # plt.colorbar()
        # plt.title('Frequency')
        # plt.subplot(2, 1, 2)
        # plt.imshow(energy, cmap='gist_heat_r', aspect=4.5)
        # plt.yticks(p_ticks, regions, rotation="horizontal")
        # for y in y_ticks:
        #     plt.axhline(y=y, color='black', linewidth=2)
        # # plt.hlines(y=y_ticks, color='black', linewidth=2.0)
        # plt.colorbar()
        # plt.title('Energy')
        # figname = os.path.join(self.expFolder, self.exp + '_freq and energy.eps')
        # plt.savefig(figname)

        plt.figure(figsize=(22, 8))
        v_ticks = []
        for i, cortex in enumerate(cortices):
            cortex_v = np.mean(Vol[cortex[0]:cortex[1]], axis=0)
            plt.plot(cortex_v + 4*i, color='black')
            # plt.text(-30, 1+4*i, regions[i], fontsize=12, ha='center')
            v_ticks.append(1 + i*4)
        plt.xlabel('Time', fontsize=15)
        plt.yticks(v_ticks, regions, rotation="horizontal", fontsize=15)
        plt.xticks(fontsize=15)
        plt.title('mean Vol', fontsize=18)
        figname = os.path.join(self.expFolder, self.exp + '_sync_vol.pdf')
        plt.savefig(figname)

        plt.figure(figsize=(22, 8))
        # plt.imshow(phi[:, 50000:99000], origin='lower', aspect='auto')
        plt.imshow(phi, origin='lower', aspect='auto')
        plt.xlabel('Time', fontsize=15)
        plt.yticks(p_ticks, regions, rotation="horizontal", fontsize=15)
        plt.xticks(fontsize=15)
        for y in y_ticks:
            plt.axhline(y=y, color='black', linewidth=2)
        # plt.hlines(y=y_ticks, color='black', linewidth=2.0)O
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=15)
        plt.title('Phi', fontsize=18)
        figname = os.path.join(self.expFolder, self.exp + '_sync_phi.pdf')
        plt.savefig(figname)


class mouse_connect_visualization:
    def __init__(self, config):
        self.config = config
        self.exp = config['exp']
        self.start = config['start']
        self.T_window = config['T_window']

    def make_data_folder(self):
        dataFolder = self.config['figFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def load_data(self, dataIndex, type):
        dataFolder = self.config[dataIndex + 'Folder']
        expFolder = os.path.join(dataFolder, self.exp)
        fileName = os.path.join(expFolder, type + '.npy')
        return np.load(fileName, allow_pickle=True).item()

    def run(self):
        self.make_data_folder()
        outputData = self.load_data('output', self.exp)
        analysisData = self.load_data('analysis', self.exp)
        cortices = outputData['cortices']
        regions = outputData['regions']
        matrix = outputData['matrix']
        # V0 = butter_filter(outputData['V0'])
        V0 = outputData['V0']
        freq0 = analysisData['freq0']
        energy0 = analysisData['energy0']
        S0 = analysisData['S0']
        E0 = analysisData['E0']
        y_ticks = [cortex[1] for cortex in cortices[:-1]]
        p_ticks = [cortex[0] + (cortex[1] - cortex[0])/2 for cortex in cortices]

        plt.figure()
        plt.matshow(matrix)
        plt.xticks([cortex[1] - 0.5 for cortex in cortices[:-1]], [cortex[1] for cortex in cortices[:-1]],
                   rotation="vertical")
        plt.yticks([cortex[1] - 0.5 for cortex in cortices[:-1]], [cortex[1] for cortex in cortices[:-1]])
        plt.ylabel("From")
        plt.xlabel("To")
        plt.gca().xaxis.set_label_position("top")
        plt.grid()
        plt.colorbar(ticks=[1, 2, 3])
        plt.gcf().set_size_inches(4.5, 4.5)
        figname = os.path.join(self.expFolder, self.exp + '_net.pdf')
        plt.savefig(figname)

        plt.figure(figsize=(13, 10))
        plt.subplot(2, 1, 1)
        plt.imshow(freq0[:, self.start:], origin='lower', cmap='ocean_r', aspect=4.5)
        # plt.xlabel('Time', fontsize=12)
        plt.xticks(fontsize=15)
        plt.yticks(p_ticks, regions, rotation="horizontal", fontsize=15)
        for y in y_ticks:
            plt.axhline(y=y, color='black', linewidth=2)
        # plt.hlines(y=y_ticks, color='black', linewidth=2.0)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=15)
        plt.title('Frequency    S=' + str(S0), fontsize=18)
        plt.subplot(2, 1, 2)
        plt.imshow(energy0[:, self.start:], origin='lower', cmap='gist_heat_r', aspect=4.5)
        plt.xlabel('Time', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(p_ticks, regions, rotation="horizontal", fontsize=15)
        for y in y_ticks:
            plt.axhline(y=y, color='black', linewidth=2)
        # plt.hlines(y=y_ticks, color='black', linewidth=2.0)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=15)
        plt.title('Energy   E=' + str(E0), fontsize=18)
        figname = os.path.join(self.expFolder, 'hipp_FE.pdf')
        plt.savefig(figname)

        plt.figure(figsize=(22, 8))
        v_ticks = []
        for i, cortex in enumerate(cortices):
            cortex_v = np.mean(V0[cortex[0]:cortex[1]], axis=0)
            plt.plot(cortex_v[self.start*self.T_window:] +4*(i+1), color='black')
            v_ticks.append(1 + i*4)
        plt.xlabel('Time', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(v_ticks, regions, rotation="horizontal", fontsize=15)
        plt.title('Vol', fontsize=18)
        figname = os.path.join(self.expFolder, 'hipp_Vol.pdf')
        plt.savefig(figname)

        # plt.figure(figsize=(22, 8))
        # plt.plot(V0[0])
        # plt.xlabel('Time', fontsize=15)
        # plt.xticks(fontsize=15)
        # plt.yticks(fontsize=15)
        # plt.title('Vol', fontsize=18)
        # figname = os.path.join(self.expFolder, 'check_Vol.eps')
        # plt.savefig(figname)


class mouse_control_visualization:
    def __init__(self, config):
        self.config = config
        self.exp = config['exp']
        self.start = config['start']
        self.dk_max = config['dk_max']
        self.dk_min = config['dk_min']
        self.ddk = config['ddk']

    def make_data_folder(self):
        dataFolder = self.config['figFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def load_data(self, dataIndex, type, idx):
        dataFolder = self.config[dataIndex + 'Folder']
        expFolder = os.path.join(dataFolder, type)
        fileName = os.path.join(expFolder, idx + '.npy')
        return np.load(fileName, allow_pickle=True).item()

    def save_data(self, type, data):
        fileName = os.path.join(self.expFolder, type + '.npy')
        np.save(fileName, data)

    def run(self):
        self.make_data_folder()
        connect_outputData = self.load_data('output', 'mouse_connect', 'mouse_connect')
        analysisData = self.load_data('analysis', self.exp, 'index')
        cortices = connect_outputData['cortices']
        regions = connect_outputData['regions']
        matrix = connect_outputData['matrix']
        dk_range = np.arange(self.dk_min, self.dk_max+self.ddk, self.ddk)
        S = analysisData['S']
        E = analysisData['E']
        y_ticks = [cortex[1] for cortex in cortices[:-1]]
        p_ticks = [cortex[0] + (cortex[1] - cortex[0])/2 for cortex in cortices]

        plt.figure('S_curve', figsize=(7.5, 5.5))
        S_mean = np.mean(S, axis=0)
        S_min = np.min(S, axis=0)
        S_max = np.max(S, axis=0)
        plt.fill_between(dk_range, S_min, S_max, color='lightskyblue', alpha=0.5)
        plt.plot(dk_range, S_mean, color='royalblue')
        plt.title('Sync index curve', fontsize=18)
        # plt.legend(fontsize=15, loc='upper right')
        plt.xlabel('dk range', fontsize=15)
        plt.ylabel('S', fontsize=15)
        plt.xticks([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], ['0.0', '0.4', '0.8', '1.2', '1.6', '2.0'], fontsize=15)
        plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
        figname = os.path.join(self.expFolder, 'curve_S.pdf')
        plt.savefig(figname)

        plt.figure('E_curve', figsize=(7.5, 5.5))
        E_mean = np.mean(E, axis=0)
        E_min = np.min(E, axis=0)
        E_max = np.max(E, axis=0)
        plt.fill_between(dk_range, E_min, E_max, color='pink', alpha=0.5)
        plt.plot(dk_range, E_mean, color='red')
        plt.title('Energy index curve', fontsize=18)
        # plt.legend(fontsize=15, loc='upper right')
        plt.xlabel('dk range', fontsize=15)
        plt.ylabel('E', fontsize=15)
        plt.xticks([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], ['0.0', '0.4', '0.8', '1.2', '1.6', '2.0'], fontsize=15)
        plt.yticks(fontsize=15)  # 调整纵轴刻度的大小
        figname = os.path.join(self.expFolder, 'curve_E.pdf')
        plt.savefig(figname)

        # plt.figure(figsize=(14, 10))
        # plt.subplot(2, 1, 1)
        # plt.imshow(FREQ[4][10][:, self.start:], origin='lower', cmap='ocean_r', aspect=4.5)
        # # plt.xlabel('Time', fontsize=12)
        # plt.yticks(p_ticks, regions, rotation="horizontal", fontsize=15)
        # for y in y_ticks:
        #     plt.axhline(y=y, color='black', linewidth=2)
        # # plt.hlines(y=y_ticks, color='black', linewidth=2.0)
        # cbar = plt.colorbar()
        # cbar.ax.tick_params(labelsize=15)
        # plt.title('Frequency    S=' + str(S[4, 10]), fontsize=18)
        # plt.subplot(2, 1, 2)
        # plt.imshow(ENERGY[4][10][:, self.start:], origin='lower', cmap='gist_heat_r', aspect=4.5)
        # plt.xlabel('Time', fontsize=15)
        # plt.yticks(p_ticks, regions, rotation="horizontal", fontsize=15)
        # for y in y_ticks:
        #     plt.axhline(y=y, color='black', linewidth=2)
        # # plt.hlines(y=y_ticks, color='black', linewidth=2.0)
        # cbar = plt.colorbar()
        # cbar.ax.tick_params(labelsize=15)
        # plt.title('Energy   E=' + str(E[4, 10]), fontsize=18)
        # figname = os.path.join(self.expFolder, 'hipp+0.5_FE.eps')
        # plt.savefig(figname)
        #
        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 2, 1)
        # plt.imshow(FREQ[4][10][70:81, 900+self.start:930+self.start], origin='lower', cmap='ocean_r', aspect=4.5)
        # plt.yticks([0, 10], ['70', '80'], fontsize=15)
        # plt.xticks([0, 120, 240], ['900', '915', '930'], fontsize=15)
        # plt.title('Frequency', fontsize=18)
        # plt.subplot(1, 2, 2)
        # plt.imshow(ENERGY[4][10][70:81, 900+self.start:930+self.start], origin='lower', cmap='gist_heat_r', aspect=4.5)
        # plt.yticks([0, 10], ['70', '80'], fontsize=15)
        # plt.xticks([0, 120, 240], ['900', '915', '930'], fontsize=15)
        # plt.title('Energy', fontsize=18)
        # figname = os.path.join(self.expFolder, 'magnify.eps')
        # plt.savefig(figname)

        plt.figure(figsize=(22, 8))
        fileName = os.path.join('/Users/zhougongjin/Documents/GitHub/seizure-sync/data/output/mouse_control/0', '16' + '.npy')
        Vol = np.load(fileName, allow_pickle=True)
        v_ticks = []
        for i, cortex in enumerate(cortices):
            cortex_v = np.mean(Vol[cortex[0]:cortex[1], :], axis=0)
            plt.plot(cortex_v[15000:] +4*(i+1), color='black')
            # plt.text(-30, 1+4*i, regions[i], fontsize=12, ha='center')
            v_ticks.append(1 + i*4)
        plt.xlabel('Time', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(v_ticks, regions, rotation="horizontal", fontsize=15)
        plt.title('Vol', fontsize=18)
        figname = os.path.join(self.expFolder, 'hipp+0.8_Vol.pdf')
        plt.savefig(figname)