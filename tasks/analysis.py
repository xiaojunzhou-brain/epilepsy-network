# coding:utf-8
import os
import numpy as np
from tools.network import networkAnalysis as netAna
import tools.methods as methods
import math


class base_analysis:
    def __init__(self, config):
        self.config = config
        self.netType = config['netType']
        self.exp = config['exp']
        self.start = config['start']
        self.T_window = config['T_window']
        self.sampling_rate = config['sampling_rate']
        self.tau_max = config['tau_max']

    def make_data_folder(self):
        dataFolder = self.config['analysisFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def load_data(self, type):
        dataFolder = self.config['outputFolder']
        expFolder = os.path.join(dataFolder, self.exp)
        fileName = os.path.join(expFolder, type + '.npy')
        return np.load(fileName, allow_pickle=True).item()

    def save_data(self, type, data):
        fileName = os.path.join(self.expFolder, type + '.npy')
        np.save(fileName, data)

    def run(self):
        self.make_data_folder()
        for type in self.netType:
            outputData = self.load_data(type)
            graph = outputData['graph']
            mean_path = round(netAna(graph).MSPL(), 3)
            print('mean_path: ', mean_path)
            mean_cluster = round(netAna(graph).ACC(), 3)
            print('mean_cluster: ', mean_cluster)
            centrality = netAna(graph).BC()
            Vol = outputData['Vol']
            freq, energy = methods.fft(Vol, self.T_window, self.sampling_rate)
            energy_index = round(np.mean(energy[:, self.start:]), 3)
            Sim_F = methods.simplify_freq(freq[:, self.start:])
            percent = Sim_F.shape[0] / freq.shape[0]
            print('percent: ', percent)
            if Sim_F.size > 0:
                M, _ = methods.sync_matrix(Sim_F, self.tau_max)
                mask = ~np.eye(M.shape[0], dtype=bool)
                M_powered = np.power(M, 1/4)
                index = np.mean(M_powered[mask])
            else:
                index = 0
            print('index: ', index)
            sync = round(percent * index, 3)
            data = {'mean_path': mean_path, 'mean_cluster': mean_cluster,  'centrality': centrality,
                    'freq': freq, 'energy': energy, 'sync': sync, 'energy_index': energy_index}
            self.save_data(type, data)


class number_analysis:
    def __init__(self, config):
        self.config = config
        self.netType = config['netType']
        self.exp = config['exp']
        self.start = config['start']
        self.T_window = config['T_window']
        self.sampling_rate = config['sampling_rate']
        self.tau_max = config['tau_max']
        self.nSLE_min = config['nSLE_min']
        self.nSLE_max = config['nSLE_max']
        self.dnSLE = config['dnSLE']
        self.times = config['times']

    def make_data_folder(self):
        dataFolder = self.config['analysisFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def load_data(self, type, idx, i):
        dataFolder = self.config['outputFolder']
        expFolder = os.path.join(dataFolder, self.exp)
        fileName = os.path.join(expFolder, type, str(idx), str(i) + '.npy')
        return np.load(fileName, allow_pickle=True)

    def save_data(self, type, idx, data):
        fileName = os.path.join(self.expFolder, type)
        if not os.path.exists(fileName):
            os.makedirs(fileName)
        fileSave = os.path.join(fileName, str(idx) + '.npy')
        np.save(fileSave, data)

    def index_ana(self, type, idx, nSLE_range):
        Freq = []
        Energy = []
        S_vector = np.zeros((1, len(nSLE_range)))
        E_vector = np.zeros((1, len(nSLE_range)))
        for i in range(len(nSLE_range)):
            Vol = self.load_data(type, idx, i)
            freq, energy = methods.fft(Vol, self.T_window, self.sampling_rate)
            Freq.append(freq)
            Energy.append(energy)
            E_vector[:, i] = round(np.mean(energy[:, self.start:]), 3)
            Sim_freq = methods.simplify_freq(freq[:, self.start:])
            percent = Sim_freq.shape[0] / freq.shape[0]
            if Sim_freq.size > 0:
                M, _ = methods.sync_matrix(Sim_freq, self.tau_max)
                if M.shape[0] > 1:
                    mask = ~np.eye(M.shape[0], dtype=bool)
                    M_powered = np.power(M, 1/4)
                    index = np.mean(M_powered[mask])
                else:
                    index = 0.0
                # index = math.pow(np.mean(M), 1 / 4)
                # if index > 0.99:
                #     index = 0.99
            else:
                index = 0.0
            S_vector[:, i] = round(percent * index, 3)
            print('finish ' + str(i) + ' times: ' + str(percent) + ' - ' + str(index))
        return Freq, Energy, S_vector, E_vector

    def run(self):
        self.make_data_folder()
        for type in self.netType:
            nSLE_range = np.arange(self.nSLE_min, self.nSLE_max + self.dnSLE, self.dnSLE)
            S = np.zeros((self.times, len(nSLE_range)))
            E = np.zeros_like(S)
            for idx in range(self.times):
                Freq, Energy, S[idx, :], E[idx, :] = self.index_ana(type, idx, nSLE_range)
                data_map = {'Freq': Freq, 'Energy': Energy}
                self.save_data(type, idx, data_map)
                print('FINISH ' + str(idx) + ' TIMES')
                print('----------------------------')
            data_index = {'nSLE_range': nSLE_range, 'S': S, 'E': E}
            self.save_data(type, 'index', data_index)
            print('----------------------------')
            print('finish ' + type + '-analysis')
            print('----------------------------')


class scalefree_dist_number_analysis:
    def __init__(self, config):
        self.config = config
        self.exp = config['exp']
        self.SLE_dist = config['SLE_dist']
        self.start = config['start']
        self.T_window = config['T_window']
        self.sampling_rate = config['sampling_rate']
        self.tau_max = config['tau_max']
        self.nSLE_min = config['nSLE_min']
        self.nSLE_max = config['nSLE_max']
        self.dnSLE = config['dnSLE']
        self.times = config['times']

    def make_data_folder(self):
        dataFolder = self.config['analysisFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def load_data(self, type, idx, i):
        dataFolder = self.config['outputFolder']
        expFolder = os.path.join(dataFolder, self.exp)
        fileName = os.path.join(expFolder, type, str(idx), str(i) + '.npy')
        return np.load(fileName, allow_pickle=True)

    def save_data(self, type, idx, data):
        fileName = os.path.join(self.expFolder, type)
        if not os.path.exists(fileName):
            os.makedirs(fileName)
        fileSave = os.path.join(fileName, str(idx) + '.npy')
        np.save(fileSave, data)

    def index_ana(self, type, idx, nSLE_range):
        Freq = []
        Energy = []
        S_vector = np.zeros((1, len(nSLE_range)))
        E_vector = np.zeros((1, len(nSLE_range)))
        for i in range(len(nSLE_range)):
            Vol = self.load_data(type, idx, i)
            freq, energy = methods.fft(Vol, self.T_window, self.sampling_rate)
            Freq.append(freq)
            Energy.append(energy)
            E_vector[:, i] = round(np.mean(energy[:, self.start:]), 3)
            Sim_freq = methods.simplify_freq(freq[:, self.start:])
            percent = Sim_freq.shape[0] / freq.shape[0]
            if Sim_freq.size > 0:
                M, _ = methods.sync_matrix(Sim_freq, self.tau_max)
                if M.shape[0] > 1:
                    mask = ~np.eye(M.shape[0], dtype=bool)
                    M_powered = np.power(M, 1/4)
                    index = np.mean(M_powered[mask])
                else:
                    index = 0.0
                # index = math.pow(np.mean(M), 1 / 4)
                # if index > 0.99:
                #     index = 0.99
            else:
                index = 0.0
            S_vector[:, i] = round(percent * index, 3)
            print('finish ' + str(i) + ' times: ' + str(percent) + ' - ' + str(index))
        return Freq, Energy, S_vector, E_vector

    def run(self):
        self.make_data_folder()
        for dist in self.SLE_dist:
            nSLE_range = np.arange(self.nSLE_min, self.nSLE_max + self.dnSLE, self.dnSLE)
            S = np.zeros((self.times, len(nSLE_range)))
            E = np.zeros_like(S)
            for idx in range(self.times):
                Freq, Energy, S[idx, :], E[idx, :] = self.index_ana(str(dist), idx, nSLE_range)
                data_map = {'Freq': Freq, 'Energy': Energy}
                self.save_data(str(dist), idx, data_map)
                print('FINISH ' + str(idx) + ' TIMES')
                print('----------------------------')
            data_index = {'nSLE_range': nSLE_range, 'S': S, 'E': E}
            self.save_data(str(dist), 'index', data_index)
            print('----------------------------')
            print('finish ' + str(dist) + '-analysis')
            print('----------------------------')


class randomIdx_analysis:
    def __init__(self, config):
        self.config = config
        self.netType = config['netType']
        self.exp = config['exp']
        self.start = config['start']
        self.T_window = config['T_window']
        self.sampling_rate = config['sampling_rate']
        self.tau_max = config['tau_max']

    def make_data_folder(self):
        dataFolder = self.config['analysisFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def load_data(self, type):
        dataFolder = self.config['outputFolder']
        expFolder = os.path.join(dataFolder, self.exp)
        fileName = os.path.join(expFolder, type + '.npy')
        return np.load(fileName, allow_pickle=True).item()

    def save_data(self, type, data):
        fileName = os.path.join(self.expFolder, type + '.npy')
        np.save(fileName, data)

    def run(self):
        self.make_data_folder()
        for type in self.netType:
            outputData = self.load_data(type)
            Vol = outputData['Vol']
            freq, energy = methods.fft(Vol, self.T_window, self.sampling_rate)
            energy_index = round(np.mean(energy[:, self.start:]), 3)
            Sim_freq = methods.simplify_freq(freq[:, self.start:])
            percent = Sim_freq.shape[0] / freq.shape[0]
            print('percent: ', percent)
            if Sim_freq.size > 0:
                M, _ = methods.sync_matrix(Sim_freq, self.tau_max)
                if M.shape[0] > 1:
                    mask = ~np.eye(M.shape[0], dtype=bool)
                    M_powered = np.power(M, 1/4)
                    index = np.mean(M_powered[mask])
                else:
                    index = 0.0
                # index = math.pow(np.mean(M), 1 / 4)
                # if index > 0.99:
                #     index = 0.99
            else:
                index = 0.0
            print('index: ', index)
            sync = round(percent * index, 3)
            data = {'freq': freq, 'energy': energy, 'sync': sync, 'energy_index': energy_index}
            self.save_data(type, data)


class coupling_strength_analysis:
    def __init__(self, config):
        self.config = config
        self.netType = config['netType']
        self.exp = config['exp']
        self.start = config['start']
        self.T_window = config['T_window']
        self.sampling_rate = config['sampling_rate']
        self.tau_max = config['tau_max']
        self.k_min = config['k_min']
        self.k_max = config['k_max']
        self.dk = config['dk']
        self.times = config['times']

    def make_data_folder(self):
        dataFolder = self.config['analysisFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def load_data(self, type, idx, i):
        dataFolder = self.config['outputFolder']
        expFolder = os.path.join(dataFolder, self.exp)
        fileName = os.path.join(expFolder, type, str(idx), str(i) + '.npy')
        return np.load(fileName, allow_pickle=True)

    def save_data(self, type, idx, data):
        fileName = os.path.join(self.expFolder, type)
        if not os.path.exists(fileName):
            os.makedirs(fileName)
        fileSave = os.path.join(fileName, str(idx) + '.npy')
        np.save(fileSave, data)

    def index_ana(self, type, idx, k_range):
        Freq = []
        Energy = []
        S_vector = np.zeros((1, len(k_range)))
        E_vector = np.zeros((1, len(k_range)))
        for i in range(len(k_range)):
            Vol = self.load_data(type, idx, i)
            freq, energy = methods.fft(Vol, self.T_window, self.sampling_rate)
            Freq.append(freq)
            Energy.append(energy)
            E_vector[:, i] = round(np.mean(energy[:, self.start:]), 3)
            Sim_freq = methods.simplify_freq(freq[:, self.start:])
            percent = Sim_freq.shape[0] / freq.shape[0]
            if Sim_freq.size > 0:
                M, _ = methods.sync_matrix(Sim_freq, self.tau_max)
                if M.shape[0] > 1:
                    mask = ~np.eye(M.shape[0], dtype=bool)
                    M_powered = np.power(M, 1/4)
                    index = np.mean(M_powered[mask])
                else:
                    index = 0.0
                # index = math.pow(np.mean(M), 1 / 4)
                # if index > 0.99:
                #     index = 0.99
            else:
                index = 0.0
            S_vector[:, i] = round(percent * index, 3)
            print('finish ' + str(i) + ' times: ' + str(percent) + ' - ' + str(index))
        return Freq, Energy, S_vector, E_vector

    def run(self):
        self.make_data_folder()
        for type in self.netType:
            k_range = np.arange(self.k_min, self.k_max + self.dk, self.dk)
            S = np.zeros((self.times, len(k_range)))
            E = np.zeros_like(S)
            for idx in range(self.times):
                Freq, Energy, S[idx, :], E[idx, :] = self.index_ana(type, idx, k_range)
                data_map = {'Freq': Freq, 'Energy': Energy}
                self.save_data(type, idx, data_map)
                print('FINISH ' + str(idx) + ' TIMES')
                print('----------------------------')
            data_index = {'k_range': k_range, 'S': S, 'E': E}
            self.save_data(type, 'index', data_index)
            print('----------------------------')
            print('finish ' + type + '-analysis')
            print('----------------------------')


class random_num_analysis:
    def __init__(self, config):
        self.config = config
        self.exp = config['exp']
        self.n_SLE = config['n_SLE']
        self.start = config['start']
        self.T_window = config['T_window']
        self.sampling_rate = config['sampling_rate']
        self.tau_max = config['tau_max']
        self.k_min = config['k_min']
        self.k_max = config['k_max']
        self.dk = config['dk']
        self.times = config['times']

    def make_data_folder(self):
        dataFolder = self.config['analysisFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def load_data(self, type, idx, i):
        dataFolder = self.config['outputFolder']
        expFolder = os.path.join(dataFolder, self.exp)
        fileName = os.path.join(expFolder, type, str(idx), str(i) + '.npy')
        return np.load(fileName, allow_pickle=True)

    def save_data(self, type, idx, data):
        fileName = os.path.join(self.expFolder, type)
        if not os.path.exists(fileName):
            os.makedirs(fileName)
        fileSave = os.path.join(fileName, str(idx) + '.npy')
        np.save(fileSave, data)

    def index_ana(self, type, idx, k_range):
        Freq = []
        Energy = []
        S_vector = np.zeros((1, len(k_range)))
        E_vector = np.zeros((1, len(k_range)))
        for i in range(len(k_range)):
            Vol = self.load_data(type, idx, i)
            freq, energy = methods.fft(Vol, self.T_window, self.sampling_rate)
            Freq.append(freq)
            Energy.append(energy)
            E_vector[:, i] = round(np.mean(energy[:, self.start:]), 3)
            Sim_freq = methods.simplify_freq(freq[:, self.start:])
            percent = Sim_freq.shape[0] / freq.shape[0]
            if Sim_freq.size > 0:
                M, _ = methods.sync_matrix(Sim_freq, self.tau_max)
                if M.shape[0] > 1:
                    mask = ~np.eye(M.shape[0], dtype=bool)
                    M_powered = np.power(M, 1/4)
                    index = np.mean(M_powered[mask])
                else:
                    index = 0.0
                # index = math.pow(np.mean(M), 1 / 4)
                # if index > 0.99:
                #     index = 0.99
            else:
                index = 0.0
            S_vector[:, i] = round(percent * index, 3)
            print('finish ' + str(i) + ' times: ' + str(percent) + ' - ' + str(index))
        return Freq, Energy, S_vector, E_vector

    def run(self):
        self.make_data_folder()
        for SLE in self.n_SLE:
            k_range = np.arange(self.k_min, self.k_max + self.dk, self.dk)
            S = np.zeros((self.times, len(k_range)))
            E = np.zeros_like(S)
            for idx in range(self.times):
                Freq, Energy, S[idx, :], E[idx, :] = self.index_ana(str(SLE), idx, k_range)
                data_map = {'Freq': Freq, 'Energy': Energy}
                self.save_data(str(SLE), idx, data_map)
                print('FINISH ' + str(idx) + ' TIMES')
                print('----------------------------')
            data_index = {'k_range': k_range, 'S': S, 'E': E}
            self.save_data(str(SLE), 'index', data_index)
            print('----------------------------')
            print('finish ' + str(SLE) + '-analysis')
            print('----------------------------')


class scalefree_dist_analysis:
    def __init__(self, config):
        self.config = config
        self.exp = config['exp']
        self.SLE_dist = config['SLE_dist']
        self.start = config['start']
        self.T_window = config['T_window']
        self.sampling_rate = config['sampling_rate']
        self.tau_max = config['tau_max']
        self.k_min = config['k_min']
        self.k_max = config['k_max']
        self.dk = config['dk']
        self.times = config['times']

    def make_data_folder(self):
        dataFolder = self.config['analysisFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def load_data(self, type, idx, i):
        dataFolder = self.config['outputFolder']
        expFolder = os.path.join(dataFolder, self.exp)
        fileName = os.path.join(expFolder, type, str(idx), str(i) + '.npy')
        return np.load(fileName, allow_pickle=True)

    def save_data(self, type, idx, data):
        fileName = os.path.join(self.expFolder, type)
        if not os.path.exists(fileName):
            os.makedirs(fileName)
        fileSave = os.path.join(fileName, str(idx) + '.npy')
        np.save(fileSave, data)

    def index_ana(self, type, idx, k_range):
        Freq = []
        Energy = []
        S_vector = np.zeros((1, len(k_range)))
        E_vector = np.zeros((1, len(k_range)))
        for i in range(len(k_range)):
            Vol = self.load_data(type, idx, i)
            freq, energy = methods.fft(Vol, self.T_window, self.sampling_rate)
            Freq.append(freq)
            Energy.append(energy)
            E_vector[:, i] = round(np.mean(energy[:, self.start:]), 3)
            Sim_freq = methods.simplify_freq(freq[:, self.start:])
            percent = Sim_freq.shape[0] / freq.shape[0]
            if Sim_freq.size > 0:
                M, _ = methods.sync_matrix(Sim_freq, self.tau_max)
                if M.shape[0] > 1:
                    mask = ~np.eye(M.shape[0], dtype=bool)
                    M_powered = np.power(M, 1/4)
                    index = np.mean(M_powered[mask])
                else:
                    index = 0.0
                # index = math.pow(np.mean(M), 1 / 4)
                # if index > 0.99:
                #     index = 0.99
            else:
                index = 0.0
            S_vector[:, i] = round(percent * index, 3)
            print('finish ' + str(i) + ' times: ' + str(percent) + ' - ' + str(index))
        return Freq, Energy, S_vector, E_vector

    def run(self):
        self.make_data_folder()
        for dist in self.SLE_dist:
            k_range = np.arange(self.k_min, self.k_max + self.dk, self.dk)
            S = np.zeros((self.times, len(k_range)))
            E = np.zeros_like(S)
            for idx in range(self.times):
                Freq, Energy, S[idx, :], E[idx, :] = self.index_ana(str(dist), idx, k_range)
                data_map = {'Freq': Freq, 'Energy': Energy}
                self.save_data(str(dist), idx, data_map)
                print('FINISH ' + str(idx) + ' TIMES')
                print('----------------------------')
            data_index = {'k_range': k_range, 'S': S, 'E': E}
            self.save_data(str(dist), 'index', data_index)
            print('----------------------------')
            print('finish ' + str(dist) + '-analysis')
            print('----------------------------')


class mouse_chimera_analysis:
    def __init__(self, config):
        self.config = config
        self.exp = config['exp']
        self.start = config['start']
        self.T_window = config['T_window']
        self.sampling_rate = config['sampling_rate']

    def make_data_folder(self):
        dataFolder = self.config['analysisFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def load_data(self, type):
        dataFolder = self.config['outputFolder']
        expFolder = os.path.join(dataFolder, self.exp)
        fileName = os.path.join(expFolder, type + '.npy')
        return np.load(fileName, allow_pickle=True).item()

    def save_data(self, type, data):
        fileName = os.path.join(self.expFolder, type + '.npy')
        np.save(fileName, data)

    def run(self):
        self.make_data_folder()
        outputData = self.load_data(self.exp)
        Vol = outputData['Vol']
        T = outputData['T']
        spike = outputData['spike'].T
        t_last_spike = outputData['t_last_spike'].T
        T_matrix = np.tile(T, (Vol.shape[0], 1))
        spike_T = T_matrix * spike
        spikeses = [[element for element in row if element != 0] if any(row) else [0] for row in spike_T]
        index = [i for i, spike in enumerate(spikeses) if spike == [0]]
        freq, energy = methods.fft(Vol, self.T_window, self.sampling_rate)
        T_matrix[index, :] = 0
        phi = methods.phase(spikeses, t_last_spike, T_matrix)
        data = {'freq': freq, 'energy': energy, 'phi': phi}
        self.save_data(self.exp, data)


class mouse_connect_analysis:
    def __init__(self, config):
        self.config = config
        self.exp = config['exp']
        self.start = config['start']
        self.T_window = config['T_window']
        self.sampling_rate = config['sampling_rate']
        self.tau_max = config['tau_max']

    def make_data_folder(self):
        dataFolder = self.config['analysisFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def load_data(self, type):
        dataFolder = self.config['outputFolder']
        expFolder = os.path.join(dataFolder, self.exp)
        fileName = os.path.join(expFolder, type + '.npy')
        return np.load(fileName, allow_pickle=True).item()

    def save_data(self, type, data):
        fileName = os.path.join(self.expFolder, type + '.npy')
        np.save(fileName, data)

    def run(self):
        self.make_data_folder()
        outputData = self.load_data(self.exp)
        V0 = outputData['V0']
        freq0, energy0 = methods.fft(V0, self.T_window, self.sampling_rate)
        E0 = round(np.mean(energy0[:, self.start:]), 3)
        Sim_freq = methods.simplify_freq(freq0[:, self.start:])
        percent = Sim_freq.shape[0] / freq0.shape[0]
        print('percent: ', percent)
        if Sim_freq.size > 0:
            M, _ = methods.sync_matrix(Sim_freq, self.tau_max)
            if M.shape[0] > 1:
                mask = ~np.eye(M.shape[0], dtype=bool)
                M_powered = np.power(M, 1 / 4)
                index = np.mean(M_powered[mask])
            else:
                index = 0.0
            # index = math.pow(np.mean(M), 1 / 4)
            # if index > 0.99:
            #     index = 0.99
        else:
            index = 0.0
        print('index: ', index)
        S0 = round(percent * index, 3)
        data = {'freq0': freq0, 'energy0': energy0, 'S0': S0, 'E0': E0}
        self.save_data(self.exp, data)


class mouse_control_analysis:
    def __init__(self, config):
        self.config = config
        self.exp = config['exp']
        self.start = config['start']
        self.T_window = config['T_window']
        self.sampling_rate = config['sampling_rate']
        self.tau_max = config['tau_max']
        self.dk_min = config['dk_min']
        self.dk_max = config['dk_max']
        self.ddk = config['ddk']
        self.times = config['times']

    def make_data_folder(self):
        dataFolder = self.config['analysisFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def load_data(self, idx, i):
        dataFolder = self.config['outputFolder']
        expFolder = os.path.join(dataFolder, self.exp)
        fileName = os.path.join(expFolder, str(idx), str(i) + '.npy')
        return np.load(fileName, allow_pickle=True)

    def save_data(self, idx, data):
        fileSave = os.path.join(self.expFolder, str(idx) + '.npy')
        np.save(fileSave, data)

    def index_ana(self, idx, dk_range):
        Freq = []
        Energy = []
        S_vector = np.zeros((1, len(dk_range)))
        E_vector = np.zeros((1, len(dk_range)))
        for i in range(len(dk_range)):
            Vol = self.load_data(idx, i)
            freq, energy = methods.fft(Vol, self.T_window, self.sampling_rate)
            Freq.append(freq)
            Energy.append(energy)
            E_vector[:, i] = round(np.mean(energy[:, self.start:]), 3)
            Sim_freq = methods.simplify_freq(freq[:, self.start:])
            percent = Sim_freq.shape[0] / freq.shape[0]
            if Sim_freq.size > 0:
                M, _ = methods.sync_matrix(Sim_freq, self.tau_max)
                if M.shape[0] > 1:
                    mask = ~np.eye(M.shape[0], dtype=bool)
                    M_powered = np.power(M, 1/4)
                    index = np.mean(M_powered[mask])
                else:
                    index = 0.0
                # index = math.pow(np.mean(M), 1 / 4)
                # if index > 0.99:
                #     index = 0.99
            else:
                index = 0.0
            S_vector[:, i] = round(percent * index, 3)
            print('finish ' + str(i) + ' times: ' + str(percent) + ' - ' + str(index))
        return Freq, Energy, S_vector, E_vector

    def run(self):
        self.make_data_folder()
        dk_range = np.arange(self.dk_min, self.dk_max + self.ddk, self.ddk)
        S = np.zeros((self.times, len(dk_range)))
        E = np.zeros_like(S)
        for idx in range(self.times):
            Freq, Energy, S[idx, :], E[idx, :] = self.index_ana(idx, dk_range)
            data_map = {'Freq': Freq, 'Energy': Energy}
            self.save_data(idx, data_map)
            print('FINISH ' + str(idx) + ' TIMES')
            print('----------------------------')
        data_index = {'dk_range': dk_range, 'S': S, 'E': E}
        self.save_data('index', data_index)
        print('----------------------------')
        print('finish ' + '-analysis')
        print('----------------------------')