# coding:utf-8
import random

from tools.methods import Epileptor
from tools.methods import Epileptor_sim
from tools.methods import butter_filter
import tools.network as net
import numpy as np
import brainpy as bp
import brainpy.math as bm
import os
import gc


class base_generation:
    def __init__(self, config):
        self.config = config
        self.exp = config['exp']
        self.netType = config['netType']
        self.n = config['n']
        self.n_SLE = config['n_SLE']
        self.run_dt = config['run_dt']
        self.run_t = config['run_t']

    def make_data_folder(self):
        dataFolder = self.config['outputFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def save_data(self, type, data):
        filename = os.path.join(self.expFolder, type + '.npy')
        np.save(filename, data)

    def net_gen(self, type):
        create = eval('net.' + type)
        graph, matrix = create(self.n)
        return graph, matrix

    def vol_gen(self, matrix):
        # bm.set_platform('gpu')
        index = np.zeros(self.n)
        index[0: self.n_SLE] = 1
        bm.random.seed(42)
        matrix = 8.5*matrix
        model = Epileptor(self.n, index, matrix)
        runner = bp.DSRunner(model, monitors=['x1', 'x2'], dt=self.run_dt)
        runner.run(self.run_t)
        Vol = (runner.mon.x1 + runner.mon.x2).T
        T = runner.mon.ts
        return Vol, T

    def run(self):
        self.make_data_folder()
        random.seed(111)
        for type in self.netType:
            graph, matrix = self.net_gen(type)
            Vol, T = self.vol_gen(matrix)
            data = {'graph': graph, 'matrix': matrix, 'Vol': Vol, 'T': T}
            self.save_data(type, data)

# class number_generation(base_generation):
#     def __init__(self, config):
#         super().__init__(config)
#         self.nSLE_min = config['nSLE_min']
#         self.nSLE_max = config['nSLE_max']
#         self.dnSLE = config['dnSLE']
#         self.times = config['times']
#         self.single_times = config['single_times']



class number_generation:
    def __init__(self, config):
        self.config = config
        self.exp = config['exp']
        self.netType = config['netType']
        self.n = config['n']
        self.nSLE_min = config['nSLE_min']
        self.nSLE_max = config['nSLE_max']
        self.dnSLE = config['dnSLE']
        self.run_dt = config['run_dt']
        self.run_t = config['run_t']
        self.times = config['times']
        self.single_times = config['single_times']

    def make_data_folder(self):
        dataFolder = self.config['outputFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def load_data(self, type):
        dataFolder = self.config['outputFolder']
        expFolder = os.path.join(dataFolder, 'base')
        fileName = os.path.join(expFolder, type + '.npy')
        return np.load(fileName, allow_pickle=True).item() # mmap_mode 使用内存映射文件

    def save_data(self, filename, data, i):
        fileSave = os.path.join(filename, str(i) + '.npy')
        np.save(fileSave, data)

    def number_gen(self, type, matrix, idx):
        # bm.set_platform('gpu')
        nSLE_range = np.arange(self.nSLE_min, self.nSLE_max+self.dnSLE, self.dnSLE)
        filename = os.path.join(self.expFolder, type, str(idx))
        if not os.path.exists(filename):
            os.makedirs(filename)
        index = np.zeros(self.n)
        for i, nSLE in enumerate(nSLE_range):
            index[0:nSLE] = 1
            bm.random.seed(42 + idx)
            model = Epileptor(self.n, index, matrix)
            runner = bp.DSRunner(model, monitors=['x1', 'x2'], dt=self.run_dt)
            runner.run(self.run_t)
            Vol = (runner.mon.x1 + runner.mon.x2).T
            self.save_data(filename, Vol, i)
            runner.mon.clear()
            del model, runner, Vol
            gc.collect()

    def run(self):
        self.make_data_folder()
        for type in self.netType:
            outputData = self.load_data(type)
            matrix = outputData['matrix']
            for idx in range(self.times):
                self.number_gen(type, matrix, idx)
                print('finish ' + type + '-' + str(idx) + '-generation')
                gc.collect()


class scalefree_dist_number_generation:
    def __init__(self, config):
        self.config = config
        self.exp = config['exp']
        self.n = config['n']
        self.SLE_dist = config['SLE_dist']
        self.nSLE_min = config['nSLE_min']
        self.nSLE_max = config['nSLE_max']
        self.dnSLE = config['dnSLE']
        self.run_dt = config['run_dt']
        self.run_t = config['run_t']
        self.times = config['times']
        self.single_times = config['single_times']

    def make_data_folder(self):
        dataFolder = self.config['outputFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def load_data(self, type):
            dataFolder = self.config['outputFolder']
            expFolder = os.path.join(dataFolder, 'base')
            fileName = os.path.join(expFolder, type + '.npy')
            return np.load(fileName, allow_pickle=True).item()

    def save_data(self, filename, data, i):
        fileSave = os.path.join(filename, str(i) + '.npy')
        np.save(fileSave, data)

    def number_gen(self, dist, matrix, idx):
        # bm.set_platform('gpu')
        nSLE_range = np.arange(self.nSLE_min, self.nSLE_max+self.dnSLE, self.dnSLE)
        filename = os.path.join(self.expFolder, str(dist), str(idx))
        if not os.path.exists(filename):
            os.makedirs(filename)
        index = np.zeros(self.n)
        for i, nSLE in enumerate(nSLE_range):
            index[dist:dist+nSLE] = 1
            bm.random.seed(42 + idx)
            model = Epileptor(self.n, index, matrix)
            runner = bp.DSRunner(model, monitors=['x1', 'x2'], dt=self.run_dt)
            runner.run(self.run_t)
            Vol = (runner.mon.x1 + runner.mon.x2).T
            self.save_data(filename, Vol, i)
            runner.mon.clear()
            del model, runner, Vol
            gc.collect()

    def run(self):
        self.make_data_folder()
        outputData = self.load_data('scale_free')
        matrix = outputData['matrix']
        for dist in self.SLE_dist:
            for idx in range(self.times):
                self.number_gen(dist, matrix, idx)
                print('finish ' + str(dist) + '-' + str(idx) + '-generation')
                print('------------------------------')
                gc.collect()


class randomIdx_generation:
    def __init__(self, config):
        self.config = config
        self.exp = config['exp']
        self.netType = config['netType']
        self.n = config['n']
        self.n_SLE = config['n_SLE']
        self.run_dt = config['run_dt']
        self.run_t = config['run_t']

    def make_data_folder(self):
        dataFolder = self.config['outputFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def load_data(self, type):
        dataFolder = self.config['outputFolder']
        expFolder = os.path.join(dataFolder, 'base')
        fileName = os.path.join(expFolder, type + '.npy')
        return np.load(fileName, allow_pickle=True).item()

    def save_data(self, type, data):
        filename = os.path.join(self.expFolder, type +'.npy')
        np.save(filename, data)

    def vol_gen(self, matrix):
        # bm.set_platform('gpu')
        index = np.zeros(self.n)
        # index[np.random.choice(self.n, size=self.n_SLE)] = 1
        index[0:40] = 1
        k_matrix = 7.7*matrix
        bm.random.seed(42)
        model = Epileptor(self.n, index, k_matrix)
        runner = bp.DSRunner(model, monitors=['x1', 'x2'], dt=self.run_dt)
        runner.run(self.run_t)
        Vol = (runner.mon.x1 + runner.mon.x2).T
        T = runner.mon.ts
        return Vol, T, index

    def run(self):
        self.make_data_folder()
        for type in self.netType:
            outputData = self.load_data(type)
            graph = outputData['graph']
            matrix = outputData['matrix']
            Vol, T, index = self.vol_gen(matrix)
            data = {'graph': graph, 'Vol': Vol, 'T': T, 'index': index}
            self.save_data(type, data)


class coupling_strength_generation:
    def __init__(self, config):
        self.config = config
        self.exp = config['exp']
        self.netType = config['netType']
        self.n = config['n']
        self.k_min = config['k_min']
        self.k_max = config['k_max']
        self.dk = config['dk']
        self.n_SLE = config['n_SLE']
        self.run_dt = config['run_dt']
        self.run_t = config['run_t']
        self.times = config['times']
        self.single_times = config['single_times']

    def make_data_folder(self):
        dataFolder = self.config['outputFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def load_data(self, type):
        dataFolder = self.config['outputFolder']
        expFolder = os.path.join(dataFolder, 'base')
        fileName = os.path.join(expFolder, type + '.npy')
        return np.load(fileName, allow_pickle=True).item()

    def save_data(self, filename, data, i):
        fileSave = os.path.join(filename, str(i) + '.npy')
        np.save(fileSave, data)

    def strength_gen(self, type, matrix, idx):
        # bm.set_platform('gpu')
        k_range = np.arange(self.k_min, self.k_max+self.dk, self.dk)
        filename = os.path.join(self.expFolder, type, str(idx))
        if not os.path.exists(filename):
            os.makedirs(filename)
        index = np.zeros(self.n)
        index[0:self.n_SLE] = 1
        for i, k in enumerate(k_range):
            k_matrix = k * matrix
            bm.random.seed(42 + idx)
            model = Epileptor(self.n, index, k_matrix)
            runner = bp.DSRunner(model, monitors=['x1', 'x2'], dt=self.run_dt)
            runner.run(self.run_t)
            Vol = (runner.mon.x1 + runner.mon.x2).T
            self.save_data(filename, Vol, i)
            runner.mon.clear()
            del model, runner, Vol
            gc.collect()

    def run(self):
        self.make_data_folder()
        for type in self.netType:
            outputData = self.load_data(type)
            matrix = outputData['matrix']
            for idx in range(self.times):
                self.strength_gen(type, matrix, idx)
                print('finish ' + type + '-' + str(idx) + '-generation')
                print('------------------------------')
                gc.collect()


class random_num_generation:
    def __init__(self, config):
        self.config = config
        self.exp = config['exp']
        self.n = config['n']
        self.k_min = config['k_min']
        self.k_max = config['k_max']
        self.dk = config['dk']
        self.n_SLE = config['n_SLE']
        self.run_dt = config['run_dt']
        self.run_t = config['run_t']
        self.times = config['times']
        self.single_times = config['single_times']

    def make_data_folder(self):
        dataFolder = self.config['outputFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def load_data(self, type):
            dataFolder = self.config['outputFolder']
            expFolder = os.path.join(dataFolder, 'base')
            fileName = os.path.join(expFolder, type + '.npy')
            return np.load(fileName, allow_pickle=True).item()

    def save_data(self, filename, data, i):
        fileSave = os.path.join(filename, str(i) + '.npy')
        np.save(fileSave, data)

    def number_gen(self, SLE, matrix, idx):
        # bm.set_platform('gpu')
        k_range = np.arange(self.k_min, self.k_max + self.dk, self.dk)
        filename = os.path.join(self.expFolder, str(SLE), str(idx))
        if not os.path.exists(filename):
            os.makedirs(filename)
        index = np.zeros(self.n)
        index[0: SLE] = 1
        for i, k in enumerate(k_range):
            k_matrix = k * matrix
            bm.random.seed(42 + idx)
            model = Epileptor(self.n, index, k_matrix)
            runner = bp.DSRunner(model, monitors=['x1', 'x2'], dt=self.run_dt)
            runner.run(self.run_t)
            Vol = (runner.mon.x1 + runner.mon.x2).T
            self.save_data(filename, Vol, i)
            runner.mon.clear()
            del model, runner, Vol
            gc.collect()

    def run(self):
        self.make_data_folder()
        outputData = self.load_data('random')
        matrix = outputData['matrix']
        for SLE in self.n_SLE:
            for idx in range(self.times):
                self.number_gen(SLE, matrix, idx)
                print('finish ' + str(SLE) + '-' + str(idx) + '-generation')
                print('------------------------------')
                gc.collect()


class scalefree_dist_generation:
    def __init__(self, config):
        self.config = config
        self.exp = config['exp']
        self.n = config['n']
        self.n_SLE = config['n_SLE']
        self.SLE_dist = config['SLE_dist']
        self.k_min = config['k_min']
        self.k_max = config['k_max']
        self.dk = config['dk']
        self.run_dt = config['run_dt']
        self.run_t = config['run_t']
        self.times = config['times']
        self.single_times = config['single_times']

    def make_data_folder(self):
        dataFolder = self.config['outputFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def load_data(self, type):
        dataFolder = self.config['outputFolder']
        expFolder = os.path.join(dataFolder, 'base')
        fileName = os.path.join(expFolder, type + '.npy')
        return np.load(fileName, allow_pickle=True).item()

    def save_data(self, filename, data, i):
        fileSave = os.path.join(filename, str(i) + '.npy')
        np.save(fileSave, data)

    def strength_gen(self, dist, matrix, idx):
        # bm.set_platform('gpu')
        k_range = np.arange(self.k_min, self.k_max+self.dk, self.dk)
        filename = os.path.join(self.expFolder, str(dist), str(idx))
        if not os.path.exists(filename):
            os.makedirs(filename)
        index = np.zeros(self.n)
        index[dist: dist+self.n_SLE] = 1
        for i, k in enumerate(k_range):
            k_matrix = k * matrix
            bm.random.seed(42 + idx)
            model = Epileptor(self.n, index, k_matrix)
            runner = bp.DSRunner(model, monitors=['x1', 'x2'], dt=self.run_dt)
            runner.run(self.run_t)
            Vol = (runner.mon.x1 + runner.mon.x2).T
            self.save_data(filename, Vol, i)
            runner.mon.clear()
            del model, runner, Vol
            gc.collect()

    def run(self):
        self.make_data_folder()
        outputData = self.load_data('scale_free')
        matrix = outputData['matrix']
        for dist in self.SLE_dist:
            for idx in range(self.times):
                self.strength_gen(dist, matrix, idx)
                print('finish ' + str(dist) + '-' + str(idx) + '-generation')
                print('------------------------------')
                gc.collect()


class mouse_chimera_generation:
    def __init__(self, config):
        self.config = config
        self.exp = config['exp']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.run_dt = config['run_dt']
        self.run_t = config['run_t']

    def make_data_folder(self):
        dataFolder = self.config['outputFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def save_data(self, type, data):
        filename = os.path.join(self.expFolder, type+'.npy')
        np.save(filename, data)

    def net_gen(self):
        graph, matrix, G1, G2, n1, n2, cortices, regions = net.mouse()
        return graph, matrix, G1, G2, n1, n2, cortices, regions

    def Vol_gen(self, matrix, G1, G2, n1, n2):
        # bm.set_platform('gpu')
        index = np.ones(self.n)
        # index = np.zeros(self.n)
        # index[70:81] = 1
        # model = HR(self.n, G1, G2, n1, n2, self.alpha, self.beta, coupling='mouse')
        # runner = bp.DSRunner(model, monitors=['x', 'spike', 't_last_spike'], dt=0.05)
        # model = Epileptor(self.n, index, matrix, G1, G2, n1, n2, self.alpha, self.beta, coupling='mouse')
        # runner = bp.DSRunner(model, monitors=['x1', 'x2', 'spike', 't_last_spike'], dt=0.05)
        random.seed(100)
        model = Epileptor_sim(self.n, index, matrix, G1, G2, n1, n2, self.alpha, self.beta)
        runner = bp.DSRunner(model, monitors=['X', 'Z', 'spike', 't_last_spike'], dt=self.run_dt)
        runner.run(self.run_t)
        # Vol = runner.mon.x.T
        # Vol = (runner.mon.x1 + runner.mon.x2).T
        Vol = runner.mon.X.T
        T = runner.mon.ts
        spike = runner.mon.spike
        t_last_spike = runner.mon.t_last_spike
        return Vol, T, spike, t_last_spike

    def run(self):
        self.make_data_folder()
        graph, matrix, G1, G2, n1, n2, cortices, regions = self.net_gen()
        self.n = np.shape(matrix)[0]
        Vol, T, spike, t_last_spike = self.Vol_gen(matrix, G1, G2, n1, n2)
        data = {'graph': graph, 'matrix': matrix, 'G1': G1, 'G2': G2, 'n1': n1, 'n2': n2,
                'cortices': cortices, 'regions': regions,
                'Vol': Vol, 'T': T, 'spike': spike, 't_last_spike': t_last_spike}
        self.save_data(self.exp, data)


class mouse_connect_generation:
    def __init__(self, config):
        self.config = config
        self.exp = config['exp']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.run_dt = config['run_dt']
        self.run_t = config['run_t']

    def make_data_folder(self):
        dataFolder = self.config['outputFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def save_data(self, type, data):
        filename = os.path.join(self.expFolder, type+'.npy')
        np.save(filename, data)

    def net_gen(self):
        graph, matrix, G1, G2, n1, n2, cortices, regions = net.mouse()
        return graph, matrix, G1, G2, n1, n2, cortices, regions

    def Vol_gen(self, matrix, G1, G2, n1, n2):
        # bm.set_platform('gpu')
        index = np.zeros(self.n)
        index[70:77] = 1
        bm.random.seed(42)
        # matrix[70:81, 70:81][matrix[70:81, 70:81] != 0] += 0.8
        model0 = Epileptor(self.n, index, matrix, G1, G2, n1, n2, self.alpha, self.beta)
        runner0 = bp.DSRunner(model0, monitors=['x1', 'x2'], dt=self.run_dt)
        runner0.run(self.run_t)
        V0 = (runner0.mon.x1 + runner0.mon.x2).T
        # V0 = butter_filter(V0)
        T0 = runner0.mon.ts
        return V0, T0

    def run(self):
        self.make_data_folder()
        graph, matrix, G1, G2, n1, n2, cortices, regions = self.net_gen()
        self.n = np.shape(matrix)[0]
        V0, T0 = self.Vol_gen(matrix, G1, G2, n1, n2)
        data = {'graph': graph, 'matrix': matrix, 'G1': G1, 'G2': G2, 'n1': n1, 'n2': n2,
                'cortices': cortices, 'regions': regions, 'V0': V0, 'T0': T0}
        self.save_data(self.exp, data)


class mouse_control_generation:
    def __init__(self, config):
        self.config = config
        self.exp = config['exp']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.dk_min = config['dk_min']
        self.dk_max = config['dk_max']
        self.ddk = config['ddk']
        self.run_dt = config['run_dt']
        self.run_t = config['run_t']
        self.times = config['times']
        self.single_times = config['single_times']

    def make_data_folder(self):
        dataFolder = self.config['outputFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def save_data(self, filename, data, i):
        fileSave = os.path.join(filename, str(i) + '.npy')
        np.save(fileSave, data)

    def net_gen(self):
        graph, matrix, G1, G2, n1, n2, cortices, regions = net.mouse()
        return graph, matrix, G1, G2, n1, n2, cortices, regions

    def Vol_gen(self, matrix, G1, G2, n1, n2, idx):
        # bm.set_platform('gpu')
        dk_range = np.arange(self.dk_min, self.dk_max+self.ddk, self.ddk)
        filename = os.path.join(self.expFolder, str(idx))
        if not os.path.exists(filename):
            os.makedirs(filename)
        index = np.zeros(self.n)
        index[70:77] = 1
        for i, dk in enumerate(dk_range):
            dk_matrix = matrix.copy()
            dk_matrix[70:81, 70:81][dk_matrix[70:81, 70:81] != 0] += dk
            bm.random.seed(42 + idx)
            model = Epileptor(self.n, index, dk_matrix, G1, G2, n1, n2, self.alpha, self.beta)
            runner = bp.DSRunner(model, monitors=['x1', 'x2'], dt=self.run_dt)
            runner.run(self.run_t)
            Vol = (runner.mon.x1 + runner.mon.x2).T
            self.save_data(filename, Vol, i)
            runner.mon.clear()
            del model, runner, Vol
            gc.collect()

    def run(self):
        self.make_data_folder()
        graph, matrix, G1, G2, n1, n2, cortices, regions = self.net_gen()
        self.n = np.shape(matrix)[0]
        for idx in range(self.times):
            self.Vol_gen(matrix, G1, G2, n1, n2, idx)
            print('finish ' + str(idx) + '-generation')
            print('------------------------------')
            gc.collect()


class tvb_generation:
    def __init__(self, config):
        self.config = config
        self.exp = config['exp']
        self.run_dt = config['run_dt']
        self.run_t = config['run_t']

    def make_data_folder(self):
        dataFolder = self.config['outputFolder']
        self.expFolder = os.path.join(dataFolder, self.exp)
        if not os.path.exists(self.expFolder):
            os.makedirs(self.expFolder)

    def save_data(self, type, data):
        filename = os.path.join(self.expFolder, type+'.npy')
        np.save(filename, data)

    def net_gen(self):
        graph, matrix = net.mouse()
        return graph, matrix

    def Vol_gen(self, matrix, G1, G2, n1, n2):
        # bm.set_platform('gpu')
        index = np.zeros(self.n)
        index[70:77] = 1
        bm.random.seed(42)
        # matrix[70:81, 70:81][matrix[70:81, 70:81] != 0] += 0.8
        model0 = Epileptor(self.n, index, matrix)
        runner0 = bp.DSRunner(model0, monitors=['x1', 'x2'], dt=self.run_dt)
        runner0.run(self.run_t)
        V0 = (runner0.mon.x1 + runner0.mon.x2).T
        # V0 = butter_filter(V0)
        T0 = runner0.mon.ts
        return V0, T0

    def run(self):
        self.make_data_folder()
        graph, matrix = self.net_gen()
        self.n = np.shape(matrix)[0]
        V0, T0 = self.Vol_gen(matrix)
        data = {'matrix': matrix, 'V0': V0, 'T0': T0}
        self.save_data(self.exp, data)