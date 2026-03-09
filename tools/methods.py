# coding:utf-8
import brainpy as bp
import brainpy.math as bm
import numpy as np
from scipy.signal import butter, filtfilt
from tools.utils import section, link_section


class HR(bp.NeuGroup):
    def __init__(self, size, G1=0, G2=0, n1=0, n2=0, alpha=0.04, beta=0.02, coupling='mouse', name=None):
        super(HR, self).__init__(size=size, name=name)

        self.x_0 = -1.6
        self.tau_0 = 200
        self.V_th = 0
        self.Iext = 4.4

        self.G1 = bm.asarray(G1)
        self.G2 = bm.asarray(G2)
        self.n1 = bm.asarray(n1)
        self.n2 = bm.asarray(n2)
        self.alpha = alpha
        self.beta = beta
        self.x_rev = 2
        self.lamb = 10
        self.theta = -0.25

        # bm.random.seed(123)
        self.x = bm.Variable(bm.random.randn(self.num)-1.6)
        self.z = bm.Variable(bm.ones(self.num)*1.4)
        self.y = bm.Variable(bm.ones(self.num)*-10.)
        self.t_last_spike = bm.Variable(bm.ones(self.num) * 1e-7)
        self.spike = bm.Variable(bm.zeros(self.num, dtype='bool'))

        self.coupling = coupling
        self.integral = bp.odeint(f=self.derivative, method='exp_auto')

    def Theta(self, x1):
        xk, xj = bm.meshgrid(x1, x1)
        return (xj - self.x_rev) / (1 + bm.exp(-self.lamb * (xk - self.theta)))

    @property
    def derivative(self):
        if self.coupling == 'chi':
            return bp.JointEq(self.dx_chi, self.dy, self.dz)
        elif self.coupling == 'mouse':
            return bp.JointEq(self.dx, self.dy, self.dz_mouse)
        elif self.coupling == 'no':
            return bp.JointEq(self.dx, self.dy, self.dz)
        else:
            print('Unknown coupling', self.coupling)

    def dx(self, x, t, y, z):
        dxdt = y - x**3 + 3 * x**2 - z + self.Iext
        return dxdt

    def dx_chi(self, x, t, y, z):
        dxdt = (y - x**3 + 3 * x**2 - z + self.Iext
               - self.alpha/self.n1 * bm.sum(self.G1*self.Theta(x), axis=1)
               - self.beta/self.n2 * bm.sum(self.G2*self.Theta(x), axis=1))
        return dxdt + bm.random.normal(0, 0.0025)

    def dx_mouse(self, x, t, y, z):
        dxdt = (y - x**3 + 3 * x**2 - z + self.Iext
                + self.alpha * (-x @ self.G1 + bm.sum(self.G1, axis=0) * x)
                + self.beta * (-x @ self.G2 + bm.sum(self.G2, axis=0) * x)) / self.tau_0
        return dxdt + bm.random.normal(0, 0.0025)

    def dy(self, y, t, x):
        dydt = 1 - 5 * x**2 - y
        return dydt

    def dz(self, z, t, x):
        dzdt = (4 * (x - self.x_0) - z)/self.tau_0
        return dzdt

    def dz_mouse(self, z, t, x):
        dzdt = (4 * (x - self.x_0) - z + self.alpha*(-x @ self.G1 + bm.sum(self.G1, axis=0) * x)
                + self.beta*(-x @ self.G2 + bm.sum(self.G2, axis=0) * x)) / self.tau_0
        return dzdt

    def update(self):
        t, dt = bp.share['t'], bp.share['dt']
        x, y, z = self.integral(self.x, self.y, self.z, t, dt=dt)
        self.spike.value = bm.logical_and(x >= self.V_th, self.x < self.V_th)
        self.t_last_spike.value = bm.where(self.spike, t, self.t_last_spike)
        self.x.value = x
        self.y.value = y
        self.z.value = z
        # self.z.value = bm.where(self.z >= 4.5, 4.5, z)
        # self.z.value = bm.where(self.z <= 2.5, 2.5, z)


class Epileptor(bp.NeuGroup):
    def __init__(self, size, index, S=0, G1=0, G2=0, n1=0, n2=0, alpha=0.04, beta=0.02, coupling='low'):
        super(Epileptor, self).__init__(size=size)
        # self.num = num
        self.tau_0 = 2857
        self.tau_2 = 10
        self.I1 = 3.1
        self.I2 = 0.45
        self.gamma = 0.01
        self.x_0 = -2.5 * bm.ones(self.num)
        self.x_0[index == 1] = -1.5
        self.V_rest = -1.6

        self.x_rev = 2
        self.lamb = 10
        self.theta = -0.25
        self.alpha = alpha
        self.beta = beta
        self.V_th = 0.
        self.S = bm.asarray(S)
        self.G1 = bm.asarray(G1)
        self.G2 = bm.asarray(G2)
        self.n1 = bm.asarray(n1)
        self.n2 = bm.asarray(n2)

        # bm.random.seed(111)
        # self.x1 = bm.Variable(bm.random.normal(loc=0, scale=0.5, size=self.num))
        self.x1 = bm.Variable(bm.random.rand(self.num))
        # self.x1 = bm.Variable(bm.random.rand(self.num)-2.5)
        self.y1 = bm.Variable(-5*bm.ones(self.num))
        # self.z = bm.Variable(2.5 + bm.random.rand(self.num))
        self.z = bm.Variable(3*bm.ones(self.num))
        self.x2 = bm.Variable(bm.zeros(self.num))
        self.y2 = bm.Variable(bm.zeros(self.num))
        self.g = bm.Variable(bm.zeros(self.num))
        self.spike = bm.Variable(bm.zeros(self.num, dtype='bool'))
        self.t_last_spike = bm.Variable(bm.ones(self.num) * 1e-7)

        self.coupling = coupling
        self.integral = bp.odeint(f=self.derivative, method='exp_auto')

    def f1(self, x1, x2, z):
        return bm.heaviside(x1, 1) * x1 * (x2 - 0.6 * (z - 4) ** 2) + bm.heaviside(-x1, 0) * ((x1 ** 3) - 3 * (x1 ** 2))

    def f2(self, x1, x2):
        return bm.heaviside(x2 + 0.25, 1) * 6 * (x2 + 0.25)

    def Theta(self, x1):
        xk, xj = bm.meshgrid(x1, x1)
        return (xj - self.x_rev) / (1 + bm.exp(-self.lamb * (xk - self.theta)))
        # reversal = x1 - self.x_rev
        # activation = 1 + bm.exp(-self.lamb*(x1 - self.theta))
        # return reversal @ (G / bm.tile(activation.reshape(-1, 1), (1, len(activation))))

    @property
    def derivative(self):
        if self.coupling == 'low':
            return bp.JointEq(self.dx1, self.dy1, self.dz_K, self.dx2, self.dy2, self.dg)
        elif self.coupling == 'chi':
            # return bp.JointEq(self.dx1, self.dy1, self.dz_chi, self.dx2, self.dy2, self.dg)
            return bp.JointEq(self.dx1_chi, self.dy1, self.dz, self.dx2, self.dy2, self.dg)
        elif self.coupling == 'no':
            return bp.JointEq(self.dx1, self.dy1, self.dz, self.dx2, self.dy2, self.dg)
        else:
            print('Unknown coupling', self.coupling)

    def dx1(self, x1, t, x2, z, y1):
        dx1dt = y1 - self.f1(x1, x2, z) - z + self.I1
        # return dx1dt
        return dx1dt + bm.random.normal(0, 0.025, size=x1.shape)

    def dx1_chi(self, x1, t, x2, z, y1):
        dx1dt = (y1 - self.f1(x1, x2, z) - z + self.I1
                 - self.alpha/self.n1 * bm.sum(self.G1*self.Theta(x1), axis=1)
                 - self.beta/self.n2 * bm.sum(self.G2*self.Theta(x1), axis=1))
        # return dx1dt
        return dx1dt + bm.random.normal(0, 0.025, self.x1.shape)

    def dy1(self, y1, t, x1):
        dy1dt = 1 - 5 * x1**2 - y1
        # return dy1dt
        return dy1dt + bm.random.normal(0, 0.025, size=y1.shape)

    def dz(self, z, t, x1):
        dzdt = (4 * (x1 - self.x_0) - z) / self.tau_0
        return dzdt

    def dz_K(self, z, t, x1):
        dzdt = (4 * (x1 - self.x_0) - z - x1 @ self.S + bm.sum(self.S, axis=0) * x1) / self.tau_0
        return dzdt

    def dz_chi(self, z, t, x1):
        dzdt = (4 * (x1 - self.x_0) - z + self.alpha/self.n1 * bm.sum(self.G1*self.Theta(x1), axis=1)
                 + self.beta/self.n2 * bm.sum(self.G2*self.Theta(x1), axis=1)) / self.tau_0
        return dzdt

    def dx2(self, x2, t,  y2, z, g):
        dx2dt = -y2 + x2 - x2 ** 3 + self.I2 + 0.002 * g - 0.3 * (z - 3.5)
        # return dx2dt
        return dx2dt + bm.random.normal(0, 0.25, size=x2.shape)

    def dy2(self, y2, t, x1, x2):
        dy2dt = (self.f2(x1, x2) - y2) / self.tau_2
        # return dy2dt
        return dy2dt + bm.random.normal(0, 0.25, size=y2.shape)

    def dg(self, g, t, x1):
        dgdt = x1 - self.gamma * g
        return dgdt

    def update(self):
        t, dt = bp.share['t'], bp.share['dt']
        x1, y1, z, x2, y2, g = self.integral(self.x1, self.y1, self.z, self.x2, self.y2, self.g, t, dt=dt)
        self.spike.value = bm.logical_and(x1 >= self.V_th, self.x1 < self.V_th)
        self.t_last_spike.value = bm.where(self.spike, t, self.t_last_spike)
        # self.x1.value = bm.where((self.x1 >= 3), 2, x1)
        # self.x1.value = bm.where((self.x1 <= -3), -3, x1)
        self.x1.value = x1
        self.y1.value = y1
        self.z.value = bm.where(self.z >= 4.5, 4.5, z)
        self.z.value = bm.where(self.z <= 2.5, 2.5, z)
        # self.z.value = z
        # self.x2.value = bm.where((self.x2 >= 3), 3, x2)
        # self.x2.value = bm.where((self.x2 <= -3), -3, x2)
        self.x2.value = x2
        self.y2.value = y2
        self.g.value = g


class Epileptor_sim(bp.NeuGroup):
    def __init__(self, size, index, S=0, G1=0, G2=0, n1=0, n2=0, alpha=0.04, beta=0.02, coupling='low'):
        super(Epileptor_sim, self).__init__(size=size)
        # self.num = num
        self.tau_0 = 500    # paras in phi
        # self.tau_0 = 3400
        self.I1 = 3.1
        self.x_0 = -2.5 * bm.ones(self.num)
        self.x_0[index == 1] = -1.5
        self.V_rest = -1.5

        self.x_rev = 2
        self.lamb = 10
        self.theta = -0.25
        self.alpha = alpha
        self.beta = beta
        self.V_th = 0
        self.S = bm.asarray(S)
        self.G1 = bm.asarray(G1)
        self.G2 = bm.asarray(G2)
        self.n1 = bm.asarray(n1)
        self.n2 = bm.asarray(n2)

        bm.random.seed(111)
        self.X = bm.Variable(bm.random.rand(self.num) - 3)
        # self.X = bm.Variable(bm.random.rand(self.num) - 1.6)  # phi
        # self.X = bm.Variable(bm.zeros(self.num))
        self.Z = bm.Variable(bm.random.rand(self.num) + 3)    # phi
        # self.Z = bm.Variable(3*bm.ones(self.num))
        self.spike = bm.Variable(bm.zeros(self.num, dtype='bool'))
        self.t_last_spike = bm.Variable(bm.ones(self.num) * 1e-7)

        self.coupling = coupling
        self.integral = bp.odeint(f=self.derivative, method='exp_auto')

    def Theta(self, x1):
        xk, xj = bm.meshgrid(x1, x1)
        return (xj - self.x_rev) / (1 + bm.exp(-self.lamb * (xk - self.theta)))

    @property
    def derivative(self):
        if self.coupling == 'low':
            return bp.JointEq(self.dX, self.dZ_K)
        elif self.coupling == 'chi':
            return bp.JointEq(self.dX_chi, self.dZ)
        elif self.coupling == 'mouse':
            return bp.JointEq(self.dX, self.dZ_mouse)
        elif self.coupling == 'no':
            return bp.JointEq(self.dX, self.dZ)
        else:
            print('Unknown coupling', self.coupling)

    def dX(self, X, t, Z):
        dXdt = 1 - X**3 - 2*X**2 - Z + self.I1
        return dXdt

    def dX_chi(self, X, t, Z):
        dXdt = (1 - X**3 - 2*X**2 - Z + self.I1 +
                - self.alpha / self.n1 * bm.sum(self.G1 * self.Theta(X), axis=1)
                - self.beta / self.n2 * bm.sum(self.G2 * self.Theta(X), axis=1))
        return dXdt

    def dZ(self, Z, t, X):
        dZdt = (4 * (X - self.x_0) - Z) / self.tau_0
        return dZdt

    def dZ_K(self, Z, t, X):
        dZdt = (4 * (X - self.x_0) - Z - X @ self.S + bm.sum(self.S, axis=0) * X) / self.tau_0
        return dZdt

    def dZ_chi(self, Z, t, X):
        dZdt = (4 * (X - self.x_0) - Z + self.alpha/self.n1 * bm.sum(self.G1*self.Theta(X), axis=1)
                 + self.beta/self.n2 * bm.sum(self.G2*self.Theta(X), axis=1)) / self.tau_0
        return dZdt

    def dZ_mouse(self, Z, t, X):
        dZdt = (4 * (X - self.x_0) - Z + self.alpha*(-X @ self.G1 + bm.sum(self.G1, axis=0) * X)
                + self.beta*(-X @ self.G2 + bm.sum(self.G2, axis=0) * X)) / self.tau_0
        return dZdt

    def update(self):
        t, dt = bp.share['t'], bp.share['dt']
        X, Z = self.integral(self.X, self.Z, t, dt=dt)
        self.spike.value = bm.logical_and(X >= self.V_th, self.X < self.V_th)
        self.t_last_spike.value = bm.where(self.spike, t, self.t_last_spike)
        self.X.value = X
        self.Z.value = bm.where(self.Z >= 4.5, 4.5, Z)
        self.Z.value = bm.where(self.Z <= 2.5, 2.5, Z)
        # self.Z.value = Z


def phase(spikeses, T_mask, T):
    # T_mask = np.zeros((T.shape[0], T.shape[1]))
    # Tp1_mask = T_mask.copy() + 1
    Tp1_mask = np.tile(T[:, -1], (T.shape[1], 1)).T
    t = T[0, :]
    for i, spikes in enumerate(spikeses):
        for spike in reversed(spikes):
            Tp1_mask[i, t<spike] = spike
        if i % 50 == 0:
            print('finish 50 neurons')
    # for i, spikes in enumerate(spikeses):
    #     for j, spike in enumerate(spikes[:-1]):
    #         T_mask[i, t>=spike] = spike
    #         Tp1_mask[i, t<=spike] = spikes[j + 1]
    #     if i % 50 == 0:
    #         print('finish 50 neurons')
    divisor = np.where(Tp1_mask - T_mask == 0, 1, Tp1_mask - T_mask)
    phi = np.where(divisor == 0, 0, 2 * np.pi * (T - T_mask) / divisor)
    return phi


def butter_filter(Vol, order=5, fs=500, low_cut=0.16, high_cut=97.0):
    nyquist = 0.5 * fs
    low = low_cut / nyquist
    high = high_cut / nyquist
    b, a = butter(order, [low, high], btype='band', analog=False)
    filtered_data = filtfilt(b, a, Vol, axis=1)
    return filtered_data


def EI(Vol):
    pass


def fft(Vol, T_window, sampling_rate, DC_shift=True):
    n = Vol.shape[0]
    T = Vol.shape[1]
    n_window = T // T_window
    freq = np.zeros((n, n_window))
    energy = np.zeros((n, n_window))
    if DC_shift is True:
        for i in range(n_window):
            data = Vol[:, i*T_window: (i+1)*T_window]
            num_neurons, num_time_points = data.shape

            # 计算直流分量 DC
            DC = np.mean(data, axis=1)
            DC_idx = DC < -1  # 判断是否存在直流偏置

            # 减去直流偏置
            data = data - np.mean(data, axis=1, keepdims=True)

            # 计算傅里叶变换
            fft_result = np.fft.fft(data, axis=1)

            # 计算频谱
            power_spectrum = np.abs(fft_result) ** 2
            frequencies = np.fft.fftfreq(num_time_points, d=1 / sampling_rate)

            # 截取正频谱部分
            positive_frequencies_mask = frequencies > 0
            frequencies = frequencies[positive_frequencies_mask]
            power_spectrum = power_spectrum[:, positive_frequencies_mask]

            dominant_frequencies = frequencies[np.argmax(power_spectrum, axis=1)]   # 找到主要的频率成分
            dominant_frequencies[DC_idx] = 0 # 如果存在直流偏置，则将对应频率设为0
            dominant_energies = np.max(power_spectrum, axis=1)  # main energy
            dominant_energies[DC_idx] = 0

            freq[:, i] = dominant_frequencies
            energy[:, i] = dominant_energies
    else:
        for i in range(n_window):
            data = Vol[:, i*T_window: (i+1)*T_window]
            num_neurons, num_time_points = data.shape

            # 减去直流偏置
            data = data - np.mean(data, axis=1, keepdims=True)

            # 计算傅里叶变换
            fft_result = np.fft.fft(data, axis=1)

            # 计算频谱
            power_spectrum = np.abs(fft_result) ** 2
            frequencies = np.fft.fftfreq(num_time_points, d=1 / sampling_rate)

            # 截取正频谱部分
            positive_frequencies_mask = frequencies > 0
            frequencies = frequencies[positive_frequencies_mask]
            power_spectrum = power_spectrum[:, positive_frequencies_mask]

            dominant_frequencies = frequencies[np.argmax(power_spectrum, axis=1)]   # 找到主要的频率成分
            dominant_energies = np.max(power_spectrum, axis=1)  # main energy

            freq[:, i] = dominant_frequencies
            energy[:, i] = dominant_energies

    return freq, energy


def sync_matrix(freq, tau_max):
    """
    Get sync and causality calculating by any two different channels of the matrix and them averaging

    Args:
        freq (ndarray): events matrix
        tau_max (float): max gap time to define sync

    Returns:
        sync_matrix (ndarray): sync matrix of any two channels
        cause_matrix (ndarray): cause matrix of any two channels
    """
    n = freq.shape[0]
    sync_matrix = np.zeros((n, n))
    cause_matrix = np.zeros((n, n))

    event_matrix = np.zeros_like(freq)
    non_zero_index = np.where(freq != 0)
    event_matrix[non_zero_index] = 1
    for i in range(n):
        for j in range(n):
            link_sect_i = link_section(event_matrix[i, :], window_size=10)
            link_sect_j = link_section(event_matrix[j, :], window_size=10)
            nf_i = section(link_sect_i)
            nf_j = section(link_sect_j)

            if nf_i.size == 0 or nf_j.size == 0:
                sync_matrix[i][j] = 0
            else:
                m_i = len(nf_i)
                m_j = len(nf_j)
                if nf_i.size == 1 or nf_j.size == 1:
                    tau = tau_max
                else:
                    tau_i = np.min(np.diff(nf_i))
                    tau_j = np.min(np.diff(nf_j))
                    tau_min = 1/2 * min(tau_i, tau_j)
                    tau = tau_min if tau_min < tau_max else tau_max

                NF_i = np.tile(nf_i, (nf_j.size, 1))
                NF_j = np.tile(nf_j, (nf_i.size, 1)).T
                delay_ij = NF_i - NF_j
                delay_ij = delay_ij.astype(float)
                delay_ij[(delay_ij > 0) & (delay_ij <= tau)] = 1
                delay_ij[delay_ij == 0] = 0.5
                delay_ij[(delay_ij < 0) | (delay_ij > tau)] = 0

                NF__j = np.tile(nf_j, (nf_i.size, 1))
                NF__i = np.tile(nf_i, (nf_j.size, 1)).T
                delay_ji = NF__j - NF__i
                delay_ji = delay_ji.astype(float)
                delay_ji[(delay_ji > 0) & (delay_ji <= tau)] = 1
                delay_ji[(delay_ji < 0) | (delay_ji > tau)] = 0

                sync_matrix[i, j] = (np.sum(delay_ij) + np.sum(delay_ji)) / np.sqrt(m_i * m_j)
                cause_matrix[i, j] = (np.sum(delay_ij) - np.sum(delay_ji)) / np.sqrt(m_i * m_j)
    # np.fill_diagonal(sync_matrix, 0)
    return sync_matrix, cause_matrix


def sync_index(sync_matrix):
    n = sync_matrix.shape[0]
    sync_x = np.all(sync_matrix == 0, axis=1)
    sync_matrix_true = sync_matrix[~sync_x]
    sync_y = np.all(sync_matrix_true == 0, axis=0)
    sync_matrix_true = sync_matrix_true[:, ~sync_y]

    sync_n = sync_matrix_true.shape[0]
    index = np.mean(sync_matrix_true)
    percent = sync_n / n
    global_index = percent * index
    return percent, index, global_index


def simplify_freq(freq):
    """
    Removes rows from a 2D NumPy array where all elements are zero.

    Args:
        freq (ndarray): 2D NumPy array.

    Returns:
        ndarray: Array with all-zero rows removed.
    """
    freq_x = np.all(freq == 0, axis=1)
    simple_freq = freq[~freq_x]
    return simple_freq


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import tools.network as net

    # graph, matrix, G1, G2, n1, n2, cortices, regions = net.mouse()
    # n = np.shape(matrix)[0]
    n = 200
    _, S = net.scale_free(n)
    index = np.zeros(n)
    index[0:40] = 1
    alpha, beta = 0.04, 0.02
    # model = Epileptor_sim(n, index, matrix, G1, G2, n1, n2, alpha, beta)
    model = Epileptor(n, index, 3*S, coupling='low_electronic')
    # runner = bp.DSRunner(model, monitors=['X', 'Z', 'spike'])
    runner = bp.DSRunner(model, monitors=['x1','x2', 'z', 'spike'])
    runner.run(10000)
    # Vol = runner.mon.X.T
    Vol = (runner.mon.x1 + runner.mon.x2).T
    z = runner.mon.z.T
    T = runner.mon.ts
    spike = runner.mon.spike.T
    T_matrix = np.tile(T, (Vol.shape[0], 1))
    spike_T = T_matrix * spike
    # spikeses = [[element for element in row if element != 0] for row in spike_T]
    plt.figure()
    plt.plot(T, Vol[6, :])
    plt.plot(T, z[6,:])
    plt.show()
    freq, energy = fft(Vol, T_window=50, sampling_rate=500)
    plt.figure()
    plt.imshow(freq, cmap='ocean_r', aspect=4.5)
    plt.colorbar()
    plt.show()
    # phi = phase(spikeses, T_matrix)
    # plt.figure()
    # plt.imshow(phi)

