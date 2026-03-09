# coding:utf-8
import os

import brainpy as bp
import brainpy.math as bm
import numpy as np

from tools.methods import Epileptor, Epileptor_sim, butter_filter
import matplotlib.pyplot as plt

bm.random.seed(42)

model = Epileptor(1, index=bm.zeros(1), coupling='no')

runner_single = bp.DSRunner(model, monitors=['x1', 'y1', 'z', 'x2'], dt=0.01)
runner_single.run(5000)
x1 = runner_single.mon.x1
y1 = runner_single.mon.y1
z = runner_single.mon.z
x2 = runner_single.mon.x2
T = runner_single.mon.ts
T = bm.array(T).reshape(-1, 1)

plt.figure('Vol-SLE', figsize=(12, 5))
plt.plot(T, (x1+x2), label='x1+x2', color='black')
plt.plot(T, z, label='z', color='green')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Time', fontsize=15)
plt.legend(fontsize=20, loc='upper right')
plt.savefig(os.path.join('data/figures/epileptor/' + 'Vol-NS.pdf'))

model1 = Epileptor(1, index=bm.ones(1), coupling='no')

runner_single = bp.DSRunner(model1, monitors=['x1', 'y1', 'z', 'x2'], dt=0.01)
runner_single.run(5000)
x1_1 = runner_single.mon.x1
y1_1 = runner_single.mon.y1
z_1 = runner_single.mon.z
x2_1 = runner_single.mon.x2
T_1 = runner_single.mon.ts
T_1 = bm.array(T_1).reshape(-1, 1)

plt.figure('Vol-NS', figsize=(12, 5))
plt.plot(T_1, (x1_1+x2_1), label='x1+x2', color='black')
plt.plot(T_1, z_1, label='z', color='green')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Time', fontsize=15)
plt.legend(fontsize=20, loc='upper right')
plt.savefig(os.path.join('data/figures/epileptor/' + 'Vol-SLE.pdf'))

plt.figure('magnify_x1 Vol', figsize=(9, 5))
plt.plot(T_1[170000:290000], x1_1[170000:290000], label='x1', color='black')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Time', fontsize=15)
plt.legend(fontsize=20, loc='upper right')
plt.savefig(os.path.join('data/figures/epileptor/' + 'magnify_x1.pdf'))

plt.figure('magnify_x2 Vol', figsize=(9, 5))
plt.plot(T_1[170000:290000], x2_1[170000:290000], label='x2', color='black')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Time', fontsize=15)
plt.legend(fontsize=20, loc='upper right')
plt.savefig(os.path.join('data/figures/epileptor/' + 'magnify_x2.pdf'))


fig = plt.figure('phase')
bm.enable_x64()
ppa = bp.analysis.PhasePlane2D(
    model=model1,
    target_vars={'x1': [-2.0, 0.7], 'y1': [-15.0, 1.0]},
    fixed_vars={'z': 3.1, 'x2': 1.0, 'y2': -1.0, 'g': 0.0},
    resolutions=0.05
)
ppa.plot_nullcline()
# ppa.plot_nullcline(with_return=True, tol_nullcline=1e-3)  # plot nullcline
ppa.plot_fixed_point()
ppa.plot_vector_field(plot_style=dict(color='lightgrey', density=1.))
# plt.plot(x1[0:1000, :], y1[0:1000, :], label='trajectory', color='black')
ppa.plot_trajectory({'x1':[0.], 'y1': [-4.0]}, duration=200., color='black', linewidth=2, alpha=0.9, show=False)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.legend(fontsize=15)
plt.savefig(os.path.join('data/figures/epileptor/' + 'phase.pdf'))


model2 = Epileptor(2, index=bm.array([1, 0]), S=[[0, 2], [2, 0]])
runner_coupling = bp.DSRunner(model2, monitors=['x1', 'y1', 'x2'])
runner_coupling.run(10000)
x1_2 = runner_coupling.mon.x1
y1_2 = runner_coupling.mon.y1
x2_2 = runner_coupling.mon.x2
T_2 = runner_coupling.mon.ts
T_2 = bm.array(T_2).reshape(-1, 1)

plt.figure('NS2SLE', figsize=(16, 6))
plt.plot(T_2, (x1_2[:, 0]+x2_2[:, 0]), label='Pathological node', color='lightcoral')
plt.plot(T_2, (x1_2[:, 1]+x2_2[:, 1]+4), label='Healthy node', color='dodgerblue')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Time', fontsize=15)
plt.legend(fontsize=20, loc='upper right')
plt.savefig(os.path.join('data/figures/epileptor/' + 'NS2SLE.pdf'))

plt.figure('Bifurcation', figsize=(16, 6))
bm.enable_x64()
ppa = bp.analysis.PhasePlane2D(
    model=model2,
    target_vars={'x1': [-2.5, 0.7], 'y1': [-18.0, 2.0]},
    fixed_vars={'z': 4.4, 'x2': 1.0, 'y2': -1.0, 'g': 0.0},
    resolutions=0.05
)
ppa.plot_nullcline(with_return=True, tol_nullcline=1e-3)  # plot nullcline
ppa.plot_fixed_point()
ppa.plot_vector_field(plot_style=dict(color='lightgrey', density=1.))
plt.plot(x1_2[18000:24000, 1], y1_2[18000:24000, 1], label='trajectory', color='black')
# ppa.plot_trajectory({'x1':[1.], 'y1': [-4.0]}, duration=200., color='black', linewidth=2, alpha=0.9, show=True)

bif = bp.analysis.Bifurcation2D(
    model=model1,
    target_vars={'x1': [-3., 0.], 'y1': [-19., 2.]},
    fixed_vars={'x2': 1.0, 'y2': -1.0, 'g': 0.0},
    target_pars={'z': [2.5, 4.5]},
    resolutions=0.05
)
bif.plot_bifurcation()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.legend(fontsize=20)
plt.savefig(os.path.join('data/figures/epileptor/' + 'bifurcation.pdf'), bbox_inches='tight')

# net = np.ones((10, 10))
# net = np.random.random((10, 10))
# net[net >= 0.5] = 1
# net[net < 0.5] = 0
# # for i in range(10):
# #     net[i,i] = 0
# index = bm.zeros(10)
# index[0:2] = 0
# model3 = Epileptor(10, index=index, S=net, coupling='low')
#
# runner_single = bp.DSRunner(model3, monitors=['x1', 'x2'], dt=0.05)
# runner_single.run(10000)
# x1_1 = runner_single.mon.x1
# x2_1 = runner_single.mon.x2
# T_1 = runner_single.mon.ts
# vol = (x1_1 + x2_1).T
# plt.figure('noise_Vol')
# for i in range(3):
#     plt.plot(T_1[30000:50000], vol[i, 30000:50000] - (i * 4), color='black')
# plt.axis('off')
# plt.savefig(os.path.join('data/figures/epileptor/' + 'sketch.eps'))
# # plt.show()
#
#
# plt.figure('filter_Vol')
# filter_vol = butter_filter(vol)
# for i in range(4):
#     plt.plot(T_1[30000:50000], filter_vol[i, 30000:50000] - (i * 2), color='black')
# plt.axis('off')
# plt.savefig(os.path.join('data/figures/epileptor/' + 'sketch_butter.eps'))

# # modelsim1 = Epileptor(1, index=bm.ones(1), coupling='no')
# #
# # runner_single = bp.DSRunner(modelsim1, monitors=['x1', 'y1', 'z', 'x2'], dt=0.01)
# # runner_single.run(5000)
# # x1_1 = runner_single.mon.x1
# # y1_1 = runner_single.mon.y1
# # z_1 = runner_single.mon.z
# # x2_1 = runner_single.mon.x2
# # T_1 = runner_single.mon.ts
# # T_1 = bm.array(T_1).reshape(-1, 1)
# #
# # plt.figure('Vol', figsize=(12, 5))
# # plt.plot(T_1, (x1_1+x2_1), label='x1+x2', color='black')
# # plt.plot(T_1, z_1, label='z', color='green')
# # plt.xticks(fontsize=10)
# # plt.legend(fontsize=20, loc='upper right')
# # plt.savefig(os.path.join('data/figures/mouse_chimera/' + 'Vol.eps'))
# #
# modelsim = Epileptor_sim(1, bm.ones(1), coupling='no')
# runner_sim = bp.DSRunner(modelsim, monitors=['X', 'Z'], dt=0.1)
# runner_sim.run(5000)
# X = runner_sim.mon.X
# Z = runner_sim.mon.Z
# T = runner_sim.mon.ts
# plt.figure('X-Vol', figsize=(12, 5))
# plt.plot(T, X, label='X', color='black')
# plt.plot(T, Z, label='Z', color='green')
# plt.xticks(fontsize=10)
# plt.legend(fontsize=20, loc='upper right')
# plt.savefig(os.path.join('data/figures/mouse_chimera/' + 'X-Vol.eps'))