# coding: utf-8

"""Max-Cut problem solving tools following a variety of approaches."""
import numpy as np
import riemannian
from _solvers import MaxCutBM, MaxCutSDP
from _graphs import load_gset_graph, generate_sbm
f = open('test_results.txt', 'w')
# f.write('5')
ratio_of_success = []
# names = ['pm1s_100']
names = ['g05_60', 'g05_80', 'g05_100', 'pm1d_80', 'pm1s_100', 'pw01_100', 'pw05_100', 'pw09_100',  'w01_100', 'w05_100', 'w09_100']
ks = [1, 3, 5, 7, 9]
# for name in names:
#     for i in range(10):
#         graph = load_gset_graph(f"tests/rudy/{name}.{i}")
#         f = open(f'test_results/rudy/basic_{name}.{i}_results.txt', 'w')
#         # f.write('5')
#
#         sdp = MaxCutSDP(graph).solve(f, basic=True)
#         ratio_of_success.append(sdp)
#         # print(sdp.get_results(f, 'value'), sdp.get_results(f, 'cut'))
#         # print(sdp.get_solution('value'))
#     print(name, "basic", np.mean(np.array(ratio_of_success)))
#
# for name in names:
#     for i in range(10):
#         graph = load_gset_graph(f"tests/rudy/{name}.{i}")
#         f = open(f'test_results/rudy/diag_{name}.{i}_results.txt', 'w')
#         # f.write('5')
#
#         sdp = MaxCutSDP(graph).solve(f, basic=False)
#         ratio_of_success.append(sdp)
#         # print(sdp.get_results(f, 'value'), sdp.get_results(f, 'cut'))
#         # print(sdp.get_solution('value'))
#     print(name, "diag", np.mean(ratio_of_success))



graph = load_gset_graph(f"tests/G0")
f = open(f'test_results/G0_results.txt', 'w')
# f.write('5')
L = np.array([[3 , -2 , -1 , 0 , 0] ,
[ -2 , 9 , -4 , 0 , -3] ,
 [ -1 , -4 , 10 , -5 , 0] ,
[0 , 0 , -5 , 6 , -1] ,
 [0 , -3 , 0 , -1 , 4]])
print(L.shape[0])
k = 15
sdp = MaxCutSDP(graph).diag_oracle_solve(L, k)
ratio_of_success.append(sdp)
# print(sdp.get_results(f, 'value'), sdp.get_results(f, 'cut'))
# print(sdp.get_solution('value'))
# print(name, "basic", np.mean(np.array(ratio_of_success)))
