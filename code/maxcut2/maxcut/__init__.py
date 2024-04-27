# coding: utf-8

"""Max-Cut problem solving tools following a variety of approaches."""
import numpy as np
import riemannian
import networkx as nx
import nevergrad as ng

from _solvers import MaxCutBM, MaxCutSDP
from _graphs import load_gset_graph, generate_sbm
from _solvers._sdp import kdiag_solver, diag_oracle_solve_real_cut

f = open('test_results.txt', 'w')
# f.write('5')
ratio_of_success = []
# names = ['pm1s_100']
# ks = [1, 3, 5, 7, 9]


def get_laplacian(W):
    return -np.array(W) + np.diag(np.sum(W, axis=1))


names = ['g05_60', 'g05_80', 'g05_100', 'pm1d_80', 'pm1s_100', 'pw01_100', 'pw05_100', 'pw09_100', 'w01_100', 'w05_100',
         'w09_100']

k = 13
for name in names:
    for i in range(10):
        graph = load_gset_graph(f"tests/rudy/{name}.{i}")
        f = open(f'test_results/rudy/basic_{name}.{i}_results.txt', 'w')
        W = np.array(nx.adjacency_matrix(graph).toarray())
        L = get_laplacian(W)
        ans, mat = kdiag_solver(k=k, n=W.shape[0], steps=10, L=get_laplacian(W), OPT=ng.optimizers.Powell)
        x = diag_oracle_solve_real_cut(mat, k=k)
        x = 2 * np.array(x) - np.ones(len(x))
        truecut = x.T @ L @ x / 4
        ratio = 2 * truecut / np.sum(W)
        ratio_of_success.append(ratio)
    print(name, "%0.f-diag" % k, np.mean(np.array(ratio_of_success)))

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


#
# graph = load_gset_graph(f"tests/G0")
# f = open(f'test_results/G0_results.txt', 'w')
# # f.write('5')
#
# L = np.array([
# [ 14, -10,  -4,   0   ,0],
# [-10,  14,  -3 , -1   ,0],
# [ -4,  -3  ,28 ,-15  ,-6],
# [  0 , -1, -15 , 24 , -8],
# [  0  , 0 , -6 , -8,  14]
# ])
#
#
# # L = np.array([
# # [ 14, -10,  -4,   0   ,0],
# # [-10,  14,  -3 , -1   ,0],
# # [ -4,  -3  ,22 ,-15  ,-6],
# # [  0 , -1, -15 , 16 , -8],
# # [  0  , 0 , -6 , -8,  14]
# # ])
# # L = np.array([[3 , -2 , -1 , 0 , 0] ,
# # [ -2 , 9 , -4 , 0 , -3] ,
# #  [ -1 , -4 , 10 , -5 , 0] ,
# # [0 , 0 , -5 , 6 , -1] ,
# #  [0 , -3 , 0 , -1 , 4]])
# # print(L.shape[0])
# k = 5
# # graph = load_gset_graph(f"tests/rudy/{name}.{i}")
#
# sdp = MaxCutSDP(graph).diag_oracle_solve(L, k)
# ratio_of_success.append(sdp)
# print(sdp)
# # print(sdp.get_results(f, 'value'), sdp.get_results(f, 'cut'))
# # print(sdp.get_solution('value'))
# # print(name, "basic", np.mean(np.array(ratio_of_success)))
