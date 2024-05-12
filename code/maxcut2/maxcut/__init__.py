# coding: utf-8

"""Max-Cut problem solving tools following a variety of approaches."""
import numpy as np
import riemannian
import networkx as nx
import nevergrad as ng

from _solvers import MaxCutBM, MaxCutSDP
from _graphs import load_gset_graph, generate_sbm
from _solvers._sdp import kdiag_solver, diag_oracle_solve_real_cut, laplacian, mat_to_kdiag, sdp_solver, dual_solver, \
    kdiag_to_vec, cholesky_cut, dynamic_cut, dual_solve_eps
import warnings

warnings.filterwarnings('ignore')

f = open('test_results.txt', 'w')
# f.write('5')
ratio_of_success = []
# names = ['pm1s_100']
# ks = [1, 3, 5, 7, 9]


names = ['g05_60', 'g05_80', 'g05_100', 'pm1d_80', 'pm1s_100', 'pw01_100', 'pw05_100', 'pw09_100', 'w01_100', 'w05_100',
         'w09_100']

# k is number of diagonals
k = 7

# optimizer
OPTs = [ng.optimizers.ChainNaiveTBPSACMAPowell]  # ng.optimizers.ChainMetaModelSQP, ng.optimizers.ChainNaiveTBPSACMAPowell, ng.optimizers.CMAsmall, ng.optimizers.CMApara, TwoPointsDE]
graph = load_gset_graph(f"tests/rudy/my38.0")
W = np.array(nx.adjacency_matrix(graph).toarray())
L = laplacian(W)
print(dynamic_cut(matrix=L, W=W, k=5))
# print(dynamic_cut())
my_str = "_________________|__________________|________________"
for optimizer in OPTs:
    print(str(optimizer))
    for name in names:
        for i in range(10):
            print(f"tests/rudy/{name}.{i}")
            graph = load_gset_graph(f"tests/rudy/{name}.{i}")
            f = open(f'test_results/rudy/basic_{name}.{i}_results.txt', 'w')

            # W is adjacency matrix of graph
            W = np.array(nx.adjacency_matrix(graph).toarray())

            # print("SDP solver optimal value is ", round(sdp_solver(W)))
            ans, mat = sdp_solver(W)
            print("SDP cholesky \t |\t Ans:", np.round(ans, 3), "  |\tTruecut:", cholesky_cut(matrix=mat, graph=graph))
            print(my_str)
            ans, mat = dual_solve_eps(W)
            print("Dual cholesky \t |\t Ans:", np.round(ans, 3), "  |\tTruecut:", cholesky_cut(matrix=mat, graph=graph))
            print(my_str)
            print("Dual dynamic \t |\t Ans:", np.round(ans, 3), "  |\tTruecut:", dynamic_cut(matrix=mat, W=W, k=1))
            print(my_str)
            ans, mat = kdiag_solver(k, W, steps=100, OPT=optimizer, init='eye')
            # print("D%0.f (eye)cholesky |\t Ans:" % k, np.round(ans, 3), "  |\tTruecut:", cholesky_cut(matrix=mat, graph=graph))
            print(my_str)
            print("D%0.f (eye)dynamic  |\t Ans:" % k, np.round(ans, 3), "  |\tTruecut:", dynamic_cut(matrix=mat, W=W, k=k))
            print(my_str)
            ans, mat = kdiag_solver(k, W, steps=100, OPT=optimizer, init='dual')
            print("D%0.f (dual)cholesky|\t Ans:" % k, np.round(ans, 3), " |\tTruecut:", cholesky_cut(matrix=mat, graph=graph))
            print(my_str)
            print("D%0.f (dual)dynamic |\t Ans:" % k, np.round(ans, 3), " |\tTruecut:", dynamic_cut(matrix=mat, W=W, k=k))
            print(my_str)
            print('\n')

            # ratio of truecut and total weight of graph edges
            #truecut = kdiag_solver(k, W, steps=100, OPT=optimizer)

            #ratio = 2 * truecut / np.sum(W)
            #ratio_of_success.append(ratio)

            # sdp_solver(W)
            # dual_solver(W)
            # kdiag_solver(1, W, steps=100, OPT=optimizer)
            # kdiag_solver(3, W, steps=100, OPT=optimizer)
            # kdiag_solver(5, W, steps=100, OPT=optimizer)
            # print('\n')

        #print(name, "D%0.f solver " % k, np.mean(ratio_of_success))


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
