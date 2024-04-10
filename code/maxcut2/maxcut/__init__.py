# coding: utf-8

"""Max-Cut problem solving tools following a variety of approaches."""

import riemannian
from _solvers import MaxCutBM, MaxCutSDP
from _graphs import load_gset_graph, generate_sbm
f = open('test_results.txt', 'w')
f.write('5')
print(int(float('-189.000000')))
for i in range(2, 3):
    graph = load_gset_graph(f"tests/G{i}")


    MaxCutSDP(graph).solve(f)

