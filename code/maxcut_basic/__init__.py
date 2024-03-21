# coding: utf-8

"""Max-Cut problem basic solution. The resourse: https://github.com/pandrey-fr/maxcut/tree/master"""

import riemannian
from _solvers import MaxCutBM, MaxCutSDP
from _graphs import load_gset_graph, generate_sbm
f = open('test_results.txt', 'w')
f.write('5')
for i in range(1, 10):

    graph = load_gset_graph(f"tests/G{i}")
    print()
    MaxCutSDP(graph).solve(f)

