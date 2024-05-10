import nevergrad as ng
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from _sdp import vec_to_kdiag

W = np.array([[0, 2, 1, 0, 0],
              [2, 0, 4, 0, 3],
              [1, 4, 0, 5, 0],
              [0, 0, 5, 0, 1],
              [0, 3, 0, 1, 0]])


def laplacian(W):
    return -np.array(W) + np.diag(np.sum(W, axis=1))


L = laplacian(W)
print(L)

def oracul(x):
    Q = np.diag(x)
    #print('oracul:', Q)
    #print(np.trace(Q))
    return np.trace(Q)


def print_candidate_and_value(optimizer, candidate, value):
    print(candidate, value)

def is_psd(M):
    return np.all(np.linalg.eigvals(M) >= 0)

def semidef_kdiag(x):
    U = np.diag(x) - L
    # print(U)
    return is_psd(U)

def nearest_psd_ge_diag_lambda(M, grid):
    if len(grid) == 1:
        return grid[0]
    if len(grid) == 2:
        if is_psd(grid[0] * np.eye(M.shape[0]) - M):
            return grid[0]
        else:
            return grid[1]
    mid = len(grid) // 2
    if is_psd(grid[mid] * np.eye(M.shape[0]) - M):
        return nearest_psd_ge_diag_lambda(M, grid[:mid+1])
    else:
        return nearest_psd_ge_diag_lambda(M, grid[mid+1:])


OPTs = [ng.optimizers.OnePlusLambda, ng.optimizers.OnePlusOne, ng.optimizers.CMApara, ng.optimizers.CMAsmall, ng.optimizers.Powell, ng.optimizers.RPowell, ng.optimizers.VoronoiDE, ng.optimizers.TwoPointsDE, ng.optimizers.PortfolioDiscreteOnePlusOne, ng.optimizers.TBPSA, ng.optimizers.RandomSearch]

best = 100
best_opt = ''
for opt in OPTs:
    print(str(opt))
    grid = np.linspace(1, 20, 100)
    l = nearest_psd_ge_diag_lambda(L, grid)
    print(l)
    print("NOW")

    optimizer = opt(parametrization=ng.p.Array(init=l * np.ones(5)), budget=1000)
    optimizer.parametrization.register_cheap_constraint(lambda x: np.all(np.linalg.eigvals(np.diag(x) - L) >= 0))
    recommendation = optimizer.minimize(oracul)  # triggers a print at each tell within minimize
    answer = oracul(recommendation.value)
    #print('recommendation:', recommendation.value)
    print("Dual grad-free solver optimal value is ", answer, '\n')
    if (answer < best):
        best = answer
        best_opt = str(opt)
    #print(np.linalg.eigvals(np.diag(recommendation.value) - L), '\n')

print(best_opt)
print(best)

