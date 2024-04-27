# coding: utf-8

"""Semi-Definite Programming based solver for the Max-Cut problem."""

import cvxpy as cp
import networkx as nx
import numpy as np
import nevergrad as ng

from _solvers.backend import (
    AbstractMaxCut, get_partition, get_cut_value
)


def laplacian(W):
    return -np.array(W) + np.diag(np.sum(W, axis=1))


def get_mask(arr):
    answer = 0
    n = len(arr)
    power_of_2 = 1
    for i in range(n):
        answer += arr[i] * power_of_2
        power_of_2 *= 2
    return answer


# dynamic programming to calculate maxcut in graph with k-diagonal laplacian
def diag_oracle_solve(L, k):
    L = L.copy()
    n = L.shape[0]
    depend_number = k // 2 + 1
    dp = [[0] * (2 ** depend_number) for i in range(n)]
    for i in range(n):
        for mask in range(2 ** depend_number):
            c = [0] * depend_number
            copy_mask = mask
            for bit in range(depend_number):
                c[bit] = copy_mask % 2
                copy_mask //= 2
            best_current_ans = 0
            c_previous_0 = c[1:]
            c_previous_1 = c[1:]
            c_previous_0.append(0)
            c_previous_1.append(1)
            for j in range(max(0, i - (k // 2)), i + 1):
                # print(i, j)
                best_current_ans += 2 * (c[0] * 2 - 1) * (c[i - j] * 2 - 1) * L[i][j]
            best_current_ans -= L[i][i]
            if i > 0:
                best_current_ans += max(dp[i - 1][get_mask(c_previous_0)], dp[i - 1][get_mask(c_previous_1)])
            dp[i][mask] = best_current_ans
    answer = 0
    for mask in range(2 ** depend_number):
        answer = max(answer, dp[n - 1][mask])
    return answer


# dynamic programming to calculate maxcut in graph with k-diagonal laplacian and store the actual cut
def diag_oracle_solve_real_cut(L, k):
    n = L.shape[0]
    # print(n)
    depend_number = k // 2 + 1
    depend_number = min(depend_number, n)
    dp = [[0] * (2 ** depend_number) for i in range(n)]
    for i in range(n):
        for mask in range(2 ** depend_number):
            c = [0] * depend_number
            copy_mask = mask
            for bit in range(depend_number):
                c[bit] = copy_mask % 2
                copy_mask //= 2
            best_current_ans = 0
            c_previous_0 = c[1:]
            c_previous_1 = c[1:]
            c_previous_0.append(0)
            c_previous_1.append(1)
            # print(c, c_previous_0, c_previous_1)
            for j in range(max(0, i - (k // 2)), i + 1):
                # print(i, j)
                best_current_ans += 2 * (c[0] * 2 - 1) * (c[i - j] * 2 - 1) * L[i][j]
            best_current_ans -= L[i][i]
            if i > 0:
                best_current_ans += max(dp[i - 1][get_mask(c_previous_0)], dp[i - 1][get_mask(c_previous_1)])
            dp[i][mask] = best_current_ans
        # print('dp', dp[i])
    best_x = []
    final_mask = 0
    answer = 0
    for mask in range(2 ** depend_number):
        if dp[n - 1][mask] > answer:
            final_mask = mask
        answer = max(answer, dp[n - 1][mask])

    c = [0] * depend_number
    copy_mask = final_mask
    for bit in range(depend_number):
        c[bit] = copy_mask % 2
        copy_mask //= 2
    best_x = c.copy()
    # print(best_x)
    for i in range(n - 2, depend_number - 2, -1):
        c_previous_0 = c[1:]
        c_previous_1 = c[1:]
        c_previous_0.append(0)
        c_previous_1.append(1)
        if dp[i][get_mask(c_previous_0)] > dp[i][get_mask(c_previous_1)]:
            c = c_previous_0
            best_x.append(0)
        else:
            c = c_previous_1
            best_x.append(1)
    best_x = best_x[::-1]
    # print(answer / 4)
    # print(dp[i])
    #
    # print(best_x)
    return best_x


# remains only k diagonals of matrix X
def mat_to_kdiag(X, k):
    X = X.copy()
    n = X.shape[0]
    assert k % 2 == 1
    kD = X
    for i in range(n):
        for j in range(n):
            if abs(i - j) > k // 2:
                kD[i][j] = 0
    return kD


# converts vector to k-diagonal matrix (extracts first n elements to main diagonal, then (n-1) to second diagonal etc., and symmetry over main diagonal)
def vec_to_kdiag(x, n, k):
    x = x.copy()
    assert k % 2 == 1
    assert len(x) == (n * (n + 1) - (n - (k - 1) / 2 - 1) * (n - (k - 1) / 2)) // 2
    kD = np.diag(x[:n])
    current_index = n
    for i in range(1, (k + 1) // 2):  # rows from 1 to k-1 / 2
        for j in range(0, n - i):  # columns from 0 to n-1-i
            kD[i + j][j] = x[current_index]
            kD[j][i + j] = x[current_index]
            current_index += 1
    return kD


# converts k-diagonal matrix to vector (main diagonal, then substack second diagonal etc., and symmetry over main diagonal)
def kdiag_to_vec(X, k):
    X = X.copy()
    n = X.shape[0]
    assert k % 2 == 1
    dim = (n * (n + 1) - (n - (k - 1) // 2 - 1) * (n - (k - 1) // 2)) // 2
    x = np.zeros(dim)
    current_index = 0
    for i in range(0, (k + 1) // 2):  # from 1 to k-1 / 2
        for j in range(0, n - i):  # from 0 to n-1-i
            x[current_index] = X[i + j][j]
            current_index += 1
    return x


def kdiag_solver(k, steps, W, OPT):
    W = W.copy()
    n = W.shape[0]
    L = laplacian(W)
    # L0 = np.ones((n, n))
    L0 = np.zeros((n, n))
    # dim = (n * (n + 1) - (n - (k - 1) // 2 - 1) * (n - (k - 1) // 2)) // 2
    optimizer = OPT(parametrization=ng.p.Array(init=kdiag_to_vec(mat_to_kdiag(L + L0, k), k)), budget=steps)

    # optimizer = OPT(parametrization=ng.p.Array(shape=(dim, )), budget=steps)

    def semidef_kdiag(x):
        return np.all(np.linalg.eigvals(vec_to_kdiag(x, n, k) - L) >= 0)

    optimizer.parametrization.register_cheap_constraint(semidef_kdiag)

    def oracul(x):
        return diag_oracle_solve(vec_to_kdiag(x, n, k), k)

    recommendation = optimizer.minimize(oracul)
    answer = oracul(recommendation.value)
    return answer, vec_to_kdiag(recommendation.value, n, k)


class MaxCutSDP(AbstractMaxCut):
    """Semi-Definite Programming based solver for the Max-Cut problem.

    Given a graph with non-negative weights, the method implemented
    here aims at maximizing $$\\sum_{{i < j}} w_{{ij}}(1 - x_{{ij}})$$
    where $X = (x_{{ij}}))$ is a positive semi-definite matrix with
    values equal to 1 on its diagonal.

    The implementation relies on an external solver, interfaced
    through the `cvxpy` package, thus allowing the user to select
    the precise solver to use (by default, 'scs').

    Usage:
    >>> sdp = MaxCutSDP(graph)
    >>> cut = sdp.get_solution('cut')          # solve problem here
    >>> cut_value = sdp.get_solution('value')  # get pre-computed solution
    """

    def __init__(self, graph, solver='scs'):
        """Instantiate the SDP-relaxed Max-Cut solver.

        graph  : networkx.Graph instance of the graph to cut
        solver : name of the solver to use (default 'scs')

        Note:
        'cvxopt' appears, in general, better than 'scs', but tends
        to disfunction on large (or even middle-sized) graphs, for
        an unknown reason internal to it. 'scs' is thus preferred
        as default solver.
        """
        # Declare the graph attribute and the __results backend one.
        super().__init__(graph)
        # Check that the required solver is available through cvxpy.
        solver = solver.upper()
        if solver not in cp.installed_solvers():
            raise KeyError("Solver '%s' is not installed." % solver)
        self.solver = getattr(cp, solver)

    def solve(self, f, k=1, basic=True, verbose=True):
        """Solve the SDP-relaxed max-cut problem.

        Resulting cut, value of the cut and solved matrix
        may be accessed through the `get_solution` method.
        """
        # Solve the program. Marginally adjust the matrix to be PSD if needed.
        if basic:
            matrix = self._solve_sdp()
        else:
            matrix = self._solve_diag_sdp(k)
        max_value = -1
        matrix = nearest_psd(matrix)
        # Get the cut defined by the matrix.
        vectors = np.linalg.cholesky(matrix)
        for i in range(500):
            cut = get_partition(vectors)
            # Get the value of the cut. Store results.
            value, total = get_cut_value(self.graph, cut)
            self._results = {'matrix': matrix, 'cut': cut, 'value': value}
            if value > max_value:
                max_value = value
        # Optionally be verbose about the results.
        if verbose:
            f.write(
                "Solved the SDP-relaxed max-cut problem.\n"
                "Total weight is %f\n"
                "Value of the cut is %f\n"
                "Solution cuts off %f share of total weights." % (total, max_value, max_value / total)
            )
        return max_value / total

    def _solve_sdp(self):
        """Solve the SDP-relaxed max-cut problem.

        Return the matrix maximizing <C, 1 - X>
        """
        # Gather properties of the graph to cut.
        n_nodes = len(self.graph)
        adjacent = nx.adjacency_matrix(self.graph).toarray()
        # Set up the semi-definite program.
        matrix = cp.Variable((n_nodes, n_nodes), PSD=True)
        cut = .25 * cp.sum(cp.multiply(adjacent, 1 - matrix))
        problem = cp.Problem(cp.Maximize(cut), [cp.diag(matrix) == 1])
        # Solve the program.
        problem.solve(getattr(cp, self.solver))
        # print(matrix.value)
        return matrix.value

    def get_laplacian(self, W):
        # print(W)
        L = -np.array(W)
        for i in range(W.shape[0]):
            L[i][i] = np.sum(W[i])
        # print(L)
        return L

    def _solve_diag_sdp(self, k):
        """Solve the SDP-relaxed max-cut problem.

        Return the matrix maximizing <C, 1 - X>
        """
        # Gather properties of the graph to cut.
        # k = 1
        n = len(self.graph)
        # print(np.array(nx.adjacency_matrix(self.graph).toarray()).size)

        L = self.get_laplacian(np.array(nx.adjacency_matrix(self.graph).toarray()))
        X = cp.Variable((n, n), symmetric=True)
        eps = np.diag([0.0001] * n)
        constraints = [X >> (L + eps)]
        for i in range(n):
            for j in range(n):
                if (abs(i - j) > k // 2):
                    constraints.append(X[i][j] == 0)
        func = cp.Minimize(cp.trace(X))
        optim = cp.Problem(func, constraints)
        optim.solve()
        # print(optim.value)
        # print(X.value)
        # print(np.linalg.eigvals(X.value - L))
        # print(X.value)
        adjacent = X.value - L
        # Set up the semi-definite program.
        matrix = cp.Variable((n, n), PSD=True)
        cut = .25 * cp.sum(cp.multiply(adjacent, 1 - matrix))
        problem = cp.Problem(cp.Maximize(cut), [cp.diag(matrix) == 1])
        # Solve the program.
        problem.solve(getattr(cp, self.solver))

        return matrix.value


def nearest_psd(matrix):
    """Find the nearest positive-definite matrix to input.

    Numpy can be troublesome with rounding values and stating
    a matrix is PSD. This function is thus used to enable the
    decomposition of result matrices

    (altered code from) source:
    https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd
    """
    if is_psd(matrix):
        return matrix
    # false positive warning; pylint: disable=assignment-from-no-return
    spacing = np.spacing(np.linalg.norm(matrix))
    identity = np.identity(len(matrix))
    k = 1
    while not is_psd(matrix):
        min_eig = np.min(np.real(np.linalg.eigvals(matrix)))
        matrix += identity * (- min_eig * (k ** 2) + spacing)
        k += 1
    return matrix


def is_psd(matrix):
    """Check whether a given matrix is PSD to numpy."""
    try:
        _ = np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False
