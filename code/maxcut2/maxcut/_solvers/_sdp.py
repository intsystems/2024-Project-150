# coding: utf-8

"""Semi-Definite Programming based solver for the Max-Cut problem."""

import cvxpy as cp
import networkx as nx
import numpy as np

from maxcut._solvers.backend import (
    AbstractMaxCut, get_partition, get_cut_value
)


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

    def get_mask(self, arr):
        answer = 0
        n = len(arr)
        power_of_2 = 1
        for i in range(n):
            answer += arr[i] * power_of_2
            power_of_2 *= 2
        return answer
    def diag_oracle_solve(self, L, k):
        n = L.shape[0]
        print(n)
        depend_number = k // 2 + 1
        dp = [[0] * (2 ** depend_number) for i in range (n)]
        for i in range(n):
            for mask in range (2 ** depend_number):
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
                print(c, c_previous_0, c_previous_1)
                for j in range(max(0, i - (k // 2)), i + 1):
                    # print(i, j)
                    best_current_ans += 2 * (c[0] * 2 - 1) * (c[i - j] * 2 - 1) * L[i][j]
                best_current_ans -= L[i][i]
                if i > 0:
                    best_current_ans += max(dp[i - 1][self.get_mask(c_previous_0)], dp[i - 1][self.get_mask(c_previous_1)])
                dp[i][mask] = best_current_ans
            print('dp', dp[i])
        answer = 0
        for mask in range(2 ** depend_number):
            answer = max(answer, dp[n - 1][mask])
        print(answer / 4)
        print(dp[i])
        return answer

    def solve(self, f, k = 1, basic=True, verbose=True):
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
