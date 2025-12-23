import numpy as np
from Sorter import quick_sort
from ProblemGenerator import ProblemGenerator

class SortingProblemGenerator(ProblemGenerator):
    """
    This class generates sorting problems.
    """

    def __init__(self):
        # the solver provides the optimal solution
        solver = lambda input_: quick_sort(input_)
        scorer = lambda solution, input_: np.prod(np.array(solution) == np.array(solver(input_)))
        super().__init__(solver, scorer)

    def generate_problem(self, params, time_cost, list_type="random"):
        problem = {}
        if list_type=="pos":
            problem["object"] = self.generate_partially_ordered_sequence(params[0], params[1])
        elif list_type=="pros":
            problem["object"] = self.generate_partially_reversely_ordered_sequence(params[0], params[1])
        else:
            problem["object"] = self.generate_random_sequence(params[0], params[1])
        problem["scorer"]  = self.scorer
        problem["solution"] = self.solver(self, problem["object"])
        problem["time_cost"] = time_cost
        problem["params"] = params

        return problem

    def generate_random_sequence(self, number_of_elements, range_of_elements=None):
        if range_of_elements is None:
            range_of_elements = number_of_elements
        return np.array(list(np.random.permutation(range_of_elements)[:number_of_elements] + 1))

    def generate_partially_ordered_sequence(self, number_of_elements, fraction_of_permutations):
        input_ = list(range(1, number_of_elements + 1))
        nr_of_permutations = round(number_of_elements * fraction_of_permutations)
        n_perm = min(number_of_elements, nr_of_permutations * 2)
        permutations = np.random.permutation(number_of_elements)[:n_perm]
        permutations = permutations.tolist()
        for i in range(0, len(permutations) - 1, 2):
            idx1 = permutations[i]
            idx2 = permutations[i + 1]
            input_[idx1], input_[idx2] = input_[idx2], input_[idx1]
        return np.array(input_)

    def generate_partially_reversely_ordered_sequence(self, number_of_elements, fraction_of_permutations):
        input_ = list(range(number_of_elements, 0, -1))
        nr_of_permutations = round(number_of_elements * fraction_of_permutations)
        n_perm = min(number_of_elements, nr_of_permutations * 2)
        permutations = np.random.permutation(number_of_elements)[:n_perm]
        permutations = permutations.tolist()
        for i in range(0, len(permutations) - 1, 2):
            idx1 = permutations[i]
            idx2 = permutations[i + 1]
            input_[idx1], input_[idx2] = input_[idx2], input_[idx1]
        return np.array(input_)
