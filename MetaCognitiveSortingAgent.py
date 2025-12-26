from MetaCognitiveAgent import MetaCognitiveAgent
from ProblemAnalyzer import ProblemAnalyzer
from Sorter import Sorter
from MetaLevelModel import MetaLevelModel
import numpy as np

def logit(p):
    return np.log(p / (1 - p))

class MetaCognitiveSortingAgent(MetaCognitiveAgent):
    """
    A metacognitive sorting agent with optional feature extractors.
    """
    def __init__(self, score_is_binary, algorithms=None, time_cost=1, use_features=True, seed=42):
        if algorithms is None:
            print("No algorithms provided, using default algorithms.")
            algorithms = [
                lambda data: data,  # bogoSort placeholder
                lambda data: data,  # insertionSort placeholder
                lambda data: data,  # selectionSort placeholder
                lambda data: data,  # bubbleSort placeholder
                lambda data: data,  # shellsort placeholder
                lambda data: data,  # heapsort placeholder
                lambda data: data,  # mergesort placeholder
                lambda data: data   # quickSort placeholder
            ]

        if use_features:
            def presortdeness(input_arr):
                input_arr = np.array(input_arr)
                return np.sum(input_arr[1:] < input_arr[:-1])
            feature_extractors = [lambda input_arr: len(input_arr), lambda input_arr: int(presortdeness(input_arr))]
        else:
            feature_extractors = []
        problem_analyzer = ProblemAnalyzer(feature_extractors)
        range_time = 3 * 60
        range_reward = logit(0.99)
        super().__init__(algorithms, problem_analyzer, score_is_binary, time_cost, range_time, range_reward)
        self.metalevel_model = MetaLevelModel(self.problem_analyzer.nr_features, self.algorithm_executer.nr_algorithms, score_is_binary, time_cost, seed=seed)
        self.metalevel_model.regressor_names = [
            '1','n','i','log(n)','log(i)','n*i','n*log(n)','n*log(i)',
            'i*log(n)','i*log(i)','log(n)*log(i)','l^2','i^2','log(n)^2','log(i)^2'
        ]
        self.nr_algorithms = self.algorithm_executer.nr_algorithms 