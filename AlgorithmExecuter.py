import time

class AlgorithmExecuter:
    """
    Class for executing algorithms and timing execution.
    """
    def __init__(self, algorithms):
        self.algorithms = algorithms
        self.nr_algorithms = len(algorithms)

    def execute(self, input_data, algorithm):
        start_time = time.time()
        output = self.algorithms[algorithm](input_data)
        run_time = (time.time() - start_time) * 1000  # milliseconds
        return output, run_time 