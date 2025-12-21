from MetaLevelModel import MetaLevelModel
from AlgorithmExecuter import AlgorithmExecuter

class MetaCognitiveAgent:
    """
    The metacognitive agent learns to select the resource-rational algorithm for a given problem.
    """
    def __init__(self, algorithms, problem_analyzer, score_is_binary, time_cost=1, range_time=5*60, range_reward=20):
        self.learn = True
        self.problem_analyzer = problem_analyzer
        self.nr_algorithms = len(algorithms)
        self.strategy_available = [True] * self.nr_algorithms
        self.meta_level_model = MetaLevelModel(problem_analyzer.nr_features, self.nr_algorithms, score_is_binary, time_cost)
        self.algorithm_executer = AlgorithmExecuter(algorithms)
        self.nr_algorithms = self.algorithm_executer.nr_algorithms
        self.score_is_binary = score_is_binary
        self.explore = True

    def solve_problem(self, problem, features, algorithm=None):
        import time
        start = time.time()
        #features = self.problem_analyzer.extractFeatures(problem['input']) if 'input' in problem else self.problem_analyzer.extract_features(problem)
        if algorithm is None:
            algorithm = self.select_algorithm(features, problem.get('time_cost', 1), problem.get('payoffs', None))
        solution, run_time = self.algorithm_executer.execute(problem['input'] if 'input' in problem else problem, algorithm)
        experience = {
            'total_time': (time.time() - start) * 1000,
            'run_time': run_time,
            'problem': problem,
            'features': features,
            'solution': solution,
            'algorithm': algorithm
        }   
        return self, solution, experience

    def select_algorithm(self, problem_features, time_cost, payoffs=None):
        self.meta_level_model.time_cost = time_cost
        if payoffs is not None:
            E_run_time, E_score, sigma_run_time, sigma_score = self.meta_level_model.predict_performance(problem_features, payoffs)
        else:
            E_run_time, E_score, sigma_run_time, sigma_score = self.meta_level_model.predict_performance(problem_features)
        E_VOC = E_score - time_cost * E_run_time
        if self.explore:
            nr_samples = 1
            if payoffs is not None:
                VOC_samples = self.meta_level_model.sampleVOC(problem_features, nr_samples, None, None, payoffs)
            else:
                VOC_samples = self.meta_level_model.sampleVOC(problem_features, nr_samples)
            import numpy as np
            available_indices = [i for i, available in enumerate(self.strategy_available) if available]
            a_index = int(np.argmax([np.mean(VOC_samples[i]) for i in available_indices]))
            a = available_indices[a_index]
        else:
            import numpy as np
            available_indices = [i for i, available in enumerate(self.strategy_available) if available]
            a_index = int(np.argmax([E_VOC[i] for i in available_indices]))
            a = available_indices[a_index]
        return a

    def reflect(self, experience):
        self.meta_level_model.update(experience['algorithm'], experience['features'],
                                     experience['run_time'], experience["score"])                                    
        #self.meta_level_model.model_selection(experience['algorithm'])
        return self

    def judge_performance(self, experience):
        return experience['feedback'] 