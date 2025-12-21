from MetaCognitiveAgent import MetaCognitiveAgent
import numpy as np

class SCADSAgent(MetaCognitiveAgent):
    """
    SCADSAgent: Strategy Selection Component of the SCADS Model.
    """
    def __init__(self, strategies, feature_extractor, feedback_type, relative_strengths=None, total_strength=None):
        super().__init__(strategies, feature_extractor, False)
        nr_strategies = len(strategies)
        if relative_strengths is None:
            relative_strengths = np.ones(nr_strategies) / nr_strategies
        if total_strength is None:
            total_strength = 1
        self.confidence_by_problem_type = total_strength * np.tile(relative_strengths.reshape(-1, 1), (1, feature_extractor.nr_features))
        self.general_strength = np.sum(self.confidence_by_problem_type, axis=1)
        self.reward_definition = feedback_type
        self.time_per_computation = 1

    def selectAlgorithm(self, problem_features, time_cost=None, payoffs=None):
        if np.prod(np.array(problem_features) == 0) == 1:
            strengths = self.general_strength
        else:
            strengths = self.general_strength * np.sum(self.confidence_by_problem_type[:, np.array(problem_features, dtype=bool)], axis=1)
        strengths = np.maximum(np.finfo(float).eps, strengths)
        p_choice = strengths / np.sum(strengths)
        # Sample from the discrete distribution
        a = np.random.choice(len(p_choice), p=p_choice)
        return a

    def reflect(self, experience):
        try:
            if not (experience['feedback'] is None):
                performance = self.judgePerformance(experience)
                for e in range(len(experience['algorithm'])):
                    if getattr(self, 'score_is_binary', False):
                        delta = 0.2 if performance[e] == 1 else -0.1
                    else:
                        delta = performance[e]
                    self.general_strength[experience['algorithm'][e]] += delta
                    self.confidence_by_problem_type[experience['algorithm'][e], np.array(experience['features'][e], dtype=bool)] += delta
        except:
            return self

    def judgePerformance(self, experience):
        if self.reward_definition == 1:
            return experience['feedback']
        elif self.reward_definition == 2:
            return experience['feedback'] - experience['time_cost'] * experience['run_time']
        elif self.reward_definition == 3:
            return experience['feedback'] / experience['run_time']
        else:
            return experience['feedback'] 