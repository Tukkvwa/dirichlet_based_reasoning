from SCADSAgent import SCADSAgent
import numpy as np

class SCADSSortingAgent(SCADSAgent):
    """
    Implementation of the Strategy Selection Component of the SCADS Model by Shrager and Siegler (1998).
    """
    def __init__(self, sorting_algorithms, feedback_type, categories=None, seed=42):
        if categories is None:
            categories = {
                'is_short': lambda input: len(input) <= 100,
                'is_long': lambda input: len(input) >= 1000,
                'is_presorted': lambda input: np.mean(np.diff(input) >= 0) > 0.9,
                'is_disordered': lambda input: np.mean(np.diff(input) >= 0) < 0.5
            }
        def extract_features(input_):
            return np.array([[cat(input_) for cat in categories.values()]])
        class ProblemAnalyzerStub:
            def __init__(self, extract_features):
                self.extract_features = extract_features
                self.nr_features = len(categories)
        problem_analyzer = ProblemAnalyzerStub(extract_features)
        super().__init__(sorting_algorithms, problem_analyzer, feedback_type, seed=seed) 