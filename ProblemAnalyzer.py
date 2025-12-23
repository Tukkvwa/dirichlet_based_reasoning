import numpy as np

class ProblemAnalyzer:
    """
    Perceive the problem and return its features.
    """
    def __init__(self, feature_extractors, categories=None):
        self.feature_extractors = feature_extractors
        self.nr_features = len(feature_extractors)
        self.categories = categories

    def extract_features(self, input_):
        features = []
        for extractor in self.feature_extractors:
            features.append(extractor(input_))
        return np.array([features])

    def extract_categories(self, inputs):
        if self.categories is None:
            raise ValueError('No categories defined')
        category_names = list(self.categories.keys())
        nr_inputs = inputs.shape[1]
        features = np.full((nr_inputs, self.nr_features), np.nan)
        for i in range(nr_inputs):
            for c, name in enumerate(category_names):
                features[i, c] = self.categories[name](inputs[:, i])
        return features 