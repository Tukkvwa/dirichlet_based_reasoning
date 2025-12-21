import numpy as np

class ProblemAnalyzer:
    """
    Perceive the problem and return its features.
    """
    def __init__(self, feature_extractors, categories=None):
        self.feature_extractors = feature_extractors
        self.nr_features = len(feature_extractors)
        self.categories = categories

    def extract_features(self, inputs):
        # inputs: each input is a column vector (2D numpy array)
        nr_inputs = inputs.shape[1]
        features = np.full((nr_inputs, self.nr_features), np.nan)
        for i in range(nr_inputs):
            for f, extractor in enumerate(self.feature_extractors):
                features[i, f] = extractor(inputs[:, i])
        return features

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