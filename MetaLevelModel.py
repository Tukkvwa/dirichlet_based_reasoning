import numpy as np

class MetaLevelModel:
    def __init__(self, nr_features, nr_algorithms, score_is_binary, time_cost):
        self.nr_algorithms = nr_algorithms
        self.score_is_binary = score_is_binary
        self.time_cost = time_cost
        self.nr_regressors = 1 + 3 * nr_features + 2 * nr_features ** 2 #taking into account interactions and polynomial terms
        self.max_nr_regressors = self.nr_regressors

        self.glm_runtime = []
        self.glm_score = []
        self.selected_regressors_runtime = []
        self.selected_regressors_score = []

        for a in range(self.nr_algorithms):
            glm_rt = BayesianGLM(self.nr_regressors, 10)
            glm_rt.setParameter('a_0', 1)
            glm_rt.setParameter('b_0', 10)
            self.glm_runtime.append(glm_rt)

            glm_sc = BayesianGLM(self.nr_regressors, 1)
            glm_sc.setParameter('a_0', 1)
            glm_sc.setParameter('b_0', 10)
            self.glm_score.append(glm_sc)

            self.selected_regressors_runtime.append(list(range(self.nr_regressors)))
            self.selected_regressors_score.append(list(range(self.nr_regressors)))

        self.observations = [{'features': np.empty((0, nr_features)),
                              'run_time': np.array([]),
                              'score': np.array([])} for _ in range(nr_algorithms)]

        # setting priors for the logistic regression coefficients
        self.a_gamma_p_correct = 0.01
        self.b_gamma_p_correct = 1

        self.prior_pi_gamma_p_correct = 100 * np.stack([np.eye(self.nr_regressors) for _ in range(self.nr_algorithms)], axis=2)
        self.prior_E_gamma_p_correct = np.zeros((self.nr_regressors, self.nr_algorithms))
        self.pi_gamma_p_correct = np.copy(self.prior_pi_gamma_p_correct)
        self.E_gamma_p_correct = np.copy(self.prior_pi_gamma_p_correct)

    def predict_performance(self, problem_features, payoffs=None):
        regressors, _ = self.construct_regressors(problem_features)
        E_run_time = np.zeros((self.nr_algorithms, 1)) # Expected run time
        E_score = np.zeros((self.nr_algorithms, 1)) # Expected score
        sigma_run_time = np.zeros((self.nr_algorithms, 1)) # Uncertainty in run time
        sigma_score = np.zeros((self.nr_algorithms, 1)) # Uncertainty in score

        for a in range(self.nr_algorithms):
            reg_rt = regressors[:, self.selected_regressors_runtime[a]]
            E_run_time[a, 0] = reg_rt @ self.glm_runtime[a].mu_n
            #print("E run time: ", E_run_time, E_run_time.shape)

            if self.score_is_binary:
                if payoffs is not None:
                    p_correct = sigmoid(regressors[:, self.selected_regressors_score[a]] @ self.E_gamma_p_correct[:, a])
                    E_score[a, 0] = np.dot([1 - p_correct, p_correct], payoffs)
                else:
                    E_score[a, 0] = sigmoid(regressors[:, self.selected_regressors_score[a]] @ self.E_gamma_p_correct[:, a])
                sigma_score[a, 0] = np.nan
            else:
                sigma2_epsilon_S = self.glm_score[a].b_n / self.glm_score[a].a_n
                X, _ = self.construct_regressors(self.observations[a]['features'])
                X_S = X[:, self.selected_regressors_score[a]]
                regressors_S = regressors[:, self.selected_regressors_score[a]]
                #sigma_score[a, 0] = np.sqrt(sigma2_epsilon_S * (1 + regressors_S @ np.linalg.pinv(X_S) @ np.linalg.pinv(X_S.T) @ regressors_S))

            #sigma2_epsilon_T = max(1e-4, self.glm_runtime[a].b_n / self.glm_runtime[a].a_n)
            #if self.observations[a]['features'].shape[0] > 0:
            #    X, _ = self.construct_regressors(self.observations[a]['features'])
            #else:
            #    X = np.array([])
            #regressors_T = regressors[:, self.selected_regressors_runtime[a]]
            #if X.size > 0:
            #    X_T = X[:, self.selected_regressors_runtime[a]]
            #    sigma_run_time[a, 0] = np.sqrt(sigma2_epsilon_T * (1 + max(0, regressors_T @ np.linalg.pinv(X_T) @ np.linalg.pinv(X_T.T) @ regressors_T)))
            #else:
            #    sigma_run_time[a, 0] = np.sqrt(sigma2_epsilon_T * (1 + regressors_T @ self.glm_runtime[a].Lambda_n @ regressors_T))
        return E_run_time, E_score, sigma_run_time, sigma_score

    def sampleVOC(self, problem_features, nr_samples, min_payoff=None, max_payoff=None, payoffs=None):
        regressors, _ = self.construct_regressors(problem_features)
        run_times = np.zeros((self.nr_algorithms, nr_samples))
        scores = np.zeros((self.nr_algorithms, nr_samples))

        for a in range(self.nr_algorithms):
            coefficients_T = mvnrnd_from_Pi(self.glm_runtime[a].mu_n, self.glm_runtime[a].Lambda_n, nr_samples)
            run_times[a, :] = regressors[:, self.selected_regressors_runtime[a]] @ coefficients_T

            if self.score_is_binary:
                coefficients_S = mvnrnd_from_Pi(self.E_gamma_p_correct[:, a], self.pi_gamma_p_correct[:, :, a], nr_samples)
                if payoffs is not None:
                    p_correct = sigmoid(regressors[:, self.selected_regressors_score[a]] @ coefficients_S)
                    scores[a, :] = np.dot([1 - p_correct, p_correct], payoffs)
                else:
                    scores[a, :] = sigmoid(regressors[:, self.selected_regressors_score[a]] @ coefficients_S)
            else:
                coefficients_S = mvnrnd_from_Pi(self.glm_score[a].mu_n, self.glm_score[a].Lambda_n, nr_samples)
                scores[a, :] = regressors[:, self.selected_regressors_score[a]] @ coefficients_S

        costs = self.time_cost * run_times
        VOC_samples = scores - costs
        return VOC_samples, scores, costs

    def update(self, a, features, run_time, score):
        self.observations[a]['features'] = np.vstack([self.observations[a]['features'], features])
        self.observations[a]['run_time'] = np.concatenate([self.observations[a]['run_time'], [run_time]])
        self.observations[a]['score'] = np.concatenate([self.observations[a]['score'], [score]])
        regressors, _ = self.construct_regressors(features)
        regressors_T = regressors[:, self.selected_regressors_runtime[a]]
        self.glm_runtime[a].learn_and_update_prior(regressors_T, run_time)

        if self.score_is_binary:
            regressors, _ = self.construct_regressors(self.observations[a]['features'])
            #s - strategy, r - rewards(1 or 0 I guess), 
            mu_posterior, Pi_posterior, _ = logisticRegressionLAP(
                self.observations[a]['score'],
                self.observations[a]['run_time'],
                #self.observations[a]['features'],
                regressors[:, self.selected_regressors_score[a]],
                self.prior_E_gamma_p_correct[:, a],
                self.prior_pi_gamma_p_correct[:, :, a]
            )
            
            self.pi_gamma_p_correct[:, :, a] = Pi_posterior
            self.E_gamma_p_correct[:, a] = mu_posterior
        else:
            regressors, _ = self.construct_regressors(features)
            regressors_S = regressors[:, self.selected_regressors_score[a]]
            self.glm_score[a].learn_and_update_prior(regressors_S, score)

    def model_selection(self, a):
        any_observations = np.any(~np.isnan(self.observations[a]['run_time']))
        if any_observations:
            old_model = self.glm_runtime[a]
            self.glm_runtime[a] = BayesianGLM(self.max_nr_regressors)
            regressors, _ = self.construct_regressors(self.observations[a]['features'])
            y = self.observations[a]['run_time']
            self.selected_regressors_runtime[a], p_m = self.glm_runtime[a].featureSelection(regressors, y)
            self.glm_runtime[a] = BayesianGLM(len(self.selected_regressors_runtime[a]), old_model.sigma)
            self.glm_runtime[a].a_0 = old_model.a_0
            self.glm_runtime[a].b_0 = old_model.b_0
            self.glm_runtime[a].learn(regressors[:, self.selected_regressors_runtime[a]], y)

        any_observations = np.any(~np.isnan(self.observations[a]['score']))
        if any_observations and not self.score_is_binary:
            old_model = self.glm_score[a]
            self.glm_score[a] = BayesianGLM(self.max_nr_regressors, old_model.sigma)
            self.glm_score[a].a_0 = old_model.a_0
            self.glm_score[a].b_0 = old_model.b_0

            observed_reward = ~np.isnan(self.observations[a]['score'])
            regressors, _ = self.construct_regressors(self.observations[a]['features'][observed_reward])
            y = self.observations[a]['score'][observed_reward]
            if np.sum(~np.isnan(y)) > 0:
                self.selected_regressors_score[a], p_m = self.glm_score[a].featureSelection(regressors[~np.isnan(y)], y[~np.isnan(y)])
            self.glm_score[a] = BayesianGLM(len(self.selected_regressors_score[a]), old_model.sigma)
            self.glm_score[a].a_0 = old_model.a_0
            self.glm_score[a].b_0 = old_model.b_0
            self.glm_score[a] = self.glm_score[a].learn(regressors[:, self.selected_regressors_score[a]], y)

    def construct_regressor(self, basic_features):
        # Construct a single regressor from the basic features
        
        return basic_features

    def construct_regressors(self, basic_features):
        """
        Construct regressors from basic features including interaction and polynomial terms.
        """
        nr_basic_features = basic_features.shape[1] if basic_features.size > 0 else 0
        features = np.copy(basic_features)
        extended_features = []

        if nr_basic_features > 0:
            extended_features = [f'feature{f+1}' for f in range(nr_basic_features)]
        
        for f in range(nr_basic_features):
            log_feat = np.log(basic_features[:, f] + 1)
            features = np.hstack((features, log_feat[:, None]))
            extended_features.append(f'log(feature{f+1})')

        nr_features = features.shape[1]
        regressors = np.zeros((features.shape[0], self.nr_regressors))
        feature_powers = np.zeros((nr_features, self.nr_regressors))
        index = 0
        regressors[:, index] = 1
        feature_powers[:, index] = 0
        index += 1

        for f1 in range(nr_features):
            regressors[:, index] = features[:, f1]
            feature_powers[:, index] = np.eye(nr_features)[:, f1]
            index += 1

        for f1 in range(nr_features):
            for f2 in range(f1 + 1, nr_features):
                regressors[:, index] = features[:, f1] * features[:, f2]
                powers = np.zeros(nr_features)
                powers[f1] = 1
                powers[f2] = 1
                feature_powers[:, index] = powers
                index += 1

        for f1 in range(nr_features):
            regressors[:, index] = features[:, f1] ** 2
            powers = np.zeros(nr_features)
            powers[f1] = 2
            feature_powers[:, index] = powers
            index += 1

        self.extended_features = extended_features
        self.feature_powers = feature_powers
        return regressors, self

# Helper functions/classes (to be implemented)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mvnrnd_from_Pi(mu, Lambda, n):
    # Sample n times from N(mu, inv(Lambda))
    cov = np.linalg.pinv(Lambda)
    return np.random.multivariate_normal(mu, cov, n).T

class BayesianGLM:
    def __init__(self, nr_regressors, sigma=1):
        self.mu_n = np.zeros(nr_regressors) # Mean of the posterior
        self.Lambda_n = np.eye(nr_regressors) # Precision matrix of the posterior
        self.sigma = sigma # Noise variance
        self.a_0 = 1 # Prior parameters
        self.b_0 = 10 # Prior parameters
        self.a_n = 1 # Posterior parameters
        self.b_n = 10 # Posterior parameters

    def setParameter(self, name, value):
        setattr(self, name, value)
        return self

    def learn_and_update_prior(self, X, y):
        """
        Update posterior distribution given new data.
        This is the key method that performs Bayesian updates!
        
        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,)
        """
        # Ensure X is 2D
        y = np.array([y])
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # Save old parameters for beta update
        old_mu = self.mu_n.copy()
        old_Lambda = self.Lambda_n.copy()
        
        # UPDATE 1: Precision matrix for theta
        # New precision = old precision + data information
        self.Lambda_n = old_Lambda + X.T @ X
        
        # UPDATE 2: Mean for theta
        # Weighted combination of prior mean and data-driven estimate
        self.mu_n = np.linalg.solve(
            self.Lambda_n,
            old_Lambda @ old_mu + X.T @ y
        )
        
        # UPDATE 3: Shape parameter for Inverse-Gamma
        # Increases by half the number of new observations
        self.a_n = self.a_n + 1 / 2.0
        
        # UPDATE 4: Rate parameter for Inverse-Gamma
        # Accumulates sum of squared residuals (in a specific form)
        self.b_n = self.b_n + 0.5 * (
            y.T @ y +                           # Sum of squared observations
            old_mu.T @ old_Lambda @ old_mu -    # Prior quadratic term
            self.mu_n.T @ self.Lambda_n @ self.mu_n  # Posterior quadratic term
        )
        return self

    def featureSelection(self, X, y):
        # Dummy implementation
        return list(range(X.shape[1])), np.ones(X.shape[1])

def logisticRegressionLAP(ys, xs, mu_prior, Pi_prior):
    
    mu_prior = mu_prior.copy()
    Pi_prior = Pi_prior.copy()
    for i in range(len(ys)):
        y = ys[i]
        x = xs[i, :].reshape(-1, 1)  # column vector

        # initial values
        ksi = np.sqrt(float(x.T @ np.linalg.solve(Pi_prior, x)) + float((x.T @ mu_prior)**2))
        old_values = np.concatenate([mu_prior.flatten(), Pi_prior.flatten()])

        converged = False
        nr_iterations = 0
        while not converged and nr_iterations <= 100:
            # E-step
            lam = abs(lambda x : 0.5 / x * (sigmoid(x) - 0.5 ))
            Pi_posterior = Pi_prior + 2 * lam(ksi) * (x @ x.T)
            mu_posterior = np.linalg.solve(Pi_posterior, Pi_prior @ mu_prior + x * (y - 0.5))

            # M-step
            ksi = np.sqrt(float(x.T @ np.linalg.solve(Pi_posterior, x)) + float((x.T @ mu_posterior)**2))

            # check for convergence
            new_values = np.concatenate([mu_posterior.flatten(), Pi_posterior.flatten()])
            converged = np.linalg.norm(old_values - new_values) < 0.01
            old_values = new_values

            nr_iterations += 1

            if nr_iterations > 100:
                print(f'variational logistic regression: EM did not converge within {nr_iterations} iterations.')

        mu_prior = mu_posterior
        Pi_prior = Pi_posterior

    return mu_posterior, Pi_posterior, ksi
