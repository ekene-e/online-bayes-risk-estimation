# -*- coding: utf-8 -*-

# posterior_update.py

import numpy as np

class BayesianPosterior:

    def __init__(self, param_space, prior=None):
        self.param_space = param_space
        self.num_params = len(param_space)
        if prior is None:
            prior = np.ones(self.num_params) / float(self.num_params)
        prior = np.array(prior, dtype=np.float64)
        assert prior.shape[0] == self.num_params, "Mismatch in prior dimension."
        self.log_prior = np.log(prior)
        self.log_posterior = np.copy(self.log_prior)

    def get_posterior_probs(self):
        max_lp = np.max(self.log_posterior)
        shifted = np.exp(self.log_posterior - max_lp)
        probs = shifted / np.sum(shifted)
        return probs

    def update_posterior_independent(self, data_batch, likelihood_func):
        for i in range(self.num_params):
            lp = self.log_posterior[i]
            for d in data_batch:
                ll = likelihood_func(d, self.param_space[i])
                lp += np.log(ll + 1e-300) 
            self.log_posterior[i] = lp
        self._renormalize_log_posterior()

    def update_posterior_dependent(self, data_batch, x_curr, likelihood_func):
        for i in range(self.num_params):
            lp = self.log_posterior[i]
            for d in data_batch:
                ll = likelihood_func(d, x_curr, self.param_space[i])
                lp += np.log(ll + 1e-300)
            self.log_posterior[i] = lp
        self._renormalize_log_posterior()

    def _renormalize_log_posterior(self):
        max_lp = np.max(self.log_posterior)
        self.log_posterior -= max_lp