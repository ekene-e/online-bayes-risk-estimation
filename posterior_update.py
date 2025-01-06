# -*- coding: utf-8 -*-

# posterior_update.py

import numpy as np

class BayesianPosterior:
    """
    Maintains and updates a discrete Bayesian posterior distribution over a finite set of parameters.

    Attributes
    ----------
    param_space : numpy.ndarray
        Array of all possible parameter values (discrete).
    log_prior   : numpy.ndarray
        log of the prior distribution over the param_space.
    log_posterior : numpy.ndarray
        log of the posterior distribution over the param_space.
    """

    def __init__(self, param_space, prior=None):
        """
        Initialize the BayesianPosterior object.

        Parameters
        ----------
        param_space : numpy.ndarray
            Array of shape (N,) or (N, dim_theta) representing all discrete candidate parameters.
        prior : numpy.ndarray or None
            If given, must match the shape of param_space in its first dimension.
            If None, uses uniform prior.
        """
        self.param_space = param_space
        self.num_params = len(param_space)
        if prior is None:
            prior = np.ones(self.num_params) / float(self.num_params)
        prior = np.array(prior, dtype=np.float64)
        assert prior.shape[0] == self.num_params, "Mismatch in prior dimension."
        self.log_prior = np.log(prior)
        self.log_posterior = np.copy(self.log_prior)

    def get_posterior_probs(self):
        """
        Returns the current posterior distribution as probabilities (in linear space).
        """
        max_lp = np.max(self.log_posterior)
        shifted = np.exp(self.log_posterior - max_lp)
        probs = shifted / np.sum(shifted)
        return probs

    def update_posterior_independent(self, data_batch, likelihood_func):
        """
        Bayesian update for decision-independent data:
        posterior(theta) = prior(theta) * prod_{d in data_batch} f(d; theta)

        Parameters
        ----------
        data_batch : list or np.ndarray
            Observed data in the current time stage, shape (D, ...) if batch size = D.
        likelihood_func : function
            likelihood_func(data_point, theta) -> scalar likelihood f(data_point; theta).
            Here 'theta' is from param_space.
        """
        for i in range(self.num_params):
            lp = self.log_posterior[i]
            for d in data_batch:
                ll = likelihood_func(d, self.param_space[i])
                lp += np.log(ll + 1e-300) 
            self.log_posterior[i] = lp
        self._renormalize_log_posterior()

    def update_posterior_dependent(self, data_batch, x_curr, likelihood_func):
        """
        Bayesian update for decision-dependent data:
        posterior(theta) = prior(theta) * prod_{d in data_batch} f(d; x_curr, theta)

        Parameters
        ----------
        data_batch : list or np.ndarray
            Observed data in the current time stage, shape (D, ...).
        x_curr : np.ndarray or float
            The decision that was used to generate the data_batch.
        likelihood_func : function
            likelihood_func(data_point, x, theta) -> scalar
            The likelihood of data_point given decision x and parameter theta.
        """
        for i in range(self.num_params):
            lp = self.log_posterior[i]
            for d in data_batch:
                ll = likelihood_func(d, x_curr, self.param_space[i])
                lp += np.log(ll + 1e-300)
            self.log_posterior[i] = lp
        self._renormalize_log_posterior()

    def _renormalize_log_posterior(self):
        """
        Helper function: normalizes self.log_posterior in log-space to avoid numerical overflow.
        """
        max_lp = np.max(self.log_posterior)
        self.log_posterior -= max_lp