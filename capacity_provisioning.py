# capacity_provisioning.py

import numpy as np
from math import factorial, exp

class BayesianPosterior:
    """
    Maintains and updates a discrete Bayesian posterior distribution over a
    finite set of parameters param_space = [theta_1, theta_2, ...].
    """

    def __init__(self, param_space, prior=None):
        """
        param_space : 1D np.ndarray of discrete candidate parameters
        prior       : None or 1D array matching param_space
        """
        self.param_space = param_space
        self.num_params = len(param_space)
        if prior is None:
            prior = np.ones(self.num_params) / float(self.num_params)
        prior = np.array(prior, dtype=float)
        assert prior.shape[0] == self.num_params, "Mismatch in prior dimension."
        self.log_prior = np.log(prior + 1e-300)
        self.log_posterior = np.copy(self.log_prior)

    def get_posterior_probs(self):
        """
        Returns the current posterior as probabilities in linear space.
        """
        max_lp = np.max(self.log_posterior)
        shifted = np.exp(self.log_posterior - max_lp)
        return shifted / np.sum(shifted)

    def update_posterior_independent(self, data_batch, likelihood_func):
        """
        Bayesian update for decision-independent case.
        posterior(theta) = prior(theta) * Product_{d in data_batch} likelihood(d; theta).
        data_batch : array/list of observations
        """
        for i in range(self.num_params):
            lp = self.log_posterior[i]
            for d in data_batch:
                ll = likelihood_func(d, self.param_space[i])
                lp += np.log(ll + 1e-300)
            self.log_posterior[i] = lp
        self._renormalize_log_posterior()
    def _renormalize_log_posterior(self):
        mx = np.max(self.log_posterior)
        self.log_posterior -= mx


def project_to_box(x, lower_bounds, upper_bounds):
    """
    Elementwise clip for projection onto [lower_bounds, upper_bounds].
    """
    return np.minimum(np.maximum(x, lower_bounds), upper_bounds)

def bayesian_sgd_independent(
    x_init,
    posterior,
    data_generator_func,
    gradient_estimator_func_independent,
    likelihood_func,
    step_sizes,
    T,
    batch_size=1,
    K=1,
    lower_bounds=None,
    upper_bounds=None
):
    """
    Runs Bayesian-SGD for the decision-independent setting.

    x_init : initial decision, can be float or 1D array.
    posterior : BayesianPosterior object
    data_generator_func(batch_size) -> batch of data
    gradient_estimator_func_independent(x, posterior_probs) -> unbiased gradient
    likelihood_func(data_point, theta)
    step_sizes : list or array (>= T*K in length or we reuse the last)
    T : no. of outer time stages
    batch_size : how many new data points arrive each stage
    K : no. of sub-iterations (SGD steps) per stage
    lower_bounds, upper_bounds : box constraints

    Returns a list of decisions [x_0, x_1, ..., x_T].
    """
    x = np.atleast_1d(x_init).astype(float)
    d = x.shape[0]
    if lower_bounds is None:
        lower_bounds = -np.inf * np.ones(d)
    if upper_bounds is None:
        upper_bounds =  np.inf * np.ones(d)
    trajectory = [x.copy()]
    for t in range(T):
        data_batch = data_generator_func(batch_size)
        posterior.update_posterior_independent(data_batch, likelihood_func)
        posterior_probs = posterior.get_posterior_probs()
        for j in range(K):
            idx = t*K + j
            a_t = step_sizes[idx] if idx < len(step_sizes) else step_sizes[-1]
            grad_est = gradient_estimator_func_independent(x, posterior_probs)
            x = x - a_t * grad_est
            x = project_to_box(x, lower_bounds, upper_bounds)
        trajectory.append(x.copy())

    return trajectory

def data_generator_capacity(batch_size, true_lambda=8, rng=None):
    """
    Observes batch_size arrival counts from Poisson(true_lambda).
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.poisson(lam=true_lambda, size=batch_size)

def likelihood_poisson(data_point, lam_param):
    """
    Poisson pmf = exp(-lam) * lam^k / k!.
    We'll do a minimal check for lam_param <=0 => 1e-300.
    """
    if lam_param <= 0:
        return 1e-300
    k = data_point
    pmf = np.exp(-lam_param) * (lam_param**k) / float(factorial(k))
    return pmf if pmf>1e-300 else 1e-300

def gradient_estimator_capacity(x, posterior_probs, param_space,
                                c=1.0, p=5.0, rng=None):
    """
    Monte Carlo gradient wrt x for cost h(x, xi) = c*x + p * max(0, xi - x),
    xi ~ Poisson(lambda). We do:
      1) sample lam from the posterior
      2) sample xi ~ Poisson(lam)
      3) grad_x h(x, xi) = c if xi <= x else (c - p).
    """
    if rng is None:
        rng = np.random.default_rng()
    idx_list = np.arange(len(posterior_probs))
    i = rng.choice(idx_list, p=posterior_probs)
    lam = param_space[i]
    xi = rng.poisson(lam=lam)
    x_val = x[0]
    if xi <= x_val:
        grad = c
    else:
        grad = c - p
    return np.array([grad], dtype=float)

def approximate_true_cost(x, true_lambda=8, c=1.0, p=5.0, mc_samples=2000, rng=None):
    """
    Approximate H(x) = E[c*x + p * max(0, xi - x)] under xi~Poisson(true_lambda).
    We'll do a quick Monte Carlo approach for demonstration.
    """
    if rng is None:
        rng = np.random.default_rng()
    x_val = x[0]
    arr = rng.poisson(lam=true_lambda, size=mc_samples)
    cost_vals = c*x_val + p*np.maximum(0, arr - x_val)
    return np.mean(cost_vals)

def run_capacity_provisioning(
    true_lambda=8,
    param_space=None,
    prior=None,
    T=100,
    c=1.0,
    p=5.0,
    step_coefficient=0.5,
    batch_size=1,
    K=1,
    rng=None
):
    """
    Runs Bayesian-SGD for capacity provisioning and returns the entire x(t) trajectory.

    param_space: If None, defaults to np.arange(1,31).
    prior: if None, uniform over param_space.
    T: number of time stages
    c, p: cost and penalty
    step_coefficient: for step size schedule a_t = step_coefficient/(t+5)
    """
    if rng is None:
        rng = np.random.default_rng()
    if param_space is None:
        param_space = np.arange(1,31,dtype=float)
    if prior is None:
        prior = np.ones(len(param_space))/len(param_space)
    posterior = BayesianPosterior(param_space, prior)
    step_sizes = [step_coefficient/(t+5) for t in range(T*K)]
    x_init = 0.0
    trajectory = bayesian_sgd_independent(
        x_init=np.array([x_init]),
        posterior=posterior,
        data_generator_func=lambda bs: data_generator_capacity(bs, true_lambda=true_lambda, rng=rng),
        gradient_estimator_func_independent=lambda x, post: gradient_estimator_capacity(
            x, post, param_space, c=c, p=p, rng=rng
        ),
        likelihood_func=likelihood_poisson,
        step_sizes=step_sizes,
        T=T,
        batch_size=batch_size,
        K=K,
        lower_bounds=np.array([0.0]),
        upper_bounds=np.array([100.0])
    )
    return trajectory