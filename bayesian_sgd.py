# bayesian_sgd.py

import numpy as np

def project_to_box(x, lower_bounds, upper_bounds):
    """
    Helper function: projects the vector x onto the box defined by
    lower_bounds and upper_bounds (elementwise).
    """
    return np.minimum(np.maximum(x, lower_bounds), upper_bounds)

def bayesian_sgd_independent(
    x_init,
    posterior,             # BayesianPosterior object
    data_generator_func,   # data_generator_func(batch_size) -> data_batch
    gradient_estimator_func_independent, 
    # gradient_estimator_func_independent(x, posterior_probs) -> unbiased grad
    likelihood_func,       # likelihood_func(data_point, theta)
    step_sizes,            # list or array of step sizes a_{t}
    T,                     # total number of outer iterations
    batch_size=1,
    K=1,                   # number of SGD sub-iterations per stage
    lower_bounds=None,
    upper_bounds=None
):
    """
    Implementation of Bayesian-SGD for decision-independent uncertainty.

    Parameters
    ----------
    x_init : float or np.ndarray
        Initial decision. If scalar (univariate), we convert it to a 1D array.
    posterior : BayesianPosterior
        Discrete posterior object from posterior_update.py.
    data_generator_func : callable
        data_generator_func(batch_size) -> returns a batch of observations (D of them).
    gradient_estimator_func_independent : callable
        Function that returns an unbiased gradient estimate of
        E_{theta ~ posterior} [ E_{xi ~ f(·; theta)} h(x, xi) ].
    likelihood_func : callable
        likelihood_func(data_point, theta) returning f(data_point; theta).
    step_sizes : list or np.ndarray
        Sequence of step sizes (one per iteration).
    T : int
        Number of time stages (outer loop).
    batch_size : int
        How many new data points arrive each stage.
    K : int
        How many SGD sub-iterations are performed each stage.
    lower_bounds, upper_bounds : None or float or np.ndarray
        If not None, we project onto these bounds. For univariate, can be float or array of length 1.

    Returns
    -------
    trajectory : list of np.ndarray
        The decision after each stage's update. The length is T+1 (including initial).
    """
    # --- FIX: ensure x_init is a 1D array, even if given as a float ---
    x = np.atleast_1d(x_init).astype(float)
    d = x.shape[0]

    if lower_bounds is None:
        lower_bounds = -np.inf * np.ones(d)
    if upper_bounds is None:
        upper_bounds = np.inf * np.ones(d)

    trajectory = [np.copy(x)]

    for t in range(T):
        data_batch = data_generator_func(batch_size)
        posterior.update_posterior_independent(data_batch, likelihood_func)
        posterior_probs = posterior.get_posterior_probs()
        for j in range(K):
            idx = t * K + j
            a_t = step_sizes[idx] if idx < len(step_sizes) else step_sizes[-1]
            grad_est = gradient_estimator_func_independent(x, posterior_probs)
            x = x - a_t * grad_est
            x = project_to_box(x, lower_bounds, upper_bounds)
        trajectory.append(np.copy(x))

    return trajectory


def bayesian_sgd_dependent(
    x_init,
    posterior,
    data_generator_func,   # data_generator_func(batch_size, x_current) -> data_batch
    gradient_estimator_func_dependent,
    # gradient_estimator_func_dependent(x, posterior_probs) -> unbiased grad
    likelihood_func,
    step_sizes,
    T,
    batch_size=1,
    K=1,
    lower_bounds=None,
    upper_bounds=None
):
    """
    Implementation of Bayesian-SGD for decision-dependent uncertainty.

    Parameters
    ----------
    x_init : float or np.ndarray
        Initial decision. If scalar (univariate), we convert it to a 1D array.
    posterior : BayesianPosterior
        Discrete posterior object from posterior_update.py.
    data_generator_func : callable
        data_generator_func(batch_size, x_current) -> returns data from f(·; x_current, true_theta).
    gradient_estimator_func_dependent : callable
        Unbiased gradient estimator of E_{theta ~ posterior}[ E_{xi ~ f(·; x, theta)} h(x, xi) ],
        which includes the 'endogenous' correction term.
    likelihood_func : callable
        likelihood_func(data_point, x, theta) returning f(data_point; x, theta).
    step_sizes : list or np.ndarray
        Sequence of step sizes.
    T : int
        Number of time stages (outer loop).
    batch_size : int
        How many new data points arrive each stage.
    K : int
        How many SGD sub-iterations are performed each stage.
    lower_bounds, upper_bounds : None or float or np.ndarray
        If not None, we project onto these bounds. For univariate, can be float or array of length 1.

    Returns
    -------
    trajectory : list of np.ndarray
        The decision after each stage's update. The length is T+1 (including initial).
    """
    x = np.atleast_1d(x_init).astype(float)
    d = x.shape[0]
    if lower_bounds is None:
        lower_bounds = -np.inf * np.ones(d)
    if upper_bounds is None:
        upper_bounds = np.inf * np.ones(d)
    trajectory = [np.copy(x)]
    for t in range(T):
        data_batch = data_generator_func(batch_size, x)
        posterior.update_posterior_dependent(data_batch, x, likelihood_func)
        posterior_probs = posterior.get_posterior_probs()
        for j in range(K):
            idx = t * K + j
            a_t = step_sizes[idx] if idx < len(step_sizes) else step_sizes[-1]
            grad_est = gradient_estimator_func_dependent(x, posterior_probs)

            x = x - a_t * grad_est
            x = project_to_box(x, lower_bounds, upper_bounds)

        trajectory.append(np.copy(x))
    return trajectory