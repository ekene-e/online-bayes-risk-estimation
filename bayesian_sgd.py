import numpy as np

def project_to_box(x, lower_bounds, upper_bounds):
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
    data_generator_func,  
    gradient_estimator_func_dependent,
    likelihood_func,
    step_sizes,
    T,
    batch_size=1,
    K=1,
    lower_bounds=None,
    upper_bounds=None
):
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