# example_problems.py

import numpy as np

##########################################
# 1) Univariate Decision-Independent Example
##########################################

def data_generator_indep_univar(batch_size, true_theta=9, noise_std=4, rng=None):
    """
    Data generator for the univariate test: xi ~ Normal(theta^c, noise_std^2).
    Returns an array of shape (batch_size,).
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.normal(loc=true_theta, scale=noise_std, size=batch_size)

def likelihood_normal_indep_univar(data_point, theta, noise_std=4.0):
    """
    f(data_point; theta) = pdf of N(theta, noise_std^2).
    """
    diff = data_point - theta
    return 1.0/np.sqrt(2*np.pi*noise_std**2) * np.exp(-0.5*(diff**2)/(noise_std**2))

def gradient_estimator_indep_univar(x, posterior_probs, 
                                    sample_size=1, rng=None,
                                    noise_std=4.0):
    """
    Unbiased gradient of E_{theta ~ posterior}[ E_{xi ~ N(theta, noise_std^2)}[(x-5)^2 + 0.5 xi x]].
    We'll do a small Monte Carlo approach here:
      sample param ~ posterior, sample xi ~ Normal(param, noise_std),
      grad wrt x = 2(x-5) + 0.5 * xi.
    """
    if rng is None:
        rng = np.random.default_rng()
    param_space_indices = np.arange(len(posterior_probs))
    grad_acc = 0.0
    for _ in range(sample_size):
        i = rng.choice(param_space_indices, p=posterior_probs)
        # Here we assume param_space[i] = i+1 if discrete param space is [1,2,3,...].
        theta_i = i+1
        xi = rng.normal(loc=theta_i, scale=noise_std)
        grad = 2*(x[0]-5) + 0.5*xi  # x is an array of length 1, so x[0].
        grad_acc += grad
    return np.array([grad_acc / float(sample_size)])


##########################################
# 2) Multivariate Decision-Independent Example
##########################################

def data_generator_indep_multi(batch_size, true_theta=4, rng=None):
    """
    For the multivariate example in the paper:
      xi ~ Exponential(mean=true_theta).
    We'll return shape (batch_size,) for xi. 'x' is 2D, but xi is 1D here.
    """
    if rng is None:
        rng = np.random.default_rng()
    rate = 1.0 / true_theta
    return rng.exponential(1.0/rate, size=batch_size)

def likelihood_exponential(data_point, theta):
    """
    f(xi; theta) = (1/theta)*exp(-xi/theta), xi >=0
    """
    if data_point < 0:
        return 1e-300
    return (1.0/theta)*np.exp(-data_point/theta)

def gradient_estimator_indep_multi(x, posterior_probs,
                                   sample_size=1, rng=None):
    """
    h(x, xi) = (x1-1)^2 + (x2-2)^2 + xi*(x1+x2).
    grad wrt x = [2(x1-1) + xi, 2(x2-2) + xi].
    We'll sample param -> theta, sample xi, compute grad.
    """
    if rng is None:
        rng = np.random.default_rng()
    param_space_indices = np.arange(len(posterior_probs))
    grad_acc = np.zeros(2)
    for _ in range(sample_size):
        i = rng.choice(param_space_indices, p=posterior_probs)
        theta_i = i+1
        rate = 1.0 / theta_i
        xi = rng.exponential(1.0/rate)
        grad = np.array([2*(x[0]-1) + xi, 2*(x[1]-2) + xi])
        grad_acc += grad
    return grad_acc / float(sample_size)


##########################################
# 3) Univariate Decision-Dependent Example
##########################################

def data_generator_dep_univar(batch_size, x_curr, true_theta=4, noise_std=4.0, rng=None):
    """
    xi ~ Normal(x + theta^c, noise_std^2).
    """
    if rng is None:
        rng = np.random.default_rng()
    mu = x_curr[0] + true_theta
    return rng.normal(loc=mu, scale=noise_std, size=batch_size)

def likelihood_normal_dep_univar(data_point, x, theta, noise_std=4.0):
    """
    f(data_point; x, theta) = pdf of N(x + theta, noise_std^2).
    """
    diff = data_point - (x[0] + theta)
    return 1.0/np.sqrt(2*np.pi*noise_std**2) * np.exp(-0.5*(diff**2)/(noise_std**2))

def gradient_estimator_dep_univar(x, posterior_probs, 
                                  sample_size=1, rng=None,
                                  noise_std=4.0):
    """
    h(x, xi) = (x-5)^2 + 0.5 xi x.
    grad wrt x = 2(x-5) + 0.5 xi + [h(x,xi) * partial_x log f(xi; x,theta)].
    We'll do an MC approach:
       sample param from posterior, sample xi ~ Normal(x+param, noise_std).
    """
    if rng is None:
        rng = np.random.default_rng()
    param_space_indices = np.arange(len(posterior_probs))
    grad_acc = 0.0
    for _ in range(sample_size):
        i = rng.choice(param_space_indices, p=posterior_probs)
        theta_i = i+1
        xi = rng.normal(loc=x[0] + theta_i, scale=noise_std)
        grad_term1 = 2*(x[0]-5) + 0.5*xi
        partial_log_f = (xi - (x[0] + theta_i)) / (noise_std**2)
        hval = (x[0]-5)**2 + 0.5*xi*x[0]
        grad_term2 = hval * partial_log_f
        grad_acc += (grad_term1 + grad_term2)
    return np.array([grad_acc / float(sample_size)])


##########################################
# 4) Multivariate Decision-Dependent Example
##########################################

def data_generator_dep_multi(batch_size, x_curr, true_theta=4, rng=None):
    """
    From the paper's example:
    h(x, xi) = (x1-1)^2 + (x2-2)^2 + xi,
    xi ~ Exponential(mean = (x1-x2)^2 + true_theta).
    """
    if rng is None:
        rng = np.random.default_rng()
    mean_val = (x_curr[0] - x_curr[1])**2 + true_theta
    return rng.exponential(scale=mean_val, size=batch_size)

def likelihood_exponential_dep_multi(data_point, x, theta):
    if data_point < 0:
        return 1e-300
    mean_val = (x[0]-x[1])**2 + theta
    return (1.0/mean_val)*np.exp(-data_point/mean_val)

def gradient_estimator_dep_multi(x, posterior_probs, sample_size=1, rng=None):
    """
    h(x, xi) = (x1-1)^2 + (x2-2)^2 + xi
    grad wrt x = [2(x1-1), 2(x2-2)] + [h(x, xi)* partial_x log f(...)].
    We'll do direct sampling from posterior -> param -> sample xi.
    """
    if rng is None:
        rng = np.random.default_rng()
    d = len(x)
    grad_acc = np.zeros(d)
    param_space_indices = np.arange(len(posterior_probs))

    for _ in range(sample_size):
        i = rng.choice(param_space_indices, p=posterior_probs)
        theta_i = i+1
        mean_val = (x[0]-x[1])**2 + theta_i
        xi = rng.exponential(scale=mean_val)
        grad_term1 = np.array([2*(x[0]-1), 2*(x[1]-2)])
        partial_log_f = np.zeros(d)
        partial_mean_0 = 2*(x[0]-x[1])
        partial_log_f[0] = -partial_mean_0/mean_val - (xi*partial_mean_0)/(mean_val**2)
        partial_mean_1 = -2*(x[0]-x[1])
        partial_log_f[1] = -partial_mean_1/mean_val - (xi*partial_mean_1)/(mean_val**2)
        hval = (x[0]-1)**2 + (x[1]-2)**2 + xi
        grad_term2 = hval * partial_log_f
        grad_acc += (grad_term1 + grad_term2)

    return grad_acc / float(sample_size)

##########################################
# 5) MLE Baseline for Decision-Independent (for comparison)
##########################################

def mle_update_independent(x_init, param_space, prior,
                           data_generator_func,
                           max_likelihood_func, gradient_estimator_func,
                           step_sizes, T, batch_size=1, K=1,
                           lower_bounds=None, upper_bounds=None):
    """
    MLE approach:
      1) We keep collecting data over time.
      2) Each time stage, we compute MLE from all observed data so far.
      3) We plug that MLE param into the objective, do K steps of plain SGD on x.

    x_init can be float or array, so let's ensure it's at least 1D.
    """
    x_init_array = np.atleast_1d(x_init).astype(float)
    x = np.copy(x_init_array)

    if lower_bounds is None:
        lower_bounds = -np.inf * np.ones(len(x))
    if upper_bounds is None:
        upper_bounds = np.inf * np.ones(len(x))

    trajectory = [x.copy()]
    all_data = []

    for t in range(T):
        data_batch = data_generator_func(batch_size)
        all_data.extend(data_batch)

        mle_param = max_likelihood_func(all_data, param_space)

        for j in range(K):
            idx = t*K + j
            a_t = step_sizes[idx] if idx < len(step_sizes) else step_sizes[-1]
            grad_est = gradient_estimator_func(x, mle_param)
            x = x - a_t * grad_est
            x = np.minimum(np.maximum(x, lower_bounds), upper_bounds)

        trajectory.append(x.copy())

    return trajectory


##########################################
# 6) Multi-item Newsvendor Problem
##########################################

def data_generator_newsvendor_indep(batch_size, true_mu, true_cov, rng=None):
    """
    Decision-Independent. Demand ~ Normal(true_mu, true_cov).
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.multivariate_normal(true_mu, true_cov, size=batch_size)

def likelihood_newsvendor_indep(data_point, theta_params):
    """
    Evaluate likelihood for the multi-item newsvendor example, discrete param space.
    For demonstration, we do a diagonal-cov Normal.
    """
    mu = theta_params[:3]
    var = theta_params[3:6]
    val = 1.0
    for i in range(3):
        diff = data_point[i] - mu[i]
        sigma2 = var[i]
        val *= (1.0/np.sqrt(2*np.pi*sigma2)) * np.exp(-0.5*diff**2/sigma2)
    return val if val>1e-300 else 1e-300

def cost_newsvendor(x, xi, c, p, s):
    """
    Cost function: c^T x - p^T min(x, xi+) - s^T max(0, x - xi).
    xi+ = max(xi, 0) elementwise.
    """
    xi_pos = np.maximum(xi, 0)
    return np.sum(c*x) - np.sum(p*np.minimum(x, xi_pos)) - np.sum(s*np.maximum(0, x - xi))

def grad_estimator_newsvendor_indep(x, param, c, p, s, sample_size=1, rng=None):
    """
    MC gradient estimator for multi-item newsvendor with decision-independent demand.
    param describes (mu, var, ignoring correlation).
    We sample from that normal, compute gradient wrt x.
    """
    if rng is None:
        rng = np.random.default_rng()
    d = len(x)
    grad_acc = np.zeros(d)
    for _ in range(sample_size):
        mu = param[:3]
        var = param[3:6]
        cov = np.diag(var)
        xi = rng.multivariate_normal(mu, cov)

        grad = c.copy()
        xi_pos = np.maximum(xi, 0)
        for i in range(d):
            if x[i] <= xi_pos[i]:
                grad[i] -= p[i]
            if x[i] > xi[i]:
                grad[i] -= s[i]
        grad_acc += grad
    return grad_acc / float(sample_size)

###########################################################
# Decision-Dependent Newsvendor variant
###########################################################

def data_generator_newsvendor_dep(batch_size, x, true_mu, true_cov, alpha=1.0, beta=0.5, rng=None):
    """
    Demand ~ Normal(true_mu + alpha*x^beta, true_cov).
    x^beta is elementwise.
    """
    if rng is None:
        rng = np.random.default_rng()
    shift = alpha*(x**beta)
    mu_eff = true_mu + shift
    return rng.multivariate_normal(mu_eff, true_cov, size=batch_size)

def likelihood_newsvendor_dep(data_point, x, theta_params, alpha=1.0, beta=0.5):
    """
    Evaluate f(data_point; x, theta_params).
    mu_eff = theta_params[:3] + alpha*x^beta
    diag var = theta_params[3:6]
    """
    mu = theta_params[:3]
    var = theta_params[3:6]
    shift = alpha*(x**beta)
    mu_eff = mu + shift
    val = 1.0
    for i in range(3):
        diff = data_point[i] - mu_eff[i]
        sigma2 = var[i]
        val *= (1.0/np.sqrt(2*np.pi*sigma2)) * np.exp(-0.5*(diff**2)/sigma2)
    return val if val>1e-300 else 1e-300

def grad_estimator_newsvendor_dep(x, posterior_probs, param_space,
                                  c, p, s, alpha=1.0, beta=0.5,
                                  sample_size=1, rng=None):
    """
    MC gradient wrt x of E_{param~posterior}[ E_{xi~Normal(mu+alpha x^beta,cov)}[ cost_newsvendor(x,xi) ] ].
    Includes partial_x cost(...) + cost(...) * partial_x log f(xi; x, param).
    """
    if rng is None:
        rng = np.random.default_rng()
    d = len(x)
    param_indices = np.arange(len(param_space))

    grad_acc = np.zeros(d)
    for _ in range(sample_size):
        idx = rng.choice(param_indices, p=posterior_probs)
        param = param_space[idx]
        mu = param[:3]
        var = param[3:6]
        cov = np.diag(var)
        shift = alpha*(x**beta)
        mu_eff = mu + shift
        xi = rng.multivariate_normal(mu_eff, cov)

        grad_cost = np.zeros(d)
        xi_pos = np.maximum(xi, 0)
        for i in range(d):
            grad_cost[i] += c[i]
            if x[i] <= xi_pos[i]:
                grad_cost[i] -= p[i]
            if x[i] > xi[i]:
                grad_cost[i] -= s[i]

        cost_val = cost_newsvendor(x, xi, c, p, s)

        grad_log_f = np.zeros(d)
        for i in range(d):
            diff_i = xi[i] - mu[i] - alpha*(x[i]**beta)
            grad_log_f[i] = (alpha*beta*(x[i]**(beta-1))*diff_i) / var[i]

        grad_acc += (grad_cost + cost_val*grad_log_f)
    return grad_acc / float(sample_size)