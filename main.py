import numpy as np
import matplotlib.pyplot as plt

from posterior_update import BayesianPosterior
from bayesian_sgd import (
    bayesian_sgd_independent,
    bayesian_sgd_dependent
)
from example_problems import (
    data_generator_indep_univar,
    likelihood_normal_indep_univar,
    gradient_estimator_indep_univar,

    data_generator_indep_multi,
    likelihood_exponential,
    gradient_estimator_indep_multi,

    data_generator_dep_univar,
    likelihood_normal_dep_univar,
    gradient_estimator_dep_univar,

    data_generator_dep_multi,
    likelihood_exponential_dep_multi,
    gradient_estimator_dep_multi,

    data_generator_newsvendor_indep,
    likelihood_newsvendor_indep,
    grad_estimator_newsvendor_indep,
)

from capacity_provisioning import (
    run_capacity_provisioning,
    approximate_true_cost
)

def discrete_param_space_1D(max_param=20):
    return np.arange(1, max_param+1, dtype=float)

def run_univariate_decision_independent_demo(return_trajectory=False):
    param_space = discrete_param_space_1D(20)
    prior = np.ones(len(param_space)) / len(param_space)
    posterior = BayesianPosterior(param_space, prior=prior)

    T = 10000
    x_init = 0.0
    step_sizes = [2.0/(t+5) for t in range(T)]

    trajectory = bayesian_sgd_independent(
        x_init=x_init,
        posterior=posterior,
        data_generator_func=lambda bs: data_generator_indep_univar(bs, true_theta=9, noise_std=4),
        gradient_estimator_func_independent=lambda x, post: gradient_estimator_indep_univar(x, post),
        likelihood_func=likelihood_normal_indep_univar,
        step_sizes=step_sizes,
        T=T,
        batch_size=1, K=1
    )
    if not return_trajectory:
        print(f"[Univar Indep] Final solution: {trajectory[-1]}")
    return trajectory

def univar_indep_true_cost(x):
    return (x[0]-5)**2 + 4.5*x[0]

def run_univariate_decision_dependent_demo(return_trajectory=False):
    param_space = discrete_param_space_1D(30)
    prior = np.ones(len(param_space)) / len(param_space)
    posterior = BayesianPosterior(param_space, prior=prior)

    T = 10000
    x_init = 0.0
    step_sizes = [2.0/(t+5) for t in range(T)]

    trajectory = bayesian_sgd_dependent(
        x_init=x_init,
        posterior=posterior,
        data_generator_func=lambda bs, x: data_generator_dep_univar(bs, x, true_theta=4),
        gradient_estimator_func_dependent=lambda x, post: gradient_estimator_dep_univar(x, post),
        likelihood_func=likelihood_normal_dep_univar,
        step_sizes=step_sizes,
        T=T,
        batch_size=1, K=1
    )
    if not return_trajectory:
        print(f"[Univar Dep] Final solution: {trajectory[-1]}")
    return trajectory

def univar_dep_true_cost(x):
    return (x[0]-5)**2 + 0.5*x[0]**2 + 2*x[0]

def run_multivariate_decision_independent_demo(return_trajectory=False):
    param_space = discrete_param_space_1D(20)
    prior = np.ones(len(param_space)) / len(param_space)
    posterior = BayesianPosterior(param_space, prior=prior)

    T = 10000
    x_init = np.array([5.0, 5.0])
    step_sizes = [2.0/(t+5) for t in range(T)]

    trajectory = bayesian_sgd_independent(
        x_init=x_init,
        posterior=posterior,
        data_generator_func=lambda bs: data_generator_indep_multi(bs, true_theta=4),
        gradient_estimator_func_independent=lambda x, post: gradient_estimator_indep_multi(x, post),
        likelihood_func=likelihood_exponential,
        step_sizes=step_sizes,
        T=T,
        batch_size=1, K=1
    )
    if not return_trajectory:
        print(f"[MultiVar Indep] Final solution: {trajectory[-1]}")
    return trajectory

def multivar_indep_true_cost(x):
    return (x[0]-1)**2 + (x[1]-2)**2 + 4*(x[0]+x[1])

def run_multivariate_decision_dependent_demo(return_trajectory=False):
    param_space = discrete_param_space_1D(20)
    prior = np.ones(len(param_space)) / len(param_space)
    posterior = BayesianPosterior(param_space, prior=prior)

    T = 10000
    x_init = np.array([5.0, 5.0])
    step_sizes = [2.0/(t+5) for t in range(T)]

    trajectory = bayesian_sgd_dependent(
        x_init=x_init,
        posterior=posterior,
        data_generator_func=lambda bs, x: data_generator_dep_multi(bs, x, true_theta=4),
        gradient_estimator_func_dependent=lambda x, post: gradient_estimator_dep_multi(x, post),
        likelihood_func=likelihood_exponential_dep_multi,
        step_sizes=step_sizes,
        T=T,
        batch_size=1, K=1
    )
    if not return_trajectory:
        print(f"[MultiVar Dep] Final solution: {trajectory[-1]}")
    return trajectory

def multivar_dep_true_cost(x):
    return (x[0]-1)**2 + (x[1]-2)**2 + (x[0]-x[1])**2 + 4


def run_newsvendor_indep_demo(return_trajectory=False):
    param_space = np.array([
        [10,15,20, 3,6,9, 0.0,0.0,0.0],
        [10,15,20, 4,7,10,0.0,0.0,0.0],
        [ 5,10,15, 2,3,4, 0.0,0.0,0.0]
    ])
    prior = np.ones(len(param_space)) / len(param_space)
    posterior = BayesianPosterior(param_space, prior)

    T = 10000
    x_init = np.array([15.0, 15.0, 15.0])
    step_sizes = [2.0/(t+5) for t in range(T)]

    true_mu = np.array([10,15,20])
    true_var = np.array([3,6,9])
    true_cov = np.diag(true_var)

    def data_gen(batch_size):
        return data_generator_newsvendor_indep(batch_size, true_mu, true_cov)

    import numpy.random as rnd
    rng = rnd.default_rng()

    def gradient_est(x, posterior_probs):
        param_indices = np.arange(len(posterior_probs))
        i = rng.choice(param_indices, p=posterior_probs)
        param = param_space[i]  
        return grad_estimator_newsvendor_indep(
            x, param,
            c=np.array([2,4,6]), p=np.array([4,6,8]), s=np.array([1,2,3]),
            sample_size=1
        )
    trajectory = bayesian_sgd_independent(
        x_init=x_init,
        posterior=posterior,
        data_generator_func=data_gen,
        gradient_estimator_func_independent=lambda x, post: gradient_est(x, post),
        likelihood_func=likelihood_newsvendor_indep,
        step_sizes=step_sizes,
        T=T,
        batch_size=2, K=1,
        lower_bounds=np.zeros(3),
        upper_bounds=np.array([10000.0,10000.0,10000.0])
    )
    if not return_trajectory:
        print(f"[Newsvendor Indep] Final solution: {trajectory[-1]}")
    return trajectory

def make_plots():
    traj1 = run_univariate_decision_independent_demo(return_trajectory=True)
    xs1 = np.array(traj1) 
    timesteps1 = np.arange(len(xs1))

    plt.figure()
    plt.plot(timesteps1, xs1, color='blue')
    plt.title("Scenario 1: Univar Indep - x(t)")
    plt.xlabel("Time")
    plt.ylabel("x")

    cost1 = [univar_indep_true_cost(np.atleast_1d(x)) for x in xs1]
    plt.figure()
    plt.plot(timesteps1, cost1, color='red')
    plt.title("Scenario 1: Univar Indep - cost(t)")
    plt.xlabel("Time")
    plt.ylabel("Cost")

    traj2 = run_univariate_decision_dependent_demo(return_trajectory=True)
    xs2 = np.array(traj2)
    timesteps2 = np.arange(len(xs2))

    plt.figure()
    plt.plot(timesteps2, xs2, color='blue')
    plt.title("Scenario 2: Univar Dep - x(t)")
    plt.xlabel("Time")
    plt.ylabel("x")

    cost2 = [univar_dep_true_cost(np.atleast_1d(x)) for x in xs2]
    plt.figure()
    plt.plot(timesteps2, cost2, color='red')
    plt.title("Scenario 2: Univar Dep - cost(t)")
    plt.xlabel("Time")
    plt.ylabel("Cost")

    traj3 = run_multivariate_decision_independent_demo(return_trajectory=True)
    xs3 = np.array(traj3) 
    timesteps3 = np.arange(len(xs3))

    plt.figure()
    plt.plot(timesteps3, xs3[:,0], 'b-', label='x1')
    plt.plot(timesteps3, xs3[:,1], 'g-', label='x2')
    plt.legend()
    plt.title("Scenario 3: MultiVar Indep - x(t)")
    plt.xlabel("Time")
    plt.ylabel("Decision")

    cost3 = [multivar_indep_true_cost(x) for x in xs3]
    plt.figure()
    plt.plot(timesteps3, cost3, 'r-')
    plt.title("Scenario 3: MultiVar Indep - cost(t)")
    plt.xlabel("Time")
    plt.ylabel("Cost")

    traj4 = run_multivariate_decision_dependent_demo(return_trajectory=True)
    xs4 = np.array(traj4)
    timesteps4 = np.arange(len(xs4))

    plt.figure()
    plt.plot(timesteps4, xs4[:,0], 'b-', label='x1')
    plt.plot(timesteps4, xs4[:,1], 'g-', label='x2')
    plt.legend()
    plt.title("Scenario 4: MultiVar Dep - x(t)")
    plt.xlabel("Time")
    plt.ylabel("Decision")

    cost4 = [multivar_dep_true_cost(x) for x in xs4]
    plt.figure()
    plt.plot(timesteps4, cost4, 'r-')
    plt.title("Scenario 4: MultiVar Dep - cost(t)")
    plt.xlabel("Time")
    plt.ylabel("Cost")

    traj5 = run_newsvendor_indep_demo(return_trajectory=True)
    xs5 = np.array(traj5)
    timesteps5 = np.arange(len(xs5))

    plt.figure()
    plt.plot(timesteps5, xs5[:,0], 'b-', label='x1')
    plt.plot(timesteps5, xs5[:,1], 'g-', label='x2')
    plt.plot(timesteps5, xs5[:,2], 'r-', label='x3')
    plt.legend()
    plt.title("Scenario 5: Newsvendor Indep - x(t)")
    plt.xlabel("Time")
    plt.ylabel("Decision")

    norms5 = [np.linalg.norm(x) for x in xs5]
    plt.figure()
    plt.plot(timesteps5, norms5, 'm-')
    plt.title("Scenario 5: Newsvendor Indep - ||x(t)||")
    plt.xlabel("Time")
    plt.ylabel("Norm of x")

    plt.show()
    
def run_capacity_provisioning_demo_for_plots():
    """
    This function runs the capacity provisioning scenario, returns the trajectory,
    and produces 2 separate plots:
       - x(t) vs t
       - approximate cost(t) vs t
    """
    T = 10000
    c = 1.0
    p = 5.0
    true_lambda = 8

    traj = run_capacity_provisioning(
        true_lambda=true_lambda,
        T=T,
        c=c,
        p=p,
        step_coefficient=0.5, 
        batch_size=1,
        K=1
    )
    xs = np.array(traj).flatten()  

    plt.figure()
    plt.plot(xs, markersize=3, linewidth=1)
    plt.title("Capacity Provisioning: x(t) Over Time")
    plt.xlabel("Iteration t")
    plt.ylabel("Capacity x(t)")

    cost_vals = []
    rng = np.random.default_rng(123)
    for x_val in traj:
        cost_est = approximate_true_cost(x_val, true_lambda=true_lambda,
                                         c=c, p=p,
                                         mc_samples=2000,
                                         rng=rng)
        cost_vals.append(cost_est)
    plt.figure()
    plt.plot(cost_vals, markersize=3, linewidth=1)
    plt.title("Capacity Provisioning: Approx. Cost Over Time")
    plt.xlabel("Iteration t")
    plt.ylabel("Approx. $\mathbb{E}[cx + p*(xi - x)^+]$")

    plt.show()



if __name__ == "__main__":
    run_univariate_decision_independent_demo()
    run_univariate_decision_dependent_demo()
    run_multivariate_decision_independent_demo()
    run_multivariate_decision_dependent_demo()
    run_newsvendor_indep_demo()

    make_plots()
    
    run_capacity_provisioning_demo_for_plots()