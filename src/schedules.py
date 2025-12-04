def exponential_beta(iteration, beta_0=0.1, cooling_rate=1.001):
    """β(t) = β₀ · c^t"""
    return beta_0 * (cooling_rate ** iteration)

def linear_beta(iteration, beta_0=0.1, t_max=10000):
    """β(t) = β₀ · (1 + t/t_max)"""
    return beta_0 * (1 + iteration / t_max)

def constant_beta(iteration, beta_0=0.1):
    """β(t) = β₀"""
    return beta_0

def choose_beta(beta_func):
    if beta_func == "exponential":
        return exponential_beta
    elif beta_func == "linear":
        return linear_beta
    elif beta_func == "constant":
        return constant_beta
    else:
        raise ValueError(f"Unknown beta function: {beta_func}")