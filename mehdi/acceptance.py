import numpy as np

def metropolis_hastings(delta_energy, beta):
    """Standard Metropolis-Hastings: always accept if ΔE < 0, else accept with prob exp(-β·ΔE).
    
    Args:
        delta_energy: Energy change
            beta: Inverse temperature
        
        Returns:
            Acceptance probability
        """
    if delta_energy < 0:
        return True
    return np.random.random() < np.exp(-beta * delta_energy)

def always_accept(delta_energy: float, beta: float) -> bool:
    """Always accept moves (for testing/debugging)."""
    return True

def greedy(delta_energy: float, beta: float) -> bool:
    """Only accept improving moves."""
    return delta_energy < 0

def choose_acceptance(acceptance_func):
    if acceptance_func == "metropolis":
        return metropolis_hastings
    elif acceptance_func == "always_accept":
        return always_accept
    elif acceptance_func == "greedy":
        return greedy
    else:
        raise ValueError(f"Unknown acceptance function: {acceptance_func}")