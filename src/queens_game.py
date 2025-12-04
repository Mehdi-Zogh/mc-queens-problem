from dataclasses import dataclass
from tqdm.notebook import tqdm
from src.schedules import choose_beta
from src.acceptance import choose_acceptance
import numpy as np


class QueensGame:
    def __init__(self, board_size, state=None):
        self.board_size = board_size
        self.state = self._initialize_state() if state is None else state
        self.energy = self.compute_energy(verbose=True) 


    def _initialize_state(self):
        """Initialize with N² queens placed randomly (no two queens same cell)."""
        occupied = set()
        state = []
        num_queens = self.board_size**2
        for _ in range(num_queens):
            while True:
                pos = tuple(np.random.randint(1, self.board_size + 1) for _ in range(3))
                if pos not in occupied:
                    occupied.add(pos)
                    state.append(pos)
                    break
        return state


    def minimize_energy(
        self,
        num_iterations,
        beta_func="exponential",
        acceptance_func="metropolis",
        target_energy=0,
        verbose=False,
    ):
        """
        Run MCMC chain for the 3D N^2-Queens problem.

        Args:
            board_size: Size of board (N for NxNxN)
            num_iterations: Number of MCMC steps
            beta_func: Function that returns β given iteration number
            acceptance_func: Function that decides whether to accept a move
            target_energy: Stop when energy <= this value
            verbose: Print progress

        Returns:
            MCMCResult object containing chain history
        """

        beta_func = choose_beta(beta_func)
        acceptance_func = choose_acceptance(acceptance_func)
        
        states = [self.state.copy()]
        energies = [self.energy]
        betas = []
        accepted_moves = 0

        pbar = tqdm(range(num_iterations), position=0, leave=True, desc="Minimizing energy", disable=not verbose)
        for t in pbar:
            beta = beta_func(t)
            betas.append(beta)

            # Move random queen to random position
            queen_idx, new_pos = self._new_queen_position()

            # Compute energy change
            delta_energy = self._compute_delta_energy(queen_idx, new_pos)

            # Accept or reject using Metropolis-Hastings
            if acceptance_func(delta_energy, beta):
                self.state[queen_idx] = new_pos
                self.energy += delta_energy
                accepted_moves += 1

            energies.append(self.energy)
            states.append(self.state.copy())

            # Stop if solution found
            if self.energy <= target_energy:
                if verbose:
                    pbar.write(f"✓ Solution found at iteration {t} with energy {self.energy}")
                break

        return MCMCResult(
            states=states,
            energies=energies,
            betas=betas,
            accepted_moves=accepted_moves,
            final_state=self.state,
            min_energy=min(energies),
            final_energy=self.energy,
        )  

    def compute_energy(self, verbose=False):
        """Compute total number of attacking pairs."""
        conflicts = 0
        pbar = tqdm(range(len(self.state)), disable=not verbose, desc="Computing initial energy")

        for m in pbar:
            for n in range(m + 1, len(self.state)):
                if QueensGame._attacks_each_other(self.state[m], self.state[n]):
                    conflicts += 1
        return conflicts

    def _compute_delta_energy(self, queen_idx, new_pos):
        """Compute energy change when moving one queen. O(N²) instead of O(N⁴).
        
        Args:
            queen_idx: Index of queen to move
            new_pos: New position for queen
        
        Returns:
            Energy change
        """
        old_pos = self.state[queen_idx]
        conflicts_before = sum(1 for i in range(len(self.state)) 
                            if i != queen_idx and QueensGame._attacks_each_other(old_pos, self.state[i]))
        conflicts_after = sum(1 for i in range(len(self.state)) 
                            if i != queen_idx and QueensGame._attacks_each_other(new_pos, self.state[i]))
        return conflicts_after - conflicts_before 

    def _new_queen_position(self):
        """Generate a new position for a queen ensuring it's not occupied by another queen."""
        queen_idx = np.random.randint(0, len(self.state))
        
        # Generate new position ensuring it's not occupied by another queen
        while True:
            new_pos = tuple(np.random.randint(1, self.board_size + 1) for _ in range(3))
            # Check if any other queen is at this position
            if all(self.state[i] != new_pos for i in range(len(self.state)) if i != queen_idx):
                break

        return queen_idx, new_pos



    @staticmethod
    def _attacks_each_other(pos1, pos2):
        """Check if two queens attack each other in 3D."""
        i1, j1, k1 = pos1
        i2, j2, k2 = pos2

        # Same position
        if i1 == i2 and j1 == j2 and k1 == k2:
            return True    
        # Plane attacks (exactly two coordinates match)
        if i1 == i2 and j1 == j2 and k1 != k2:
            return True
        if i1 == i2 and k1 == k2 and j1 != j2:
            return True
        if j1 == j2 and k1 == k2 and i1 != i2:
            return True
        
        # 2D diagonals within planes
        if abs(i1 - i2) == abs(j1 - j2) and k1 == k2 and abs(i1 - i2) != 0:
            return True
        if abs(i1 - i2) == abs(k1 - k2) and j1 == j2 and abs(i1 - i2) != 0:
            return True
        if abs(j1 - j2) == abs(k1 - k2) and i1 == i2 and abs(j1 - j2) != 0:
            return True
        
        # 3D diagonals
        if (abs(i1 - i2) == abs(j1 - j2) == abs(k1 - k2) and abs(i1 - i2) != 0):
            return True
        
        return False


@dataclass
class MCMCResult:
    """Results from MCMC run."""

    states: list
    energies: list
    betas: list
    accepted_moves: int
    final_state: list
    min_energy: float
    final_energy: float