"""
Definition of a state of the 3D N^2-Queens problem.
"""

import numpy as np
from collections import defaultdict
from tqdm import tqdm

# ============================================================================
# QUEEN STATE
# ============================================================================

class QueenState:
    """
    Represents a state of the 3D N^2-Queens problem.
    """

    def __init__(self, board_size):
        """
        Randomly initialize the state and compute initial energy.
        """
        queens = []
        num_queens = board_size ** 2
        occupied = set()
        
        for _ in range(num_queens):
            while True:
                pos = tuple(np.random.randint(1, board_size + 1) for _ in range(3))
                if pos not in occupied:
                    occupied.add(pos)
                    queens.append(pos)
                    break    
        self.queens = queens
        
        self.xy_planes = defaultdict(int)
        self.xz_planes = defaultdict(int)
        self.yz_planes = defaultdict(int)
        self.face_diagonals = [defaultdict(int) for _ in range(6)]
        self.space_diagonals = [defaultdict(int) for _ in range(4)]
        
        for queen in self.queens:
            self._update_state(queen, 1)
            
        collisions = 0
        for table in [self.xy_planes, self.xz_planes, self.yz_planes, *self.face_diagonals, *self.space_diagonals]:
            for value in table.values():
                collisions += value * (value - 1) // 2
            
        self.initial_energy = collisions
        
    def compute_delta_energy(self, queen_index, new_pos):
        """
        Compute the delta in energy caused by moving a queen to a new position, without modifying the state.
        """
        old_pos = self.queens[queen_index]
        old_collisions = self._calculate_collisions(old_pos)
        self.apply_move(queen_index, new_pos)
        new_collisions = self._calculate_collisions(new_pos)
        self.apply_move(queen_index, old_pos)
        
        return new_collisions - old_collisions
    
    def apply_move(self, queen_index, new_pos):
        """
        Apply a move by moving a queen to a new position, updating all tracking structures.
        """
        old_pos = self.queens[queen_index]
        self._update_state(old_pos, -1)
        self.queens[queen_index] = new_pos
        self._update_state(new_pos, 1)
    
    def _update_state(self, pos, delta):
        """
        Update plane and diagonal counts for a position.
        """
        x, y, z = pos
        self.xy_planes[(x, y)] += delta
        self.xz_planes[(x, z)] += delta
        self.yz_planes[(y, z)] += delta
        
        self.face_diagonals[0][(x - y, z)] += delta
        self.face_diagonals[1][(x + y, z)] += delta
        self.face_diagonals[2][(x - z, y)] += delta
        self.face_diagonals[3][(x + z, y)] += delta
        self.face_diagonals[4][(y - z, x)] += delta
        self.face_diagonals[5][(y + z, x)] += delta
        
        self.space_diagonals[0][(x - y, x - z)] += delta
        self.space_diagonals[1][(x - y, x + z)] += delta
        self.space_diagonals[2][(x + y, x - z)] += delta
        self.space_diagonals[3][(x + y, x + z)] += delta
        
    
    def _calculate_collisions(self, pos):
        """
        Calculate the number of queens that a queen at position (x, y, z) collides with.
        Two queens attack if they share exactly two coordinates (plane attack) or are on a diagonal.
        """
        x, y, z = pos
        attacks = (self.xy_planes[(x, y)] - 1) + (self.xz_planes[(x, z)] - 1) + (self.yz_planes[(y, z)] - 1)
        attacks += (
            (self.face_diagonals[0][(x - y, z)] - 1) +
            (self.face_diagonals[1][(x + y, z)] - 1) +
            (self.face_diagonals[2][(x - z, y)] - 1) +
            (self.face_diagonals[3][(x + z, y)] - 1) +
            (self.face_diagonals[4][(y - z, x)] - 1) +
            (self.face_diagonals[5][(y + z, x)] - 1)
        )
        attacks += (
            (self.space_diagonals[0][(x - y, x - z)] - 1) +
            (self.space_diagonals[1][(x - y, x + z)] - 1) +
            (self.space_diagonals[2][(x + y, x - z)] - 1) +
            (self.space_diagonals[3][(x + y, x + z)] - 1)
        )
        
        return attacks
    
