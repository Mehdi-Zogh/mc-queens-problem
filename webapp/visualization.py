"""
3D visualization data generation for the N^2-Queens problem.
Returns data structure for Three.js rendering.
"""
from typing import List, Tuple, Set, Dict

def queens_attack_3d(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> bool:
    """
    Check if two queens attack each other in 3D.
    Two queens attack if they share exactly two coordinates (plane attack) 
    or are on a diagonal (face or space diagonal).
    """
    x1, y1, z1 = a
    x2, y2, z2 = b
    
    # Same position
    if x1 == x2 and y1 == y2 and z1 == z2:
        return False
    
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    
    # Plane attacks: two coordinates match
    if (dx == 0 and dy == 0) or (dx == 0 and dz == 0) or (dy == 0 and dz == 0):
        return True
    
    # Face diagonals: two coordinates differ by same amount, third matches
    if dz == 0 and abs(dx) == abs(dy) and dx != 0:
        return True
    if dy == 0 and abs(dx) == abs(dz) and dx != 0:
        return True
    if dx == 0 and abs(dy) == abs(dz) and dy != 0:
        return True
    
    # Space diagonals: all three coordinates differ by same amount
    if abs(dx) == abs(dy) == abs(dz) and dx != 0:
        return True
    
    return False

def compute_attacked_cells(queens: List[Tuple[int, int, int]], n: int) -> Set[Tuple[int, int, int]]:
    """
    Compute all cells attacked by queens.
    Returns a set of (x, y, z) coordinates that are attacked.
    """
    attacked = set()
    queen_positions = set(queens)
    
    for queen in queens:
        qx, qy, qz = queen
        
        # All possible attack directions
        directions = [
            # Axes (plane attacks)
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
            # Face diagonals in XY plane
            (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
            # Face diagonals in XZ plane
            (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
            # Face diagonals in YZ plane
            (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
            # Space diagonals
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
            (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)
        ]
        
        for dx, dy, dz in directions:
            for step in range(1, n):
                x, y, z = qx + dx * step, qy + dy * step, qz + dz * step
                if 0 <= x < n and 0 <= y < n and 0 <= z < n:
                    attacked.add((x, y, z))
                else:
                    break
    
    # Remove queen positions themselves
    attacked -= queen_positions
    return attacked

def create_3d_visualization(n: int, queens: List[Tuple[int, int, int]]) -> Dict:
    """
    Create visualization data for the N^2-Queens problem.
    
    Args:
        n: Board size (n x n x n cube)
        queens: List of (x, y, z) tuples representing queen positions (0-based)
    
    Returns:
        Dictionary containing data for Three.js rendering:
        {
            'n': int,
            'cells': [{'x': int, 'y': int, 'z': int, 'type': 'queen'|'attacked'|'safe'}, ...],
            'queens': [{'x': int, 'y': int, 'z': int, 'index': int}, ...],
            'attacks': [{'queen1': int, 'queen2': int, 'from': [x,y,z], 'to': [x,y,z]}, ...],
            'grid_lines': {'x': [...], 'y': [...], 'z': [...]}
        }
    """
    # Validate queens
    for q in queens:
        if any(c < 0 or c >= n for c in q):
            raise ValueError(f"Queen {q} outside bounds [0, {n-1}]")
    
    # Compute attacked cells
    attacked_cells = compute_attacked_cells(queens, n)
    queen_positions = set(queens)
    
    # Build cells data
    cells = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                cell = (x, y, z)
                if cell in queen_positions:
                    cell_type = 'queen'
                elif cell in attacked_cells:
                    cell_type = 'attacked'
                else:
                    cell_type = 'safe'
                
                cells.append({
                    'x': x,
                    'y': y,
                    'z': z,
                    'type': cell_type
                })
    
    # Build queens data
    queens_data = []
    for i, q in enumerate(queens):
        queens_data.append({
            'x': q[0],
            'y': q[1],
            'z': q[2],
            'index': i
        })
    
    # Build attack lines
    attacks = []
    for i in range(len(queens)):
        for j in range(i + 1, len(queens)):
            if queens_attack_3d(queens[i], queens[j]):
                attacks.append({
                    'queen1': i,
                    'queen2': j,
                    'from': [queens[i][0] + 0.5, queens[i][1] + 0.5, queens[i][2] + 0.5],
                    'to': [queens[j][0] + 0.5, queens[j][1] + 0.5, queens[j][2] + 0.5]
                })
    
    # Build grid lines as pairs of points
    grid_lines = []
    
    # Lines parallel to X axis
    for y in range(n + 1):
        for z in range(n + 1):
            grid_lines.append({
                'type': 'x',
                'from': [0, y, z],
                'to': [n, y, z]
            })
    
    # Lines parallel to Y axis
    for x in range(n + 1):
        for z in range(n + 1):
            grid_lines.append({
                'type': 'y',
                'from': [x, 0, z],
                'to': [x, n, z]
            })
    
    # Lines parallel to Z axis
    for x in range(n + 1):
        for y in range(n + 1):
            grid_lines.append({
                'type': 'z',
                'from': [x, y, 0],
                'to': [x, y, n]
            })
    
    return {
        'n': n,
        'cells': cells,
        'queens': queens_data,
        'attacks': attacks,
        'grid_lines': grid_lines,
        'attack_count': len(attacks)
    }
