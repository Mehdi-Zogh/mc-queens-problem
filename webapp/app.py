"""
Flask webapp for 3D N^2-Queens visualization using Three.js.
"""
import sys
import os
import json
import threading
import csv
from datetime import datetime

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from visualization import create_3d_visualization
from src.mcmc import mcmc_chain, exponential_beta, linear_beta, constant_beta, metropolis_hastings, greedy

app = Flask(__name__)

@app.route('/')
def index():
    """Main page with visualization interface."""
    return render_template('index.html')

@app.route('/visualize', methods=['POST'])
def visualize():
    """
    Endpoint to generate 3D visualization data.
    Expects JSON: {'n': int, 'queens': [[x, y, z], ...]}
    Returns JSON with Three.js scene data.
    """
    try:
        data = request.get_json()
        n = int(data.get('n', 3))
        queens_raw = data.get('queens', [])
        
        # Validate input
        if n < 1:
            return jsonify({'error': 'n must be at least 1'}), 400
        
        if not isinstance(queens_raw, list):
            return jsonify({'error': 'queens must be a list'}), 400
        
        # Convert queens to tuples and handle coordinate system
        # The MCMC code uses 1-based indexing, but we'll accept both
        queens = []
        for i, q in enumerate(queens_raw):
            if not isinstance(q, (list, tuple)) or len(q) != 3:
                return jsonify({'error': f'Queen {i} must be a 3-tuple/list'}), 400
            
            x, y, z = q
            # Convert to 0-based for visualization (if 1-based, subtract 1)
            # We'll detect: if any coordinate is 0, assume 0-based; otherwise assume 1-based
            if all(coord >= 1 for coord in [x, y, z]):
                # Looks like 1-based, convert to 0-based
                x, y, z = x - 1, y - 1, z - 1
            
            # Validate bounds
            if not (0 <= x < n and 0 <= y < n and 0 <= z < n):
                return jsonify({'error': f'Queen {i} at ({x}, {y}, {z}) is out of bounds [0, {n-1}]'}), 400
            
            queens.append((x, y, z))
        
        # Check for duplicates
        if len(set(queens)) != len(queens):
            return jsonify({'error': 'Duplicate queen positions found'}), 400
        
        # Generate visualization data
        viz_data = create_3d_visualization(n, queens)
        
        return jsonify(viz_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/run_mcmc', methods=['POST'])
def run_mcmc():
    """
    Run MCMC algorithm with real-time updates via Server-Sent Events (SSE).
    Expects JSON: {
        'board_size': int,
        'num_iterations': int,
        'beta_func': str ('exponential', 'linear', 'constant'),
        'acceptance_func': str ('metropolis', 'greedy')
    }
    Streams updates every 500 iterations.
    """
    try:
        data = request.get_json()
        board_size = int(data.get('board_size', 3))
        num_iterations = int(data.get('num_iterations', 10000))
        beta_func_name = data.get('beta_func', 'exponential')
        acceptance_func_name = data.get('acceptance_func', 'metropolis')
        save_csv = bool(data.get('save_csv', False))
        
        # Validate input
        if board_size < 1:
            return jsonify({'error': 'board_size must be at least 1'}), 400
        if num_iterations < 1:
            return jsonify({'error': 'num_iterations must be at least 1'}), 400
        
        # Select beta function
        beta_funcs = {
            'exponential': exponential_beta,
            'linear': linear_beta,
            'constant': constant_beta
        }
        beta_func = beta_funcs.get(beta_func_name, exponential_beta)
        
        # Select acceptance function
        accept_funcs = {
            'metropolis': metropolis_hastings,
            'greedy': greedy
        }
        acceptance_func = accept_funcs.get(acceptance_func_name, metropolis_hastings)
        
        def generate():
            """Generator function for SSE streaming."""
            import queue
            import time
            
            # Queue to store updates from callback
            update_queue = queue.Queue()
            mcmc_complete = threading.Event()
            mcmc_error = [None]  # Use list to allow modification in nested function
            
            def progress_callback(iteration, queens, energy, beta, accepted_moves):
                """Callback called every 500 iterations."""
                # Convert 1-based to 0-based coordinates for visualization
                queens_0based = [(q[0] - 1, q[1] - 1, q[2] - 1) for q in queens]
                viz_data = create_3d_visualization(board_size, queens_0based)
                
                # Put update in queue
                update = {
                    'type': 'progress',
                    'iteration': iteration,
                    'energy': energy,
                    'beta': float(beta),
                    'accepted_moves': accepted_moves,
                    'visualization': viz_data
                }
                update_queue.put(update)
            
            def run_mcmc_thread():
                """Run MCMC in a separate thread."""
                try:
                    final_queens, final_energies, final_betas, final_accepted = mcmc_chain(
                        board_size=board_size,
                        num_iterations=num_iterations,
                        target_energy=0,
                        beta_func=beta_func,
                        acceptance_func=acceptance_func,
                        verbose=False,
                        progress_callback=progress_callback
                    )
                    
                    # Send final result
                    final_queens_0based = [(q[0] - 1, q[1] - 1, q[2] - 1) for q in final_queens[-1]]
                    final_viz_data = create_3d_visualization(board_size, final_queens_0based)
                    csv_file = None
                    if save_csv:
                        # Match CLI behavior: write final queen positions (1-based) with header x,y,z
                        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                        results_dir = os.path.join(repo_root, 'results')
                        os.makedirs(results_dir, exist_ok=True)
                        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"results_N{board_size}_it{num_iterations}_{ts}.csv"
                        csv_path = os.path.join(results_dir, filename)
                        with open(csv_path, "w", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(["x", "y", "z"])
                            for queen_pos in final_queens[-1]:
                                writer.writerow(queen_pos)
                        csv_file = os.path.relpath(csv_path, repo_root)

                    update_queue.put({
                        'type': 'complete',
                        'final_energy': final_energies[-1],
                        'total_iterations': len(final_energies),
                        'accepted_moves': final_accepted,
                        'visualization': final_viz_data,
                        'csv_file': csv_file
                    })
                except Exception as e:
                    mcmc_error[0] = str(e)
                finally:
                    mcmc_complete.set()
            
            # Send initial message
            yield f"data: {json.dumps({'type': 'start', 'board_size': board_size, 'num_iterations': num_iterations})}\n\n"
            
            # Start MCMC in a separate thread
            thread = threading.Thread(target=run_mcmc_thread)
            thread.daemon = True
            thread.start()
            
            # Stream updates from queue
            while not mcmc_complete.is_set() or not update_queue.empty():
                try:
                    # Wait for update with timeout
                    update = update_queue.get(timeout=0.1)
                    yield f"data: {json.dumps(update)}\n\n"
                except queue.Empty:
                    # Check if MCMC is complete
                    if mcmc_complete.is_set():
                        break
                    continue
            
            # Check for errors
            if mcmc_error[0]:
                yield f"data: {json.dumps({'type': 'error', 'message': mcmc_error[0]})}\n\n"
        
        return Response(stream_with_context(generate()), mimetype='text/event-stream')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
