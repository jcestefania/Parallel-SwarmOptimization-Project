import csv
import time
from tqdm import tqdm
from PSO.pso_algorithm import PSO
from func.objective_functions import sphere_function, rastrigin_function, ackley_function
import itertools
import numpy as np
from typing import Tuple, List, Union

# File where the optimization results will be saved
CSV_FILE = "../analysis/results.csv"

# Function to save the PSO results to a CSV file
def save_results(
    function_name: str,
    best_position: np.ndarray,
    best_value: float,
    exec_time: float,
    dimensions: int,
    num_particles: int,
    bounds: Union[Tuple[float, float], List[Tuple[float, float]]],
    max_iterations: int,
    w: float,
    c1: float,
    c2: float
) -> None:
    """
    Saves the optimization results obtained by the PSO algorithm to a CSV file.
    """
    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            function_name,
            best_position.tolist(),
            best_value,
            exec_time,
            dimensions,
            num_particles,
            bounds,
            max_iterations,
            w,
            c1,
            c2
        ])

if __name__ == "__main__":
    # Set seeds for reproducibility
    seed = 42
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    # Start global timer to measure total execution time
    global_start_time = time.time()

    # Range of values to combine
    param_ranges = {
    "dimensions": [10, 20],
    "num_particles": [30, 50],
    "max_iterations": [200, 500],
    "w": np.linspace(0.4, 0.9, 3),  # PROBAR CON VALORES MAS ALTOS
    "c1": np.linspace(0.5, 1.5, 3), # PROBAR CON VALORES MAS ALTOS
    "c2": np.linspace(0.3, 1.5, 3) # PROBAR CON VALORES MAS ALTOS
}

    # Unified configuration for each benchmark function
    functions_config = {
        "Sphere": {
            "function": sphere_function,
            "bounds": [(-5, 5), (-10, 10)]
        },
        "Rastrigin": {
            "function": rastrigin_function,
            "bounds": [(-5.12, 5.12), (-10, 10)]
        },
        "Ackley": {
            "function": ackley_function,
            "bounds": [(-32.768, 32.768)]
        }
    }

    # Initialize CSV file with header (overwrites on each execution)
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Function", "Best Position", "Best Value", "Execution Time",
            "Dimensions", "Num Particles", "Bounds", "Max Iterations",
            "Inertia (w)", "Cognitive (c1)", "Social (c2)"
        ])

    # Dictionary to store the best result per function to show a final summary
    best_overall = {}

    for name, config in functions_config.items():
        print(f"Starting optimization for function: {name}")
        best_func_value = float('inf')
        best_func_data = None

        # Start timer for the current function
        func_start_time = time.time()

        # Generate parameter combinations using only valid bounds for the current function
        function = config["function"]
        bounds_list = config["bounds"]
        param_combinations = list(itertools.product(
            bounds_list,
            param_ranges["dimensions"],
            param_ranges["num_particles"],
            param_ranges["max_iterations"],
            param_ranges["w"],
            param_ranges["c1"],
            param_ranges["c2"]
        ))

        print(f"Total combinations for {name}: {len(param_combinations)}")

        with tqdm(total=len(param_combinations), desc=f"{name} Progress", ncols=100) as pbar:
            for bounds, dimensions, num_particles, max_iterations, w, c1, c2 in param_combinations:
                # Run PSO
                pso = PSO(num_particles, dimensions, bounds, function, w=w, c1=c1, c2=c2, max_iterations=max_iterations)
                best_position, best_value, exec_time = pso.optimize()

                # Save results to CSV
                save_results(name, best_position, best_value, exec_time, dimensions, num_particles, bounds, max_iterations, w, c1, c2)

                # Compare with the best value for this function
                if best_value < best_func_value:
                    best_func_value = best_value
                    best_func_data = {
                        "position": best_position,
                        "value": best_value,
                        "time": exec_time,
                        "dimensions": dimensions,
                        "particles": num_particles,
                        "bounds": bounds,
                        "iterations": max_iterations,
                        "w": w,
                        "c1": c1,
                        "c2": c2
                    }

                # Update progress bar
                pbar.set_postfix({
                    "Dim": dimensions,
                    "Part": num_particles,
                    "w": round(w, 2),
                    "c1": round(c1, 2),
                    "c2": round(c2, 2)
                })
                pbar.update(1)

        # Total time spent optimizing this function
        func_total_time = time.time() - func_start_time
        print(f"Total execution time for {name}: {func_total_time:.2f} seconds\n")

        best_overall[name] = best_func_data
        print(f"Finished optimization for function: {name}\n")

    # Total execution time of the entire script
    total_time = time.time() - global_start_time
    print(f"Total execution time: {total_time:.2f} seconds\n")

    # Final summary of the best results
    print("<<< Summary of Best Results >>>")
    for name, data in best_overall.items():
        print(f"\nFunction: {name}")
        print(f"  Best Value: {data['value']:.6f}")
        print(f"  Best Position: {data['position']}")
        print(f"  Dimensions: {data['dimensions']}, Particles: {data['particles']}, Iterations: {data['iterations']}")
        print(f"  Bounds: {data['bounds']}")
        print(f"  Inertia w: {data['w']}, c1: {data['c1']}, c2: {data['c2']}")
        print(f"  Execution Time: {data['time']:.4f} seconds")
