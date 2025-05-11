import csv
import time
import random
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, List, Union, Dict, Callable

from PSO.pso_algorithm import PSO
from func.objective_functions import sphere_function, rastrigin_function, ackley_function

# File to save results
CSV_FILE = "../analysis/results_multiprocessing.csv"

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)

# Unified configuration for each benchmark function
functions_config: Dict[str, Dict[str, Union[Callable, List[Tuple[float, float]]]]] = {
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

# Ranges of hyperparameters using linspace
param_ranges = {
    "dimensions": [10, 20],
    "num_particles": [30, 50],
    "max_iterations": [200, 500],
    "w": np.linspace(0.4, 0.9, 3),
    "c1": np.linspace(0.5, 1.5, 3),
    "c2": np.linspace(0.3, 1.5, 3)
}

# Function executed by each process
def run_single_experiment(
    name: str,
    function,
    bounds: Tuple[float, float],
    dimensions: int,
    num_particles: int,
    max_iterations: int,
    w: float,
    c1: float,
    c2: float
) -> Dict:
    pso = PSO(num_particles, dimensions, bounds, function, w, c1, c2, max_iterations)
    best_position, best_value, exec_time = pso.optimize()

    return {
        "function_name": name,
        "best_position": best_position,
        "best_value": best_value,
        "exec_time": exec_time,
        "dimensions": dimensions,
        "num_particles": num_particles,
        "bounds": bounds,
        "max_iterations": max_iterations,
        "w": w,
        "c1": c1,
        "c2": c2
    }

# Save results to CSV
def save_results(result: Dict) -> None:
    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            result["function_name"],
            result["best_position"].tolist(),
            result["best_value"],
            result["exec_time"],
            result["dimensions"],
            result["num_particles"],
            result["bounds"],
            result["max_iterations"],
            result["w"],
            result["c1"],
            result["c2"]
        ])

if __name__ == "__main__":
    # Initialize CSV file with headers
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Function", "Best Position", "Best Value", "Execution Time",
            "Dimensions", "Num Particles", "Bounds", "Max Iterations",
            "Inertia (w)", "Cognitive (c1)", "Social (c2)"
        ])

    best_overall = {}
    total_start = time.time()

    for name, config in functions_config.items():
        print(f"\nStarting optimization for function: {name}")
        function = config["function"]
        bounds_list = config["bounds"]
        param_combinations = [
            (bounds, dimensions, num_particles, max_iterations, w, c1, c2)
            for bounds in bounds_list
            for dimensions in param_ranges["dimensions"]
            for num_particles in param_ranges["num_particles"]
            for max_iterations in param_ranges["max_iterations"]
            for w in param_ranges["w"]
            for c1 in param_ranges["c1"]
            for c2 in param_ranges["c2"]
        ]


        print(f"Total combinations for {name}: {len(param_combinations)}")

        best_value = float("inf")
        best_result = None
        start_time = time.time()

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(run_single_experiment, name, function, *params)
                for params in param_combinations
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{name} Progress", ncols=100):
                result = future.result()
                save_results(result)
                if result["best_value"] < best_value:
                    best_value = result["best_value"]
                    best_result = result

        print(f"Total execution time for {name}: {time.time() - start_time:.2f} seconds\n")
        best_overall[name] = best_result

    print(f"\n<<< Summary of Best Results >>>")
    for name, result in best_overall.items():
        print(f"\nFunction: {name}")
        print(f"  Best Value: {result['best_value']:.6f}")
        print(f"  Best Position: {result['best_position']}")
        print(f"  Dimensions: {result['dimensions']}, Particles: {result['num_particles']}, Iterations: {result['max_iterations']}")
        print(f"  Bounds: {result['bounds']}")
        print(f"  Inertia w: {result['w']}, c1: {result['c1']}, c2: {result['c2']}")
        print(f"  Execution Time: {result['exec_time']:.4f} seconds")

    print(f"\nTotal execution time: {time.time() - total_start:.2f} seconds")
