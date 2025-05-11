import csv
import time
import random
import numpy as np
from typing import Tuple, List, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PSO.pso_algorithm import PSO
from func.objective_functions import sphere_function, rastrigin_function, ackley_function

# File where the optimization results will be saved
CSV_FILE = "../analysis/results_threading.csv"

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)

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

# Parameter ranges using linspace where appropriate
param_ranges = {
    "dimensions": [10, 20],
    "num_particles": [30, 50],
    "max_iterations": [200, 500],
    "w": np.linspace(0.4, 0.9, 3),   # PROBAR CON VALORES MAS ALTOS
    "c1": np.linspace(0.5, 1.5, 3), # PROBAR CON VALORES MAS ALTOS
    "c2": np.linspace(0.3, 1.5, 3)  # PROBAR CON VALORES MAS ALTOS
}

# Function to run a single experiment
def run_single_experiment(
    name: str,
    function,
    bounds: Union[Tuple[float, float], List[Tuple[float, float]]],
    dimensions: int,
    num_particles: int,
    max_iterations: int,
    w: float,
    c1: float,
    c2: float
) -> Dict[str, Any]:
    pso = PSO(num_particles, dimensions, bounds, function, w=w, c1=c1, c2=c2, max_iterations=max_iterations)
    best_position, best_value, exec_time = pso.optimize()

    return {
        "function": name,
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
def save_results(result: Dict[str, Any]) -> None:
    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            result["function"],
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
    # Create CSV with header
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Function", "Best Position", "Best Value", "Execution Time",
            "Dimensions", "Num Particles", "Bounds", "Max Iterations",
            "Inertia (w)", "Cognitive (c1)", "Social (c2)"
        ])

    total_start = time.time()
    best_overall = {}

    for name, config in functions_config.items():
        print(f"Starting threaded optimization for function: {name}")
        func_start = time.time()

        function = config["function"]
        bounds_list = config["bounds"]
        param_combinations = list([
            (bounds, dimensions, num_particles, max_iterations, w, c1, c2)
            for bounds in bounds_list
            for dimensions in param_ranges["dimensions"]
            for num_particles in param_ranges["num_particles"]
            for max_iterations in param_ranges["max_iterations"]
            for w in param_ranges["w"]
            for c1 in param_ranges["c1"]
            for c2 in param_ranges["c2"]
        ])

        print(f"Total combinations for {name}: {len(param_combinations)}")

        results = []
        best_value = float('inf')
        best_data = None

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(run_single_experiment, name, function, *params)
                for params in param_combinations
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{name} Progress", ncols=100):
                result = future.result()
                results.append(result)
                save_results(result)

                if result["best_value"] < best_value:
                    best_value = result["best_value"]
                    best_data = result

        func_time = time.time() - func_start
        print(f"Finished {name} in {func_time:.2f} seconds")
        best_overall[name] = best_data

    total_time = time.time() - total_start
    print(f"\nTotal execution time: {total_time:.2f} seconds")

    print("\n<<< Summary of Best Results >>>")
    for name, data in best_overall.items():
        print(f"\nFunction: {name}")
        print(f"  Best Value: {data['best_value']:.6f}")
        print(f"  Best Position: {data['best_position']}")
        print(f"  Dimensions: {data['dimensions']}, Particles: {data['num_particles']}, Iterations: {data['max_iterations']}")
        print(f"  Bounds: {data['bounds']}")
        print(f"  Inertia w: {data['w']}, c1: {data['c1']}, c2: {data['c2']}")
        print(f"  Execution Time: {data['exec_time']:.4f} seconds")
