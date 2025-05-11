import asyncio
import csv
import time
import random
import numpy as np
import tqdm.asyncio
from typing import Callable, Dict, List, Tuple, Union
from PSO.pso_algorithm import PSO
from func.objective_functions import sphere_function, rastrigin_function, ackley_function

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)

# CSV file to store results
CSV_FILE = "../analysis/results_asyncio.csv"

# Central configuration of functions and bounds
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

# Parameter space (generated with linspace where appropriate)
param_ranges = {
    "dimensions": [10, 20],
    "num_particles": [30, 50],
    "max_iterations": [200, 500],
    "w": np.linspace(0.4, 0.9, 3),
    "c1": np.linspace(0.5, 1.5, 3),
    "c2": np.linspace(0.3, 1.5, 3)
}

# Save results to CSV
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

# Async function to run one PSO experiment
async def run_single_experiment_async(
    name: str,
    function: Callable,
    bounds: Tuple[float, float],
    dimensions: int,
    num_particles: int,
    max_iterations: int,
    w: float,
    c1: float,
    c2: float
) -> Dict:
    start = time.time()
    pso = PSO(num_particles, dimensions, bounds, function, w, c1, c2, max_iterations)
    best_position, best_value, _ = pso.optimize()
    exec_time = time.time() - start

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

# Main async execution
async def main_async():
    # Initialize CSV
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Function", "Best Position", "Best Value", "Execution Time",
            "Dimensions", "Num Particles", "Bounds", "Max Iterations",
            "Inertia (w)", "Cognitive (c1)", "Social (c2)"
        ])

    best_overall = {}

    for name, config in functions_config.items():
        print(f"\nStarting async optimization for function: {name}")
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

        tasks = [
            run_single_experiment_async(name, function, *params)
            for params in param_combinations
        ]

        # Async progress bar
        results = await tqdm.asyncio.tqdm.gather(*tasks, desc="Async Progress", ncols=100)

        # Track best result
        best_func_value = float('inf')
        best_result = None

        for result in results:
            if result["best_value"] < best_func_value:
                best_func_value = result["best_value"]
                best_result = result

        best_overall[name] = best_result


        # Save results to CSV
        for result in results:
            save_results(
                result["function_name"],
                result["best_position"],
                result["best_value"],
                result["exec_time"],
                result["dimensions"],
                result["num_particles"],
                result["bounds"],
                result["max_iterations"],
                result["w"],
                result["c1"],
                result["c2"]
            )

        print(f"Finished {name} | Best value: {best_result['best_value']:.6f} in {best_result['exec_time']:.4f}s")


    print("\n<<< Summary of Best Results >>>")
    for name, data in best_overall.items():
        print(f"\nFunction: {name}")
        print(f"  Best Value: {data['best_value']:.6f}")
        print(f"  Best Position: {data['best_position']}")
        print(f"  Dimensions: {data['dimensions']}, Particles: {data['num_particles']}, Iterations: {data['max_iterations']}")
        print(f"  Bounds: {data['bounds']}")
        print(f"  Inertia w: {data['w']}, c1: {data['c1']}, c2: {data['c2']}")
        print(f"  Execution Time: {data['exec_time']:.4f} seconds")

# Entry point
if __name__ == "__main__":
    start = time.time()
    asyncio.run(main_async())
    total_time = time.time() - start
    print(f"\nTotal async execution time: {total_time:.2f} seconds")
