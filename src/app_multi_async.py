import asyncio
import csv
import time
import random
import numpy as np
from typing import Callable, Dict, List, Tuple, Union
from multiprocessing import Process
from tqdm.asyncio import tqdm
from PSO.pso_algorithm import PSO
from func.objective_functions import sphere_function, rastrigin_function, ackley_function

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)

# Output CSV file
CSV_FILE = "../analysis/results_multi_async.csv"

# Central configuration of benchmark functions and bounds
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

# Parameter space
param_ranges = {
    "dimensions": [10, 20],
    "num_particles": [30, 50],
    "max_iterations": [200, 500],
    "w": np.linspace(0.4, 0.9, 3),
    "c1": np.linspace(0.5, 1.5, 3),
    "c2": np.linspace(0.3, 1.5, 3)
}

# Save a single result to CSV
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

# Run a single PSO optimization asynchronously
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

# Run all parameter combinations asynchronously (runs in a separate process)
async def async_optimize_function(name: str, config: Dict):
    print(f"\n[Process] Starting optimization for function: {name}")
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

    print(f"[Process] Total combinations for {name}: {len(param_combinations)}")

    tasks = [
        run_single_experiment_async(name, function, *params)
        for params in param_combinations
    ]

    # Use tqdm.asyncio for progress bar
    results = await tqdm.gather(*tasks, desc=f"{name} Progress", ncols=100)

    best_func_value = float("inf")
    best_result = None

    for result in results:
        if result["best_value"] < best_func_value:
            best_func_value = result["best_value"]
            best_result = result


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

    print(f"[Process] Finished {name} | Best value: {best_result['best_value']:.6f}")

# Wrapper to launch asyncio loop from a process
def run_process(name: str, config: Dict):
    asyncio.run(async_optimize_function(name, config))

# Main process entry point
if __name__ == "__main__":
    # Create CSV header
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Function", "Best Position", "Best Value", "Execution Time",
            "Dimensions", "Num Particles", "Bounds", "Max Iterations",
            "Inertia (w)", "Cognitive (c1)", "Social (c2)"
        ])

    # Global timer
    global_start = time.time()

    # Create and start a process per function
    processes = []
    for name, config in functions_config.items():
        p = Process(target=run_process, args=(name, config))
        p.start()
        processes.append(p)

    # Wait for all to finish
    for p in processes:
        p.join()

    print(f"\nTotal execution time: {time.time() - global_start:.2f} seconds")
