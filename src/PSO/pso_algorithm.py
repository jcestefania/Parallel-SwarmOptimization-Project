from PSO.particle import Particle
import numpy as np
import time
from typing import Callable, Tuple, List, Union

class PSO:
    def __init__(
        self,
        num_particles: int,
        dimensions: int,
        bounds: Union[Tuple[float, float], List[Tuple[float, float]]],
        objective_function: Callable[[np.ndarray], float],
        # default values for PSO parameters stated in the official pyswarm library
        w: float = 0.5,
        c1: float = 0.5,
        c2: float = 0.5,
        max_iterations: int = 100
    ):
        """
        Initializes the Particle Swarm Optimization (PSO) algorithm.

        Parameters:
        - num_particles: Total number of particles in the swarm.
        - dimensions: Number of dimensions in the search space.
        - bounds: Tuple (min, max) or a list of tuples specifying the range for each dimension.
        - objective_function: Function to be minimized.
        - w: Inertia weight (default 0.5).
        - c1: Cognitive component weight (default 0.5).
        - c2: Social component weight (default 0.5).
        - max_iterations: Number of iterations to run the algorithm.

        Attributes:
        - particles: List of Particle instances.
        - global_best_position: Best position found across all particles.
        - global_best_value: Corresponding objective function value of the global best position.
        """
        self.num_particles: int = num_particles
        self.dimensions: int = dimensions
        self.bounds: Union[Tuple[float, float], List[Tuple[float, float]]] = bounds
        self.objective_function: Callable[[np.ndarray], float] = objective_function
        self.w: float = w
        self.c1: float = c1
        self.c2: float = c2
        self.max_iterations: int = max_iterations
        self.particles: List[Particle] = [Particle(dimensions, bounds) for _ in range(num_particles)]
        self.global_best_position: np.ndarray = np.copy(self.particles[0].best_position)
        self.global_best_value: float = float('inf')

    def optimize(self) -> Tuple[np.ndarray, float, float]:
        """
        Executes the PSO optimization loop.

        Process:
        - For each iteration, evaluates the objective function for each particle.
        - Updates personal best and global best positions if improvements are found.
        - Updates particle velocity and position based on bests.
        - Measures total execution time.

        Returns:
        - global_best_position: Best position found by the swarm.
        - global_best_value: Best objective function value.
        - execution_time: Total optimization time in seconds.
        """
        start_time = time.time()  # Start timer

        for _ in range(self.max_iterations):
            for particle in self.particles:
                value = self.objective_function(particle.position)

                # If the particle improves its personal best, update it
                if value < particle.best_value:
                    particle.best_value = value
                    particle.best_position = np.copy(particle.position)

                    # If this value is also better than the global best, update it
                    if value < self.global_best_value:
                        self.global_best_value = value
                        self.global_best_position = np.copy(particle.best_position)

                # Update velocity and position relative to global best
                particle.update_velocity(self.global_best_position, self.w, self.c1, self.c2)
                particle.update_position(self.bounds)

        end_time = time.time()  # End timer
        execution_time = end_time - start_time

        return self.global_best_position, self.global_best_value, execution_time
