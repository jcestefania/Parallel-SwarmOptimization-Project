import random
import numpy as np
from typing import Tuple

class Particle:
    def __init__(self, dimensions: int, bounds: Tuple[float, float]):
        """
        Initializes a particle in a PSO search space.
        
        Parameters:
        - dimensions: Number of dimensions in the search space.
        - bounds: Tuple (min, max) defining the same limits for each dimension.
        
        Attributes:
        - position: Initial position vector, randomly generated within bounds.
        - velocity: Initial velocity vector with random values between -1 and 1.
        - best_position: Best position found by the particle (initially its current position).
        - best_value: Best value found by the particle (initially set to infinity).
        """
        self.position: np.ndarray = np.array([random.uniform(bounds[0], bounds[1]) for _ in range(dimensions)])
        self.velocity: np.ndarray = np.array([random.uniform(-1, 1) for _ in range(dimensions)])
        self.best_position: np.ndarray = np.copy(self.position)
        self.best_value: float = float('inf')  # Initialized to a very high value (infinity)

    def update_velocity(self, global_best: np.ndarray, w: float, c1: float, c2: float) -> None:
        """
        Updates the particle's velocity using the PSO equation.
        
        Parameters:
        - global_best: Best global position found by the swarm.
        - w: Inertia factor that influences the previous velocity.
        - c1: Acceleration coefficient for the cognitive component (individual learning).
        - c2: Acceleration coefficient for the social component (collective learning).
        
        The new velocity is computed as:
        v = w * v + c1 * r1 * (best_position - position) + c2 * r2 * (global_best - position)
        where r1 and r2 are random values in [0, 1].
        """
        r1, r2 = random.random(), random.random()
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self, bounds: Tuple[float, float]) -> None:
        """
        Updates the particle's position based on its velocity, keeping values within bounds.
        
        Parameters:
        - bounds: Tuple (min, max) that defines the limits of the search space.
        
        The new position is obtained by adding the current velocity.
        Then it is clipped to ensure it remains within the valid range.
        """
        self.position += self.velocity
        self.position = np.clip(self.position, bounds[0], bounds[1])
