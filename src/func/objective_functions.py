import numpy as np
from typing import Sequence

def sphere_function(x: Sequence[float]) -> float:
    """
    Sphere function (minimization).
    
    f(x) = sum(x_i^2)
    Global minimum at x = 0

    Parameters:
    - x: Input vector.

    Returns:
    - Function value at x.
    """
    return sum(xi**2 for xi in x)

def rastrigin_function(x: Sequence[float]) -> float:
    """
    Rastrigin function (multimodal, many local minima).

    f(x) = 10n + sum[x_i^2 - 10*cos(2πx_i)]
    Global minimum at x = 0

    Parameters:
    - x: Input vector.

    Returns:
    - Function value at x.
    """
    n = len(x)
    return 10 * n + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)

def ackley_function(x: Sequence[float]) -> float:
    """
    Ackley function (complex multimodal function).

    Global minimum at x = 0

    Parameters:
    - x: Input vector.

    Returns:
    - Function value at x.
    """
    a, b, c = 20, 0.2, 2 * np.pi
    n = len(x)
    sum1 = sum(xi**2 for xi in x) / n
    sum2 = sum(np.cos(c * xi) for xi in x) / n
    return -a * np.exp(-b * np.sqrt(sum1)) - np.exp(sum2) + a + np.e
