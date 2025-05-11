# Parallel-SwarmOptimization-Project

**Parallel-SwarmOptimization-Project** is an optimization project based on the PSO (Particle Swarm Optimization) algorithm, developed to evaluate and compare different parallelization strategies in Python. It was created as part of a practical assignment for the Parallel Programming course.

## 🧠 Objective

The goal of the project is to compare the performance of various parallel execution models when running multiple parameter combinations of the PSO algorithm across several benchmark functions (`Sphere`, `Rastrigin`, `Ackley`).

The following PSO variants were implemented:

- `app.py`: sequential version.
- `app_threading.py`: multithreaded version using `ThreadPoolExecutor`.
- `app_multiprocessing.py`: multiprocess version using `multiprocessing`.
- `app_asyncio.py`: asynchronous version using `asyncio`.
- `app_multi_asyncio.py`: combined multiprocessing + asyncio version.

Each script generates its own CSV output and records its total execution time.

---

## ⚙️ Project Structure
```
pso_project/
│
├── src/
│ ├── app.py
│ ├── app_threading.py
│ ├── app_multiprocessing.py
│ ├── app_asyncio.py
│ ├── app_multi_asyncio.py
│ ├── PSO/
│ │ ├── pso_algorithm.py
│ │ └── particle.py
│ └── func/
│ └── objective_functions.py
│
├── analysis/
│ ├── results.csv
│ ├── results_threading.csv
│ ├── results_multiprocessing.csv
│ ├── results_asyncio.csv
│ ├── results_multi_async.csv
│ └── notebook.ipynb
│
├── requirements.txt
└── README.md
```
---

## 📦 Installation

This project requires Python 3.9 or later.

Install the dependencies with:

```bash
pip install -r requirements.txt
```

---

## 📊 Analysis

The notebook notebook.ipynb includes:

Execution of all algorithm variants from Jupyter.

A visual comparison of execution times.

Calculation of Speedup for each version relative to the sequential baseline.

A brief discussion on parallelization decisions, the GIL, and the impact of using different strategies in CPU-bound scenarios.

---

## 📝 Notes

The PSO algorithm is adapted from the pyswarm library.

Execution times may vary slightly depending on the system load.

In this project, the most efficient implementation was the Multi + Asyncio version, which combines multiprocessing with asynchronous task execution.

