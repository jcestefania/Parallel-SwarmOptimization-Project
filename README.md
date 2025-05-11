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

- Execution of all algorithm variants from Jupyter.

- A visual comparison of execution times.

- Calculation of Speedup for each version relative to the sequential baseline.

- A brief discussion on parallelization decisions, the GIL, and the impact of using different strategies in CPU-bound scenarios.

---

## 📝 Notes

- Execution times may vary slightly depending on the system load.

- In this project, the most efficient implementation was the Multi + Asyncio version, which combines multiprocessing with asynchronous task execution.

---

## ▶️ How to Run

To run any of the PSO implementations manually, open a terminal and navigate to the `src` directory. Then, execute the desired script using Python. For example:

```bash
cd src
python app.py
```

You can replace app.py with any of the following:

`app_threading.py`

`app_multiprocessing.py`

`app_asyncio.py`

`app_multi_asyncio.py`

Each script will run the optimization process, print progress to the console, and generate a CSV file with the results in the analysis folder.

- Note: Running all combinations may take a few minutes depending on your system.

---