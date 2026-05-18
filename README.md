# Machine Learning from Scratch: Matrix Primitives to Native Execution

A collection of foundational machine learning algorithms implemented entirely from scratch. This repository is developed using a strict, two-phase evolutionary design pattern to isolate mathematical prototyping from low-level infrastructure and memory architecture.

## Architectural Vision & Roadmap

To maximize understanding of both machine learning mathematics and high-performance computing, every algorithm in this repository undergoes a two-stage lifecycle:

1. **Phase 1: Mathematical Validation (Python + Strict 2D NumPy)**
   * Implementing core equations, loss tracking, and statistical estimators using clean vector abstractions.
   * **Rule of Design:** All vectors are strictly enforced as 2D matrices (`shape=(N, 1)` or `shape=(1, N)`) from day one. This keeps the execution linear and prevents dimension decay, smoothing the eventual transition to native code.

2. **Phase 2: High-Performance Infrastructure Swap (C++ + `pybind11`)**
   * Building a lightweight, proprietary BLAS (Basic Linear Algebra Subprograms) engine from raw primitives in C++ (`custom_numpy`).
   * Implementing explicit cache-aligned structures, row-major flat data vectors, operator overloading (`*` for matrix multiplication), and the **Rule of 5** for manual heap/move memory management.
   * Exposing the native backend to Python via `pybind11` using a zero-copy data pointer transfer interface, allowing a seamless drop-in replacement: changing `import numpy as np` to `import custom_numpy as np`.

---

## Repository Structure

The directory is flatly categorized by learning paradigms to optimize scannability and avoid multi-layered nested folder paths:

```text
ml-from-scratch/
├── supervised_learning/
│   ├── linear_regression.py      # Normal Equation solver with full diagnostics
│   └── tree_algorithms/          # (Coming Soon) Recursive pointer-based branching
│
├── unsupervised_learning/        # (Coming Soon) Clustering and dimensionality primitives
│
├── reinforcement_learning/       # (Coming Soon) Exact DP and model-free decision engines
│
└── deep_learning/                # (Coming Soon) Vectorized neural layers & backprop graph