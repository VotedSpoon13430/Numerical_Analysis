# Numerical_Analysis
A custom Numerical Analysis library built entirely from scratch in Python 3. It features algorithms for optimized interpolation, root finding, numerical integration, and advanced, noisy curve fitting.
# Numerical Analysis Course Project

## Overview
[cite_start]This repository contains a comprehensive set of solutions for a Numerical Analysis course project[cite: 1]. [cite_start]The codebase is written entirely in Python 3[cite: 3]. [cite_start]The primary goal of this project is to implement core numerical methods from scratch, strictly avoiding the use of standard out-of-the-box library functions for root finding, interpolation, matrix decomposition, and integration[cite: 7, 8]. 

[cite_start]All implementations are rigorously optimized to balance two Key Performance Indicators (KPIs): minimal running time and maximum numerical accuracy[cite: 26, 50].

---

## Assignments Breakdown

### Assignment 1: Function Interpolation
* [cite_start]**Objective:** Interpolate a given continuous function `f` within a closed range `[a, b]` using a maximum of `n` points[cite: 63, 64].
* [cite_start]**Output:** Returns a new callable function `g` that represents the interpolation, aiming to minimize the average absolute error across random test points[cite: 65, 70].
* [cite_start]**Implementation Approach:** To achieve a highly efficient $O(n)$ running time complexity instead of the trivial $O(n^2)$[cite: 68, 69], the solution utilizes **Chebyshev nodes** and weights. This prevents Runge's phenomenon and ensures fast, accurate evaluation of the interpolated points.

### Assignment 2: Finding Intersections
* [cite_start]**Objective:** Find approximate intersection points between two functions, `f1` and `f2`[cite: 77, 78].
* **Constraints:** For each found intersection point `x`, the absolute difference must satisfy $|f_1(x) - f_2(x)| [cite_start]< maxerr$[cite: 80].
* **Implementation Approach:** The algorithm divides the given range into discrete steps to detect sign changes between the functions. Once a bracket is found, it uses a robust combination of the **Bisection method** (for safety) and the **Secant method** (for rapid convergence) to pinpoint the exact roots.

### Assignment 3: Numerical Integration & Area Computation
* [cite_start]**Objective 3.1 (Integrate):** Approximate the definite integral of a function `f` over a range `[a, b]` calling the function at most `n` times[cite: 84, 85, 86]. 
  * *Approach:* Implemented using **Gauss-Legendre Quadrature** for high precision. It features smart variable substitutions (e.g., $u = 1/x$) to successfully handle highly oscillatory functions.
* [cite_start]**Objective 3.2 (Area Between):** Calculate the exact enclosed area between two given functions `f1` and `f2`[cite: 88, 89].
  * [cite_start]*Approach:* Automatically detects all intersection points within the range $x \in [1, 100]$[cite: 91, 92], and integrates the absolute difference function segment by segment.

### Assignment 4: Curve Fitting with Noisy Data
* [cite_start]**Objective:** Given a function that returns normally distributed noisy results, return a clean function `g` that perfectly fits the underlying true data[cite: 97, 98, 99].
* [cite_start]**Constraints:** The function receives sampling boundaries `a` and `b`, an expected polynomial degree `d`, and is strictly bound by a maximum allowed runtime `maxtime`[cite: 101, 102, 105].
* **Implementation Approach:** * Implements dynamic time-budgeting to sample as many points as possible without exceeding `maxtime`.
  * Features an advanced **heuristic classifier** that detects the nature of the data (Polynomial, Gaussian, Exponential, or Oscillatory).
  * Uses **Least Squares Fitting** by solving Normal Equations manually via Gaussian Elimination, utilizing Chebyshev matrices for numerical stability.

### Assignment 5: Geometric Shape Analysis
* [cite_start]**Objective 5.1 (Contour Area):** Calculate the approximate area of a shape given a contour function[cite: 111, 112]. 
  * [cite_start]*Approach:* Implements the **Shoelace formula** over adaptively sampled points, utilizing early stopping when the area converges within the allowed error margin to save running time[cite: 117].
* [cite_start]**Objective 5.2 (Shape Fitting):** Receive a generator yielding noisy coordinate samples and return an `AbstractShape` object (like a Circle or Polygon)[cite: 119, 120, 122].
  * [cite_start]*Approach:* Since standard numerical optimization tools are permitted for this assignment[cite: 125], the algorithm uses **Delaunay Triangulation** (via SciPy) and highly optimized vectorized spatial binning. [cite_start]It automatically classifies shapes into Circles, elongated shapes, or generic polygons under strict time constraints[cite: 124].

---

## Testing & Evaluation
The project includes internal `unittest` suites to ensure correctness. [cite_start]Grading policies dictate that each function is rigorously tested against various combinations of target errors and target execution times[cite: 21, 29]. [cite_start]Test cases include polynomials (up to 65% of test cases in Assignment 4) and complex non-polynomial functions (e.g., highly oscillatory math equations)[cite: 107, 108].
