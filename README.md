# Numerical_Analysis
A custom Numerical Analysis library built entirely from scratch in Python 3. It features algorithms for optimized interpolation, root finding, numerical integration, and advanced, noisy curve fitting.
NUMERICAL ANALYSIS COURSE PROJECT

OVERVIEW
This repository contains a comprehensive set of solutions for a Numerical Analysis course project. The codebase is written entirely in Python 3. The primary goal of this project is to implement core numerical methods from scratch, strictly avoiding the use of standard out-of-the-box library functions for root finding, interpolation, matrix decomposition, and integration. 

All implementations are rigorously optimized to balance two Key Performance Indicators (KPIs): minimal running time and maximum numerical accuracy.

--------------------------------------------------

ASSIGNMENTS BREAKDOWN

Assignment 1: Function Interpolation
- Objective: Interpolate a given continuous function 'f' within a closed range [a, b] using a maximum of 'n' points.
- Output: Returns a new callable function 'g' that represents the interpolation, aiming to minimize the average absolute error across random test points.
- Implementation Approach: To achieve a highly efficient O(n) running time complexity instead of the trivial O(n^2), the solution utilizes Chebyshev nodes and weights. This prevents Runge's phenomenon and ensures fast, accurate evaluation of the interpolated points.

Assignment 2: Finding Intersections
- Objective: Find approximate intersection points between two functions, 'f1' and 'f2'.
- Constraints: For each found intersection point 'x', the absolute difference must satisfy |f1(x) - f2(x)| < maxerr.
- Implementation Approach: The algorithm divides the given range into discrete steps to detect sign changes between the functions. Once a bracket is found, it uses a robust combination of the Bisection method (for safety) and the Secant method (for rapid convergence) to pinpoint the exact roots.

Assignment 3: Numerical Integration & Area Computation
- Objective 3.1 (Integrate): Approximate the definite integral of a function 'f' over a range [a, b] calling the function at most 'n' times. 
  * Approach: Implemented using Gauss-Legendre Quadrature for high precision. It features smart variable substitutions (e.g., u = 1/x) to successfully handle highly oscillatory functions.
- Objective 3.2 (Area Between): Calculate the exact enclosed area between two given functions 'f1' and 'f2'.
  * Approach: Automatically detects all intersection points within the range x in [1, 100], and integrates the absolute difference function segment by segment.

Assignment 4: Curve Fitting with Noisy Data
- Objective: Given a function that returns normally distributed noisy results, return a clean function 'g' that perfectly fits the underlying true data.
- Constraints: The function receives sampling boundaries 'a' and 'b', an expected polynomial degree 'd', and is strictly bound by a maximum allowed runtime 'maxtime'.
- Implementation Approach: 
  * Implements dynamic time-budgeting to sample as many points as possible without exceeding 'maxtime'.
  * Features an advanced heuristic classifier that detects the nature of the data (Polynomial, Gaussian, Exponential, or Oscillatory).
  * Uses Least Squares Fitting by solving Normal Equations manually via Gaussian Elimination, utilizing Chebyshev matrices for numerical stability.

Assignment 5: Geometric Shape Analysis
- Objective 5.1 (Contour Area): Calculate the approximate area of a shape given a contour function. 
  * Approach: Implements the Shoelace formula over adaptively sampled points, utilizing early stopping when the area converges within the allowed error margin to save running time.
- Objective 5.2 (Shape Fitting): Receive a generator yielding noisy coordinate samples and return an AbstractShape object (like a Circle or Polygon).
  * Approach: Since standard numerical optimization tools are permitted for this assignment, the algorithm uses Delaunay Triangulation (via SciPy) and highly optimized vectorized spatial binning. It automatically classifies shapes into Circles, elongated shapes, or generic polygons under strict time constraints.

--------------------------------------------------

TESTING & EVALUATION
The project includes internal unittest suites to ensure correctness. Grading policies dictate that each function is rigorously tested against various combinations of target errors and target execution times. Test cases include polynomials (up to 65% of test cases in Assignment 4) and complex non-polynomial functions (e.g., highly oscillatory math equations).
