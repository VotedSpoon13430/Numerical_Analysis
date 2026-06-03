

import numpy as np


class Assignment3:
    def __init__(self):
        pass

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        if a == b or n <= 0:
            return np.float32(0.0)

        if a > b:
            a, b = b, a
            sign = -1
        else:
            sign = 1

        a, b = np.float64(a), np.float64(b)
        n = int(n)

        nodes, weights = np.polynomial.legendre.leggauss(n)

        # Hard oscillatory case: use inverse substitution u = 1/x
        if a > 1e-9 and a < 0.5 and b > 2.0:
            ua = 1.0 / b
            ub = 1.0 / a
            h = (ub - ua) / 2.0
            c = (ub + ua) / 2.0
            u = h * nodes + c
            x = 1.0 / u
            jac = 1.0 / (u * u)

            vals = np.array([f(float(p)) for p in x], dtype=np.float64)
            vals = np.nan_to_num(vals * jac, nan=0.0, posinf=0.0, neginf=0.0)

            result = h * np.sum(weights * vals)
            return np.float32(sign * result)

        # Mild case near zero: use t^2 substitution
        if a >= 0 and a < 1.0 and b > 1.0:
            ta = np.sqrt(max(a, 0))
            tb = np.sqrt(b)
            h = (tb - ta) / 2.0
            c = (tb + ta) / 2.0
            t = h * nodes + c
            x = t * t
            jac = 2.0 * t

            vals = np.array([f(float(p)) for p in x], dtype=np.float64)
            vals = np.nan_to_num(vals * jac, nan=0.0, posinf=0.0, neginf=0.0)

            result = h * np.sum(weights * vals)
            return np.float32(sign * result)

        # Regular Gauss-Legendre
        h = (b - a) / 2.0
        c = (a + b) / 2.0
        pts = h * nodes + c

        vals = np.array([f(float(p)) for p in pts], dtype=np.float64)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

        result = h * np.sum(weights * vals)
        return np.float32(sign * result)

    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        g = lambda x: f1(x) - f2(x)  # difference function
        roots = []
        xs = np.linspace(1, 100, 1000)
        ys = [g(x) for x in xs]  # sampled values
        for i in range(len(xs) - 1):
            y1, y2 = ys[i], ys[i + 1]  # consecutive values
            if y1 * y2 <= 0:  # sign change
                a, b = xs[i], xs[i + 1]  # bracket
                root = None
                if abs(y1) < 1e-9:
                    root = a  # exact root
                elif abs(y2) < 1e-9:
                    root = b
                else:
                    c = 0
                    while c < 50:  # bisection
                        mid = (a + b) / 2
                        fm = g(mid)
                        if abs(fm) < 1e-9 or (b - a) < 1e-9: break
                        if g(a) * fm < 0:
                            b = mid
                        else:
                            a = mid
                        c += 1
                    root = (a + b) / 2  # final root
                if not roots or abs(roots[-1] - root) > 1e-5:
                    roots.append(root)  # unique roots
        if len(roots) < 2: return np.float32(np.nan)  # no enclosed area
        area = 0.0
        i = 0
        while i < len(roots) - 1:
            area += abs(self.integrate(g, roots[i], roots[i + 1], 100))  # segment area
            i += 1
        return np.float32(area)


##########################################################################

import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment3(unittest.TestCase):

    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        r = ass3.integrate(f1, -1, 1, 10)
        self.assertEqual(r.dtype, np.float32)

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()
        r = ass3.integrate(f1, 0.09, 10, 20)
        true_result = -7.78662 * 10 ** 33
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

if __name__ == "__main__":
    unittest.main()