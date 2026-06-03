"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable
def bisection(g,left,right,maxerr,maxiter=50):
    for itr in range(maxiter):
        mid=(left+right)/2
        g_mid=g(mid)
        if abs(g_mid)<=maxerr or (right-left)/2<maxerr:
            return mid
        if g(left)*g_mid<0:
            right=mid
        else:
            left=mid
    return (left+right)/2

'Helping func2 secant'
def secant(g,left,right,maxerr,maxiter=50):
    for i in range(maxiter):
        g_left=g(left)
        g_right=g(right)
        if abs(g_right)<=maxerr:return right
        if abs(g_left)<=maxerr:return left
        if g_left==g_right:return right
        x2=right-g_right*(right-left)/(g_right-g_left)
        left,right=right,x2
    return right


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.

        This function may not work correctly if there is infinite number of
        intersection points.


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """
        X=[]
        n=100
        'g is short of function like g(x)=x^2'
        g=lambda x:f1(x)-f2(x)
        step=(b-a)/n
        xs=[a+i*step for i in range(n+1)]
        for i in range(n):
            left=xs[i]
            right=xs[i+1]
            g_left=g(left)
            g_right=g(right)
            if g_left*g_right<=0:
                robust_root=bisection(g,left,right,maxerr)
                root=secant(g,robust_root-maxerr,robust_root+maxerr,maxerr)
                if abs(g(root))<=maxerr:
                    if not any(abs(x-root)<maxerr*10 for x in X):
                        X.append(root)
        return X


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


if __name__ == "__main__":
    unittest.main()
