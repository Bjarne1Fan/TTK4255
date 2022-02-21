from ast import Call
from unittest.mock import CallableMixin
import numpy as np
from numpy import ndarray
from typing import Callable

def jacobian2point(
      resfun  : Callable, 
      p       : ndarray, 
      epsilon : float
    ):
  r = resfun(p)
  J = np.empty((len(r), len(p)))
  for j in range(len(p)):
    pj0 = p[j]
    p[j] = pj0 + epsilon
    rpos = resfun(p)
    p[j] = pj0 - epsilon
    rneg = resfun(p)
    p[j] = pj0
    J[:,j] = rpos - rneg
  return J / (2.0*epsilon)

def gauss_newton(
      resfun    : Callable, 
      jacfun    : Callable, 
      p0        : ndarray, 
      step_size : float, 
      tolerance : float,
      num_steps : int
    ):
  r = resfun(p0)
  J = jacfun(p0)
  p = p0.copy()
  p_prev = p
  for iteration in range(num_steps):
    A = J.T@J
    b = -J.T@r
    d = np.linalg.solve(A, b)
    p = p + step_size * d

    r = resfun(p)
    J = jacfun(p)

    if np.linalg.norm(p - p_prev) < tolerance:
      break
    p_prev = p
  return p
