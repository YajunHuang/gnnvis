"""
"""
import os
import struct
import random
import time
from pathlib import Path

import numpy as np
from scipy.special import expi
import scipy.sparse
from collections import defaultdict

from .base import BaseDataset
from .utils import download, extract_archive


__all__ =['PRIMEDataset']


def primesbelow(N):
    # http://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
    # """ Input N>=6, Returns a list of primes, 2 <= p < N """
    correction = N % 6 > 1
    N = {0: N, 1: N - 1, 2: N + 4, 3: N + 3, 4: N + 2, 5: N + 1}[N % 6]
    sieve = [True] * (N // 3)
    sieve[0] = False
    for i in range(int(N ** .5) // 3 + 1):
        if sieve[i]:
            k = (3 * i + 1) | 1
            sieve[k * k // 3::2 * k] = [False] * ((N // 6 - (k * k) // 6 - 1) // k + 1)
            sieve[(k * k + 4 * k - 2 * k * (i % 2)) // 3::2 * k] = [False] * (
                    (N // 6 - (k * k + 4 * k - 2 * k * (i % 2)) // 6 - 1) // k + 1)
    return [2, 3] + [(3 * i + 1) | 1 for i in range(1, N // 3 - correction) if sieve[i]]


def gcd(a, b):
    if a == b: return a
    while b > 0: a, b = b, a % b
    return a


# https://comeoncodeon.wordpress.com/2010/09/18/pollard-rho-brent-integer-factorization/
def pollard_brent(n):
    if n % 2 == 0: return 2
    if n % 3 == 0: return 3

    y, c, m = random.randint(1, n - 1), random.randint(1, n - 1), random.randint(1, n - 1)
    g, r, q = 1, 1, 1
    while g == 1:
        x = y
        for i in range(r):
            y = (pow(y, 2, n) + c) % n

        k = 0
        while k < r and g == 1:
            ys = y
            for i in range(min(m, r - k)):
                y = (pow(y, 2, n) + c) % n
                q = q * abs(x - y) % n
            g = gcd(q, n)
            k += m
        r *= 2
    if g == n:
        while True:
            ys = (pow(ys, 2, n) + c) % n
            g = gcd(abs(x - ys), n)
            if g > 1:
                break

    return g


class PRIMEGenerator(object):
    def __init__(self):
        self.smallprimes = primesbelow(10000000)
        self.prime_ix = {p: i for i, p in enumerate(self.smallprimes)}
        self._known_factors = {}
        self._smallprimeset = 1000000
        self.smallprimeset = set(primesbelow(1000000))
    
    def generate(self, N):
        t0 = time.time()
        X = self.factor_vector_lil(N)
        print("get X time: {:.4f}".format(time.time() - t0))
        return X.todense(), np.arange(N)
        
    def factor_vector_lil(self, N):
        ## approximate prime counting function (upper bound for the values we are interested in)
        ## gives us the number of rows (dimension of our space)
        d = int(np.ceil(expi(np.log(N))))
        x = scipy.sparse.lil_matrix((N, d))
        for i in range(2, N):
            for k, v in self.factorization(i).items():
                x[i, self.prime_ix[k]] = 1
    
            if i % 100000 == 0:  # just check it is still alive...
                print(i)
        return x
    
    def factorization(self, N):
        factors = defaultdict(int)
        for p1 in self.primefactors(N):
            factors[p1] += 1
        return factors
        
    def primefactors(self, N, sort=False):
        if N in self._known_factors:
            return self._known_factors[N]
    
        result = self._primefactors(N)
        self._known_factors[N] = result
        return result
    
    def _primefactors(self, n, sort=False):
        factors = []
    
        for checker in self.smallprimes:
            while n % checker == 0:
                factors.append(checker)
                n //= checker
                # early exit memoization
                if n in self._known_factors:
                    return factors + self._known_factors[n]
            if checker > n: break
    
        if n < 2: return factors
    
        while n > 1:
            if self.isprime(n):
                factors.append(n)
                break
            factor = pollard_brent(n)  # trial division did not fully factor, switch to pollard-brent
            factors.extend(
                primefactors(factor))  # recurse to factor the not necessarily prime factor returned by pollard-brent
            n //= factor
    
        if sort: factors.sort()
    
        return factors
    
    def isprime(self, n, precision=7):
        # http://en.wikipedia.org/wiki/Miller-Rabin_primality_test#Algorithm_and_running_time
        if n < 1:
            raise ValueError("Out of bounds, first argument must be > 0")
        elif n <= 3:
            return n >= 2
        elif n % 2 == 0:
            return False
        elif n < self._smallprimeset:
            return n in self.smallprimeset
    
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
    
        for repeat in range(precision):
            a = random.randrange(2, n - 2)
            x = pow(a, d, n)
    
            if x == 1 or x == n - 1: continue
    
            for r in range(s - 1):
                x = pow(x, 2, n)
                if x == 1: return False
                if x == n - 1: break
            else:
                return False
    
        return True





class PRIMEDataset(BaseDataset):
    """ PRIME dataset object.
    """
    
    def __init__(self, data_dir, n_samples=60000, k=30, split_rates=None):
        self.name = 'prime'
        super().__init__(data_dir, n_samples, k, split_rates)

    def _load_data(self):
        """ Download dataset and prepocess data to features, 
        labels and adjacency matrix of knn graph.
        """
        print(self.n_samples)
        pg = PRIMEGenerator()
        self.features, self.labels = pg.generate(self.n_samples)
        print(self.features.shape)
