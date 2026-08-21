"""Reproducible directions for the VMD-style random-ray classifier."""

from __future__ import annotations

import ctypes
import math
from functools import lru_cache

import numpy as np


VMD_RAND_MAX = 2**31 - 1
VMD_POISSON_SEED = 512346


def fibonacci_sphere(nrays: int) -> np.ndarray:
    """Deterministic, nearly uniform directions; useful for convergence tests."""

    if nrays < 1:
        raise ValueError("nrays must be positive")
    i = np.arange(nrays, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * i / nrays
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = (math.pi * (3.0 - math.sqrt(5.0))) * i
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta), z)).astype(np.float32)


def poisson_disk_sphere(nrays: int, seed: int = 103467829, candidates: int = 40) -> np.ndarray:
    """Generate a generic deterministic Poisson-like set on the unit sphere."""

    if nrays < 1:
        raise ValueError("nrays must be positive")
    if candidates < 1:
        raise ValueError("candidates must be positive")
    rng = np.random.default_rng(seed)
    target_angle = 0.72 * math.sqrt(4.0 * math.pi / nrays)
    min_dot = math.cos(target_angle)
    points: list[np.ndarray] = []
    max_trials = max(1000, nrays * candidates * 100)
    trials = 0
    while len(points) < nrays and trials < max_trials:
        trials += 1
        z = rng.uniform(-1.0, 1.0)
        a = rng.uniform(0.0, 2.0 * math.pi)
        r = math.sqrt(max(0.0, 1.0 - z * z))
        candidate = np.asarray((r * math.cos(a), r * math.sin(a), z), dtype=np.float32)
        if all(float(np.dot(candidate, point)) < min_dot for point in points):
            points.append(candidate)

    if len(points) < nrays:
        fallback = fibonacci_sphere(nrays)
        for point in fallback:
            if len(points) == nrays:
                break
            if all(float(np.dot(point, old)) < min_dot for old in points):
                points.append(point)
    if len(points) < nrays:
        points.extend(fibonacci_sphere(nrays - len(points)))
    return np.asarray(points[:nrays], dtype=np.float32)


def _vmd_random_functions(seed: int):
    """Return Linux libc ``random``/``srandom`` matching VMD's utilities.C."""

    try:
        libc = ctypes.CDLL(None)
        libc.srandom.argtypes = [ctypes.c_uint]
        libc.srandom.restype = None
        libc.random.argtypes = []
        libc.random.restype = ctypes.c_long
    except AttributeError:  # pragma: no cover - non-Unix fallback
        rng = np.random.RandomState(seed)

        def reseed(value: int) -> None:
            rng.seed(value)

        def draw() -> int:
            return int(rng.randint(0, VMD_RAND_MAX + 1))

        reseed(seed)
        return draw, reseed

    def reseed(value: int) -> None:
        libc.srandom(ctypes.c_uint(int(value)))

    def draw() -> int:
        return int(libc.random())

    reseed(seed)
    return draw, reseed


def _vmd_arc_distance(lambda1: np.float32, lambda2: np.float32,
                      phi1: np.float32, phi2: np.float32) -> float:
    """Single-precision equivalent of VMD's ``arcdistance``."""

    sl1 = np.float32(math.sin(float(lambda1)))
    cl1 = np.float32(math.cos(float(lambda1)))
    sl2 = np.float32(math.sin(float(lambda2)))
    cl2 = np.float32(math.cos(float(lambda2)))
    sp1 = np.float32(math.sin(float(phi1)))
    cp1 = np.float32(math.cos(float(phi1)))
    sp2 = np.float32(math.sin(float(phi2)))
    cp2 = np.float32(math.cos(float(phi2)))
    cos_a = np.float32(
        np.float32(np.float32(cl1 * sp1) * np.float32(cl2 * sp2))
        + np.float32(np.float32(sl1 * sp1) * np.float32(sl2 * sp2))
        + np.float32(cp1 * cp2)
    )
    return float(np.float32(math.acos(float(np.clip(cos_a, -1.0, 1.0)))))


def _vmd_correction(nrays: int) -> np.float32:
    n = np.float32(nrays)
    ans = np.sqrt(np.float32(8.0 * np.pi) / (n * np.float32(3.0 * np.sqrt(3.0))))
    min_d = np.sqrt(np.float32(ans * ans) + np.float32((np.pi / 6.0) * ans) ** 2)
    return np.float32(1.1275) * np.float32(min_d)


def _vmd_k_candidates(
    k: int,
    nrays: int,
    idx: int,
    testpt: int,
    min_d: np.float32,
    population: np.ndarray,
    random_int,
) -> tuple[int, np.ndarray]:
    rand_inv = np.float32(1.0 / VMD_RAND_MAX)
    lambda1 = np.float32(population[testpt, 0])
    phi1 = np.float32(population[testpt, 1])
    min_lambda = np.float32(2.0 * np.pi) * min_d
    min_phi = np.float32(np.pi) * min_d
    candidates = np.zeros((k, 2), dtype=np.float32)
    for i in range(k):
        dl = np.float32(lambda1 - np.float32(min_lambda + np.float32(rand_inv * random_int()) * min_lambda))
        dp = np.float32(phi1 - np.float32(min_phi + np.float32(rand_inv * random_int()) * min_phi))
        candidates[i, 0] = np.float32(lambda1 + dl)
        candidates[i, 1] = np.float32(phi1 + dp)

    best_idx = -1
    best_dist = 0.0
    for j in range(k):
        curr_dist = 0.0
        count = 0
        for jj in range(idx):
            dist = _vmd_arc_distance(
                population[jj, 0], candidates[j, 0],
                population[jj, 1], candidates[j, 1],
            )
            if np.float32(dist - float(min_d)) > np.float32(1.0e-8):
                curr_dist += dist
                count += 1
        if count == idx and curr_dist > best_dist:
            best_idx = j
            best_dist = curr_dist
    return best_idx, candidates


def _sphere_from_lambda_phi(population: np.ndarray) -> np.ndarray:
    dirs = np.empty((len(population), 3), dtype=np.float32)
    for i, (lam, phi) in enumerate(population):
        sin_phi = np.float32(math.sin(float(phi)))
        cos_phi = np.float32(math.cos(float(phi)))
        dirs[i, 0] = np.float32(math.cos(float(lam))) * sin_phi
        dirs[i, 1] = np.float32(math.sin(float(lam))) * sin_phi
        dirs[i, 2] = cos_phi
    return dirs


def vmd_poisson_disk_sphere(
    nrays: int,
    candidates: int = 40,
    seed: int = VMD_POISSON_SEED,
) -> np.ndarray:
    """Replicate VMD's ``poisson_sample_on_sphere`` direction generator.

    VMD currently seeds libc ``random`` with 512346 inside the sampler and
    stores N+1 angular samples, while the ray loop consumes the first N.
    """

    if nrays < 1:
        raise ValueError("nrays must be positive")
    if candidates < 1:
        raise ValueError("candidates must be positive")

    random_int, reseed = _vmd_random_functions(seed)
    rand_inv = np.float32(1.0 / VMD_RAND_MAX)
    min_d = _vmd_correction(nrays)
    population = np.zeros((nrays + 1, 2), dtype=np.float32)
    active = np.ones(nrays, dtype=np.int8)
    population[0, 0] = np.float32(rand_inv * random_int() * (2.0 * np.pi))
    population[0, 1] = np.float32(rand_inv * random_int() * np.pi)
    popul = 1
    testpt = 0
    numactive = nrays
    attempts = 0
    converged = False

    while not converged:
        while active[testpt] == 1:
            result, cand = _vmd_k_candidates(
                candidates, nrays, popul, testpt, min_d, population, random_int
            )
            if result != -1:
                population[popul] = cand[result]
                popul += 1
                if popul == nrays + 1:
                    converged = True
                    break
                testpt = random_int() % popul
            else:
                active[testpt] = 0
                numactive -= 1
                break

        if popul == nrays + 1:
            converged = True
            break
        if numactive >= 2:
            while active[testpt] != 1:
                testpt = random_int() % popul
        if numactive <= 2 and popul != nrays + 1:
            active[:] = 1
            popul = 1
            numactive = nrays
            testpt = 0
            attempts += 1
        elif popul == nrays + 1:
            converged = True
            break
        if attempts > 10:
            break

    if converged:
        return _sphere_from_lambda_phi(population[:nrays])

    # This is TclMeasure.C's fallback when the Poisson sampler fails.
    dirs = np.empty((nrays, 3), dtype=np.float32)
    for i in range(nrays):
        u1 = np.float32(rand_inv * random_int())
        u2 = np.float32(rand_inv * random_int())
        z = np.float32(2.0 * u1 - 1.0)
        phi = np.float32(2.0 * np.pi * u2)
        r = np.sqrt(np.float32(1.0 - z * z))
        dirs[i, 0] = np.float32(r * math.cos(float(phi)))
        dirs[i, 1] = np.float32(r * math.sin(float(phi)))
        dirs[i, 2] = z
    return dirs


@lru_cache(maxsize=32)
def _make_directions_cached(nrays: int, scheme: str, seed: int, candidates: int) -> np.ndarray:
    """Build one immutable-in-practice direction set per parameter tuple.

    The VMD-compatible sampler is deterministic for a fixed tuple but is
    intentionally CPU-heavy. Trajectory frames reuse the same set, so caching
    removes repeated generation without changing any direction values.
    """

    if scheme == "vmd_poisson":
        return vmd_poisson_disk_sphere(nrays, candidates=candidates, seed=seed)
    if scheme == "poisson":
        return poisson_disk_sphere(nrays, seed=seed, candidates=candidates)
    if scheme == "fibonacci":
        return fibonacci_sphere(nrays)
    raise ValueError(f"unknown ray scheme: {scheme}")


def make_directions(nrays: int, scheme: str, seed: int, candidates: int) -> np.ndarray:
    # Return a copy so callers cannot mutate the shared cached array.
    return _make_directions_cached(int(nrays), str(scheme), int(seed), int(candidates)).copy()
